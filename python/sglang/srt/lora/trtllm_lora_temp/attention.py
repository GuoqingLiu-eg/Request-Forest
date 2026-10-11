import torch

from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
from sglang.srt.lora.trtllm_lora_temp import (
    get_lora_side_stream,
    get_original_column_forward,
    get_original_qkv_forward,
    get_original_replicated_forward,
    get_original_row_forward,
    is_two_stream_active,
    lora_overlap_alloc_stream,
    supports_two_stream_dense_lora,
)
from sglang.srt.runtime_context import get_parallel


def qkv_proj_lora_forward(self, input_: torch.Tensor):

    if (
        not self.lora_active
        or not is_two_stream_active(input_)
        or not supports_two_stream_dense_lora(self.A_buffer_qkv, self.B_buffer_qkv)
    ):
        return get_original_qkv_forward()(self, input_)

    from sglang.kernels.ops.gemm.trtllm_lora_temp.qkv_lora_b import qkv_lora_b_fwd
    from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a import sgemm_lora_a_fwd

    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    side_stream = get_lora_side_stream()

    sgemm_info = self.lora_backend._sgemm_info()

    _alloc = lora_overlap_alloc_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        shrink_intermediate = sgemm_lora_a_fwd(
            input_, self.A_buffer_qkv, sgemm_info, stack_num=3, out_alloc_stream=_alloc
        )


    output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_, bias)


    torch.cuda.current_stream().wait_stream(side_stream)
    output_parallel = qkv_lora_b_fwd(
        shrink_intermediate,
        self.B_buffer_qkv,
        sgemm_info,
        self.output_offset,
        self.max_qkv_out_dim,
        output_parallel,
        n_slices=3,
    )

    if self.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias


def row_parallel_lora_forward(
    self, input_: torch.Tensor, skip_all_reduce: bool = False, forward_batch=None
):


    if self.base_layer.input_is_parallel:
        input_parallel = input_
    else:
        tp_rank = get_parallel().tp_rank
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.base_layer.tp_size
        )
        input_parallel = splitted_input[tp_rank].contiguous()

    if (
        not self.lora_active
        or not is_two_stream_active(input_parallel)
        or not supports_two_stream_dense_lora(self.A_buffer, self.B_buffer)
    ):
        return get_original_row_forward()(self, input_, skip_all_reduce, forward_batch)

    bias_ = (
        None
        if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)
        else self.base_layer.bias
    )

    side_stream = get_lora_side_stream()
    sgemm_info = self.lora_backend._sgemm_info()
    _alloc = lora_overlap_alloc_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a import (
            sgemm_lora_a_fwd,
        )
        from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_b import (
            sgemm_lora_b_fwd,
        )

        lora_a_output = sgemm_lora_a_fwd(
            input_parallel,
            self.A_buffer,
            sgemm_info,
            out_alloc_stream=_alloc,
        )


    output_parallel = self.base_layer.quant_method.apply(
        self.base_layer, input_parallel, bias=bias_
    )

    torch.cuda.current_stream().wait_stream(side_stream)

    should_reduce = (
        self.base_layer.reduce_results
        and self.base_layer.tp_size > 1
        and not skip_all_reduce
        and not should_skip_mlp_all_reduce()
    )

    if should_reduce:
        output_ = tensor_model_parallel_all_reduce(output_parallel)
        lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
        output_ = sgemm_lora_b_fwd(lora_a_output, self.B_buffer, sgemm_info, output_)
    else:


        output_parallel = sgemm_lora_b_fwd(
            lora_a_output, self.B_buffer, sgemm_info, output_parallel
        )
        output_ = output_parallel

    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output_, output_bias


def column_parallel_lora_forward(self, input_: torch.Tensor):

    if (
        not self.lora_active
        or not is_two_stream_active(input_)
        or not supports_two_stream_dense_lora(self.A_buffer, self.B_buffer)
    ):
        return get_original_column_forward()(self, input_)

    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    side_stream = get_lora_side_stream()
    sgemm_info = self.lora_backend._sgemm_info()

    _alloc = lora_overlap_alloc_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a import (
            sgemm_lora_a_fwd,
        )
        from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_b import (
            sgemm_lora_b_fwd,
        )

        lora_a_output = sgemm_lora_a_fwd(
            input_,
            self.A_buffer,
            sgemm_info,
            out_alloc_stream=_alloc,
        )


    output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_, bias)


    torch.cuda.current_stream().wait_stream(side_stream)
    output_parallel = sgemm_lora_b_fwd(
        lora_a_output, self.B_buffer, sgemm_info, output_parallel
    )

    if self.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias


def replicated_lora_forward(self, x: torch.Tensor):

    if (
        not self.lora_active
        or not is_two_stream_active(x)
        or not supports_two_stream_dense_lora(self.A_buffer, self.B_buffer)
    ):
        return get_original_replicated_forward()(self, x)

    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    side_stream = get_lora_side_stream()
    first_dim = self.first_output_dim
    sgemm_info = self.lora_backend._sgemm_info()

    _alloc = lora_overlap_alloc_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_a import (
            sgemm_lora_a_fwd,
        )
        from sglang.kernels.ops.gemm.trtllm_lora_temp.sgemm_lora_b import (
            sgemm_lora_b_fwd,
        )

        lora_a_output = sgemm_lora_a_fwd(
            x,
            self.A_buffer,
            sgemm_info,
            stack_num=(2 if first_dim > 0 else 1),
            out_alloc_stream=_alloc,
        )


    output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

    torch.cuda.current_stream().wait_stream(side_stream)
    if first_dim == 0:
        output = sgemm_lora_b_fwd(lora_a_output, self.B_buffer, sgemm_info, output)
    else:
        from sglang.kernels.ops.gemm.trtllm_lora_temp.qkv_lora_b import qkv_lora_b_fwd

        output = qkv_lora_b_fwd(
            lora_a_output,
            self.B_buffer,
            sgemm_info,
            self._output_offset,
            self._max_out_dim,
            output,
            n_slices=2,
        )
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias
