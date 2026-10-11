from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo, get_lm_head_lora_b_shard_size
from sglang.srt.runtime_context import get_parallel

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()


def unwrap_lora_layer(module: nn.Module) -> nn.Module:

    return module.base_layer if isinstance(module, BaseLayerWithLoRA) else module


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.bias = self.base_layer.bias


        if hasattr(self.base_layer, "reduce_results"):
            self.reduce_results = self.base_layer.reduce_results

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    @property
    def lora_active(self) -> bool:

        return self.set_lora and self.lora_backend.batch_info is not None

    def set_lora_info(self, *args):
        pass


    def slice_lora_a_weights(self, A: torch.Tensor):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):


    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size
        self.num_embeddings = base_layer.num_embeddings


        if hasattr(base_layer, "tp_size") and base_layer.tp_size > 1:
            from sglang.srt.layers.communicator import get_attn_tp_context

            assert (
                not get_attn_tp_context().allow_input_scattered
            ), "VocabParallelEmbeddingWithLoRA with TP > 1 under input_scattered mode (e.g., DeepSeek-v2 MLA with --enable-attn-tp-input-scattered) is not fully supported and may produce incorrect results. Consider disabling input_scattered or removing embed_tokens from LoRA target modules."
        offsets = [0, self.embed_dim]
        self.output_offset = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=next(base_layer.parameters()).device,
        )
        self.output_offset_cpu = torch.tensor(
            offsets,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

    def set_lora_info(
        self,
        new_embeddings_buffer: Optional[torch.Tensor],
        embedding_A_buffer: torch.Tensor,
        embedding_B_buffer: torch.Tensor,
    ):

        self.set_lora = True
        self.new_embeddings_buffer = new_embeddings_buffer
        self.embedding_A_buffer = embedding_A_buffer
        self.embedding_B_buffer = embedding_B_buffer

    def apply_lora(
        self, base_output: torch.Tensor, input_: torch.Tensor, batch_info
    ) -> torch.Tensor:


        lora_a_output = self.run_lora_a_embedding(input_, batch_info)


        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.embedding_B_buffer,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            base_output=base_output,
        )
        return lora_output

    def run_lora_a_embedding(
        self, input_: torch.Tensor, batch_info: LoRABatchInfo
    ) -> torch.Tensor:


        lora_a_output = self.lora_backend.run_lora_a_embedding(
            input_ids=input_,
            weights=self.embedding_A_buffer,
            vocab_size=self.vocab_size,
            extra_embeddings=(
                self.new_embeddings_buffer
                if hasattr(self, "new_embeddings_buffer")
                and self.new_embeddings_buffer is not None
                else None
            ),
        )

        return lora_a_output

    def extra_token_embedding(
        self, input_: torch.Tensor, base_output: torch.Tensor
    ) -> torch.Tensor:


        raise NotImplementedError(
            "Error in sglang/python/sglang/srt/lora/layers.py - VocabParallelEmbeddingWithLoRA \n"
            "Current SGLang codebase did not support tuned lora with extra/added tokens. \n"
            "[TODO]: \n"
            "1. Refer to this commit: https://github.com/yushengsu-thu/sglang/commit/90415211eee8a28a316de262583d4d33fa615d10#diff-191177438bcc223837963de63c005850371f8c8a860acb153b26744b66ecc623 to complete \n"
            "2. And then you need to modified the en/decoder tokenizer - tokenizer_manager.py to support extra_token_embedding in-place. \n"
        )

    def forward(self, input_: torch.Tensor):

        batch_info = self.lora_backend.batch_info


        added_tokens_mask = input_ > self.vocab_size - 1
        base_output = self.base_layer.forward(input_.masked_fill(added_tokens_mask, 0))


        if (
            hasattr(self, "new_embeddings_buffer")
            and self.new_embeddings_buffer is not None
        ):
            base_output = self.extra_token_embedding(input_, base_output)


        if self.lora_active:


            base_output = self.apply_lora(base_output, input_, batch_info)

        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor):


        return A

    def slice_lora_b_weights(self, B: torch.Tensor):


        return B


class ParallelLMHeadWithLoRA(BaseLayerWithLoRA):


    def __init__(
        self,
        base_layer: ParallelLMHead,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size

        offsets = [0, self.vocab_size]

        tp_size = base_layer.tp_size if hasattr(base_layer, "tp_size") else 1


        if tp_size > 1:
            from sglang.srt.layers.communicator import get_attn_tp_context

            if get_attn_tp_context().allow_input_scattered:
                raise ValueError(
                    "ParallelLMHeadWithLoRA is not compatible with "
                    "input_scattered mode (e.g., DeepSeek-v2 MLA with "
                    "--enable-attn-tp-input-scattered). Please disable "
                    "input_scattered or remove lm_head from LoRA "
                    "target modules."
                )

            self.shard_vocab_size = get_lm_head_lora_b_shard_size(
                self.vocab_size,
                shard_indices=base_layer.shard_indices,
            )
            offsets = [0, self.shard_vocab_size]

        self.output_offset = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=next(base_layer.parameters()).device,
        )
        self.output_offset_cpu = torch.tensor(
            offsets,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

    def set_lora_info(
        self,
        lm_head_A_buffer: torch.Tensor,
        lm_head_B_buffer: torch.Tensor,
    ):

        self.set_lora = True
        self.lm_head_A_buffer = lm_head_A_buffer
        self.lm_head_B_buffer = lm_head_B_buffer

    def _get_lm_head_batch_info(self, num_tokens: int):

        pass_idx = self.lora_backend._lm_head_pass_idx
        if (
            pass_idx is not None
            and self.lora_backend.lm_head_pass_batch_infos is not None
        ):
            batch_info = self.lora_backend.lm_head_pass_batch_infos[pass_idx]
        else:
            batch_info = self.lora_backend.lm_head_batch_info

        if batch_info is not None:
            if batch_info.use_cuda_graph:
                raise RuntimeError(
                    "lm_head LoRA with pruned batch info is not supported "
                    "under CUDA graph. lm_head pruning should only occur "
                    "during extend, which does not use CUDA graph."
                )
            if num_tokens != batch_info.expected_tokens:
                raise RuntimeError(
                    f"lm_head LoRA input token count mismatch: got "
                    f"{num_tokens} tokens but lm_head_batch_info expects "
                    f"{batch_info.expected_tokens}. This likely means "
                    f"a pruning step in LogitsProcessor._get_pruned_states is "
                    f"not reflected in get_lm_head_pruned_lens()."
                )

        return batch_info

    def apply_lora(
        self,
        base_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        lm_head_batch_info = self._get_lm_head_batch_info(hidden_states.shape[0])


        lora_a_output = self.lora_backend.run_lora_a_sgemm(
            hidden_states,
            self.lm_head_A_buffer,
            pruned_batch_info=lm_head_batch_info,
        )


        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lm_head_B_buffer,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            base_output=base_output,
            pruned_batch_info=lm_head_batch_info,
        )

        return lora_output

    def forward(self, hidden_states: torch.Tensor):

        base_output = F.linear(
            hidden_states, self.weight, bias=getattr(self.base_layer, "bias", None)
        )

        if self.lora_active:
            base_output = self.apply_lora(base_output, hidden_states)

        return base_output


    def set_lm_head_pass(self, pass_idx: int):

        self.lora_backend._lm_head_pass_idx = pass_idx

    def reset_lm_head_pass(self):

        self.lora_backend._lm_head_pass_idx = None

    def slice_lora_a_weights(self, A: torch.Tensor):


        return A

    def slice_lora_b_weights(self, B: torch.Tensor):


        tp_size = self.base_layer.tp_size if hasattr(self.base_layer, "tp_size") else 1
        if tp_size <= 1:
            return B
        start_idx = self.base_layer.shard_indices.org_vocab_start_index
        end_idx = self.base_layer.shard_indices.org_vocab_end_index
        return B[start_idx:end_idx, :]


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        shard_size = self.base_layer.output_partition_sizes[0]
        offsets = [0, shard_size]
        self.output_offset = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = torch.tensor(
            offsets,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor):

        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.lora_active:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor):
        local_tp_rank = self.base_layer.tp_rank
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = local_tp_rank * shard_size
        end_idx = (local_tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.n_slices = len(self.base_layer.output_partition_sizes)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer


        lora_n_slices = self._get_lora_n_slices()
        if lora_n_slices <= 0 or lora_n_slices > self.n_slices:
            raise ValueError(
                f"Invalid LoRA slice count {lora_n_slices} for "
                f"{self.n_slices} base output partitions."
            )
        partition_sizes = list(self.base_layer.output_partition_sizes[:lora_n_slices])
        offsets = [0]
        for ps in partition_sizes:
            offsets.append(offsets[-1] + ps)
        if offsets[-1] != B_buffer.shape[-2]:
            raise ValueError(
                f"LoRA B output dim {B_buffer.shape[-2]} does not match "
                f"base partition prefix dim {offsets[-1]} for {lora_n_slices} slices."
            )
        self.output_offset = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = self.output_offset.cpu().pin_memory()
        self.max_out_dim = max(partition_sizes)
        self.use_gate_up_lora = (
            lora_n_slices == 2 and partition_sizes[0] == partition_sizes[1]
        )

    def _get_lora_n_slices(self) -> int:

        lora_rank = self.B_buffer.shape[-1]
        if lora_rank == 0:
            return self.n_slices
        return self.A_buffer.shape[-2] // lora_rank

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_n_slices = self._get_lora_n_slices()
        if lora_n_slices == 2 and self.use_gate_up_lora:
            lora_output = self.lora_backend.run_gate_up_lora(
                x=x,
                gate_up_lora_a=self.A_buffer,
                gate_up_lora_b=self.B_buffer,
                output_offset=self.output_offset,
                output_offset_cpu=self.output_offset_cpu,
                base_output=base_output,
            )
        else:
            lora_output = self.lora_backend.run_qkv_lora(
                x=x,
                qkv_lora_a=self.A_buffer,
                qkv_lora_b=self.B_buffer,
                output_offset=self.output_offset,
                output_offset_cpu=self.output_offset_cpu,
                max_qkv_out_dim=self.max_out_dim,
                base_output=base_output,
                n_slices=lora_n_slices,
            )
        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor):
        local_tp_rank = self.base_layer.tp_rank
        partition_sizes = self.base_layer.output_partition_sizes
        output_sizes = self.base_layer.output_sizes
        slices = []
        offset = 0
        for full_size, part_size in zip(output_sizes, partition_sizes):
            start_idx = local_tp_rank * part_size
            end_idx = start_idx + part_size
            slices.append(B[offset + start_idx : offset + end_idx, :])
            offset += full_size
        return torch.concat(slices, dim=0)


class InklingQKVRLinearWithLoRA(MergedColumnParallelLinearWithLoRA):


    def slice_lora_b_weights(self, B: torch.Tensor):
        bl = self.base_layer
        tp_rank = bl.tp_rank
        hd, nkv, nh, dr, tp = (
            bl.inkling_head_dim,
            bl.inkling_num_kv_heads,
            bl.inkling_num_heads,
            bl.inkling_d_rel,
            bl.inkling_tp_size,
        )
        q_size, kv_size, r_size = hd * nh, hd * nkv, dr * nh
        q_off, k_off, v_off, r_off = 0, q_size, q_size + kv_size, q_size + 2 * kv_size
        q_per, r_per = q_size // tp, r_size // tp
        q = B[q_off + tp_rank * q_per : q_off + (tp_rank + 1) * q_per, :]
        r = B[r_off + tp_rank * r_per : r_off + (tp_rank + 1) * r_per, :]
        if tp > nkv:

            replicas = tp // nkv
            kv_idx = tp_rank // replicas
            k = B[k_off + kv_idx * hd : k_off + (kv_idx + 1) * hd, :]
            v = B[v_off + kv_idx * hd : v_off + (kv_idx + 1) * hd, :]
        else:
            kv_per = kv_size // tp
            k = B[k_off + tp_rank * kv_per : k_off + (tp_rank + 1) * kv_per, :]
            v = B[v_off + tp_rank * kv_per : v_off + (tp_rank + 1) * kv_per, :]
        return torch.concat([q, k, v, r], dim=0)


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        offsets = [
            0,
            q_proj_shard_size,
            q_proj_shard_size + kv_proj_shard_size,
            q_proj_shard_size + 2 * kv_proj_shard_size,
        ]
        self.output_offset = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = torch.tensor(
            offsets,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )


        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_qkv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_qkv = B_buffer_qkv

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer_qkv,
            qkv_lora_b=self.B_buffer_qkv,
            base_output=base_output,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            max_qkv_out_dim=self.max_qkv_out_dim,
        )

        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas
        local_tp_rank = base_layer.tp_rank

        q_start_idx = q_proj_shard_size * local_tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = local_tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        q_size = base_layer.output_sizes[0]
        k_size = base_layer.output_sizes[1] // num_kv_head_replicas
        B_q_shard = B[q_start_idx:q_end_idx, :]
        B_k_shard = B[q_size + kv_start_idx : q_size + kv_end_idx, :]
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]

        return torch.concat(
            (
                B_q_shard,
                B_k_shard,
                B_v_shard,
            ),
            dim=0,
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        output_size = self.base_layer.output_size
        offsets = [0, output_size]
        self.output_offset = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = torch.tensor(
            offsets,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor, skip_all_reduce=False, forward_batch=None):
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_parallel().tp_rank
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()

        bias_ = (
            None
            if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)
            else self.base_layer.bias
        )
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel, bias=bias_
        )

        should_reduce = (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
            and not should_skip_mlp_all_reduce()
        )


        if self.base_layer.use_dp_attention_reduce:
            all_reduce = get_parallel().attn_tp_group.all_reduce
        else:
            all_reduce = tensor_model_parallel_all_reduce
        lora_active = self.lora_active
        if lora_active and should_reduce:
            lora_a_output = self.lora_backend.run_lora_a_sgemm(
                input_parallel, self.A_buffer
            )
            output_ = all_reduce(output_parallel)
            lora_a_output = all_reduce(lora_a_output)
            output_ = self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.B_buffer,
                output_offset=self.output_offset,
                output_offset_cpu=self.output_offset_cpu,
                base_output=output_,
            )
        else:
            if lora_active:
                output_parallel = self.apply_lora(output_parallel, input_parallel)
            if should_reduce:
                output_ = all_reduce(output_parallel)
            else:
                output_ = output_parallel

        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output_, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor):
        local_tp_rank = self.base_layer.tp_rank
        shard_size = self.base_layer.input_size_per_partition
        start_idx = local_tp_rank * shard_size
        end_idx = (local_tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor):
        return B


class ReplicatedLinearWithLoRA(BaseLayerWithLoRA):


    first_output_dim: int = 0

    def __init__(
        self,
        base_layer: ReplicatedLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.output_size = base_layer.output_size

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        first_dim = self.first_output_dim
        if first_dim > 0:
            second_dim = B_buffer.shape[-2] - first_dim
            self._output_offset = torch.tensor(
                [0, first_dim, first_dim + second_dim],
                dtype=torch.int32,
                device=B_buffer.device,
            )
            self._output_offset_cpu = self._output_offset.cpu()
            self._max_out_dim = max(first_dim, second_dim)
        else:


            self._output_offset = torch.tensor(
                [0, B_buffer.shape[-2]],
                dtype=torch.int32,
                device=B_buffer.device,
            )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        first_dim = self.first_output_dim

        if first_dim == 0:

            lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
            lora_output = self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.B_buffer,
                output_offset=self._output_offset,
                base_output=base_output,
            )
            return lora_output


        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer,
            qkv_lora_b=self.B_buffer,
            output_offset=self._output_offset,
            output_offset_cpu=self._output_offset_cpu,
            max_qkv_out_dim=self._max_out_dim,
            base_output=base_output,
            n_slices=2,
        )
        return lora_output

    def forward(self, x: torch.Tensor):
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        if self.lora_active:
            output = self.apply_lora(output, x)
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor):
        return B


class FusedMoEWithLoRA(BaseLayerWithLoRA):


    def __init__(
        self,
        base_layer: FusedMoE,
        lora_backend: BaseLoRABackend,
    ):

        super().__init__(base_layer, lora_backend)

        lora_backend.is_moe_lora = True

        self.experts_shared_outer_loras: bool = False
        self.lora_use_virtual_experts: bool = False
        self.quant_method = base_layer.quant_method
        self.moe_runner_config = base_layer.moe_runner_config


        self.moe_runner_config.inplace = False
        self.dispatcher = base_layer.dispatcher
        self.num_local_experts = base_layer.num_local_experts
        self.should_fuse_routed_scaling_factor_in_topk = (
            base_layer.should_fuse_routed_scaling_factor_in_topk
        )

        self.tp_size = getattr(base_layer, "moe_tp_size", 1)
        self.tp_rank = getattr(base_layer, "moe_tp_rank", 0)
        self.intermediate_size_per_partition = getattr(
            base_layer, "intermediate_size_per_partition", None
        )
        self._uses_interleaved_gate_up = (
            getattr(base_layer.moe_runner_config, "gemm1_alpha", None) is not None
        )


        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.utils import get_moe_runner_backend


        global_backend = get_moe_runner_backend()
        if not global_backend.is_auto():
            runner_backend = global_backend
        elif (
            hasattr(base_layer.quant_method, "runner")
            and base_layer.quant_method.runner is not None
        ):
            runner_backend = base_layer.quant_method.runner.runner_backend
        else:
            runner_backend = MoeRunnerBackend.TRITON


        if runner_backend.is_marlin():
            from sglang.srt.layers.quantization.unquant import (
                UnquantizedFusedMoEMethod,
            )

            if isinstance(base_layer.quant_method, UnquantizedFusedMoEMethod):
                runner_backend = MoeRunnerBackend.TRITON


        self._lora_runner_backend = runner_backend
        if runner_backend.is_experimental_sgl_trtllm():
            from sglang.srt.lora.trtllm_lora_temp.lora_layer import (
                init_experimental_sgl_trtllm_lora,
            )

            init_experimental_sgl_trtllm_lora(self, base_layer)
            return
        if runner_backend.is_experimental_sgl_marlin():
            from sglang.srt.lora.marlin_lora_temp.lora_layer import (
                init_experimental_sgl_marlin_lora,
            )

            init_experimental_sgl_marlin_lora(self, base_layer)
            return


        self._lora_runner = MoeRunner(
            runner_backend,
            base_layer.moe_runner_config,
            lora_enabled=True,
        )

        if runner_backend.is_marlin():
            from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsFusedMoEMethod,
            )
            from sglang.srt.layers.quantization.modelopt_quant import (
                ModelOptNvFp4FusedMoEMethod,
            )

            assert isinstance(
                base_layer.quant_method,
                (CompressedTensorsFusedMoEMethod, ModelOptNvFp4FusedMoEMethod),
            ), (
                f"Marlin MoE backend requires a quant method exposing "
                f"get_marlin_quant_info, got {type(base_layer.quant_method).__name__}"
            )
            self._quant_info = base_layer.quant_method.get_marlin_quant_info(base_layer)
        elif runner_backend.is_triton():
            assert base_layer.quant_method is not None, "Quant method must be set"
            self._quant_info = base_layer.quant_method.get_triton_quant_info(base_layer)
        else:
            raise NotImplementedError(
                f"LoRA MoE not supported for backend {runner_backend}"
            )

    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,
        gate_up_lora_b_weights: torch.Tensor,
        down_lora_a_weights: torch.Tensor = None,
        down_lora_b_weights: torch.Tensor = None,
    ):

        self.set_lora = True
        self.gate_up_lora_a_weights = gate_up_lora_a_weights
        self.gate_up_lora_b_weights = gate_up_lora_b_weights
        self.down_lora_a_weights = down_lora_a_weights
        self.down_lora_b_weights = down_lora_b_weights

    def _get_lora_info(self):

        from sglang.srt.lora.lora_moe_runners import LoRAInfo

        batch_info = self.lora_backend.batch_info

        lora_ranks = batch_info.lora_ranks
        max_lora_rank = self.down_lora_a_weights.shape[2]
        cg_buffers = getattr(self.lora_backend, "moe_cg_buffers", None)
        moe_lora_info = batch_info.moe_lora_info
        assert moe_lora_info is not None


        has_active_lora = bool(getattr(batch_info, "has_active_lora", False))

        if self._lora_runner_backend.is_experimental_sgl_trtllm() or (
            self._lora_runner_backend.is_experimental_sgl_marlin()
        ):
            num_experts = (
                self.down_lora_a_weights.shape[1]
                if self.down_lora_a_weights is not None
                else self.base_layer.num_local_experts
            )
        else:
            num_experts = self.base_layer.num_experts

        return LoRAInfo(
            gate_up_lora_a_weights=self.gate_up_lora_a_weights,
            gate_up_lora_b_weights=self.gate_up_lora_b_weights,
            down_lora_a_weights=self.down_lora_a_weights,
            down_lora_b_weights=self.down_lora_b_weights,
            seg_indptr=moe_lora_info.seg_indptr,
            req_to_lora=moe_lora_info.req_to_lora,
            lora_ranks=lora_ranks,
            adapter_enabled=moe_lora_info.adapter_enabled,
            token_lora_mapping=moe_lora_info.token_lora_mapping,
            max_lora_rank=max_lora_rank,
            num_experts=num_experts,
            has_active_lora=has_active_lora,
            experts_shared_outer_loras=self.experts_shared_outer_loras,
            cg_buffers=cg_buffers,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            hidden_size=getattr(self.base_layer, "hidden_size", 0),
            lora_use_virtual_experts=self.lora_use_virtual_experts,
        )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):


        if self.lora_backend.batch_info is None:
            return self.base_layer.forward(hidden_states, topk_output, **kwargs)


        lora_info = self._get_lora_info()


        return self._forward_with_lora(hidden_states, topk_output, lora_info, **kwargs)

    def _forward_with_lora(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        lora_info,
        **kwargs,
    ):


        base_layer = self.base_layer


        dispatch_output = base_layer.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )


        quant_info = self._quant_info


        if self._lora_runner_backend.is_experimental_sgl_trtllm():
            from sglang.srt.lora.trtllm_lora_temp.lora_layer import (
                dispatch_experimental_sgl_trtllm_lora,
            )

            combine_input = dispatch_experimental_sgl_trtllm_lora(
                dispatch_output, quant_info, base_layer, lora_info
            )
        elif self._lora_runner_backend.is_experimental_sgl_marlin():
            from sglang.srt.lora.marlin_lora_temp.lora_layer import (
                dispatch_experimental_sgl_marlin_lora,
            )

            combine_input = dispatch_experimental_sgl_marlin_lora(
                dispatch_output,
                quant_info,
                base_layer,
                lora_info,
            )

        else:
            combine_input = self._lora_runner.run(
                dispatch_output, quant_info, lora_info=lora_info
            )

        final_hidden_states = base_layer.dispatcher.combine(combine_input=combine_input)

        return final_hidden_states

    def slice_lora_a_weights(self, A: torch.Tensor):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor):
        return B

    def slice_moe_lora_a_weights(
        self,
        A: Union[torch.Tensor, Dict[int, torch.Tensor]],
        tp_rank: int,
        target_module: str,
    ):

        if self.tp_size <= 1:
            return A
        if target_module != "down_proj_moe":
            return A
        if isinstance(A, dict):
            return {
                eid: self._slice_moe_a(w, tp_rank, target_module)
                for eid, w in A.items()
            }
        return self._slice_moe_a(A, tp_rank, target_module)

    def _slice_moe_a(
        self, A: torch.Tensor, tp_rank: int, target_module: str
    ) -> torch.Tensor:
        shard_size = self.intermediate_size_per_partition
        start = tp_rank * shard_size
        end = start + shard_size
        return A[..., start:end].contiguous()

    def slice_moe_lora_b_weights(
        self,
        B: Union[torch.Tensor, Dict[int, torch.Tensor]],
        tp_rank: int,
        target_module: str,
    ):

        needs_processing = (self.tp_size > 1) or (
            target_module == "gate_up_proj_moe" and self._uses_interleaved_gate_up
        )
        if not needs_processing:
            return B
        if target_module != "gate_up_proj_moe":
            return B
        if isinstance(B, dict):
            return {
                eid: self._slice_moe_b_2d(w, tp_rank, target_module)
                for eid, w in B.items()
            }
        if isinstance(B, torch.Tensor) and B.dim() == 3:
            return torch.stack(
                [
                    self._slice_moe_b_2d(B[i], tp_rank, target_module)
                    for i in range(B.shape[0])
                ]
            )
        return self._slice_moe_b_2d(B, tp_rank, target_module)

    def _slice_moe_b_2d(
        self, B: torch.Tensor, tp_rank: int, target_module: str
    ) -> torch.Tensor:
        if target_module == "gate_up_proj_moe":


            is_gated = self.base_layer.moe_runner_config.is_gated
            if not is_gated:
                if self.tp_size > 1:
                    shard_size = self.intermediate_size_per_partition
                    start = tp_rank * shard_size
                    end = start + shard_size
                    return B[start:end, :]
                return B

            shard_size = self.intermediate_size_per_partition
            start = tp_rank * shard_size
            end = start + shard_size
            full_inter = B.shape[0] // 2
            gate_b = B[start:end, :]
            up_b = B[full_inter + start : full_inter + end, :]
            if self._uses_interleaved_gate_up:
                return torch.stack([gate_b, up_b], dim=1).reshape(-1, B.shape[-1])
            return torch.cat([gate_b, up_b], dim=0).contiguous()
        return B


def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    supported_layer_types = {

        FusedMoE: FusedMoEWithLoRA,
        ParallelLMHead: ParallelLMHeadWithLoRA,
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        ReplicatedLinear: ReplicatedLinearWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }


    if getattr(layer, "is_inkling_qkvr", False):
        return InklingQKVRLinearWithLoRA(layer, lora_backend)
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")


if _SGLANG_EXPERIMENTAL_LORA_OPTI:
    from sglang.srt.lora.trtllm_lora_temp import (
        install_two_stream_overrides as _install_lora_two_stream,
    )

    _install_lora_two_stream()
