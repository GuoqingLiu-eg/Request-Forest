from typing import Optional, Tuple

import torch

from sglang.srt.utils import is_cuda, is_sm90_supported, is_sm100_supported

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import (
        apply_shuffle_mul_sum,
        es_fp8_blockwise_scaled_grouped_mm,
        es_sm100_mxfp8_blockscaled_grouped_mm,
        es_sm100_mxfp8_blockscaled_grouped_quant,
        fp8_blockwise_scaled_grouped_mm,
        prepare_moe_input,
        shuffle_rows,
    )

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.shuffle_rows_with_scales import (
        shuffle_rows_with_scales,
    )


def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
    workspace: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    use_fp8_blockscale: bool = True,
    use_mxfp8: bool = False,
    output: Optional[torch.Tensor] = None,
    enable_es: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:

    assert use_fp8_blockscale, "Only support fp8 blockscale for now"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    if is_cuda:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
    es_up, es_down = enable_es
    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)
    device = a.device

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    if use_mxfp8:
        assert es_up and es_down, "MXFP8 requires expert-specialization for both GEMMs"
        assert is_sm100_supported(), "MXFP8 requires SM100"
        assert k % 32 == 0, "MXFP8 requires hidden size to be divisible by 32"
        assert n % 32 == 0, "MXFP8 requires intermediate size to be divisible by 32"
        assert w1_scale.dtype == torch.uint8, "MXFP8 w1_scale must be uint8"
        assert w2_scale.dtype == torch.uint8, "MXFP8 w2_scale must be uint8"
        expected_w1_scale_shape = (
            num_experts,
            w1_q.shape[1] // 32,
            w1_q.shape[2],
        )
        expected_w2_scale_shape = (
            num_experts,
            w2_q.shape[1] // 32,
            w2_q.shape[2],
        )
        assert (
            w1_scale.shape == expected_w1_scale_shape
        ), f"MXFP8 w1_scale must be {expected_w1_scale_shape}, got {w1_scale.shape}"
        assert (
            w2_scale.shape == expected_w2_scale_shape
        ), f"MXFP8 w2_scale must be {expected_w2_scale_shape}, got {w2_scale.shape}"

        mxfp8_blockscale_align = 128
        total_tokens = m * topk
        nonzero_experts = min(num_experts, total_tokens)
        max_total = total_tokens + (mxfp8_blockscale_align - 1) * nonzero_experts
        max_blockscale = (
            (max_total + mxfp8_blockscale_align - 1) // mxfp8_blockscale_align
        ) * mxfp8_blockscale_align

    blockscale_offsets = None
    if use_mxfp8 and (es_up or es_down):
        blockscale_offsets = torch.empty(
            (num_experts + 1,), dtype=torch.int32, device=device
        )

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
        blockscale_offsets,
    )

    if use_mxfp8 and es_up:
        rep_a = shuffle_rows(a, a_map, (m * topk, k))
        rep_a_q = torch.empty_like(rep_a, dtype=torch.float8_e4m3fn)
        rep_a1_scales = torch.empty(
            (max_blockscale, k // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            rep_a,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
            rep_a_q,
            rep_a1_scales,
        )
    else:
        a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)


        rep_a_q, rep_a1_scales = shuffle_rows_with_scales(
            a_q, a1_scale, a_map, m * topk
        )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    a_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    w_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)

    if is_sm90_supported() and es_up:
        es_fp8_blockwise_scaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            a1_strides,
            a1_strides,
            c1_strides,
            problem_sizes1,
            expert_offsets[:-1],
            workspace,
        )
    elif use_mxfp8 and es_up:
        es_sm100_mxfp8_blockscaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    else:
        fp8_blockwise_scaled_grouped_mm(
            c1,
            a_ptrs,
            b_ptrs,
            out_ptrs,
            a_scales_ptrs,
            b_scales_ptrs,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            a1_strides,
            a1_strides,
            c1_strides,
            a_sf_layout,
            w_sf_layout,
            problem_sizes1,
            expert_offsets[:-1],
            workspace,
        )

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(c1, intermediate)

    if use_mxfp8 and es_down:
        intemediate_q = torch.empty_like(intermediate, dtype=torch.float8_e4m3fn)
        a2_scale = torch.empty(
            (max_blockscale, n // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            intermediate,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
            intemediate_q,
            a2_scale,
        )
    else:
        intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)

    if is_sm90_supported() and es_down:
        es_fp8_blockwise_scaled_grouped_mm(
            c2,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            a2_strides,
            a2_strides,
            c2_strides,
            problem_sizes2,
            expert_offsets[:-1],
            workspace,
        )
    elif use_mxfp8 and es_down:
        es_sm100_mxfp8_blockscaled_grouped_mm(
            c2,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    else:
        fp8_blockwise_scaled_grouped_mm(
            c2,
            a_ptrs,
            b_ptrs,
            out_ptrs,
            a_scales_ptrs,
            b_scales_ptrs,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            a2_strides,
            a2_strides,
            c2_strides,
            a_sf_layout,
            w_sf_layout,
            problem_sizes2,
            expert_offsets[:-1],
            workspace,
        )

    if output is None:
        output = torch.empty((m, k), device=device, dtype=out_dtype)

    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(out_dtype))
    return output
