import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import safe_exp
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
)
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd

_MERGE_DOT_PRECISION = tl.constexpr("ieee")


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [16, 32, 64]
        for num_warps in [2, 4, 8, 16, 32]
    ],
    key=["H", "Hg", "K", "BC", "BK", "USE_G", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel_low_reg(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):

    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]


    for i_b in tl.static_range(4):
        i_tci = i_tc0 + i_b * BC
        m_tci = (i_tci + o_i) < T
        p_bi = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
        )
        b_bi = tl.load(p_bi, boundary_check=(0,)).to(tl.float32)
        if USE_G:
            p_gi = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
            )
            b_gi = tl.load(p_gi, boundary_check=(0,)).to(tl.float32)
        b_A = tl.zeros([BC, BC], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_tci, i_k * BK), (BC, BK), (1, 0)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))
        if USE_G:
            b_A *= safe_exp(b_gi[:, None] - b_gi[None, :])
        b_A = (
            tl.where(m_d & (m_tci[:, None] & m_tci[None, :]), b_A, 0.0) * b_bi[:, None]
        )


        b_Ai = -b_A
        for i in range(2, BC):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        p_Aii = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_tci, i_b * BC), (BC, BC), (1, 0)
        )
        tl.store(p_Aii, b_Ai.to(A.dtype.element_ty), boundary_check=(0, 1))


    for d in tl.static_range(1, 4):
        for j in tl.static_range(0, 4 - d):
            i = j + d
            i_tci = i_tc0 + i * BC
            i_tcj = i_tc0 + j * BC
            p_bi = tl.make_block_ptr(
                beta + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
            )
            b_bi = tl.load(p_bi, boundary_check=(0,)).to(tl.float32)
            if USE_G:
                p_gi = tl.make_block_ptr(
                    g + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
                )
                p_gj = tl.make_block_ptr(
                    g + bos * H + i_h, (T,), (H,), (i_tcj,), (BC,), (0,)
                )
                b_gi = tl.load(p_gi, boundary_check=(0,)).to(tl.float32)
                b_gj = tl.load(p_gj, boundary_check=(0,)).to(tl.float32)

            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_ki = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tci, i_k * BK), (BC, BK), (1, 0)
                )
                p_kj = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tcj, i_k * BK), (BC, BK), (1, 0)
                )
                b_A += tl.dot(
                    tl.load(p_ki, boundary_check=(0, 1)),
                    tl.trans(tl.load(p_kj, boundary_check=(0, 1))),
                )
            if USE_G:
                b_A *= safe_exp(b_gi[:, None] - b_gj[None, :])
            b_A *= b_bi[:, None]


            if i < 3:
                sc = 1 if (i == 1 and j == 0) else (2 if (i == 2 and j == 1) else 3)
                p_s = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tc0, sc * BC), (BC, BC), (1, 0)
                )
                tl.store(p_s, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


            b_corr = tl.zeros([BC, BC], dtype=tl.float32)
            if d >= 2:
                m1 = j + 1
                sc_m1j = (
                    1 if (m1 == 1 and j == 0) else (2 if (m1 == 2 and j == 1) else 3)
                )
                p_s_m1j = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tc0, sc_m1j * BC), (BC, BC), (1, 0)
                )
                b_A_m1j = tl.load(p_s_m1j, boundary_check=(0, 1)).to(tl.float32)
                p_Ai_im1 = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tci, m1 * BC), (BC, BC), (1, 0)
                )
                b_corr += tl.dot(
                    tl.load(p_Ai_im1, boundary_check=(0, 1)).to(tl.float32),
                    b_A_m1j,
                    input_precision=_MERGE_DOT_PRECISION,
                )
            if d >= 3:
                m2 = j + 2
                sc_m2j = (
                    1 if (m2 == 1 and j == 0) else (2 if (m2 == 2 and j == 1) else 3)
                )
                p_s_m2j = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tc0, sc_m2j * BC), (BC, BC), (1, 0)
                )
                b_A_m2j = tl.load(p_s_m2j, boundary_check=(0, 1)).to(tl.float32)
                p_Ai_im2 = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tci, m2 * BC), (BC, BC), (1, 0)
                )
                b_corr += tl.dot(
                    tl.load(p_Ai_im2, boundary_check=(0, 1)).to(tl.float32),
                    b_A_m2j,
                    input_precision=_MERGE_DOT_PRECISION,
                )

            p_Ai_ii = tl.make_block_ptr(
                A, (T, BT), (H * BT, 1), (i_tci, i * BC), (BC, BC), (1, 0)
            )
            p_Ai_jj = tl.make_block_ptr(
                A, (T, BT), (H * BT, 1), (i_tcj, j * BC), (BC, BC), (1, 0)
            )
            b_Ai_ii = tl.load(p_Ai_ii, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_jj = tl.load(p_Ai_jj, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_ij = -tl.dot(
                tl.dot(b_Ai_ii, b_A, input_precision=_MERGE_DOT_PRECISION) + b_corr,
                b_Ai_jj,
                input_precision=_MERGE_DOT_PRECISION,
            )
            p_Ai_ij = tl.make_block_ptr(
                A, (T, BT), (H * BT, 1), (i_tci, j * BC), (BC, BC), (1, 0)
            )
            tl.store(p_Ai_ij, b_Ai_ij.to(A.dtype.element_ty), boundary_check=(0, 1))


    b_zero = tl.zeros([BC, BC], dtype=tl.float32)
    for sc in tl.static_range(1, BT // BC):
        p_scratch = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_tc0, sc * BC), (BC, BC), (1, 0)
        )
        tl.store(p_scratch, b_zero.to(A.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)


    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    kernel = chunk_gated_delta_rule_fwd_kkt_solve_kernel_low_reg
    kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=BC,
    )


    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A
