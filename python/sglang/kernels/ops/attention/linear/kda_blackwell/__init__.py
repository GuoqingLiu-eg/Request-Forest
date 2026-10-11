# SPDX-License-Identifier: Apache-2.0
# KDA (Kimi Delta Attention) SM100/Blackwell CuteDSL prefill pipeline.
#
# Mirrors gdn_blackwell but for KDA's PER-CHANNEL decay gate. A fused Triton
# prologue computes the per-chunk cumsum g_cu and five pre-scaled key/query
# tensors; three cutedsl kernels then run the chunked gated delta rule:
#   prologue -> kkt_inv_uw (U,W) -> h (V_new, per-chunk state, final state) -> o
import torch

from .kernel_h import kda_h_cutedsl
from .kernel_kkt_inv_uw import kkt_inv_uw_cutedsl
from .kernel_o import kda_o_cutedsl
from .prologue import kda_prologue

__all__ = ["chunk_kda_cutedsl", "prepare_metadata"]


def prepare_metadata(cu_seqlens: torch.Tensor, chunk_size: int = 64):

    dev = cu_seqlens.device
    cs = cu_seqlens.to(torch.int64)
    seqlens = cs[1:] - cs[:-1]
    nchunks = (seqlens + chunk_size - 1) // chunk_size
    n = seqlens.numel()
    chunk_offsets = torch.zeros(n + 1, dtype=torch.int32, device=dev)
    chunk_offsets[1:] = nchunks.cumsum(0).to(torch.int32)
    total = int(chunk_offsets[-1].item())
    seq_id = torch.repeat_interleave(torch.arange(n, device=dev), nchunks)
    local = torch.arange(total, device=dev) - chunk_offsets[seq_id].to(torch.int64)
    chunk_indices = torch.stack(
        [seq_id.to(torch.int32), local.to(torch.int32)], dim=1
    ).contiguous()
    total_chunks = torch.tensor([total], dtype=torch.int32, device=dev)
    return chunk_indices, chunk_offsets, total_chunks, total


_KDA_WS: dict = {}


def _kda_workspace(q, T, Hv, K, V, cu_seqlens):
    import torch as _t

    dev = q.device


    stream = _t.cuda.current_stream(device=dev).cuda_stream
    key = (Hv, K, V, dev, q.dtype, stream)
    ws = _KDA_WS.get(key)


    if ws is None or ws["cu"] is not cu_seqlens:
        ci, co, tcs, total = prepare_metadata(cu_seqlens)
    else:
        ci, co, tcs, total = ws["ci"], ws["co"], ws["tcs"], ws["total"]
    pad_t = total * 64

    if ws is None or ws["Tcap"] < T or ws["padcap"] < pad_t or ws["totalcap"] < total:
        Tcap = T if ws is None else max(T, ws["Tcap"])
        padcap = pad_t if ws is None else max(pad_t, ws["padcap"])
        totalcap = total if ws is None else max(total, ws["totalcap"])
        ws = {
            "kL": q.new_zeros(Tcap, Hv, K, dtype=_t.bfloat16),
            "qg2": q.new_zeros(Tcap, Hv, K, dtype=_t.bfloat16),
            "eye": q.new_zeros(Tcap, Hv, K, dtype=_t.bfloat16),
            "U": q.new_empty(padcap, Hv, V, dtype=_t.bfloat16),
            "W": q.new_empty(padcap, Hv, K, dtype=_t.bfloat16),
            "Vn": q.new_empty(padcap, Hv, V, dtype=_t.bfloat16),
            "hc": q.new_empty(totalcap, Hv, V, K, dtype=_t.bfloat16),
            "Tcap": Tcap,
            "padcap": padcap,
            "totalcap": totalcap,
            "cu": None,
            "eye_hw": 0,
        }
        _KDA_WS[key] = ws

    ws["ci"], ws["co"], ws["tcs"], ws["total"] = ci, co, tcs, total


    if ws["cu"] is not cu_seqlens:
        eye = ws["eye"]
        hw = max(ws["eye_hw"], T)
        eye[:hw].zero_()


        tok = _t.arange(T, device=dev, dtype=cu_seqlens.dtype)
        seq_of = _t.searchsorted(cu_seqlens, tok, right=True) - 1
        pos = (tok - cu_seqlens[seq_of]) % 64
        eye[tok, :, pos] = 1.0
        ws["eye_hw"] = T
        ws["cu"] = cu_seqlens
    return ws, ci, co, tcs, total, pad_t


def chunk_kda_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    num_sms: int | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    h0_indices: torch.Tensor | None = None,
):

    import torch.nn.functional as F

    T, Hv, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count


    if A_log is not None:
        if lower_bound is not None:
            raise NotImplementedError(
                "KDA cutedsl: safe_gate (lower_bound) not yet supported"
            )
        x = g.float()
        if dt_bias is not None:
            x = x + dt_bias.float().view(1, Hv, K)
        g_act = -torch.exp(A_log.float()).view(1, Hv, 1) * F.softplus(x)
    else:
        g_act = g.float()


    ws, chunk_indices, chunk_offsets, total_chunks, total, pad_t = _kda_workspace(
        q, T, Hv, K, V, cu_seqlens
    )


    _, KR, KG, qg, _, g_cu = kda_prologue(
        q, k, g_act, float(scale), cu_seqlens, chunk_indices, total
    )


    from sglang.kernels.ops.attention.fla.kda import (
        RCP_LN2,
        chunk_kda_scaled_dot_kkt_fwd,
    )

    ones_beta = q.new_ones(1, T, Hv, dtype=torch.float32)


    M_kk, M_qk = chunk_kda_scaled_dot_kkt_fwd(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        gk=g_cu.unsqueeze(0),
        gk_scale=RCP_LN2,
        beta=ones_beta,
        scale=float(scale),
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )


    kL_inj = ws["kL"][:T]
    qg2_inj = ws["qg2"][:T]
    kL_inj[:, :, :64] = M_kk[0].to(torch.bfloat16)
    qg2_inj[:, :, :64] = M_qk[0].to(torch.bfloat16)
    eye = ws["eye"][:T]

    U = ws["U"][:pad_t]
    W = ws["W"][:pad_t]
    kkt_inv_uw_cutedsl(
        kL_inj,
        eye,
        KG,
        v,
        U,
        W,
        beta,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_sms=num_sms,
    )

    V_new = ws["Vn"][:pad_t]
    h_chunks = ws["hc"][:total]
    if h0_indices is None:

        ht = torch.empty_like(h0)
        state_indices = torch.arange(
            cu_seqlens.numel() - 1, device=q.device, dtype=torch.int32
        )
    else:

        ht = h0
        state_indices = h0_indices
    kda_h_cutedsl(
        KR,
        U,
        W,
        V_new,
        g_cu,
        h_chunks,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
        state_indices,
    )

    o = q.new_empty(T, Hv, V, dtype=torch.bfloat16)
    kda_o_cutedsl(
        qg,
        qg2_inj,
        eye,
        V_new,
        h_chunks,
        o,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_sms=num_sms,
    )
    return o, ht
