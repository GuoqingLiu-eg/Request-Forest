# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined.
# ruff: noqa  -- vendored kernel library, minimal local deltas


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

RCP_LN2 = 1.4426950408889634
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
)

try:
    from .Akk_inverse_lower_triangle_bf16 import akk_inv_host as _akk_inv_host
    from .fuse_k4_only_persistent import BYTES_PER_TENSORMAP as _K4P_BTM
    from .fuse_k4_only_persistent import NUM_TENSORMAPS as _K4P_NTM
    from .fuse_k4_only_persistent import make_host_fn as _k4p_make_host
    from .fuse_kernel123_persistent import CHUNKS_PER_BLOCK as _K123_CHUNKS_PER_BLOCK
    from .fuse_kernel123_persistent import make_host_function as _fused_make_host
except ImportError:
    from Akk_inverse_lower_triangle_bf16 import akk_inv_host as _akk_inv_host
    from fuse_k4_only_persistent import BYTES_PER_TENSORMAP as _K4P_BTM
    from fuse_k4_only_persistent import NUM_TENSORMAPS as _K4P_NTM
    from fuse_k4_only_persistent import make_host_fn as _k4p_make_host
    from fuse_kernel123_persistent import CHUNKS_PER_BLOCK as _K123_CHUNKS_PER_BLOCK
    from fuse_kernel123_persistent import make_host_function as _fused_make_host


def _ct(t, etype):

    r = from_dlpack(t, assumed_align=16)
    r.element_type = etype
    return r


_eqlen_dummy_cache = {}


def _get_eqlen_dummies(device, idx_dtype=torch.int64):

    key = (device.index if device.index is not None else 0, idx_dtype)
    if key not in _eqlen_dummy_cache:
        cu_t = torch.empty(2, dtype=idx_dtype, device=device)
        ci_t = torch.empty(1, 2, dtype=idx_dtype, device=device)
        cu_etype = cutlass.Int64 if idx_dtype == torch.int64 else cutlass.Int32
        _eqlen_dummy_cache[key] = (_ct(cu_t, cu_etype), _ct(ci_t, cu_etype))
    return _eqlen_dummy_cache[key]


def _cute_int_type(dtype):

    if dtype == torch.int32:
        return cutlass.Int32
    elif dtype == torch.int64:
        return cutlass.Int64
    else:
        raise ValueError(f"Unsupported integer dtype: {dtype}")


_fused_k123_cache = {}


_varlen_pure_cache = {}


_varlen_single_seqlen_cache = {}


_input_wrap_cache = {}


def _tkey(t):

    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _ct_cached(t, etype):

    key = (_tkey(t), etype)
    w = _input_wrap_cache.get(key)
    if w is None:
        w = _ct(t, etype)
        _input_wrap_cache[key] = w
    return w


_dt_bias_cache = {}


def _get_dt_bias_ct(dt_bias, H, K):

    key = (_tkey(dt_bias), H, K)
    entry = _dt_bias_cache.get(key)
    if entry is None:
        bias_t = dt_bias.float().contiguous().view(H, K)
        entry = (bias_t, _ct(bias_t, cutlass.Float32))
        _dt_bias_cache[key] = entry
    return entry[1]


_empty_bias_cache = {}


def _get_empty_bias_ct(device):
    idx = device.index if device.index is not None else 0
    if idx not in _empty_bias_cache:
        t = torch.empty(1, 1, dtype=torch.float32, device=device)
        _empty_bias_cache[idx] = _ct(t, cutlass.Float32)
    return _empty_bias_cache[idx]


_k4_varlen_cu_co_cache = {}


def _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets):
    key = (_tkey(cu_seqlens), _tkey(chunk_offsets))
    entry = _k4_varlen_cu_co_cache.get(key)
    if entry is None:
        cu_int32 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int32
            else cu_seqlens.to(torch.int32)
        )
        cu_int32 = cu_int32.contiguous()
        co_int32 = (
            chunk_offsets
            if chunk_offsets.dtype == torch.int32
            else chunk_offsets.to(torch.int32)
        )
        co_int32 = co_int32.contiguous()
        cu_ct = from_dlpack(cu_int32, assumed_align=4).mark_layout_dynamic()
        cu_ct.element_type = cutlass.Int32
        co_ct = from_dlpack(co_int32, assumed_align=4).mark_layout_dynamic()
        co_ct.element_type = cutlass.Int32


        entry = (cu_int32, co_int32, cu_ct, co_ct)
        _k4_varlen_cu_co_cache[key] = entry
    return entry[2], entry[3]


_v_ct_cache = {}


_varlen_k4_input_cache = {}


def _get_varlen_k4_inputs(cu_seqlens, BT):


    cu_int32 = (
        cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    )
    cu_int32 = cu_int32.contiguous()
    seq_lens = cu_int32[1:] - cu_int32[:-1]
    chunk_counts = (seq_lens + (BT - 1)) // BT
    zero = torch.zeros(1, dtype=torch.int32, device=cu_int32.device)
    co_int32 = torch.cat([zero, torch.cumsum(chunk_counts, dim=0).to(torch.int32)])
    co_int32 = co_int32.contiguous()
    return cu_int32, co_int32


_akk_inv_cache = {}

_k4p_cache = {}
_k4p_tm_ws = {}


@torch.compile(fullgraph=True)
def _v_absorb(v, beta):
    return v * beta.unsqueeze(-1).to(v.dtype)


@torch.compile(fullgraph=True)
def _ks_absorb(k_scaled, beta):
    return k_scaled * beta.unsqueeze(-1).to(k_scaled.dtype)


_side_streams = {}


def _get_side_stream(dev):
    idx = dev.index or 0
    if idx not in _side_streams:
        _side_streams[idx] = torch.cuda.Stream(device=dev)
    return _side_streams[idx]


_TIMING_ENABLED = False
_TIMING_STATS = {"k123_us": 0.0, "k4_us": 0.0, "count": 0}


def enable_timing(enabled=True):

    global _TIMING_ENABLED
    _TIMING_ENABLED = enabled


def reset_timings():
    _TIMING_STATS["k123_us"] = 0.0
    _TIMING_STATS["k4_us"] = 0.0
    _TIMING_STATS["count"] = 0


def get_timings():

    n = max(_TIMING_STATS["count"], 1)
    return (
        _TIMING_STATS["k123_us"] / n,
        _TIMING_STATS["k4_us"] / n,
        _TIMING_STATS["count"],
    )


_buf_cache = {}


_padded_input_cache = {}


_g_sentinel_cache = {}


def _get_g_sentinel_buffer(B, T_padded, H, K, dtype_g, device, real_T):
    key = (
        B,
        T_padded,
        H,
        K,
        dtype_g,
        device.index if device.index is not None else 0,
        real_T,
    )
    e = _g_sentinel_cache.get(key)
    if e is None:
        e = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            e[:, real_T:] = -1000.0
        _g_sentinel_cache[key] = e
    return e


def _get_padded_input_buffers(
    B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta, device, real_T
):
    key = (
        B,
        T_padded,
        H,
        K,
        dtype_qkv,
        dtype_g,
        dtype_beta,
        device.index if device.index is not None else 0,
        real_T,
    )
    e = _padded_input_cache.get(key)
    if e is None:
        q_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        k_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        v_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        beta_pad = torch.zeros(B, T_padded, H, dtype=dtype_beta, device=device)


        g_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            g_pad[:, real_T:] = -1000.0
        e = (q_pad, k_pad, v_pad, g_pad, beta_pad)
        _padded_input_cache[key] = e
    return e


_multiseq_repack_cache = {}

_caller_layout_O_cache = {}


def _get_caller_layout_O_buffer(multiseq_info, dtype, V_dim, device):

    caller_T = multiseq_info["caller_T"]
    H_x_V = multiseq_info.get("_H_V")
    key = (caller_T, V_dim, dtype, device.index if device.index is not None else 0)
    e = _caller_layout_O_cache.get(key)
    if e is None:

        e = torch.empty(
            1,
            caller_T,
            multiseq_info["q_pad"].shape[2],
            V_dim,
            dtype=dtype,
            device=device,
        )
        _caller_layout_O_cache[key] = e
    return e


def _get_multiseq_repack_info(cu_seqlens, q, k, v, g, beta, BT, device):

    import weakref

    key = id(cu_seqlens)
    cached = _multiseq_repack_cache.get(key)
    if cached is not None:
        wref, e = cached
        if wref() is cu_seqlens:
            return e

        del _multiseq_repack_cache[key]
    cu_cpu = cu_seqlens.cpu().tolist()
    seq_lens = [cu_cpu[i + 1] - cu_cpu[i] for i in range(len(cu_cpu) - 1)]
    if all(sl % BT == 0 for sl in seq_lens):
        _multiseq_repack_cache[key] = (weakref.ref(cu_seqlens), None)
        return None
    padded_lens = [((sl + BT - 1) // BT) * BT for sl in seq_lens]
    new_cu = [0]
    for pl in padded_lens:
        new_cu.append(new_cu[-1] + pl)
    new_T_total = new_cu[-1]
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]


    q_pad = torch.zeros(B, new_T_total, H, K, dtype=q.dtype, device=device)
    k_pad = torch.zeros_like(q_pad)
    v_pad = torch.zeros(B, new_T_total, H, v.shape[3], dtype=v.dtype, device=device)
    beta_pad = torch.zeros(B, new_T_total, H, dtype=beta.dtype, device=device)
    g_pad = torch.zeros(B, new_T_total, H, K, dtype=g.dtype, device=device)
    for i, (sl, pl) in enumerate(zip(seq_lens, padded_lens)):
        if sl < pl:
            tail_start = new_cu[i] + sl
            tail_end = new_cu[i + 1]
            g_pad[:, tail_start:tail_end] = -1000.0
    new_cu_tensor = torch.tensor(new_cu, dtype=cu_seqlens.dtype, device=device)
    new_chunk_indices = prepare_chunk_indices(new_cu_tensor, BT)


    T_total_orig = cu_cpu[-1]
    dst_indices_list = []
    for i, sl in enumerate(seq_lens):
        for j in range(sl):
            dst_indices_list.append(new_cu[i] + j)
    dst_indices = torch.tensor(dst_indices_list, dtype=torch.long, device=device)


    _varlen_pure_cache[id(new_cu_tensor)] = True
    e = {
        "seq_lens": seq_lens,
        "padded_lens": padded_lens,
        "new_cu": new_cu,
        "new_T_total": new_T_total,
        "new_cu_tensor": new_cu_tensor,
        "new_chunk_indices": new_chunk_indices,
        "q_pad": q_pad,
        "k_pad": k_pad,
        "v_pad": v_pad,
        "g_pad": g_pad,
        "beta_pad": beta_pad,
        "dst_indices": dst_indices,
        "T_total_orig": T_total_orig,
    }
    _multiseq_repack_cache[key] = (weakref.ref(cu_seqlens), e)
    return e


def _get_buffers(dev, dtype_k, B, T, H, K_dim, V_dim, NT, N_seqs, BT):

    key = (dev.index or 0, B, T, H, K_dim, V_dim, NT, N_seqs)
    if key not in _buf_cache:
        bf16 = cutlass.BFloat16
        fp32 = cutlass.Float32
        k_scaled = torch.empty(
            B, T, H, K_dim, device=dev, dtype=dtype_k
        )
        kg = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        q_scaled = torch.empty(B, T, H, K_dim, device=dev, dtype=dtype_k)
        gk_last_exp = torch.empty(B, NT, H, K_dim, device=dev, dtype=torch.float32)
        A_qk = torch.zeros(B, T, H, BT, device=dev, dtype=dtype_k)
        A_kk = torch.zeros(B, T, H, BT, device=dev, dtype=dtype_k)
        O_flat = torch.empty(B, T, H, V_dim, device=dev, dtype=dtype_k)


        S_out = torch.empty(N_seqs, H, K_dim, V_dim, device=dev, dtype=torch.float32)
        cu_eqlen = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
        co_eqlen = torch.arange(
            0, (B + 1) * (T // BT), T // BT, dtype=torch.int32, device=dev
        )

        T_total = B * T
        A_kk_flat = A_kk.reshape(T_total, H, BT)
        A_qk_flat = A_qk.reshape(T_total, H, BT)
        KS_flat = k_scaled.reshape(T_total, H, K_dim)
        QS_flat = q_scaled.reshape(T_total, H, K_dim)
        KG_flat = kg.reshape(T_total, H, K_dim)
        O_token = O_flat.reshape(T_total, H, V_dim)
        gk_flat = gk_last_exp.reshape(-1, H, K_dim)

        def _wrap(t, etype):
            r = from_dlpack(t, assumed_align=16).mark_layout_dynamic()
            r.element_type = etype
            return r

        a_ct = _wrap(A_kk_flat, bf16)
        aqc_ct = _wrap(A_qk_flat, bf16)
        ks_ct = _wrap(KS_flat, bf16)
        qs_ct = _wrap(QS_flat, bf16)
        kg_ct = _wrap(KG_flat, bf16)
        o_ct = _wrap(O_token, bf16)
        gk_ct = _wrap(gk_flat, fp32)
        cu_eqlen_ct = from_dlpack(cu_eqlen, assumed_align=4).mark_layout_dynamic()
        cu_eqlen_ct.element_type = cutlass.Int32
        co_eqlen_ct = from_dlpack(co_eqlen, assumed_align=4).mark_layout_dynamic()
        co_eqlen_ct.element_type = cutlass.Int32


        akk_in_view = from_dlpack(A_kk, assumed_align=16)
        akk_in_view.element_type = fp32
        akk_out_view = from_dlpack(A_kk, assumed_align=16)
        akk_out_view.element_type = fp32


        s_ct = from_dlpack(S_out, assumed_align=16)
        s_ct.element_type = fp32


        main_stream_cached = torch.cuda.current_stream(dev)
        side_stream_cached = _get_side_stream(dev)

        cute_wrappers = dict(
            a_ct=a_ct,
            aqc_ct=aqc_ct,
            ks_ct=ks_ct,
            qs_ct=qs_ct,
            kg_ct=kg_ct,
            o_ct=o_ct,
            gk_ct=gk_ct,
            cu_eqlen_ct=cu_eqlen_ct,
            co_eqlen_ct=co_eqlen_ct,
            akk_in_view=akk_in_view,
            akk_out_view=akk_out_view,
            s_ct=s_ct,
            main_stream=main_stream_cached,
            side_stream=side_stream_cached,


            _k123_fns={},
            _akk_inv_fn=None,
            _k4_fn=None,
        )

        _buf_cache[key] = (
            k_scaled,
            kg,
            q_scaled,
            gk_last_exp,
            A_qk,
            A_kk,
            O_flat,
            S_out,
            cu_eqlen,
            co_eqlen,
            cute_wrappers,
        )
    return _buf_cache[key]


def _launch_k4_persistent(
    cute_wrappers,
    v_beta,
    S_in,
    S_out,
    cu_seqlens,
    chunk_offsets,
    cu_eqlen_passed=False,
    num_sm=148,
    H=None,
    V_dim=None,
    use_fast_sync=False,
):


    launch_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)


    fast_key = (
        _tkey(v_beta),
        0 if cu_eqlen_passed else _tkey(cu_seqlens),
        _tkey(S_in),
        _tkey(S_out),
    )
    fast_cache = cute_wrappers.get("_k4_fast_cache")
    if fast_cache is None:
        fast_cache = {}
        cute_wrappers["_k4_fast_cache"] = fast_cache
    fast_entry = fast_cache.get(fast_key)
    if fast_entry is not None:
        k4_fn, args = fast_entry
        k4_fn(*args, launch_stream)
        return

    bf16 = cutlass.BFloat16
    fp32 = cutlass.Float32
    N_seqs = cu_seqlens.shape[0] - 1
    BH = N_seqs * H
    nsm = min(BH, num_sm)
    dev = v_beta.device
    dev_idx = dev.index or 0

    s_ct = cute_wrappers["s_ct"]
    a_ct = cute_wrappers["a_ct"]
    b_ct = cute_wrappers["ks_ct"]
    q_ct = cute_wrappers["qs_ct"]
    aqc_ct = cute_wrappers["aqc_ct"]
    kg_ct = cute_wrappers["kg_ct"]
    o_ct = cute_wrappers["o_ct"]
    gk_ct = cute_wrappers["gk_ct"]

    v_key = _tkey(v_beta)
    v_entry = _v_ct_cache.get(v_key)
    if v_entry is None:
        v_view = v_beta.reshape(-1, H, V_dim) if v_beta.dim() == 4 else v_beta
        v_ct = from_dlpack(v_view, assumed_align=16).mark_layout_dynamic()
        v_ct.element_type = bf16
        _v_ct_cache[v_key] = (v_view, v_ct)
    else:
        v_ct = v_entry[1]

    if cu_eqlen_passed:
        cu_ct = cute_wrappers["cu_eqlen_ct"]
        co_ct = cute_wrappers["co_eqlen_ct"]
    else:
        cu_ct, co_ct = _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets)

    tm_key = (dev_idx, nsm)
    if tm_key not in _k4p_tm_ws:
        tm_ws_t = torch.zeros(nsm * _K4P_NTM * _K4P_BTM, dtype=torch.uint8, device=dev)
        tm_ct = from_dlpack(tm_ws_t, assumed_align=16)
        tm_ct.element_type = cutlass.Uint8
        _k4p_tm_ws[tm_key] = (tm_ws_t, tm_ct)
    else:
        tm_ws_t, tm_ct = _k4p_tm_ws[tm_key]

    k4_fn = cute_wrappers.get("_k4_fn")
    if k4_fn is None:
        cache_key = (dev_idx, nsm, N_seqs, H)
        k4_fn = _k4p_cache.get(cache_key)
        if k4_fn is None:
            host_fn = _k4p_make_host(num_sm=nsm)
            k4_fn = cute.compile(
                host_fn,
                a_ct,
                b_ct,
                v_ct,
                q_ct,
                aqc_ct,
                kg_ct,
                o_ct,
                gk_ct,
                s_ct,
                cu_ct,
                co_ct,
                tm_ct,
                launch_stream,
            )
            _k4p_cache[cache_key] = k4_fn
        cute_wrappers["_k4_fn"] = k4_fn

    args = (
        a_ct,
        b_ct,
        v_ct,
        q_ct,
        aqc_ct,
        kg_ct,
        o_ct,
        gk_ct,
        s_ct,
        cu_ct,
        co_ct,
        tm_ct,
    )
    fast_cache[fast_key] = (k4_fn, args)
    k4_fn(*args, launch_stream)


def _launch_fused_k123_inv(
    q,
    k,
    g,
    A_log,
    beta,
    scale,
    k_scaled,
    kg,
    q_scaled,
    gk_last_exp,
    A_qk,
    A_kk_inv,
    cu_seqlens,
    chunk_indices,
    is_varlen,
    NT,
    dt_bias=None,
    safe_gate=False,
    lower_bound=None,
    akk_in_view=None,
    akk_out_view=None,
    cute_wrappers=None,
    varlen_pure_override=None,
):


    launch_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)


    if cute_wrappers is not None:
        fast_key = (
            _tkey(q),
            _tkey(k),
            _tkey(g),
            _tkey(beta),
            _tkey(A_log),
            _tkey(dt_bias) if dt_bias is not None else 0,
            _tkey(cu_seqlens) if cu_seqlens is not None else 0,
            _tkey(chunk_indices) if chunk_indices is not None else 0,
            bool(safe_gate),
            float(lower_bound) if lower_bound is not None else 0.0,
        )
        fast_cache = cute_wrappers.setdefault("_k123_fast_cache", {})
        fast_entry = fast_cache.get(fast_key)
        if fast_entry is not None:
            k123_fn, k123_args, akk_fn, akk_args = fast_entry
            k123_fn(*k123_args, launch_stream)
            akk_fn(*akk_args, launch_stream)
            return

    B, T, H, K = q.shape
    BT = 64
    dev = q.device.index or 0
    T_padded = T if is_varlen else None
    has_bias = dt_bias is not None


    varlen_pure = False
    if varlen_pure_override is not None:


        varlen_pure = bool(varlen_pure_override) if is_varlen else False
    elif is_varlen and cu_seqlens is not None:
        _vp_key = id(cu_seqlens)
        if _vp_key not in _varlen_pure_cache:
            cu_cpu = cu_seqlens.cpu().tolist()
            seq_lens = [cu_cpu[i + 1] - cu_cpu[i] for i in range(len(cu_cpu) - 1)]
            _varlen_pure_cache[_vp_key] = all((sl % BT) == 0 for sl in seq_lens)
        varlen_pure = _varlen_pure_cache[_vp_key]
    cache_key = (B, NT, H, is_varlen, T_padded, dev, has_bias, safe_gate, varlen_pure)


    q_ct = _ct_cached(q, cutlass.BFloat16)
    k_ct = _ct_cached(k, cutlass.BFloat16)
    g_ct = _ct_cached(g, cutlass.BFloat16)
    alog_ct = _ct_cached(
        A_log if A_log.dtype == torch.float32 else A_log.float(), cutlass.Float32
    )
    beta_ct = _ct_cached(beta, cutlass.BFloat16)

    ks_ct = _ct_cached(k_scaled, cutlass.BFloat16)
    kg_ct = _ct_cached(kg, cutlass.BFloat16)
    qs_ct = _ct_cached(q_scaled, cutlass.BFloat16)
    gk_ct = _ct_cached(gk_last_exp, cutlass.Float32)
    aqk_ct = _ct_cached(A_qk, cutlass.BFloat16)
    akk_ct = _ct_cached(A_kk_inv, cutlass.BFloat16)

    if is_varlen:
        cu_ct = _ct_cached(cu_seqlens, _cute_int_type(cu_seqlens.dtype))
        ci_ct = _ct_cached(chunk_indices, _cute_int_type(chunk_indices.dtype))
    else:
        cu_ct, ci_ct = _get_eqlen_dummies(q.device, torch.int64)

    if dt_bias is not None:
        bias_ct = _get_dt_bias_ct(dt_bias, H, K)
    else:
        bias_ct = _get_empty_bias_ct(q.device)
    lb_val = float(lower_bound) if lower_bound is not None else 0.0

    ct_args = (
        q_ct,
        k_ct,
        g_ct,
        alog_ct,
        beta_ct,
        scale,
        ks_ct,
        kg_ct,
        qs_ct,
        gk_ct,
        aqk_ct,
        akk_ct,
        cu_ct,
        ci_ct,
        bias_ct,
        lb_val,
    )

    if cache_key not in _fused_k123_cache:
        host_fn = _fused_make_host(
            B,
            NT,
            H,
            is_varlen=is_varlen,
            T_padded=T_padded,
            has_bias=has_bias,
            use_safe_gate=safe_gate,
            varlen_pure=varlen_pure,
        )
        _fused_k123_cache[cache_key] = cute.compile(host_fn, *ct_args, launch_stream)
    k123_fn = _fused_k123_cache[cache_key]
    k123_fn(*ct_args, launch_stream)


    if akk_in_view is None:
        akk_in_view = from_dlpack(A_kk_inv, assumed_align=16)
        akk_in_view.element_type = cutlass.Float32
        akk_out_view = from_dlpack(A_kk_inv, assumed_align=16)
        akk_out_view.element_type = cutlass.Float32

    if is_varlen:

        akk_cu_ct = cu_ct
        akk_ci_ct = ci_ct
        is_varlen_int = 1
        T_val = T
    else:
        akk_cu_ct, akk_ci_ct = cu_ct, ci_ct
        is_varlen_int = 0
        T_val = NT * BT

    akk_cache_key = (B, NT, H, is_varlen, dev, T_val)
    if akk_cache_key not in _akk_inv_cache:
        _akk_inv_cache[akk_cache_key] = cute.compile(
            _akk_inv_host,
            akk_in_view,
            akk_out_view,
            beta_ct,
            B,
            NT,
            H,
            akk_cu_ct,
            akk_ci_ct,
            is_varlen_int,
            T_val,
            launch_stream,
        )
    akk_fn = _akk_inv_cache[akk_cache_key]
    akk_args = (akk_in_view, akk_out_view, beta_ct, akk_cu_ct, akk_ci_ct)
    akk_fn(*akk_args, launch_stream)


    if cute_wrappers is not None:
        cute_wrappers["_k123_fast_cache"][fast_key] = (
            k123_fn,
            ct_args,
            akk_fn,
            akk_args,
        )


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context=None,
    varlen_single_real_T: int | None = None,
    varlen_pure: bool | None = None,
):

    if safe_gate and lower_bound is None:
        lower_bound = -5.0

    is_varlen = cu_seqlens is not None

    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    device = q.device
    BT = 64


    real_T = T
    CPB_BT = _K123_CHUNKS_PER_BLOCK * BT
    needs_eqlen_pad = (not is_varlen) and (T % CPB_BT != 0)
    if needs_eqlen_pad:
        if B != 1 and T % BT != 0:
            raise NotImplementedError(
                f"eqlen with B>1 and T % {BT} != 0 not supported "
                f"(got B={B}, T={T})."
            )
        T_padded = ((T + CPB_BT - 1) // CPB_BT) * CPB_BT


        q_pad, k_pad, v_pad, g_pad, beta_pad = _get_padded_input_buffers(
            B, T_padded, H, K, q.dtype, g.dtype, beta.dtype, q.device, real_T
        )


        q_pad[:, :real_T].copy_(q)
        k_pad[:, :real_T].copy_(k)
        v_pad[:, :real_T].copy_(v)
        beta_pad[:, :real_T].copy_(beta)


        g_pad[:, :real_T].copy_(g)
        q, k, v, g, beta = q_pad, k_pad, v_pad, g_pad, beta_pad
        T = T_padded


    needs_varlen_single_pad = False
    if (
        varlen_single_real_T is not None
        and is_varlen
        and cu_seqlens is not None
        and cu_seqlens.shape[0] == 2
        and B == 1
    ):


        if varlen_single_real_T % BT != 0:
            real_T = varlen_single_real_T
            needs_varlen_single_pad = True
    elif is_varlen and cu_seqlens is not None and cu_seqlens.shape[0] == 2 and B == 1:
        _vl_key = id(cu_seqlens)
        if _vl_key not in _varlen_pure_cache:
            cu_cpu = cu_seqlens.cpu().tolist()
            sl = cu_cpu[1] - cu_cpu[0]
            _varlen_pure_cache[_vl_key] = sl % BT == 0
            _varlen_single_seqlen_cache[_vl_key] = sl
        if not _varlen_pure_cache[_vl_key]:
            real_T = _varlen_single_seqlen_cache[_vl_key]
            needs_varlen_single_pad = True
    if needs_varlen_single_pad:


        cur_T = q.shape[1]
        g_pad = _get_g_sentinel_buffer(B, cur_T, H, K, g.dtype, g.device, real_T)
        g_pad[:, :real_T].copy_(g[:, :real_T])
        g = g_pad

        _varlen_pure_cache[id(cu_seqlens)] = True


    multiseq_info = None

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        N_seqs = len(cu_seqlens) - 1
    else:
        NT = T // BT
        N_seqs = B


    (
        k_scaled,
        kg,
        q_scaled,
        gk_last_exp,
        A_qk,
        A_kk,
        O_flat,
        S_out,
        cu_eqlen,
        co_eqlen,
        cute_wrappers,
    ) = _get_buffers(device, k.dtype, B, T, H, K, V_dim, NT, N_seqs, BT)


    if initial_state is None:
        S_in = torch.zeros(N_seqs, H, K, V_dim, dtype=torch.float32, device=device)
    else:
        S_in = initial_state


    main_stream = torch.cuda.current_stream(device)
    needs_copy = S_in.data_ptr() != S_out.data_ptr()
    if needs_copy:
        S_out.copy_(S_in)


    if _TIMING_ENABLED:
        k123_s = torch.cuda.Event(enable_timing=True)
        k123_e = torch.cuda.Event(enable_timing=True)
        k123_s.record(stream=main_stream)

    _launch_fused_k123_inv(
        q,
        k,
        g,
        A_log,
        beta,
        scale,
        k_scaled,
        kg,
        q_scaled,
        gk_last_exp,
        A_qk,
        A_kk,
        cu_seqlens,
        chunk_indices,
        is_varlen,
        NT,
        dt_bias=dt_bias,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        akk_in_view=cute_wrappers["akk_in_view"],
        akk_out_view=cute_wrappers["akk_out_view"],
        cute_wrappers=cute_wrappers,
        varlen_pure_override=(True if needs_varlen_single_pad else varlen_pure),
    )

    if _TIMING_ENABLED:
        k123_e.record(stream=main_stream)


    if is_varlen:

        cu_for_k4, chunk_offsets_for_k4 = _get_varlen_k4_inputs(cu_seqlens, BT)
    else:
        cu_for_k4 = cu_eqlen
        chunk_offsets_for_k4 = co_eqlen


    if _TIMING_ENABLED:
        k4_s = torch.cuda.Event(enable_timing=True)
        k4_e = torch.cuda.Event(enable_timing=True)
        k4_s.record(stream=main_stream)

    _launch_k4_persistent(
        cute_wrappers,
        v,
        S_in,
        S_out,
        cu_for_k4,
        chunk_offsets_for_k4,
        cu_eqlen_passed=(not is_varlen),
        H=H,
        V_dim=V_dim,
        use_fast_sync=(not is_varlen),
    )

    if _TIMING_ENABLED:
        k4_e.record(stream=cute_wrappers["main_stream"])
        torch.cuda.synchronize(device)
        _TIMING_STATS["k123_us"] += k123_s.elapsed_time(k123_e) * 1000.0
        _TIMING_STATS["k4_us"] += k4_s.elapsed_time(k4_e) * 1000.0
        _TIMING_STATS["count"] += 1

    o = O_flat
    A_qk_out = A_qk
    A_kk_out = A_kk
    if needs_eqlen_pad:


        o = o[:, :real_T]
        A_qk_out = A_qk_out[:, :real_T]
        A_kk_out = A_kk_out[:, :real_T]

    final_state = S_out if output_final_state else None

    return (
        o,
        final_state,
        None,
        A_qk_out,
        A_kk_out,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_state,
    )
