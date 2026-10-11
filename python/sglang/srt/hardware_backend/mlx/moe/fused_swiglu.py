from __future__ import annotations

import logging
import weakref

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


_BITS = 4
_GROUP_SIZE = 64
_SIMD_SIZE = 32
_PACK_FACTOR = 8
_BYTES_PER_PACK = 4
_PACKS_PER_THREAD = 2
_NUM_SIMDGROUPS = 2
_RESULTS_PER_SIMDGROUP = 4
_VALUES_PER_THREAD = _PACK_FACTOR * _PACKS_PER_THREAD
_BLOCK_SIZE = _VALUES_PER_THREAD * _SIMD_SIZE
_ROWS_PER_TG = _NUM_SIMDGROUPS * _RESULTS_PER_SIMDGROUP


_KERNEL_SOURCE = r"""
    // Mirrors qmv_fast_impl<T, group_size=64, bits=4> from MLX's quantized.h
    // with a silu(result) * x_up write epilogue.
    //
    // Inputs:
    //   x       [M_tok, K]                — pre-gather activations (T)
    //   w       [E, N, K * 4 / 32]        — packed 4-bit weights (uint32)
    //   s       [E, N, K / GROUP_SIZE]    — affine scales (T)
    //   b       [E, N, K / GROUP_SIZE]    — affine biases (T)
    //   idx     [M_tok * TOPK]            — expert per (token, topk) pair (uint32)
    //   x_up    [M_tok * TOPK, N]         — precomputed up output (T)
    // Output:
    //   y       [M_tok * TOPK, N]         — silu(gate_qmv(x)) * x_up

    constexpr int BITS = 4;
    constexpr int GROUP_SIZE = 64;
    constexpr int SIMD_SIZE = 32;
    constexpr int PACK_FACTOR = 8;
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACKS_PER_THREAD = 2;
    constexpr int NUM_SIMDGROUPS = 2;
    constexpr int RESULTS_PER_SIMDGROUP = 4;
    constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;   // 16
    constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;            // 512
    constexpr int SCALE_STEP_PER_THREAD = GROUP_SIZE / VALUES_PER_THREAD; // 4

    // Compile-time problem dims (template params)
    constexpr int K = IN_VEC_SIZE;
    constexpr int N = OUT_VEC_SIZE;
    constexpr int TOPK = TOP_K;

    constexpr int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;    // K/2 bytes per row
    constexpr int in_vec_size_g = K / GROUP_SIZE;                       // groups per row

    // tid.x = (token, topk) pair index in M_tok*TOPK
    // tid.y = output row block index (each block writes 8 rows)
    uint mt = threadgroup_position_in_grid.x;
    uint out_row_block = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    uint m = mt / TOPK;                      // index into pre-gather x
    uint e = idx[mt];                        // expert id

    // Base pointers for this (expert) row block.
    // Weights / scales / biases live in [E, N, ...] tensors.
    const device uint8_t* ws = (const device uint8_t*)(w
        + uint64_t(e) * uint64_t(N) * uint64_t(in_vec_size_w / BYTES_PER_PACK));
    const device T* scales_p = s
        + uint64_t(e) * uint64_t(N) * uint64_t(in_vec_size_g);
    const device T* biases_p = b
        + uint64_t(e) * uint64_t(N) * uint64_t(in_vec_size_g);

    // out_row indexes the first of RESULTS_PER_SIMDGROUP rows this simdgroup handles.
    int out_row = out_row_block * NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP
                + simd_gid * RESULTS_PER_SIMDGROUP;

    ws += out_row * in_vec_size_w + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
    scales_p += out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;
    biases_p += out_row * in_vec_size_g + simd_lid / SCALE_STEP_PER_THREAD;

    // Pre-gather x: shape [M_tok, K]
    const device T* x_p = x + uint64_t(m) * uint64_t(K)
                            + uint64_t(simd_lid) * uint64_t(VALUES_PER_THREAD);

    float result[RESULTS_PER_SIMDGROUP] = {0, 0, 0, 0};
    thread float x_thread[VALUES_PER_THREAD];

    // Outer loop over K in BLOCK_SIZE-wide chunks.
    // K % BLOCK_SIZE == 0 is required (checked in Python wrapper).
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // --- load_vector for bits=4 ---
        float sum = 0;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            float a0 = float(x_p[i]);
            float a1 = float(x_p[i + 1]);
            float a2 = float(x_p[i + 2]);
            float a3 = float(x_p[i + 3]);
            sum += a0 + a1 + a2 + a3;
            x_thread[i]     = a0;
            x_thread[i + 1] = a1 / 16.0f;
            x_thread[i + 2] = a2 / 256.0f;
            x_thread[i + 3] = a3 / 4096.0f;
        }

        // For each of the 4 output rows this simdgroup is responsible for...
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
            const device uint16_t* ws_u16 =
                (const device uint16_t*)(ws + row * in_vec_size_w);
            float scale_v = float(scales_p[row * in_vec_size_g]);
            float bias_v  = float(biases_p[row * in_vec_size_g]);

            // --- qdot for bits=4, values_per_thread=16 ---
            float accum = 0;
            for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
                uint16_t packed = ws_u16[i];
                accum += (x_thread[4 * i]     * float(packed & 0x000f) +
                          x_thread[4 * i + 1] * float(packed & 0x00f0) +
                          x_thread[4 * i + 2] * float(packed & 0x0f00) +
                          x_thread[4 * i + 3] * float(packed & 0xf000));
            }
            result[row] += scale_v * accum + sum * bias_v;
        }

        ws        += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;  // += 256 bytes
        scales_p  += BLOCK_SIZE / GROUP_SIZE;                    // += 8 groups
        biases_p  += BLOCK_SIZE / GROUP_SIZE;
        x_p       += BLOCK_SIZE;
    }

    // Write epilogue: simd-sum across lanes, then silu(gate) * x_up.
    device T* y_p = y + uint64_t(mt) * uint64_t(N) + uint64_t(out_row);
    const device T* x_up_p = x_up + uint64_t(mt) * uint64_t(N) + uint64_t(out_row);

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        float gate_v = simd_sum(result[row]);
        if (simd_lid == 0) {
            // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float silu_v = gate_v / (1.0f + metal::precise::exp(-gate_v));
            y_p[row] = T(silu_v * float(x_up_p[row]));
        }
    }
"""


_kernel_cache: dict = {}


_aot_warmed: set = set()


def _get_kernel(dtype: mx.Dtype):

    if dtype not in _kernel_cache:


        dtype_tag = str(dtype).replace("mlx.core.", "").replace(".", "_")
        _kernel_cache[dtype] = mx.fast.metal_kernel(
            name=f"affine_gather_qmv_silu_mul_4bit_gs64_{dtype_tag}",
            input_names=["x", "w", "s", "b", "idx", "x_up"],
            output_names=["y"],
            source=_KERNEL_SOURCE,
        )
    return _kernel_cache[dtype]


def fused_gate_qmv_silu_mul(
    x: mx.array,
    gate_w: mx.array,
    gate_s: mx.array,
    gate_b: mx.array,
    indices: mx.array,
    x_up: mx.array,
) -> mx.array:


    K = gate_s.shape[-1] * _GROUP_SIZE
    N = gate_w.shape[-2]
    if K % _BLOCK_SIZE != 0:
        raise ValueError(
            f"fused_gate_qmv_silu_mul: K={K} not divisible by {_BLOCK_SIZE}. "
            f"Use the unfused path."
        )
    if N % _ROWS_PER_TG != 0:
        raise ValueError(
            f"fused_gate_qmv_silu_mul: N={N} not divisible by {_ROWS_PER_TG}."
        )

    if gate_s.dtype != x.dtype or gate_b.dtype != x.dtype:
        raise ValueError(
            f"fused_gate_qmv_silu_mul: dtype mismatch x={x.dtype} "
            f"s={gate_s.dtype} b={gate_b.dtype}"
        )


    assert x.shape[-1] == K, f"x last dim {x.shape[-1]} != K={K}"
    M_tok = x.size // K
    T = indices.shape[-1]
    assert (
        M_tok * T == indices.size
    ), f"M_tok({M_tok}) * T({T}) != indices.size({indices.size})"
    x_flat = x.reshape(M_tok, K)
    idx_flat = indices.reshape(M_tok * T)
    if idx_flat.dtype != mx.uint32:
        idx_flat = idx_flat.astype(mx.uint32)


    assert (
        x_up.shape[-1] == N and x_up.size == M_tok * T * N
    ), f"x_up shape {x_up.shape} does not match M_tok({M_tok})*T({T})*N({N})"
    x_up_flat = x_up.reshape(M_tok * T, N)

    kernel = _get_kernel(x.dtype)
    (y_flat,) = kernel(
        inputs=[x_flat, gate_w, gate_s, gate_b, idx_flat, x_up_flat],
        template=[
            ("T", x.dtype),
            ("IN_VEC_SIZE", K),
            ("OUT_VEC_SIZE", N),
            ("TOP_K", T),
        ],


        grid=(M_tok * T * 64, N // _ROWS_PER_TG, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(M_tok * T, N)],
        output_dtypes=[x.dtype],
    )


    return y_flat.reshape(x_up.shape)


def _aot_warm_kernel(switch_mlp, top_k: int) -> None:

    gate = switch_mlp.gate_proj
    K = gate.scales.shape[-1] * _GROUP_SIZE
    N = gate.weight.shape[-2]
    dtype = gate.scales.dtype
    for T in (top_k, 1):
        key = (dtype, K, N, T)
        if key in _aot_warmed:
            continue
        M_tok = 1
        x_dummy = mx.zeros((M_tok, 1, 1, K), dtype=dtype)
        idx_dummy = mx.zeros((M_tok, T), dtype=mx.uint32)
        x_up_dummy = mx.zeros((M_tok, T, 1, N), dtype=dtype)
        out = fused_gate_qmv_silu_mul(
            x_dummy,
            gate["weight"],
            gate["scales"],
            gate.get("biases"),
            idx_dummy,
            x_up_dummy,
        )
        mx.eval(out)
        _aot_warmed.add(key)
        logger.info(
            "Path B AOT: warmed fused kernel dtype=%s K=%d N=%d T=%d",
            dtype,
            K,
            N,
            T,
        )


def can_fuse(switch_mlp) -> bool:

    try:
        from mlx_lm.models.switch_layers import (
            QuantizedSwitchLinear,
            SwiGLU,
            SwitchGLU,
        )
    except ImportError:
        return False
    up = switch_mlp.up_proj
    gate = switch_mlp.gate_proj
    if not isinstance(up, QuantizedSwitchLinear) or not isinstance(
        gate, QuantizedSwitchLinear
    ):
        return False


    if type(switch_mlp).__call__ is not SwitchGLU.__call__:
        return False


    if type(getattr(switch_mlp, "activation", None)) is not SwiGLU:
        return False


    if "bias" in gate:
        return False
    if up.bits != 4 or up.group_size != 64 or up.mode != "affine":
        return False
    if gate.bits != 4 or gate.group_size != 64 or gate.mode != "affine":
        return False
    if up.biases is None or gate.biases is None:
        return False


    param_dtype = gate.scales.dtype
    if (
        gate.biases.dtype != param_dtype
        or up.scales.dtype != param_dtype
        or up.biases.dtype != param_dtype
    ):
        return False


    K = gate.scales.shape[-1] * _GROUP_SIZE
    N = gate.weight.shape[-2]
    if K % _BLOCK_SIZE != 0 or N % _ROWS_PER_TG != 0:
        return False
    return True


_fallback_warned = False


def _fused_gate_or_fallback(gate_proj, x, idx, x_up, sorted_indices=False):


    gate_idx = idx.reshape(-1, 1) if sorted_indices else idx
    gw = gate_proj["weight"]
    gs = gate_proj["scales"]
    gb = gate_proj.get("biases")
    try:
        return fused_gate_qmv_silu_mul(x, gw, gs, gb, gate_idx, x_up)
    except ValueError as e:
        global _fallback_warned
        if not _fallback_warned:
            logger.warning(
                "Path B: fused gate kernel declined inputs (%s); using the "
                "unfused gate path for this and matching calls.",
                e,
            )
            _fallback_warned = True


        return nn.silu(gate_proj(x, idx, sorted_indices=sorted_indices)) * x_up


class FusedSwitchSwiGLU(nn.Module):


    def __init__(self, switch_mlp):
        super().__init__()


        self._switch_mlp = weakref.proxy(switch_mlp)

    def fused_forward(self, x, indices):

        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        sw = self._switch_mlp
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x_up = sw.up_proj(x, idx, sorted_indices=do_sort)

        swiglu = _fused_gate_or_fallback(
            sw.gate_proj, x, idx, x_up, sorted_indices=do_sort
        )

        out = sw.down_proj(swiglu, idx, sorted_indices=do_sort)

        if do_sort:
            out = _scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)


def patch_switch_glu_with_fused_swiglu(model) -> int:

    from mlx_lm.models.switch_layers import SwitchGLU

    patched = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        sw = getattr(mlp, "switch_mlp", None)
        if not isinstance(sw, SwitchGLU):
            continue
        if not can_fuse(sw):
            continue

        if getattr(sw, "_path_b_installed", False):
            continue


        top_k = getattr(mlp, "top_k", None)
        if top_k is not None:
            _aot_warm_kernel(sw, int(top_k))


        sw._path_b_call = FusedSwitchSwiGLU(sw).fused_forward
        cls = type(sw)
        if "_PathBSubclass" not in cls.__dict__:
            cls._PathBSubclass = type(
                f"{cls.__name__}_PathB",
                (cls,),
                {"__call__": lambda self, *a, **kw: self._path_b_call(*a, **kw)},
            )
        sw.__class__ = cls._PathBSubclass
        sw._path_b_installed = True
        patched += 1

    if patched == 0:
        logger.warning(
            "patch_switch_glu_with_fused_swiglu: no eligible SwitchGLU found"
        )
    else:
        logger.info(f"patch_switch_glu_with_fused_swiglu: patched {patched} layers")
    return patched
