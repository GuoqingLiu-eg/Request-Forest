from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


_INKLING_AR_V4_REGION = 16 * 6144


_INKLING_AR_V5_REGION = 160 * 6144


_INKLING_AR_VEC = 8


_INKLING_AR_SSCONV_OUT_REGION = 16384 * 6144


_INKLING_AR_WORLD_SIZES = (4, 6, 8)


class _InklingArResources(msgspec.Struct):


    rank: int
    world: int
    buffer_ptrs_dev: int
    multicast_ptr: int
    flag_ptrs_dev: int
    state_ptr: int
    v4_in: tuple[int, int]
    v4_out: int
    v5_in: tuple[int, int]
    v5_out: int
    ssconv_out: int
    v4_cur: int = 0
    v5_cur: int = 0
    refs: tuple = ()


_INKLING_AR_CACHE: dict[str, _InklingArResources] = {}


@functools.cache
def _ar_jit():

    if not is_cuda():
        return None
    from sglang.kernels.ops.communication import inkling_all_reduce

    return inkling_all_reduce


@functools.cache
def _ar_fused_jit():
    if not is_cuda():
        return None
    from sglang.kernels.ops.communication import inkling_ar_fused

    return inkling_ar_fused


def _get_inkling_ar_resources(comm) -> _InklingArResources | None:

    key = comm.group.group_name
    cached = _INKLING_AR_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    if not is_cuda():
        return None
    import torch.distributed._symmetric_memory as torch_symm_mem

    jit = _ar_jit()
    world = comm.world_size


    if world & (world - 1) != 0:
        return None
    dev = comm.buffer.device
    hdl = torch_symm_mem.rendezvous(comm.buffer, key)
    flags = torch_symm_mem.empty(jit.flags_numel(world), device=dev, dtype=torch.uint32)
    flags.zero_()
    hflags = torch_symm_mem.rendezvous(flags, key)


    hflags.barrier()
    state = torch.zeros(jit.STATE_SIZE, device=dev, dtype=torch.uint32)
    jit.compile_inkling_all_reduce(comm.dtype, world)


    total = comm.buffer.numel()
    v4reg = _INKLING_AR_V4_REGION
    v5stage = world * _INKLING_AR_V5_REGION
    v5_base = total - 3 * v4reg - 2 * v5stage - _INKLING_AR_V5_REGION


    ssconv_out = v5_base - _INKLING_AR_SSCONV_OUT_REGION
    res = _InklingArResources(
        rank=hdl.rank,
        world=world,
        buffer_ptrs_dev=hdl.buffer_ptrs_dev,
        multicast_ptr=hdl.multicast_ptr,
        flag_ptrs_dev=hflags.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        v4_in=(total - 3 * v4reg, total - 2 * v4reg),
        v4_out=total - v4reg,
        v5_in=(v5_base, v5_base + v5stage),
        v5_out=v5_base + 2 * v5stage,
        ssconv_out=ssconv_out,
        refs=(flags, state, hdl, hflags),
    )
    _INKLING_AR_CACHE[key] = res
    return res


def ensure_inkling_ar_resources(group: GroupCoordinator) -> None:

    comm = group.torch_symm_mem_comm
    if (
        comm is not None
        and not comm.disabled
        and group.world_size in _INKLING_AR_WORLD_SIZES
    ):
        _get_inkling_ar_resources(comm)


def _v4_enabled(comm, num_tokens: int) -> bool:
    if not is_cuda():
        return False
    return _ar_jit().select_ar_config(num_tokens, comm.world_size)[0] == "v4"


_INKLING_AR_FUSED_MAX_TOKENS = 96


_INKLING_AR_FUSED_MAX_TOKENS_VERIFY = 160


_PENDING_AR_SHARED: list = []


def stash_ar_shared(shared: torch.Tensor) -> None:
    assert not _PENDING_AR_SHARED, "unconsumed fused-AR shared partials"
    _PENDING_AR_SHARED.append(shared)


def take_ar_shared(num_tokens: int) -> torch.Tensor | None:

    if not _PENDING_AR_SHARED:
        return None
    shared = _PENDING_AR_SHARED.pop()
    if shared.shape[0] != num_tokens:
        shared = shared[:num_tokens]
    return shared


def ar_sconv_norm_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:

    if not is_cuda():
        return False
    if not (
        envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM.get()
    ):
        return False
    if get_exec().comm.enable_scattered_sconv:


        return False
    fm = forward_batch.forward_mode
    if fm.is_decode():
        max_tokens = _INKLING_AR_FUSED_MAX_TOKENS
    elif fm.is_target_verify():
        max_tokens = _INKLING_AR_FUSED_MAX_TOKENS_VERIFY
    else:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in _INKLING_AR_WORLD_SIZES
        or dtype != comm.dtype
    ):
        return False
    if (
        num_tokens > max_tokens
        or num_tokens > _ar_jit().MAX_BARRIER_BLOCKS
        or hidden % _INKLING_AR_VEC != 0
        or hidden // _INKLING_AR_VEC > 1024
        or num_tokens * hidden > _INKLING_AR_V5_REGION
    ):
        return False
    return _get_inkling_ar_resources(comm) is not None


def ar_sconv_norm_fused(
    input: torch.Tensor,
    residual: torch.Tensor,
    sconv,
    norm,
    forward_batch,
    group: GroupCoordinator,
    shared: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    if shared is None:
        shared = take_ar_shared(input.shape[0])
    hs_out = torch.empty_like(input)
    residual_out = torch.empty_like(residual)
    cur = res.v5_cur
    stage_off = res.v5_in[cur]
    esz = comm.buffer.element_size()
    mc = res.multicast_ptr + stage_off * esz
    local = comm.buffer.data_ptr() + stage_off * esz
    if forward_batch.forward_mode.is_target_verify():
        sconv_cache, cache_indices, cache_mask, conv_weight, inter_out = (
            sconv.verify_fused_ar_inputs(forward_batch)
        )
        _ar_fused_jit().inkling_ar_sconv_norm_verify(
            input,
            residual,
            residual_out,
            hs_out,
            norm.weight,
            norm.variance_epsilon,
            sconv_cache,
            cache_indices,
            cache_mask,
            conv_weight,
            inter_out,
            forward_batch.spec_info.draft_token_num,
            mc,
            local,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
            shared=shared,
        )
    else:
        sconv_cache, cache_indices, cache_mask, conv_weight = (
            sconv.decode_fused_ar_inputs(forward_batch)
        )
        _ar_fused_jit().inkling_ar_sconv_norm(
            input,
            residual,
            residual_out,
            hs_out,
            norm.weight,
            norm.variance_epsilon,
            sconv_cache,
            cache_indices,
            cache_mask,
            conv_weight,
            mc,
            local,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
            track_mask=forward_batch.mamba_track_mask,
            track_indices=forward_batch.mamba_track_indices,
            shared=shared,
        )
    res.v5_cur = 1 - cur
    return hs_out, residual_out


def get_ar_buffer(
    group: GroupCoordinator,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:

    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in _INKLING_AR_WORLD_SIZES
        or dtype != comm.dtype
    ):
        return None
    n = num_tokens * hidden
    nbytes = n * dtype.itemsize
    if nbytes % 4 != 0:
        return None
    if (
        envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()


        and not get_exec().comm.enable_scattered_sconv
    ):
        res = _get_inkling_ar_resources(comm)
        if (
            res is not None
            and _v4_enabled(comm, num_tokens)
            and n <= _INKLING_AR_V4_REGION
            and n % _INKLING_AR_VEC == 0
        ):
            off = res.v4_in[res.v4_cur]
            return comm.buffer[off : off + n].view(num_tokens, hidden)
    if nbytes >= comm.max_size:
        return None
    return comm.buffer[:n].view(num_tokens, hidden)


def symm_mem_all_reduce(
    input: torch.Tensor,
    group: GroupCoordinator,
    *,
    output: torch.Tensor | None = None,
    input_is_ar_buffer: bool = False,
    num_sms: int = 32,
    shared: torch.Tensor | None = None,
) -> torch.Tensor:

    _ = num_sms
    if group.world_size == 1:
        if shared is not None:
            input = input + shared
        if output is None:
            return input
        output.copy_(input)
        return output

    comm = group.torch_symm_mem_comm
    if (
        output is None
        and comm is not None
        and not comm.disabled
        and group.world_size in _INKLING_AR_WORLD_SIZES
        and comm.should_torch_symm_mem_allreduce(input)
    ):
        n = input.numel()
        num_tokens = input.shape[0] if input.dim() >= 2 else n


        res = (
            _get_inkling_ar_resources(comm)
            if envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
            and not get_exec().deterministic.enable_deterministic_inference
            else None
        )


        if res is not None and n % _INKLING_AR_VEC == 0:
            jit = _ar_jit()
            kernel, nb, bs = jit.select_ar_config(num_tokens, res.world)
            if (
                kernel == "v5"
                and n <= _INKLING_AR_V5_REGION
                and input.data_ptr() % 16 == 0
            ):


                cur = res.v5_cur
                stage_off = res.v5_in[cur]
                out_view = comm.buffer[res.v5_out : res.v5_out + n]
                esz = comm.buffer.element_size()
                jit.inkling_multimem_push_oneshot(
                    input.view(-1),
                    out_view,
                    res.multicast_ptr + stage_off * esz,
                    comm.buffer.data_ptr() + stage_off * esz,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                    per_block_barrier=True,
                    shared=shared.view(-1) if shared is not None else None,
                )
                res.v5_cur = 1 - cur
                return out_view.view(input.shape)
            if kernel == "v4" and n <= _INKLING_AR_V4_REGION:

                cur = res.v4_cur
                in_off = res.v4_in[cur]
                out_off = res.v4_out
                in_view = comm.buffer[in_off : in_off + n]
                out_view = comm.buffer[out_off : out_off + n]
                v4_shared = shared
                if not input_is_ar_buffer:
                    if v4_shared is not None:
                        torch.add(input.view(-1), v4_shared.view(-1), out=in_view)
                        v4_shared = None
                    else:
                        in_view.copy_(input.view(-1))
                mc = res.multicast_ptr + in_off * comm.buffer.element_size()
                jit.inkling_multimem_full_oneshot(
                    in_view,
                    out_view,
                    mc,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                    shared=v4_shared.view(-1) if v4_shared is not None else None,
                )
                res.v4_cur = 1 - cur
                return out_view.view(input.shape)

            buf = comm.buffer[:n]
            if not input_is_ar_buffer:
                if shared is not None:
                    torch.add(input.view(-1), shared.view(-1), out=buf)
                else:
                    buf.copy_(input.view(-1))
            elif shared is not None:
                buf.add_(shared.view(-1))
            if kernel in ("v3", "v3b"):
                jit.inkling_multimem_one_shot_fused(
                    buf,
                    res.multicast_ptr,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                    per_block_barrier=(kernel == "v3b"),
                )
                return buf.view(input.shape)
            if kernel == "v2":
                jit.inkling_two_shot_all_reduce_fused(
                    buf,
                    res.buffer_ptrs_dev,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                )
                return buf.view(input.shape)

            torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", comm.group.group_name)
            return buf.view(input.shape)


        buf = comm.buffer[:n]
        if not input_is_ar_buffer:
            if shared is not None:
                torch.add(input.view(-1), shared.view(-1), out=buf)
            else:
                buf.copy_(input.view(-1))
        elif shared is not None:
            buf.add_(shared.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", comm.group.group_name)
        return buf.view(input.shape)

    if shared is not None:
        input = input + shared
    result = group.all_reduce(input)
    if output is None:
        return result
    output.copy_(result)
    return output


def _symm_mem_comm(group: GroupCoordinator, input: torch.Tensor, full_numel: int):

    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        return None
    if not input.is_cuda or input.dtype != comm.dtype:
        return None
    if group.world_size not in _INKLING_AR_WORLD_SIZES:
        return None
    nbytes = full_numel * input.element_size()
    if nbytes % 4 != 0 or nbytes >= comm.max_size:
        return None
    return comm


def reduce_scatter_hidden(
    input: torch.Tensor,
    group: GroupCoordinator,
    *,
    input_is_ar_buffer: bool = False,
) -> torch.Tensor:

    p = group.world_size
    if p == 1:
        return input
    t, h = input.shape
    assert h % p == 0, f"hidden {h} not divisible by tp size {p}"

    comm = _symm_mem_comm(group, input, t * h)
    if comm is not None:


        symm_in = comm.buffer[: t * h].view(t, h)
        if not input_is_ar_buffer:
            symm_in.copy_(input)
        out = torch.empty((t, h // p), dtype=input.dtype, device=input.device)
        torch.ops.symm_mem.reduce_scatter_out(symm_in, comm.group.group_name, True, out)
        return out


    x = input.view(t, p, h // p).movedim(1, 0).reshape(p * t, h // p).contiguous()
    out = torch.empty((t, h // p), dtype=input.dtype, device=input.device)
    group.reduce_scatter_tensor(out, x)
    return out


def all_gather_hidden(input: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:

    p = group.world_size
    if p == 1:
        return input
    t, hp = input.shape

    comm = _symm_mem_comm(group, input, t * hp * p)
    if comm is not None:


        symm_out = comm.buffer[: p * t * hp].view(p * t, hp)
        torch.ops.symm_mem.multimem_all_gather_out(
            input, comm.group.group_name, symm_out
        )
        return symm_out.view(p, t, hp).movedim(0, 1).reshape(t, p * hp)

    return group.all_gather(input, dim=-1)


@functools.cache
def _ar_ssconv_jit():
    if not is_cuda():
        return None
    from sglang.kernels.ops.communication import inkling_ar_scattered_sconv

    return inkling_ar_scattered_sconv


def scattered_ar_sconv_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:

    if not is_cuda():
        return False
    if not (
        get_exec().comm.enable_scattered_sconv
        and envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV.get()
    ):
        return False
    fm = forward_batch.forward_mode
    if fm.is_draft_extend_v2():
        return False
    if not (fm.is_extend() or fm.is_decode()):
        return False


    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    tc_ctx = get_tc_piecewise_forward_context()
    if tc_ctx is not None and not tc_ctx.full_graph:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in (4, 8)
        or dtype != comm.dtype
    ):
        return False
    n = num_tokens * hidden
    if num_tokens == 0 or hidden % (group.world_size * _INKLING_AR_VEC) != 0:
        return False
    res = _get_inkling_ar_resources(comm)
    return res is not None and n <= res.ssconv_out


def ar_scattered_sconv_fused(
    input: torch.Tensor,
    sconv,
    forward_batch,
    group: GroupCoordinator,
    norm=None,
    norm_residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    shared = take_ar_shared(input.shape[0])
    if shared is not None:


        _buf = get_ar_buffer(group, input.shape[0], input.shape[1], input.dtype)
        if _buf is not None:
            torch.add(input, shared, out=_buf)
            input = _buf
        else:
            input = input + shared

    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    jit = _ar_ssconv_jit()
    t, h = input.shape
    n = t * h
    world = res.world
    hc = h // world


    buf = comm.buffer[:n].view(t, h)
    if input.data_ptr() != buf.data_ptr():
        buf.copy_(input)

    (
        sconv_cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        weight,
        query_start_loc,
        cache_indices,
        has_initial_state,
        track_rows,
        track_mask,
        track_dst,
    ) = sconv.extend_fused_ar_inputs(forward_batch)
    del query_start_loc
    fm = forward_batch.forward_mode
    is_verify = fm.is_target_verify()
    kernel_ci = cache_indices.to(torch.int32)
    if is_verify:


        kernel_ci = torch.full_like(cache_indices, -1)


    track_from_cache = False
    if fm.is_decode() and forward_batch.mamba_track_mask is not None:
        b = forward_batch.batch_size
        track_mask = forward_batch.mamba_track_mask[:b]
        track_dst = forward_batch.mamba_track_indices[:b]
        track_from_cache = True

    esz = comm.buffer.element_size()

    if norm is not None and fm.is_decode():


        assert norm_residual is not None
        out_local = comm.buffer[res.ssconv_out : res.ssconv_out + n].view(t, h)
        hs_out = torch.empty_like(norm_residual)
        residual_out = torch.empty_like(norm_residual)
        jit.inkling_ar_col_decode(
            buf,
            out_local,
            norm_residual,
            residual_out,
            hs_out,
            norm.weight,
            float(norm.variance_epsilon),
            sconv_cache,
            kernel_ci,
            cache_mask,
            weight,
            track_mask,
            track_dst,
            res.multicast_ptr,
            res.multicast_ptr + res.ssconv_out * esz,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
        )
        return hs_out, residual_out


    per_block = False
    stream_walk = 0
    use_stream = False
    if is_verify:
        if world == 8:
            if t <= 48:
                nb, bs = 48, 384
            elif t <= 80:
                nb, bs = 80, 384
            elif t <= 144:
                nb, bs = 80, 768
            else:
                nb, bs = 96, 384
        else:
            if t <= 8:
                nb, bs = 32, 384
            elif t <= 48:
                nb, bs = 48, 384
            elif t <= 80:
                nb, bs = 80, 384
            elif t <= 144:
                nb, bs = 148, 384
            elif t <= 192:
                nb, bs = 96, 768
            else:
                nb, bs = 128, 768
    elif world == 8:
        if t < 3072:
            if t <= 128:
                nb, bs = 48, 384
            elif t <= 256:
                nb, bs = 80, 384
            elif t <= 512:
                nb, bs = 192, 768
            elif t <= 768:
                nb, bs = 96, 512
            elif t <= 1024:
                nb, bs = 96, 768
            elif t <= 1536:
                nb, bs = 192, 768
            else:
                nb, bs = 148, 1024
        elif 8192 <= t < 10240:
            nb, bs = 48, 1024
        else:
            use_stream = True
            if t < 4096:
                nb, bs, stream_walk = 148, 256, 16
            elif t < 6144:
                nb, bs, stream_walk = 96, 512, 24
            elif t < 8192:
                nb, bs, stream_walk = 64, 512, 32
            elif t < 16384:
                nb, bs, stream_walk = 148, 128, 0
            else:
                nb, bs, stream_walk = 96, 192, 0
    elif t >= 3072:
        use_stream = True
        if t < 4096:
            nb, bs, stream_walk = 148, 256, 16
        elif t < 6144:
            nb, bs, stream_walk = 0, 0, 24
        elif t < 8192:
            nb, bs, stream_walk = 148, 256, 32
        elif t < 10240:
            nb, bs, stream_walk = 96, 512, 0
        else:
            nb, bs, stream_walk = 148, 256, 0
    elif t <= 128:
        nb, bs = 96, 384
    elif t <= 256:
        nb, bs = 148, 512
    elif t <= 768:
        nb, bs = 128, 256
    elif t <= 3072:
        nb, bs = 148, 384
    else:
        nb, bs = 0, 0

    x_scratch = torch.empty((t, hc), dtype=input.dtype, device=input.device)
    out_local = comm.buffer[res.ssconv_out : res.ssconv_out + n].view(t, h)
    if norm is not None:
        assert norm_residual is not None
        norm_kwargs = dict(
            track_from_cache=track_from_cache,
            out_local=out_local,
            norm_gamma=norm.weight,
            norm_residual=norm_residual,
            norm_out=torch.empty_like(norm_residual),
            norm_eps=float(norm.variance_epsilon),
        )
    else:
        norm_kwargs = dict(track_from_cache=track_from_cache)
    jit.inkling_ar_scattered_sconv(
        buf,
        x_scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        kernel_ci,
        has_initial_state,
        cu,
        si,
        weight,
        track_rows,
        track_mask,
        track_dst,
        res.multicast_ptr,
        res.multicast_ptr + res.ssconv_out * esz,
        res.flag_ptrs_dev,
        res.state_ptr,
        res.rank,
        world,
        activation=sconv.activation,
        use_residual=sconv.use_residual,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=per_block,


        need_scratch=is_verify,


        use_stream=use_stream,
        stream_walk=stream_walk,
        **norm_kwargs,
    )
    if is_verify:

        sconv.verify_fused_ar_finish(forward_batch, x_scratch, cache_indices)
    if norm is not None:
        return norm_kwargs["norm_out"], norm_residual
    return out_local


_INKLING_AR_FW_MIN_TOKENS = 3072


def fullwidth_ar_sconv_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:

    if not is_cuda():
        return False
    if not (
        not get_exec().comm.enable_scattered_sconv
        and envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV.get()
    ):
        return False
    fm = forward_batch.forward_mode
    if fm.is_draft_extend_v2():
        return False
    if fm.is_target_verify():
        return False
    if not fm.is_extend():
        return False
    if num_tokens < _INKLING_AR_FW_MIN_TOKENS:
        return False


    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    tc_ctx = get_tc_piecewise_forward_context()
    if tc_ctx is not None and not tc_ctx.full_graph:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in (4, 8)
        or dtype != comm.dtype
    ):
        return False
    n = num_tokens * hidden
    if num_tokens == 0 or hidden % (group.world_size * _INKLING_AR_VEC) != 0:
        return False
    res = _get_inkling_ar_resources(comm)
    return res is not None and n <= res.ssconv_out


def ar_fullwidth_sconv_fused(
    input: torch.Tensor,
    sconv,
    forward_batch,
    group: GroupCoordinator,
) -> torch.Tensor:

    shared = take_ar_shared(input.shape[0])
    if shared is not None:


        _buf = get_ar_buffer(group, input.shape[0], input.shape[1], input.dtype)
        if _buf is not None:
            torch.add(input, shared, out=_buf)
            input = _buf
        else:
            input = input + shared

    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    jit = _ar_ssconv_jit()
    t, h = input.shape
    n = t * h
    world = res.world
    hc = h // world
    rank = res.rank

    buf = comm.buffer[:n].view(t, h)
    if input.data_ptr() != buf.data_ptr():
        buf.copy_(input)

    (
        sconv_cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        weight,
        query_start_loc,
        cache_indices,
        has_initial_state,
        track_rows,
        track_mask,
        track_dst,
    ) = sconv.extend_fused_ar_inputs(forward_batch)
    del query_start_loc
    kernel_ci = cache_indices.to(torch.int32)


    if world == 8:
        if t < 4096:
            nb, bs, walk = 0, 0, 16
        elif t < 6144:
            nb, bs, walk = 96, 512, 24
        elif t < 8192:
            nb, bs, walk = 64, 512, 32
        elif t < 10240:
            nb, bs, walk = 0, 0, 0
        elif t < 16384:
            nb, bs, walk = 148, 128, 0
        else:
            nb, bs, walk = 96, 192, 0
    elif t < 4096:
        nb, bs, walk = 0, 0, 16
    elif t < 6144:
        nb, bs, walk = 96, 512, 24
    elif t < 8192:
        nb, bs, walk = 148, 256, 32
    elif t < 10240:
        nb, bs, walk = 0, 0, 0
    else:
        nb, bs, walk = 148, 256, 0

    esz = comm.buffer.element_size()
    x_scratch = torch.empty((t, hc), dtype=input.dtype, device=input.device)
    out_local = comm.buffer[res.ssconv_out : res.ssconv_out + n].view(t, h)
    jit.inkling_ar_scattered_sconv(
        buf,
        x_scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        kernel_ci,
        has_initial_state,
        cu,
        si,
        weight[rank * hc : (rank + 1) * hc],
        track_rows,
        track_mask,
        track_dst,
        res.multicast_ptr,
        res.multicast_ptr + res.ssconv_out * esz,
        res.flag_ptrs_dev,
        res.state_ptr,
        rank,
        world,
        activation=sconv.activation,
        use_residual=sconv.use_residual,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=False,
        need_scratch=False,
        use_stream=True,
        stream_walk=walk,
        full_update=True,
        cache_col0=rank * hc,
    )
    return out_local
