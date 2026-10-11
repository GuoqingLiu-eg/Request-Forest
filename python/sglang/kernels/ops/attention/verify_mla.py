import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_splitkv import _AMD_LAUNCH_KWARGS

MAX_N_SPLITS = 32
TARGET_PROGRAMS = 512

DEFAULT_BLOCK_H = (
    4
)
DEFAULT_BLOCK_N = 64
DEFAULT_NUM_WARPS = 8
_BLOCK_CONFIG = {

    256: (4, 64, 8),
    576: (4, 64, 8),
    64: (4, 256, 4),
}


def block_config(head_dim):

    return _BLOCK_CONFIG.get(
        head_dim, (DEFAULT_BLOCK_H, DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS)
    )


@triton.jit
def _active_splits(seqlen, num_splits, BLOCK_N: tl.constexpr):

    return tl.maximum(1, tl.minimum(num_splits, seqlen // BLOCK_N))


@triton.jit
def _verify_mla_prefix_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    k_scale,
    v_scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_buf_kh,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    H_Q: tl.constexpr,
    L_EXT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    head_block = tl.program_id(1)
    split_kv_id = tl.program_id(2)


    R: tl.constexpr = BLOCK_H * L_EXT

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    active = _active_splits(cur_batch_seq_len, num_splits, BLOCK_N)


    if split_kv_id < active:
        head_start = head_block * BLOCK_H
        offs_h = head_start + tl.arange(0, BLOCK_H)


        if HAS_KV_HEADS:
            kv_head_off_k = (head_start // KV_GROUP_NUM) * stride_buf_kh
            kv_head_off_v = (head_start // KV_GROUP_NUM) * stride_buf_vh
        else:
            kv_head_off_k = 0
            kv_head_off_v = 0
        offs_l = tl.arange(0, L_EXT)
        offs_dn = tl.arange(0, BLOCK_DNOPE)
        offs_dp = tl.arange(0, BLOCK_DPE)
        offs_dv = tl.arange(0, BLOCK_DV)

        cur_q_start = tl.load(qo_indptr + cur_batch)
        l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
        row_mask = tl.reshape((offs_h[:, None] < H_Q) & (offs_l[None, :] < l_ext), (R,))


        kv_len_per_split = cur_batch_seq_len // active
        split_start = kv_len_per_split * split_kv_id
        split_end = tl.where(
            split_kv_id == active - 1, cur_batch_seq_len, split_start + kv_len_per_split
        )


        q_row = tl.reshape(
            (cur_q_start + offs_l)[None, :] * stride_qbs + offs_h[:, None] * stride_qh,
            (R,),
        )
        q_nope = tl.load(
            Q + q_row[:, None] + offs_dn[None, :],
            mask=row_mask[:, None] & (offs_dn[None, :] < NOPE_DIM),
            other=0.0,
        ).to(K_Buffer.dtype.element_ty)
        if PE_DIM > 0:
            q_pe = tl.load(
                Q + q_row[:, None] + (NOPE_DIM + offs_dp)[None, :],
                mask=row_mask[:, None] & (offs_dp[None, :] < PE_DIM),
                other=0.0,
            ).to(K_Buffer.dtype.element_ty)

        e_max = tl.zeros([R], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([R], dtype=tl.float32)
        acc = tl.zeros([R, BLOCK_DV], dtype=tl.float32)


        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_mask, other=0
            )
            base = kv_loc[None, :] * stride_buf_kbs + kv_head_off_k
            k_nope = tl.load(
                K_Buffer + base + offs_dn[:, None],
                mask=(offs_dn[:, None] < NOPE_DIM) & n_mask[None, :],
                other=0.0,
            )
            qk = tl.dot(q_nope, k_nope)
            if PE_DIM > 0:
                k_pe = tl.load(
                    K_Buffer + base + (NOPE_DIM + offs_dp)[:, None],
                    mask=(offs_dp[:, None] < PE_DIM) & n_mask[None, :],
                    other=0.0,
                )
                qk += tl.dot(q_pe, k_pe)
            qk *= sm_scale * k_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))


            v = tl.load(
                V_Buffer
                + kv_loc[:, None] * stride_buf_vbs
                + kv_head_off_v
                + offs_dv[None, :],
                mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                other=0.0,
            )
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max


        acc *= v_scale
        o_row = tl.reshape(
            cur_batch * stride_ob
            + offs_h[:, None] * stride_oh
            + split_kv_id * stride_os
            + offs_l[None, :] * stride_ol,
            (R,),
        )
        tl.store(
            Att_Out + o_row[:, None] + offs_dv[None, :],
            (acc / e_sum[:, None]).to(Att_Out.dtype.element_ty),
            mask=row_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )
        lse_row = tl.reshape(
            cur_batch * stride_lb
            + offs_h[:, None] * stride_lh
            + split_kv_id * stride_ls
            + offs_l[None, :],
            (R,),
        )
        tl.store(Att_Lse + lse_row, e_max + tl.log(e_sum), mask=row_mask)


@triton.jit
def _verify_mla_combine_stage2(
    Att_Out,
    Att_Lse,
    Q,
    K_Extend,
    V_Extend,
    O_Out,
    sm_scale,
    qo_indptr,
    kv_indptr,
    num_splits,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_vebs,
    stride_keh,
    stride_veh,
    stride_oobs,
    stride_ooh,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    if HAS_KV_HEADS:
        kv_head_off_ke = (cur_head // KV_GROUP_NUM) * stride_keh
        kv_head_off_ve = (cur_head // KV_GROUP_NUM) * stride_veh
    else:
        kv_head_off_ke = 0
        kv_head_off_ve = 0

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext


    seqlen = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    active = _active_splits(seqlen, num_splits, BLOCK_N)

    m = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    l_acc = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)
    for s in range(active):
        lse_s = tl.load(
            Att_Lse
            + cur_batch * stride_lb
            + cur_head * stride_lh
            + s * stride_ls
            + offs_l,
            mask=mask_l,
            other=float("-inf"),
        )
        o_s = tl.load(
            Att_Out
            + cur_batch * stride_ob
            + cur_head * stride_oh
            + s * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :],
            mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        new_m = tl.maximum(m, lse_s)
        alpha = tl.exp(m - new_m)
        beta = tl.exp(lse_s - new_m)
        acc = acc * alpha[:, None] + o_s * beta[:, None]
        l_acc = l_acc * alpha + beta
        m = new_m
    o_prefix = acc / l_acc[:, None]
    lse_prefix = m + tl.log(l_acc)


    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + offs_q, mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0
    ).to(tl.float32)
    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs + kv_head_off_ke + offs_d[None, :]
    )
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + kv_head_off_ve
        + offs_dv[None, :]
    )
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)


    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale


    both_valid = mask_l[None, :] & mask_l[:, None]
    if IS_CAUSAL:
        vis = (offs_l[None, :] <= offs_l[:, None]) & both_valid
    else:
        vis = both_valid
    qk = tl.where(vis, qk, float("-inf"))
    m_d = tl.max(qk, 1)
    pd = tl.exp(qk - m_d[:, None])
    denom_d = tl.sum(pd, 1)
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1) / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)


    mm = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - mm)
    wd = tl.exp(lse_draft - mm)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(
        O_Out + offs_oo,
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
    )


class VerifyMLA:
    def __init__(
        self,
        max_bs,
        h_q,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        block_h=DEFAULT_BLOCK_H,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
        kv_group_num=None,
    ):
        self.h_q = h_q


        self.kv_group_num = h_q if kv_group_num is None else kv_group_num


        self.has_kv_heads = self.kv_group_num < h_q
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.nope_dim = v_head_dim
        self.pe_dim = head_dim - v_head_dim
        self.l_ext = l_ext

        min_l_pad = triton.next_power_of_2(triton.cdiv(16, block_h))
        self.l_pad = max(min_l_pad, triton.next_power_of_2(l_ext))
        self.device = device
        self.block_h = block_h
        self.block_n = block_n
        self.num_warps = num_warps
        self.n_head_blocks = triton.cdiv(h_q, block_h)
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        self.max_bs = max_bs

        self.att_out = torch.empty(
            (max_bs, self.h_q, MAX_N_SPLITS, self.l_pad, self.v_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, MAX_N_SPLITS, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _num_splits(self, bs):
        budget = TARGET_PROGRAMS // max(1, bs * self.n_head_blocks)
        return max(1, min(MAX_N_SPLITS, budget))

    def _run_prefix_kernel(
        self,
        bs,
        num_splits,
        q_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
    ):
        grid = (bs, self.n_head_blocks, num_splits)
        _verify_mla_prefix_stage1[grid](
            q_extend,
            k_buffer,
            v_buffer,
            sm_scale,
            k_scale,
            v_scale,
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.att_out,
            self.att_lse,
            num_splits,
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            v_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(1),
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            H_Q=self.h_q,
            L_EXT=self.l_pad,
            BLOCK_H=self.block_h,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            NOPE_DIM=self.nope_dim,
            PE_DIM=self.pe_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DNOPE=triton.next_power_of_2(self.nope_dim),
            BLOCK_DPE=max(1, triton.next_power_of_2(self.pe_dim)),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            num_warps=self.num_warps,
            num_stages=1,
            **_AMD_LAUNCH_KWARGS,
        )

    def _run_combine_kernel(
        self,
        bs,
        num_splits,
        q_extend,
        k_extend,
        v_extend,
        o_out,
        qo_indptr,
        kv_indptr,
        sm_scale,
        is_causal=True,
    ):
        grid = (bs, self.h_q)
        _verify_mla_combine_stage2[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            kv_indptr,
            num_splits,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            v_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=1,
        )

    def __call__(
        self,
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=None,
        k_scale=1.0,
        v_scale=1.0,
        is_causal=True,
    ):
        if o_out is None:
            o_out = torch.empty(
                (q_extend.shape[0], self.h_q, self.v_head_dim),
                dtype=q_extend.dtype,
                device=q_extend.device,
            )
        bs = qo_indptr.shape[0] - 1

        num_splits = self._num_splits(bs)
        self._run_prefix_kernel(
            bs,
            num_splits,
            q_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
        )
        self._run_combine_kernel(
            bs,
            num_splits,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            kv_indptr,
            sm_scale,
            is_causal=is_causal,
        )
        return o_out


_VMLA_CACHE = {}


def _get_vmla(max_bs, h_q, head_dim, v_head_dim, l_ext, device, kv_group_num=None):
    key = (h_q, head_dim, v_head_dim, l_ext, str(device), kv_group_num)
    vk = _VMLA_CACHE.get(key)
    if vk is None:
        block_h, block_n, num_warps = block_config(head_dim)
        if (
            kv_group_num is not None
            and kv_group_num < h_q
            and kv_group_num % block_h != 0
        ):


            block_h = kv_group_num
        vk = VerifyMLA(
            max_bs,
            h_q,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            block_h=block_h,
            block_n=block_n,
            num_warps=num_warps,
            kv_group_num=kv_group_num,
        )
        _VMLA_CACHE[key] = vk
    else:
        vk.grow_buffers(max_bs)
    return vk


def can_handle(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sliding_window_size=-1,
    sinks=None,
    logit_cap=0.0,
    xai_temperature_len=-1,
):


    if sinks is not None:
        return False
    if sliding_window_size is not None and sliding_window_size > 0:
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if not is_causal and custom_mask is not None:
        return False

    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False

    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        return False
    kv_group_num = h_q // h_kv
    if h_kv > 1 and (kv_group_num & (kv_group_num - 1)) != 0:
        return False

    if k_buffer.shape[1] != h_kv or v_buffer.shape[1] != h_kv:
        return False
    if q_extend.shape[2] != k_extend.shape[2]:
        return False
    if q_extend.shape[2] != k_buffer.shape[2]:
        return False
    if v_extend.shape[2] != v_buffer.shape[2]:
        return False


    bs = qo_indptr.shape[0] - 1
    if bs < 1:
        return False


    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    if mle < 1:
        return False


    if q_extend.shape[0] != bs * mle:
        return False
    return True


def verify_shared_kv_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    max_bs=None,
):

    if not can_handle(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
    ):
        return False
    if q_extend.shape[2] < v_extend.shape[2]:
        return False
    if kv_indices.numel() == 0:
        return False

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)
    kv_group_num = h_q // h_kv

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0

    if max_bs is None or max_bs < bs:
        max_bs = bs
    vk = _get_vmla(
        max_bs, h_q, head_dim, v_head_dim, l_ext, q_extend.device, kv_group_num
    )
    vk(
        q_extend,
        k_extend.contiguous(),
        v_extend.contiguous(),
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=o_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        is_causal=is_causal,
    )
    return True
