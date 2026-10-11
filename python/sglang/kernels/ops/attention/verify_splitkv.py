import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_MIN_BLOCK_KV = 32


_IS_HIP = is_hip()
_AMD_LAUNCH_KWARGS = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16} if _IS_HIP else {}


DEFAULT_N_SPLITS = 8
DEFAULT_BLOCK_N = 32
DEFAULT_NUM_WARPS = 4
_BLOCK_CONFIG = {

    256: (32, 4),
}


def block_config(head_dim):

    return _BLOCK_CONFIG.get(head_dim, (DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS))


ADAPTIVE_SPLITS = True
MAX_N_SPLITS = 16
MIN_N_SPLITS = 4


def choose_n_splits(avg_seqlen):

    if not ADAPTIVE_SPLITS:
        return DEFAULT_N_SPLITS
    s = int(avg_seqlen)
    if s < 4096:
        n = 4
    elif s < 8192:
        n = 8
    else:
        n = 16
    if n < MIN_N_SPLITS:
        n = MIN_N_SPLITS
    if n > MAX_N_SPLITS:
        n = MAX_N_SPLITS
    return n


@triton.jit
def _verify_prefix_stage1(
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
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)


    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx


    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:

        offs_q = (
            (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q + offs_q,
            mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        q_k = q.to(K_Buffer.dtype.element_ty)

        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )

            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_d[:, None] < HEAD_DIM) & n_mask[None, :],
                other=0.0,
            )
            qk = tl.dot(q_k, k)
            qk *= sm_scale * k_scale

            qk = tl.where(n_mask[None, :], qk, float("-inf"))


            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
            v = tl.load(
                V_Buffer + offs_buf_v,
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

        offs_o = (
            cur_batch * stride_ob
            + cur_head * stride_oh
            + split_kv_id * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_o,
            acc / e_sum[:, None],
            mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )

        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_l)
    else:

        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(
            Att_Lse + offs_lse,
            tl.zeros([L_EXT], tl.float32) - float("inf"),
            mask=mask_l,
        )


@triton.jit
def _verify_combine_stage2(
    Att_Out,
    Att_Lse,
    Q,
    K_Extend,
    V_Extend,
    O_Out,
    sm_scale,
    qo_indptr,
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
    stride_keh,
    stride_vebs,
    stride_veh,
    stride_oobs,
    stride_ooh,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)
    offs_s = tl.arange(0, N_SPLITS)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext


    offs_lse = (
        cur_batch * stride_lb
        + cur_head * stride_lh
        + offs_s[:, None] * stride_ls
        + offs_l[None, :]
    )
    lse = tl.load(
        offs_lse + Att_Lse,
        mask=mask_l[None, :],
        other=float("-inf"),
    )
    m_p = tl.max(lse, 0)
    w = tl.exp(lse - m_p[None, :])
    denom_p = tl.sum(w, 0)


    offs_ao = (
        cur_batch * stride_ob
        + cur_head * stride_oh
        + offs_s[:, None, None] * stride_os
        + offs_l[None, :, None] * stride_ol
        + offs_dv[None, None, :]
    )
    ao = tl.load(
        offs_ao + Att_Out,
        mask=mask_l[None, :, None] & (offs_dv[None, None, :] < V_HEAD_DIM),
        other=0.0,
    )
    o_prefix = tl.sum(ao * w[:, :, None], 0)
    o_prefix = o_prefix / denom_p[:, None]
    lse_prefix = m_p + tl.log(denom_p)


    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + offs_q,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs
        + cur_kv_head * stride_keh
        + offs_d[None, :]
    )
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + cur_kv_head * stride_veh
        + offs_dv[None, :]
    )
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)


    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale

    causal = (offs_l[None, :] <= offs_l[:, None]) & mask_l[None, :] & mask_l[:, None]
    qk = tl.where(causal, qk, float("-inf"))
    m_d = tl.max(qk, 1)
    pd = tl.exp(qk - m_d[:, None])
    denom_d = tl.sum(pd, 1)
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1)
    o_draft = o_draft / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)


    m = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - m)
    wd = tl.exp(lse_draft - m)
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


class VerifySplitKV:


    def __init__(
        self,
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        n_splits=DEFAULT_N_SPLITS,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
    ):
        self.h_q = h_q
        self.h_kv = h_kv
        self.group = h_q // h_kv
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.l_ext = l_ext
        self.l_pad = triton.next_power_of_2(l_ext)
        self.device = device
        self.n_splits = n_splits
        self.block_n = block_n
        self.num_warps = num_warps
        self._alloc(max_bs)

    def _alloc(self, max_bs):

        self.max_bs = max_bs
        self.att_out = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _run_prefix_kernel(
        self,
        bs,
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
        grid = (bs, self.h_q, self.n_splits)
        _verify_prefix_stage1[grid](
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
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            num_warps=self.num_warps,
            num_stages=1,
            **_AMD_LAUNCH_KWARGS,
        )

    def _run_combine_kernel(
        self, bs, q_extend, k_extend, v_extend, o_out, qo_indptr, sm_scale
    ):
        grid = (bs, self.h_q)
        _verify_combine_stage2[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
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
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            num_warps=1,
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
    ):
        if o_out is None:
            o_out = torch.empty(
                (q_extend.shape[0], self.h_q, self.v_head_dim),
                dtype=q_extend.dtype,
                device=q_extend.device,
            )


        bs = qo_indptr.shape[0] - 1

        self._run_prefix_kernel(
            bs,
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
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            sm_scale,
        )
        return o_out


_VK_CACHE = {}


def _get_vk(
    max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device, n_splits=DEFAULT_N_SPLITS
):
    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, str(device), n_splits)
    vk = _VK_CACHE.get(key)
    if vk is None:
        block_n, num_warps = block_config(head_dim)
        vk = VerifySplitKV(
            max_bs,
            h_q,
            h_kv,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            n_splits=n_splits,
            block_n=block_n,
            num_warps=num_warps,
        )
        _VK_CACHE[key] = vk
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
    if not is_causal:
        return False

    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False

    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        return False

    if k_buffer.shape[1] != h_kv or v_buffer.shape[1] != h_kv:
        return False
    if q_extend.shape[2] != k_extend.shape[2]:
        return False
    if q_extend.shape[2] != k_buffer.shape[2]:
        return False
    if v_extend.shape[2] != v_buffer.shape[2]:
        return False


    if q_extend.shape[2] != v_extend.shape[2]:
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


def verify_splitkv_fwd(
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

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)

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


    avg_seqlen = kv_indices.shape[0] / max(1, bs)
    n_splits = choose_n_splits(avg_seqlen)


    if max_bs is None or max_bs < bs:
        max_bs = bs
    vk = _get_vk(
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        q_extend.device,
        n_splits=n_splits,
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
    )

    return True
