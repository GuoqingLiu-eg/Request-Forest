import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


def get_pdl_launch_metadata() -> tuple[bool, dict]:

    enable_pdl = is_arch_support_pdl()
    return enable_pdl, ({"launch_pdl": True} if enable_pdl else {})


@triton.jit
def _resolve_token_positions(
    sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER: tl.constexpr
):

    if SORTED_BY_ADAPTER:
        return tl.load(
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len
        ).to(tl.int64)
    return (seg_start + s_offset).to(tl.int64)
