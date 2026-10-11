from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec


_TRITON_KERNELS = [
    ("bitmask_ops", "apply_token_bitmask_inplace_triton"),
    ("token_filter_ops", "set_token_filter_triton"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"grammar.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.grammar.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__ = []
