from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec


_TRITON_KERNELS = [
    ("allocator", "alloc_extend_kernel"),
    ("allocator", "alloc_decode_kernel"),
    ("common", "write_req_to_token_pool_triton"),
    ("common", "get_last_loc_triton"),
    ("common", "get_last_loc_triton_safe"),
    ("virtual_slot", "alloc_bind_inplace"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"memory.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.memory.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__ = []


register_kernel(
    KernelSpec(
        op="memory.gpu_tensor_hash",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.memory.gpu_tensor_hash:gpu_tensor_hash",
    )
)
register_kernel(
    KernelSpec(
        op="memory.memcpy_triton",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.memory.memcpy_triton:memcpy_triton",
    )
)
