from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


__all__ = ["SRTPlatform", "PlatformEnum"]


class SRTPlatform(DeviceMixin):


    supported_quantization: list[str] = []


    def apply_server_args_defaults(self, server_args) -> None:

        pass


    def get_default_attention_backend(self) -> str:

        raise NotImplementedError

    def get_graph_runner_cls(self) -> type:

        raise NotImplementedError

    def get_mha_kv_pool_cls(self) -> type:

        raise NotImplementedError

    def get_mla_kv_pool_cls(self) -> type:

        raise NotImplementedError

    def get_dsa_kv_pool_cls(self) -> type:

        raise NotImplementedError

    def get_paged_allocator_cls(self) -> type:

        raise NotImplementedError

    def get_compile_backend(self, mode: str | None = None) -> str:

        return "inductor"

    def get_piecewise_backend_cls(self) -> type:

        raise NotImplementedError

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:

        return None


    def supports_fp8(self) -> bool:

        return False

    def support_cuda_graph(self) -> bool:

        return False

    def support_piecewise_cuda_graph(self) -> bool:

        return False


    def init_backend(self) -> None:

        pass


    def get_dispatch_key_name(self) -> str:

        return "native"
