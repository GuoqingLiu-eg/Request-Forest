# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from enum import Enum

import torch


class FP4KVCacheRecipe(Enum):
    MXFP4 = 1
    NVFP4 = 2


E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0


E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


class FP4MXBlock16KVQuantizeUtil:


    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        b, m, n = tensor.shape


        reshaped = tensor.view(b, m * n // 16, 16)


        block_max = reshaped.abs().max(dim=-1, keepdim=True).values
        scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)


        scaled = reshaped / torch.exp2(scale_exp)


        sign_bits = (scaled < 0).to(torch.uint8) << 3
        abs_vals = scaled.abs()


        bounds = tensor.new_tensor(E2M1_BOUNDS, dtype=torch.float32)
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= bounds, dim=-1)


        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)


        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, scale_factors

    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:

        b, m, n_half = quant_tensor.shape
        n = n_half * 2


        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F


        sign_mask = (fp4_vals & 0x08) != 0
        magnitude_idx = fp4_vals & 0x07


        values = quant_tensor.new_tensor(E2M1_VALUES[:8], dtype=torch.float32)
        float_vals = values[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)


        reshaped = float_vals.view(b, m * n // 16, 16)


        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)


class NVFP4KVQuantizeUtil:


    @staticmethod
    def quantize(
        tensor: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        from sglang.srt.utils import (
            is_sm90_supported,
            is_sm100_supported,
            is_sm120_supported,
        )

        assert (
            is_sm100_supported() or is_sm120_supported() or is_sm90_supported()
        ), "NVFP4 KV cache quantize requires SM100/SM120 or SM90 fallback GPU"

        b, m, n = tensor.shape
        tensor_2d = tensor.reshape(b * m, n)


        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)
        elif global_scale.device != tensor.device:
            raise ValueError(
                "NVFP4 global scale tensor must already be on the KV tensor device."
            )

        if is_sm100_supported() or is_sm120_supported():
            from flashinfer import nvfp4_kv_quantize


            fp4_2d, scales_2d = nvfp4_kv_quantize(tensor_2d, global_scale)
        else:

            from flashinfer import fp4_quantize

            global_scale_inv = 1.0 / global_scale
            fp4_2d, scales_2d = fp4_quantize(
                tensor_2d,
                global_scale_inv,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
                is_sf_8x4_layout=False,
                enable_pdl=None,
            )

        fp4_data = fp4_2d.view(b, m, fp4_2d.shape[-1])
        block_scales = scales_2d.view(b, m, scales_2d.shape[-1]).view(
            torch.float8_e4m3fn
        )
        return fp4_data, block_scales, global_scale

    @staticmethod
    def dequantize(
        quant_tensor: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:

        from sglang.srt.utils import (
            is_sm90_supported,
            is_sm100_supported,
            is_sm120_supported,
        )

        b, m, n_half = quant_tensor.shape


        if isinstance(global_scale, (int, float)):
            global_scale = torch.tensor(
                [global_scale], dtype=torch.float32, device=quant_tensor.device
            )
        elif global_scale.dim() == 0:
            global_scale = global_scale.unsqueeze(0)
        elif global_scale.device != quant_tensor.device:
            raise ValueError(
                "NVFP4 global scale tensor must already be on the KV tensor device."
            )

        if is_sm100_supported() or is_sm120_supported():
            from flashinfer import nvfp4_kv_dequantize

            quant_2d = quant_tensor.view(torch.uint8).reshape(b * m, n_half)
            scales_2d = block_scales.view(torch.uint8).reshape(b * m, -1)
            output_2d = nvfp4_kv_dequantize(
                quant_2d, scales_2d, global_scale, output_dtype=dtype
            )
            return output_2d.reshape(b, m, -1)
        else:
            assert (
                is_sm90_supported()
            ), "NVFP4 KV cache dequantize requires SM100/SM120 or SM90 fallback GPU"

            n = n_half * 2
            fp4_vals = torch.empty(
                b, m, n, dtype=torch.uint8, device=quant_tensor.device
            )
            fp4_vals[..., 0::2] = quant_tensor & 0x0F
            fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F
            values = quant_tensor.new_tensor(E2M1_VALUES, dtype=torch.float32)
            float_vals = values[fp4_vals.long()]
            reshaped = float_vals.view(b, m * n // 16, 16)
            block_scales_float = block_scales.float().unsqueeze(-1)
            scaled = reshaped * block_scales_float
            return (scaled.view(b, m, n) * global_scale).to(dtype)
