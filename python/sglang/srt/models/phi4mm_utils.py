# Copyright 2024 SGLang Team
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
#!/usr/bin/env python3
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BlockBase(nn.Module):


    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


def get_activation(name="relu"):

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    return nn.Identity()


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):

    chunk_start_idx = torch.Tensor(
        chunk_start_idx
    ).long()
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )
    seq_range = torch.arange(0, x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[
        :, 1
    ]

    seq_range_expand = (
        torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    )
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class Swish(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:

        return x * self.act_fn(x)


class GLU(nn.Module):


    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()

        if self.act_name == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = Swish()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:

        half_x, gate = x.chunk(2, dim=self.dim)
        return half_x * self.act_fn(gate)


class GLUPointWiseConv(nn.Module):


    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        glu_type="sigmoid",
        bias_in_glu=True,
        causal=False,
    ):
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        if causal:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=(kernel_size - 1),
            )
        else:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = Swish()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_act}")

        if bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, output_dim, 1))
            self.b2 = nn.Parameter(torch.zeros(1, output_dim, 1))

    def forward(self, x):


        x = x.permute([0, 2, 1])
        x = self.ext_pw_conv_1d(x)
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * (
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * (
                    x[:, self.output_dim : self.output_dim * 2, :]
                )
        else:
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :]
                )

        x = x.permute([0, 2, 1])
        return x


class DepthWiseSeperableConv1d(nn.Module):


    def __init__(
        self,
        input_dim,
        depthwise_seperable_out_channel,
        kernel_size,
        depthwise_multiplier,
        padding=0,
    ):
        super().__init__()

        self.dw_conv = nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            1,
            padding=padding,
            groups=input_dim,
        )

        if depthwise_seperable_out_channel != 0:
            self.pw_conv = nn.Conv1d(
                input_dim * depthwise_multiplier,
                depthwise_seperable_out_channel,
                1,
                1,
                0,
            )
        else:
            self.pw_conv = nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def forward(self, x):

        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x


class ConvModule(nn.Module):


    def __init__(
        self,
        input_dim,
        ext_pw_out_channel,
        depthwise_seperable_out_channel,
        ext_pw_kernel_size,
        kernel_size,
        depthwise_multiplier,
        dropout_rate,
        causal=False,
        batch_norm=False,
        chunk_se=0,
        chunk_size=18,
        activation="relu",
        glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        export=False,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal

        self._add_ext_pw_layer()

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(input_dim)

        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        self.export = export

        if causal:
            padding = 0 if export else kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = nn.Linear(depthwise_seperable_out_channel, input_dim)
        else:
            if depthwise_multiplier != 1:
                self.ln2 = nn.Linear(input_dim * depthwise_multiplier, input_dim)

    def _add_ext_pw_layer(self):

        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = (
            nn.Identity()
        )
        self.squeeze_excitation = nn.Identity()
        self.apply_ln1 = self.fix_len1 = False

        if self.ext_pw_out_channel != 0:
            if self.causal:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1),
                )
                if self.ext_pw_kernel_size > 1:
                    self.fix_len1 = True
                else:
                    self.fix_len1 = False
            else:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1) // 2,
                )
                self.fix_len1 = False

            if self.linear_glu_in_convm:
                self.glu = GLULinear(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.glu_type,
                    self.bias_in_glu,
                )
            else:
                self.glu = GLUPointWiseConv(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    self.glu_type,
                    self.bias_in_glu,
                    self.causal,
                )

            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = torch.nn.Parameter(torch.ones(3))
            self.pw_conv_simplify_b = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x):

        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, : -(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1

        x = x.permute([0, 2, 1])

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.permute([0, 2, 1])
            x = self.ln2(x)
            x = x.permute([0, 2, 1])
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, : -(self.ext_pw_kernel_size - 1)]

            if self.apply_ln1:
                x = x.permute([0, 2, 1])
                x = self.ln1(x)
                x = x.permute([0, 2, 1])

            x = x.permute([0, 2, 1])
        else:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = x.squeeze(1)

        x = self.dropout(x)
        return x


class GLULinear(nn.Module):


    def __init__(
        self,
        input_dim,
        output_dim,
        glu_type="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type)

    def forward(self, x):

        x = self.linear(x)
        return self.glu_act(x)


class FeedForward(nn.Module):


    def __init__(
        self,
        d_model,
        d_inner,
        dropout_rate,
        activation="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, activation, bias_in_glu)
        self.net = nn.Sequential(
            module,
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):

        out = self.net(self.layer_norm(x))

        return out


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class T5RelativeAttentionLogitBias(nn.Module):


    def __init__(self, num_heads, num_buckets=-1, max_distance=1000, symmetric=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.symmetric = symmetric
        self._skip_bucketing = self.num_buckets < 0
        if self._skip_bucketing:
            self.num_buckets = max_distance
        else:
            raise NotImplementedError(
                "T5 attention bias with bucketed positions is not yet tested"
            )
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, x):

        maxpos = x.size(1)
        context_position = torch.arange(maxpos, device=x.device, dtype=torch.long)[
            :, None
        ]
        memory_position = torch.arange(maxpos, device=x.device, dtype=torch.long)[
            None, :
        ]
        relative_position = memory_position - context_position


        relative_position = relative_position.masked_fill(
            relative_position < -self.max_distance, -self.max_distance
        )
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )


        if self._skip_bucketing:
            bias_idx = relative_position
        else:
            bias_idx = self._bucket_relative_position(relative_position)
        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx += self.num_buckets // 2

        t5_rel_att_bias = self.bias_values(bias_idx)
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(0)

        return t5_rel_att_bias

    def _bucket_relative_position(self, relative_position):


        relative_buckets = 0
        if not self.causal:
            self.num_buckets //= 2
            relative_buckets += (relative_position > 0).to(
                torch.long
            ) * self.num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )


        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact


        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, self.num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets


class AbsolutePositionalEncoding(nn.Module):


    def __init__(self, d_model, dropout_rate, max_len=5000):

        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):

        if self.pe is not None and self.pe.size(1) >= x.size(1):
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):

        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MeanVarianceNormLayer(nn.Module):


    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.global_mean = nn.Parameter(torch.zeros(input_size))
        self.global_invstd = nn.Parameter(torch.ones(input_size))

    def forward(self, input_: Tensor) -> Tensor:

        return (input_ - self.global_mean) * self.global_invstd


class CausalConv1D(nn.Conv1d):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (
                isinstance(padding, list)
                and len(padding) == 2
                and padding[0] + padding[1] == kernel_size - 1
            ):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache


class CausalConv2D(nn.Conv2d):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self,
        x,
    ):
        x = F.pad(
            x,
            pad=(self._left_padding, self._right_padding, 0, 0),
        )
        x = super().forward(x)
        return x


class NemoConvSubsampling(torch.nn.Module):


    def __init__(
        self,
        feat_in,
        feat_out,
        subsampling_factor=4,
        subsampling="dw_striding",
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_causal_cond = subsampling in (
            "dw_striding",
            "striding",
            "striding_conv1d",
        )

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a " "power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0


            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=(
                                feat_out
                                if self._sampling_num == i + 1
                                else conv_channels
                            ),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=(
                                feat_out
                                if self._sampling_num == i + 1
                                else conv_channels
                            ),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "dw_striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2


            layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=(
                            feat_out if self._sampling_num == 1 else conv_channels
                        ),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=(
                                feat_out
                                if self._sampling_num == i + 2
                                else conv_channels
                            ),
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["dw_striding", "striding"]:
            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = torch.nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, mask):

        x = x.unsqueeze(1) if self.conv2d_subsampling else x.transpose(1, 2)


        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:


                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                need_to_split = torch.numel(x) > x_ceil
            else:

                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:
                    if self._subsampling == "dw_striding":
                        x = self.conv_split_by_channel(x)
                    else:
                        x = self.conv(x)
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)


        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))

        else:
            x = x.transpose(1, 2)

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        if self.is_causal and self.subsampling_causal_cond:
            feature_lens_remainder = feature_lens % self.subsampling_factor
            padding_length[feature_lens_remainder != 1] += 1
        pad_mask = torch.arange(0, max_audio_length, device=x.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        return x, pad_mask.unsqueeze(1)

    def reset_parameters(self):

        if self._subsampling == "dw_striding":
            with torch.no_grad():

                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size**2) ** -0.5
                pw_max = self._conv_channels**-0.5

                torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
                torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)


                fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
                torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x):

        b, _, _, _ = x.size()
        if b == 1:
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:


            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        return (
            torch.cat(
                [self.conv(chunk) for chunk in torch.split(x, new_batch_size, 0)]
            ),
            True,
        )

    def conv_split_by_channel(self, x):

        x = self.conv[0](x)
        x = self.conv[1](x)

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:


                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(
                self.conv[i * 3 + 2], new_c, x
            )


            x = torch.cat(
                [self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)],
                2,
            )
            x = self.conv[i * 3 + 4](x)
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):


        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk,
                    pad=(
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(
        self, subsampling_conv_chunking_factor: int
    ):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a " "power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):

    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        lengths = torch.ceil(lengths) if ceil_mode else torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class AttModule(nn.Module):


    def __init__(self):
        super().__init__()
        self.export_mode = False

    def set_export(self, mode=True):

        self.export_mode = mode

    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        att_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:

        return x, memory, pos_emb, att_mask


class AttBlock(BlockBase, AttModule):


    def memory_dims(self, max_len=False):

        return (1, self.input_size)


def masked_softmax(
    scores,
    mask: Optional[Tensor],
):
    if mask is not None:
        mask = mask.unsqueeze(1).eq(0)
        scores = scores.masked_fill(mask, -torch.inf)
        attn = torch.softmax(scores, dim=-1).masked_fill(
            mask, 0.0
        )
    else:
        attn = torch.softmax(scores, dim=-1)
    return attn


class MultiHeadedAttention(nn.Module):


    inv_sqrt_d_k: torch.jit.Final[float]
    h: torch.jit.Final[int]
    h_k: torch.jit.Final[int]
    g: torch.jit.Final[int]

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        attention_inner_dim=-1,
        glu_type="swish",
        bias_in_glu=True,
        use_pt_scaled_dot_product_attention=False,
        n_value=-1,
        group_size: int = 1,
    ):
        super().__init__()
        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0


        self.d_k = attention_inner_dim // n_head
        self.inv_sqrt_d_k = 1.0 / math.sqrt(self.d_k)
        self.h = n_head
        assert n_head % group_size == 0, "group_size must divide n_head"
        self.g = group_size
        self.h_k = n_head // group_size

        self.linear_q = nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = nn.Linear(n_value, attention_inner_dim // group_size)
        self.linear_out = nn.Linear(attention_inner_dim // group_size, n_value)

        self.attn = torch.jit.Attribute(None, Optional[Tensor])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.use_pt_scaled_dot_product_attention = use_pt_scaled_dot_product_attention

        if use_pt_scaled_dot_product_attention and group_size > 1:
            raise ValueError("Cannot use PT Scaled Attention with GQA")


        self.quant_q = torch.ao.quantization.QuantStub()
        self.quant_x = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.ffunc = torch.ao.nn.quantized.FloatFunctional()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_k: Tensor,
        pos_v: Tensor,
        mask: Optional[Tensor],
        relative_attention_bias: Optional[Tensor] = None,
    ):

        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h_k, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h_k, self.d_k)
        q = (
            q.transpose(1, 2)
            if self.use_pt_scaled_dot_product_attention and not torch.jit.is_scripting()
            else q.transpose(1, 2) * self.inv_sqrt_d_k
        )
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_pt_scaled_dot_product_attention and not torch.jit.is_scripting():
            attn_mask = None
            if mask is not None:
                mask = mask.unsqueeze(1)
                if relative_attention_bias is not None:
                    attn_mask = mask + relative_attention_bias
                else:
                    attn_mask = mask
                if mask.dtype != q.dtype:
                    attn_mask = attn_mask.to(q.dtype)

            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                    torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                ]
            ):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_rate,
                )
        else:
            if self.h != self.h_k:
                q = q.reshape(n_batch, self.g, self.h_k, -1, self.d_k)
                A = torch.einsum("b g h t d, b h s d -> b h t s", q, k)
            else:
                A = torch.matmul(q, k.transpose(-2, -1))
            if pos_k is not None:
                if self.h != self.h_k:
                    B = torch.einsum("b g h t d, t s d -> b h t s", q, pos_k)
                else:
                    reshape_q = (
                        q.contiguous()
                        .view(n_batch * self.h, -1, self.d_k)
                        .transpose(0, 1)
                    )
                    B = torch.matmul(
                        reshape_q, pos_k.transpose(-2, -1)
                    )
                    B = B.transpose(0, 1).view(
                        n_batch, self.h, pos_k.size(0), pos_k.size(1)
                    )
                scores = A + B
            else:
                scores = A

            if relative_attention_bias is not None:
                scores = scores + relative_attention_bias

            attn = masked_softmax(scores, mask)

            self.attn = attn

            p_attn = self.dropout(attn)
            x = torch.matmul(p_attn.to(v.dtype), v)
            if pos_v is not None:
                reshape_attn = (
                    p_attn.contiguous()
                    .view(n_batch * self.h, pos_v.size(0), pos_v.size(1))
                    .transpose(0, 1)
                )

                attn_v = (
                    torch.matmul(reshape_attn, pos_v)
                    .transpose(0, 1)
                    .contiguous()
                    .view(n_batch, self.h, pos_v.size(0), self.d_k)
                )
                x = x + attn_v
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h_k * self.d_k)
        )

        return self.linear_out(x)


class MultiSequential(torch.nn.Sequential):


    @torch.jit.ignore
    def forward(self, *args):

        for m in self:
            args = m(*args)
        return args


def get_offset(input_layer: str, time_reduction: int):

    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 4:
        return 3
    if input_layer in ("conv2d",) and time_reduction == 6:
        return 1
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 8:
        return 7
    return 0


def unfold_tensor(xs_pad, max_seq_len):

    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)

    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )
    new_bsz, _, slen = xs_pad.shape

    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)

    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()

    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad
