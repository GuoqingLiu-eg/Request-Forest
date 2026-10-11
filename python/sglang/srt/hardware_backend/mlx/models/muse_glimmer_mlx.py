# Copyright 2023-2026 SGLang Team
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


import math
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache


MUSE_GLIMMER_MLX_FORMAT_VERSION = 1


_OFFSET_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attn_norm.weight",
    "post_attention_layernorm.weight",
    "post_ffn_norm.weight",
)


_VISION_KEY_MARKERS = (
    "vision_encoder",
    "vision_adapter",
    "vision_projection",
    "vision_tower",
    "perception_emb_norm",
)


_RAW_ONLY_KEY_MARKERS = (
    "output_gate_proj",
    "rotary_emb",
    "language_model",
) + _VISION_KEY_MARKERS


_RC_SUFFIX_RENAMES = (
    ("self_attn.gate_proj.weight", "self_attn.output_gate_proj.weight"),
    ("post_attention_layernorm.weight", "post_attn_norm.weight"),
    ("pre_feedforward_layernorm.weight", "post_attention_layernorm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffn_norm.weight"),
)

_RC_PREFIX = "model.language_model."


def flatten_rc_config(config: dict) -> dict:

    text = config["text_config"]

    activation = text.get("hidden_activation", "silu")
    if activation != "silu":
        raise ValueError(
            f"RC config has hidden_activation={activation!r}; this port "
            "hardcodes silu"
        )

    head_dim = int(text.get("head_dim", 128))
    rope_params = text.get("rope_parameters") or {}
    layer_rope_theta = text.get("layer_rope_theta")

    flat = {
        "model_type": "muse_glimmer",
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": head_dim,
        "intermediate_size": text["intermediate_size"],
        "vocab_size": text["vocab_size"],
        "rms_norm_eps": text["rms_norm_eps"],
        "post_norm_eps": text["post_norm_eps"],
        "rope_theta": rope_params.get("rope_theta", text.get("rope_theta", 500_000.0)),
        "max_position_embeddings": text["max_position_embeddings"],
        "qk_scale_factor": text["qk_scale_factor"] * math.sqrt(head_dim),
        "output_multiplier": text["output_multiplier"],
        "output_soft_cap_temp": text.get("final_logit_softcapping"),
        "rope_is_neox_style": True,
        "sliding_window": text["sliding_window"],
    }
    if "layer_types" in text:
        flat["layer_types"] = list(text["layer_types"])
    if layer_rope_theta is not None:
        flat["no_rope_layers"] = [0 if not theta else 1 for theta in layer_rope_theta]
    return flat


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_glimmer"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 19968
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    rope_theta: float = 500_000.0
    max_position_embeddings: int = 16384
    use_qk_norm: bool = True
    qk_scale_factor: float = 43.7840518911
    use_attn_output_gate: bool = True
    output_multiplier: float = 0.19611613513818404
    output_soft_cap_temp: Optional[float] = 20.0
    rope_is_neox_style: bool = False
    normalize_tok_embeddings: bool = True
    sliding_window: int = 2048
    every_n_layers_nope: int = 4
    no_rope_layers: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None


    muse_glimmer_mlx_format: Optional[int] = None

    @classmethod
    def from_dict(cls, params):


        if "text_config" in params:
            params = flatten_rc_config(params)
        return super().from_dict(params)

    def __post_init__(self):


        if self.every_n_layers_nope <= 0:
            raise ValueError(
                f"every_n_layers_nope must be positive, got {self.every_n_layers_nope}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})"
            )

        derived_no_rope = [
            0 if (self.num_hidden_layers - i - 1) % self.every_n_layers_nope == 0 else 1
            for i in range(self.num_hidden_layers)
        ]
        if self.no_rope_layers is None:
            self.no_rope_layers = derived_no_rope
        else:
            if len(self.no_rope_layers) != self.num_hidden_layers:
                raise ValueError(
                    f"no_rope_layers has {len(self.no_rope_layers)} entries but "
                    f"num_hidden_layers is {self.num_hidden_layers}"
                )
            bad_flags = sorted(set(self.no_rope_layers) - {0, 1})
            if bad_flags:
                raise ValueError(
                    f"no_rope_layers contains non-binary entries {bad_flags}; "
                    "each entry must be 0 (NoPE) or 1 (RoPE)"
                )


        derived_layer_types = [
            "full_attention" if rope_flag == 0 else "sliding_attention"
            for rope_flag in self.no_rope_layers
        ]
        if self.layer_types is None:
            self.layer_types = derived_layer_types
        else:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries but "
                    f"num_hidden_layers is {self.num_hidden_layers}"
                )
            bad = sorted(
                set(self.layer_types) - {"full_attention", "sliding_attention"}
            )
            if bad:
                raise ValueError(
                    f"layer_types contains unknown entries {bad}; expected only "
                    "'full_attention' or 'sliding_attention'"
                )
            if self.layer_types != derived_layer_types:
                mismatches = [
                    i
                    for i, (got, want) in enumerate(
                        zip(self.layer_types, derived_layer_types)
                    )
                    if got != want
                ]
                raise ValueError(
                    "layer_types disagrees with no_rope_layers (NoPE layers "
                    "must be the full_attention layers) at layer indices "
                    f"{mismatches}"
                )

        if self.muse_glimmer_mlx_format is not None and (
            self.muse_glimmer_mlx_format != MUSE_GLIMMER_MLX_FORMAT_VERSION
        ):
            raise ValueError(
                f"muse_glimmer_mlx_format {self.muse_glimmer_mlx_format} is not supported by "
                f"this model file (expected {MUSE_GLIMMER_MLX_FORMAT_VERSION}); "
                "regenerate the artifact with a matching packager"
            )


class ScalelessRMSNorm(nn.Module):


    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.dims = dims
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class NoPE(nn.Module):


    dims = 0
    traditional = True

    def __call__(self, x: mx.array, offset: Any = 0) -> mx.array:
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.use_attn_output_gate = args.use_attn_output_gate

        q_dim = args.num_attention_heads * args.head_dim
        kv_dim = args.num_key_value_heads * args.head_dim


        self.q_proj = nn.Linear(
            args.hidden_size,
            2 * q_dim if self.use_attn_output_gate else q_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, args.hidden_size, bias=False)


        self.use_qk_norm = args.use_qk_norm
        if args.use_qk_norm:
            self.q_norm = ScalelessRMSNorm(args.head_dim, args.rms_norm_eps)
            self.k_norm = ScalelessRMSNorm(args.head_dim, args.rms_norm_eps)


            self.scale = args.qk_scale_factor / args.head_dim
        else:
            self.scale = args.head_dim**-0.5

        use_rope = args.no_rope_layers[layer_idx] == 1
        self.rope = (
            nn.RoPE(
                args.head_dim,
                traditional=not args.rope_is_neox_style,
                base=args.rope_theta,
            )
            if use_rope
            else NoPE()
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        H, Hk, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim

        q = self.q_proj(x)
        gate = None
        if self.use_attn_output_gate:


            q, gate = mx.split(q.reshape(B, L, H, 2 * D), 2, axis=-1)
        else:
            q = q.reshape(B, L, H, D)
        k = self.k_proj(x).reshape(B, L, Hk, D)
        v = self.v_proj(x).reshape(B, L, Hk, D)


        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3)
        if gate is not None:
            out = mx.sigmoid(gate) * out
        return self.o_proj(out.reshape(B, L, -1))


class MuseGlimmerMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = MuseGlimmerAttention(args, layer_idx)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, args.post_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = MuseGlimmerMLP(args)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, args.post_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:


        x = x + self.post_attn_norm(
            self.self_attn(self.input_layernorm(x), mask, cache)
        )
        return x + self.post_ffn_norm(self.mlp(self.post_attention_layernorm(x)))


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = (
            ScalelessRMSNorm(args.hidden_size, args.rms_norm_eps)
            if args.normalize_tok_embeddings
            else None
        )
        self.layers = [
            MuseGlimmerDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)


        self.layer_types = list(args.layer_types)
        self.sliding_window = args.sliding_window

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        x = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if self.embed_norm is not None:
            x = self.embed_norm(x)

        if cache is None:
            cache = [None] * len(self.layers)


        masks = {}
        for layer_type in ("full_attention", "sliding_attention"):
            try:
                idx = self.layer_types.index(layer_type)
            except ValueError:
                continue
            window = self.sliding_window if layer_type == "sliding_attention" else None
            if window is not None:
                masks[layer_type] = create_attention_mask(
                    x, cache[idx], window_size=window
                )
            else:
                masks[layer_type] = create_attention_mask(x, cache[idx])

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            x = layer(x, masks[layer_type], c)

        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[Any]:


        return [KVCache() for _ in range(len(self.model.layers))]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache, input_embeddings)

        logits = self.lm_head(hidden).astype(mx.float32)
        if self.args.output_soft_cap_temp is not None:
            cap = self.args.output_soft_cap_temp
            logits = cap * mx.tanh(logits * self.args.output_multiplier / cap)
        else:
            logits = logits * self.args.output_multiplier
        return logits


    def _expected_raw_keys(self) -> set:

        keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for i in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{i}."
            keys.update(
                prefix + suffix
                for suffix in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "input_layernorm.weight",
                    "post_attn_norm.weight",
                    "post_attention_layernorm.weight",
                    "post_ffn_norm.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                )
            )
            if self.args.use_attn_output_gate:
                keys.add(prefix + "self_attn.output_gate_proj.weight")
        return keys

    def sanitize(self, weights: dict) -> dict:
        if self.args.muse_glimmer_mlx_format == MUSE_GLIMMER_MLX_FORMAT_VERSION:


            stray = sorted(
                k
                for k in weights
                if any(marker in k for marker in _RAW_ONLY_KEY_MARKERS)
            )
            if stray:
                raise ValueError(
                    "config.json claims a packaged Muse Glimmer MLX artifact "
                    f"(muse_glimmer_mlx_format={MUSE_GLIMMER_MLX_FORMAT_VERSION}) but the "
                    f"weights contain raw-checkpoint keys {stray[:4]}"
                    f"{'...' if len(stray) > 4 else ''}; the marker belongs "
                    "on packaged artifacts only — repackage from the raw HF export"
                )
            return weights


        if any(k.startswith(_RC_PREFIX) for k in weights):
            weights = _normalize_rc_layout(weights)

        text_keys = {
            k
            for k in weights
            if not any(marker in k for marker in _VISION_KEY_MARKERS)
            and "rotary_emb" not in k
        }
        expected = self._expected_raw_keys()
        missing = sorted(expected - text_keys)
        unexpected = sorted(text_keys - expected)
        if missing or unexpected:
            hint = ""
            gate_missing = all("output_gate_proj" in k for k in missing) and missing
            if gate_missing and not unexpected:
                hint = (
                    " (weights look already fused: if this is a packaged "
                    'artifact, its config.json must carry "muse_glimmer_mlx_format": '
                    f"{MUSE_GLIMMER_MLX_FORMAT_VERSION})"
                )
            raise ValueError(
                "not a complete raw Muse Glimmer HF checkpoint: "
                f"{len(missing)} missing keys {missing[:4]}"
                f"{'...' if len(missing) > 4 else ''}, "
                f"{len(unexpected)} unexpected keys {unexpected[:4]}"
                f"{'...' if len(unexpected) > 4 else ''}{hint}"
            )

        H = self.args.num_attention_heads
        D = self.args.head_dim
        hidden = self.args.hidden_size

        embed_shape = tuple(weights["model.embed_tokens.weight"].shape)
        if embed_shape != (self.args.vocab_size, hidden):
            raise ValueError(
                f"embed_tokens.weight has shape {embed_shape} but config says "
                f"(vocab_size, hidden_size) = ({self.args.vocab_size}, {hidden})"
            )
        raw_q_shape = tuple(weights["model.layers.0.self_attn.q_proj.weight"].shape)
        if raw_q_shape != (H * D, hidden):
            raise ValueError(
                f"raw q_proj.weight has shape {raw_q_shape}, expected "
                f"({H * D}, {hidden}); a width of {2 * H * D} means the gate "
                "is already fused — such artifacts must carry "
                f'"muse_glimmer_mlx_format": {MUSE_GLIMMER_MLX_FORMAT_VERSION} in config.json'
            )

        new_weights = {}
        for name, w in weights.items():

            if "rotary_emb" in name:
                continue
            if any(marker in name for marker in _VISION_KEY_MARKERS):
                continue

            if name.endswith("output_gate_proj.weight"):
                continue


            if name.endswith(_OFFSET_NORM_SUFFIXES):
                w = w + 1.0

            if name.endswith("q_proj.weight") and self.args.use_attn_output_gate:
                gate_name = name.replace("q_proj.weight", "output_gate_proj.weight")
                g = weights[gate_name]
                if tuple(g.shape) != (H * D, hidden):
                    raise ValueError(
                        f"{gate_name} has shape {tuple(g.shape)}, expected "
                        f"({H * D}, {hidden})"
                    )


                w = mx.concatenate(
                    [w.reshape(H, D, hidden), g.reshape(H, D, hidden)], axis=1
                ).reshape(2 * H * D, hidden)

            new_weights[name] = w

        return new_weights


def _normalize_rc_layout(weights: dict) -> dict:

    out = {}
    for name, w in weights.items():
        if any(marker in name for marker in _VISION_KEY_MARKERS):
            continue
        if name.startswith(_RC_PREFIX):
            name = "model." + name[len(_RC_PREFIX) :]
        for rc_suffix, raw_suffix in _RC_SUFFIX_RENAMES:
            if name.endswith(rc_suffix):
                name = name[: -len(rc_suffix)] + raw_suffix
                break
        out[name] = w
    return out


EntryClass = Model
