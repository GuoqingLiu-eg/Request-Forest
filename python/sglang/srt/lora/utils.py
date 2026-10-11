import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


def warn_if_adapter_targets_embeddings(
    lora_name: str,
    embedding_layer_names: Iterable[str],
    speculative_algorithm: Optional[str],
) -> None:

    if speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return
    modules = sorted(embedding_layer_names)
    if not modules:
        return
    logger.warning(
        "LoRA adapter '%s' targets embedding modules (%s) while EAGLE-family "
        "speculative decoding is enabled. The shared draft consumes their "
        "base weights, so those deltas do not influence drafting and may "
        "reduce the accept rate. Outputs are unaffected.",
        lora_name,
        ", ".join(modules),
    )


@dataclass
class MoELoRABatchInfo:

    seg_indptr: torch.Tensor


    req_to_lora: torch.Tensor


    adapter_enabled: torch.Tensor


    token_lora_mapping: torch.Tensor


@dataclass
class LoRABatchInfo:

    use_cuda_graph: bool


    bs: int


    num_segments: int


    seg_indptr: torch.Tensor


    weight_indices: torch.Tensor


    lora_ranks: torch.Tensor


    scalings: torch.Tensor


    max_len: Optional[int]


    seg_lens: Optional[torch.Tensor]


    permutation: Optional[torch.Tensor]


    expected_tokens: Optional[int] = None


    has_active_lora: bool = False


    req_seg_indptr: Optional[torch.Tensor] = None


    req_weight_indices: Optional[torch.Tensor] = None


    moe_lora_info: Optional[MoELoRABatchInfo] = None


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def copy_weight_into_buffer(
    buffer_view: torch.Tensor,
    weight: torch.Tensor,
) -> None:

    if weight.dtype == buffer_view.dtype:
        buffer_view.copy_(weight, non_blocking=True)
        return

    if weight.device.type == "cpu" and buffer_view.device.type != "cpu":
        weight = weight.to(device=buffer_view.device, non_blocking=True)

    buffer_view.copy_(weight.to(dtype=buffer_view.dtype), non_blocking=True)


def get_hidden_dim(
    module_name: str,
    config: AutoConfig,
    base_model: torch.nn.Module,
    layer_idx: int,
    lora_added_vocab_size: int = 0,
) -> Tuple[int]:


    if hasattr(base_model, "get_hidden_dim"):
        return base_model.get_hidden_dim(module_name, layer_idx)
    else:
        """
        WARNING: get_hidden_dim() is not defined,
        which is used to get the hidden dim for different lora modules
        Use the default one, but please check if it is correct for your model.
        Please implement the function in the model class if it is not.
        You can reference this function in llama.py.
        """
        return get_default_hidden_dim(
            module_name, config, layer_idx, lora_added_vocab_size
        )


def get_default_hidden_dim(
    module_name: str,
    config: AutoConfig,
    layer_idx: int,
    lora_added_vocab_size: int = 0,
) -> Tuple[int]:

    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    if module_name == "qkv_proj":
        return config.hidden_size, head_dim * (
            config.num_attention_heads + config.num_key_value_heads * 2
        )
    elif module_name == "o_proj":
        o_head_dim = getattr(config, "v_head_dim", None) or head_dim
        return (
            o_head_dim * config.num_attention_heads,
            config.hidden_size,
        )
    elif module_name == "gate_up_proj":
        inter = config.intermediate_size
        first_k = getattr(config, "first_k_dense_replace", None)
        moe_freq = getattr(config, "moe_layer_freq", 1)
        if first_k is not None and layer_idx >= first_k and layer_idx % moe_freq == 0:
            moe_inter = getattr(config, "moe_intermediate_size", None)
            n_shared = getattr(config, "n_shared_experts", None)
            if moe_inter is not None and n_shared is not None:
                inter = moe_inter * n_shared
        return config.hidden_size, inter * 2
    elif module_name == "down_proj":
        inter = config.intermediate_size
        first_k = getattr(config, "first_k_dense_replace", None)
        moe_freq = getattr(config, "moe_layer_freq", 1)
        if first_k is not None and layer_idx >= first_k and layer_idx % moe_freq == 0:
            moe_inter = getattr(config, "moe_intermediate_size", None)
            n_shared = getattr(config, "n_shared_experts", None)
            if moe_inter is not None and n_shared is not None:
                inter = moe_inter * n_shared
        return inter, config.hidden_size
    elif module_name == "fused_qkv_a_proj_with_mqa":
        q_lora_rank = getattr(config, "q_lora_rank", None) or 0
        kv_lora_rank = config.kv_lora_rank
        qk_rope_head_dim = config.qk_rope_head_dim
        return (
            config.hidden_size,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
        )
    elif module_name == "q_b_proj":
        return (
            config.q_lora_rank,
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.qk_rope_head_dim),
        )
    elif module_name == "kv_b_proj":
        return (
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
    elif module_name in DSA_INDEXER_LORA_NAMES:
        from sglang.srt.configs.model_config import (
            get_dsa_index_head_dim,
            get_dsa_index_n_heads,
        )

        if module_name == "indexer.wq_b":
            return (
                config.q_lora_rank,
                get_dsa_index_n_heads(config) * get_dsa_index_head_dim(config),
            )
        elif module_name == "indexer.wk":
            return config.hidden_size, get_dsa_index_head_dim(config)
        else:
            return config.hidden_size, get_dsa_index_n_heads(config)
    elif module_name == "gate_up_proj_moe":
        moe_inter = (
            getattr(config, "moe_intermediate_size", None) or config.intermediate_size
        )
        return config.hidden_size, moe_inter * 2
    elif module_name == "down_proj_moe":
        moe_inter = (
            getattr(config, "moe_intermediate_size", None) or config.intermediate_size
        )
        return moe_inter, config.hidden_size
    elif module_name == "embed_tokens":


        return config.vocab_size + lora_added_vocab_size, config.hidden_size
    elif module_name == "lm_head":


        return config.hidden_size, config.vocab_size + lora_added_vocab_size
    else:
        raise NotImplementedError("get_hidden_dim not implemented for " + module_name)


def get_normalized_target_modules(
    target_modules: Union[str, Iterable[str]],
) -> set[str]:


    if isinstance(target_modules, str):
        if target_modules not in ["all", "all-linear"]:
            raise ValueError(
                "Only 'all' or 'all-linear' can be used as the string for target module"
            )
        return {"all"}

    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "out_proj": "out_proj",
        "embed_tokens": "embed_tokens",
        "vocab_emb": "embed_tokens",
        "embeddings": "embed_tokens",
        "word_embeddings": "embed_tokens",
        "lm_head": "lm_head",
        "output": "lm_head",
        "unembed_tokens": "lm_head",
        "q_a_proj": "fused_qkv_a_proj_with_mqa",
        "kv_a_proj_with_mqa": "fused_qkv_a_proj_with_mqa",


        "wq_b": "indexer.wq_b",
        "wk": "indexer.wk",
        "weights_proj": "indexer.weights_proj",
    }

    result = set()
    for name in target_modules:
        base_name = name.split(".")[-1]
        normalized_name = params_mapping.get(base_name, base_name)
        result.add(normalized_name)
    return result


def get_stacked_multiply(
    module_name: str, base_model: Optional[torch.nn.Module] = None
) -> int:

    if base_model is not None and hasattr(base_model, "get_stacked_multiply"):
        return base_model.get_stacked_multiply(module_name)
    stacked_rank = {
        "qkv_proj": 3,
        "in_proj_qkvz": 4,
        "gate_up_proj": 2,
        "gate_up_proj_moe": 2,
        "gate_up_proj_shared_moe": 2,
        "in_proj": 2,
        "fused_qkv_a_proj_with_mqa": 2,
    }
    return stacked_rank[module_name] if module_name in stacked_rank else 1


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:

    best = None
    for target_module in target_modules:
        if target_module in full_module_name:
            if best is None or len(target_module) > len(best):
                best = target_module
    if best is not None:
        return best
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )


EMBEDDING_NAMES = ["embed_tokens", "lm_head"]
ROW_PARALLELISM_LINEAR_LORA_NAMES = [
    "o_proj",
    "out_proj",
    "down_proj",
    "down_proj_moe",
    "down_proj_shared_moe",
    "wo_ud",
]
DSA_INDEXER_LORA_NAMES = frozenset(
    {"indexer.wq_b", "indexer.wk", "indexer.weights_proj"}
)
REPLICATED_LINEAR_LORA_NAMES = [
    "fused_qkv_a_proj_with_mqa",
    "fc1_latent_proj",
    "fc2_latent_proj",
    *DSA_INDEXER_LORA_NAMES,
]


ATTN_TP_LORA_MODULE_NAMES = frozenset(
    {
        "qkv_proj",
        "qkvr",
        "q_b_proj",
        "kv_b_proj",
        "o_proj",
        "out_proj",
        "wo_ud",
        "in_proj",
        "in_proj_qkvz",
    }
)


_KNOWN_LORA_TARGET_MODULES = frozenset(
    {
        "qkv_proj",
        "qkvr",
        "o_proj",
        "wo_ud",
        "out_proj",
        "in_proj",
        "in_proj_qkvz",
        "up_proj",
        "gate_up_proj",
        "down_proj",
        "fc1_latent_proj",
        "fc2_latent_proj",
        "embed_tokens",
        "lm_head",
        "fused_qkv_a_proj_with_mqa",
        "q_b_proj",
        "kv_b_proj",
    }
    | DSA_INDEXER_LORA_NAMES
)


def auto_detect_lora_target_modules(model: "torch.nn.Module") -> set:

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    raw_names: set = set()
    dsa_indexer_leaf_names = {
        target_name.split(".")[-1] for target_name in DSA_INDEXER_LORA_NAMES
    }
    for name, module in model.named_modules():
        if isinstance(module, FusedMoE):
            raw_names.add("gate_up_proj")
            raw_names.add("down_proj")
        elif isinstance(module, ParallelLMHead):
            raw_names.add("lm_head")
        elif isinstance(module, VocabParallelEmbedding):
            raw_names.add("embed_tokens")
        elif isinstance(module, LinearBase):
            parts = name.split(".")
            leaf_name = parts[-1]
            parent_qualified_name = ".".join(parts[-2:])
            if parent_qualified_name in DSA_INDEXER_LORA_NAMES:
                raw_names.add(parent_qualified_name)
            elif leaf_name in dsa_indexer_leaf_names:


                continue
            else:
                raw_names.add(leaf_name)

    normalized = get_normalized_target_modules(raw_names)
    result = normalized & _KNOWN_LORA_TARGET_MODULES


    if hasattr(model, "supported_lora_modules"):
        result.update(set(model.supported_lora_modules) & _KNOWN_LORA_TARGET_MODULES)

    return result


def get_lm_head_lora_b_shard_size(output_dim: int, shard_indices=None) -> int:

    if shard_indices is not None:
        return shard_indices.num_org_elements
    return output_dim


def get_batch_token_counts(forward_batch: ForwardBatch) -> Tuple[int, int]:

    mode = forward_batch.forward_mode
    if mode.is_decode():
        return forward_batch.batch_size, 1
    if mode.is_target_verify():
        num_tokens_per_req = forward_batch.spec_info.draft_token_num
        return forward_batch.batch_size * num_tokens_per_req, num_tokens_per_req
    if mode.is_extend():
        return forward_batch.extend_num_tokens, max(forward_batch.extend_seq_lens_cpu)
    raise ValueError(f"Unsupported forward mode: {mode}")


def generate_sequence_lengths(
    forward_batch: ForwardBatch, device: Optional[torch.device] = None
) -> torch.Tensor:

    device = torch.get_default_device() if device is None else device
    with torch.device(device):
        if forward_batch.forward_mode.is_decode():
            seg_lens = torch.ones(forward_batch.batch_size, dtype=torch.int32)
        elif forward_batch.forward_mode.is_target_verify():
            seg_lens = torch.full(
                size=(forward_batch.batch_size,),
                fill_value=forward_batch.spec_info.draft_token_num,
                dtype=torch.int32,
            )
        elif forward_batch.forward_mode.is_extend():
            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.extend_seq_lens.device == device
                else torch.tensor(
                    forward_batch.extend_seq_lens_cpu,
                    dtype=torch.int32,
                )
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")
    return seg_lens


def get_lm_head_pruned_lens(
    forward_batch: ForwardBatch,
) -> Optional[List[int]]:

    lm_head_pruning = (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    )

    if not lm_head_pruning:
        return None

    if forward_batch.return_logprob:
        pruned_lens = []
        for ext_len, start_len in zip(
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_logprob_start_lens_cpu,
        ):
            pruned_lens.append(1 if ext_len == start_len else ext_len - start_len)
    else:
        pruned_lens = [1] * forward_batch.batch_size

    return pruned_lens


def merge_and_chunk_segments(
    weight_indices: list[int],
    pruned_lens: List[int],
    chunk_size: int,
) -> Tuple[List[int], List[int]]:

    seg_weight_indices: List[int] = []
    seg_lens: List[int] = []
    for wi, pl in zip(weight_indices, pruned_lens):
        if seg_weight_indices and seg_weight_indices[-1] == wi:
            seg_lens[-1] += pl
        else:
            seg_weight_indices.append(wi)
            seg_lens.append(pl)

        while seg_lens[-1] > chunk_size:
            remainder = seg_lens[-1] - chunk_size
            seg_lens[-1] = chunk_size
            seg_weight_indices.append(wi)
            seg_lens.append(remainder)

    return seg_weight_indices, seg_lens


def build_lm_head_pass_segments(
    weight_indices: List[int],
    pruned_lens: List[int],
    logprobs_chunk_size: int,
) -> List[Tuple[List[int], List[int]]]:


    token_wi: List[int] = []
    for wi, pl in zip(weight_indices, pruned_lens):
        token_wi.extend([wi] * pl)
    total = len(token_wi)
    num_passes = (total + logprobs_chunk_size - 1) // logprobs_chunk_size

    result: List[Tuple[List[int], List[int]]] = []
    for i in range(num_passes):
        start = i * logprobs_chunk_size
        end = min((i + 1) * logprobs_chunk_size, total)


        seg_wi: List[int] = []
        seg_lens: List[int] = []
        for t in range(start, end):
            if seg_wi and seg_wi[-1] == token_wi[t]:
                seg_lens[-1] += 1
            else:
                seg_wi.append(token_wi[t])
                seg_lens.append(1)
        result.append((seg_wi, seg_lens))

    return result
