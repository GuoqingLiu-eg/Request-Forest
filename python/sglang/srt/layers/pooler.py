from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1
    MEAN = 2


@dataclass
class EmbeddingPoolerOutput:


    embeddings: torch.Tensor | list[torch.Tensor]
    pooled_hidden_states: Optional[torch.Tensor | list[torch.Tensor]] = None


def pool_hidden_states(
    pooling_type: PoolingType,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:

    if pooling_type == PoolingType.LAST:
        last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        return hidden_states[last_token_indices]
    elif pooling_type == PoolingType.CLS:
        prompt_lens = forward_batch.extend_seq_lens
        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return hidden_states[first_token_flat_indices]
    elif pooling_type == PoolingType.MEAN:
        prompt_lens = forward_batch.extend_seq_lens
        end_indices = torch.cumsum(prompt_lens, dim=0) - 1
        cumulative_hidden_states = torch.cumsum(hidden_states, dim=0)
        sums = cumulative_hidden_states[end_indices]
        preceding_sums = torch.zeros_like(sums)
        preceding_sums[1:] = cumulative_hidden_states[end_indices[:-1]]
        return (sums - preceding_sums) / prompt_lens.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")


def pool_at_delimiter_positions(
    data: torch.Tensor,
    forward_batch: ForwardBatch,
    device: torch.device,
) -> List[torch.Tensor]:

    all_index_tensors: List[torch.Tensor] = []
    delim_counts: List[int] = []
    offset = 0
    for req_idx, req_seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
        indices_tensor = forward_batch.multi_item_delimiter_indices[req_idx]
        n = len(indices_tensor)
        if n > 0:


            all_index_tensors.append(indices_tensor + (offset - 1))
        delim_counts.append(n)
        offset += req_seq_len

    if all_index_tensors:
        index_tensor = torch.cat(all_index_tensors).to(device, non_blocking=True)
    else:
        index_tensor = torch.tensor([], dtype=torch.long, device=device)
    return list(data[index_tensor].split(delim_counts))


def score_and_pool(
    score_head: nn.Module,
    pooler: Pooler,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_ids: torch.Tensor,
) -> EmbeddingPoolerOutput:

    if (
        forward_batch.multi_item_delimiter_indices is not None
        and forward_batch.is_prefill_only
    ):


        per_request_phs = pool_at_delimiter_positions(
            hidden_states, forward_batch, input_ids.device
        )
        phs_flat = torch.cat(per_request_phs, dim=0)
        scores_flat = score_head(phs_flat)
        delim_counts = [t.shape[0] for t in per_request_phs]
        per_request_scores = list(scores_flat.split(delim_counts))
        return EmbeddingPoolerOutput(
            embeddings=per_request_scores,
            pooled_hidden_states=(
                per_request_phs if forward_batch.return_pooled_hidden_states else None
            ),
        )


    pooled_hs = pool_hidden_states(pooler.pooling_type, hidden_states, forward_batch)
    scores = score_head(pooled_hs)
    return EmbeddingPoolerOutput(
        embeddings=scores,
        pooled_hidden_states=(
            pooled_hs if forward_batch.return_pooled_hidden_states else None
        ),
    )


class Pooler(nn.Module):


    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:
        pooled_data = pool_hidden_states(
            self.pooling_type, hidden_states, forward_batch
        )

        if forward_batch.dimensions is not None:
            all_same_dimensions = len(set(forward_batch.dimensions)) == 1
            if all_same_dimensions:
                pooled_data = pooled_data[..., : forward_batch.dimensions[0]]
            else:
                pooled_data = [
                    tensor[..., :dim]
                    for tensor, dim in zip(pooled_data, forward_batch.dimensions)
                ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.normalize(tensor, p=2, dim=-1)
                    for tensor in pooled_data
                ]
            else:
                pooled_data = nn.functional.normalize(pooled_data, p=2, dim=-1)

        return EmbeddingPoolerOutput(embeddings=pooled_data)


class CrossEncodingPooler(nn.Module):


    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:


        prompt_lens = forward_batch.extend_seq_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset : offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i, forward_batch)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:

            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        return EmbeddingPoolerOutput(embeddings=scores)
