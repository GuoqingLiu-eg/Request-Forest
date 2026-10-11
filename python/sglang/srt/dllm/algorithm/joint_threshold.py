from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

_is_npu = is_npu()


def joint_threshold_update_step_vectorized(
    input_ids_1d: torch.Tensor,
    full_logits_2d: torch.Tensor,
    prompt_masks: torch.Tensor,
    finished: torch.Tensor,
    post_edit_steps: torch.Tensor,
    mask_id: int,
    blk: int,
    threshold: float,
    edit_threshold: float,
    max_post_edit_steps: int,
    penalty_lambda: float,
):

    B = input_ids_1d.shape[0] // blk
    V = full_logits_2d.shape[1]

    input_ids = input_ids_1d.view(B, blk)
    logits = full_logits_2d.view(B, blk, V)

    active = ~finished


    if penalty_lambda > 0:
        prev_ids = input_ids[:, :-1]
        logits[:, 1:, :].scatter_(
            dim=2,
            index=prev_ids.unsqueeze(-1),
            src=torch.full_like(
                prev_ids.unsqueeze(-1), -penalty_lambda, dtype=logits.dtype
            ),
            reduce="add",
        )


    x = torch.argmax(logits, dim=-1)
    p = torch.gather(F.softmax(logits, dim=-1), dim=-1, index=x.unsqueeze(-1)).squeeze(
        -1
    )

    mask_pos = input_ids.eq(mask_id)
    has_mask = mask_pos.any(dim=1)


    no_mask_active = active & (~has_mask)
    post_edit_steps.add_(no_mask_active.to(post_edit_steps.dtype))
    exceeded = post_edit_steps > max_post_edit_steps
    finished |= no_mask_active & exceeded


    eligible = active & (~(no_mask_active & exceeded))


    neg_inf = torch.full_like(p, float("-inf"))
    conf_m2t = torch.where(mask_pos, p, neg_inf)

    m2t = (conf_m2t > threshold) & (eligible & has_mask).view(B, 1)


    hit_any = m2t.any(dim=1)
    need_force = (eligible & has_mask) & (~hit_any)


    best_idx = torch.topk(conf_m2t, k=1, dim=1).indices.squeeze(1)
    rows = torch.arange(B, device=input_ids.device)

    m2t[rows, best_idx] |= need_force


    edit_mask = (~mask_pos) & (~prompt_masks)
    t2t = (p > edit_threshold) & (input_ids != x) & edit_mask
    t2t = t2t & eligible.view(B, 1)


    transfer = m2t | t2t
    any_transfer_row = transfer.any(dim=1)

    finished |= eligible & (~any_transfer_row)


    input_ids.copy_(torch.where(transfer, x, input_ids))

    return any_transfer_row.any()


class JointThreshold(DllmAlgorithm):


    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)


        self.vectorized_decoding = config.algorithm_config.get(
            "vectorized_decoding", _is_npu
        )


        self._use_shared_state = self.vectorized_decoding and not self.fdfo

    def max_steps(self, block_size: int) -> int:
        return block_size + self.max_post_edit_steps + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)


        prompt_mask = input_ids != self.mask_id
        if self._use_shared_state:


            device = forward_batch.input_ids.device
            shared = {
                "prompt_masks": prompt_mask,
                "finished": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "post_edit_steps": torch.zeros(
                    batch_size, dtype=torch.int32, device=device
                ),
            }
            return [shared] * batch_size
        return [
            {
                "post_edit_steps": 0,
                "finished": False,
                "prompt_mask": prompt_mask[i],
            }
            for i in range(batch_size)
        ]

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        if self._use_shared_state:
            return self._step_vectorized_shared(
                forward_batch=forward_batch, full_logits=full_logits, states=states
            )
        if self.vectorized_decoding:
            return self._step_vectorized_fdfo(
                forward_batch=forward_batch, full_logits=full_logits, states=states
            )
        return self._step_per_row(
            forward_batch=forward_batch, full_logits=full_logits, states=states
        )

    def _step_vectorized_shared(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        shared = states[0]
        joint_threshold_update_step_vectorized(
            input_ids_1d=forward_batch.input_ids,
            full_logits_2d=full_logits,
            prompt_masks=shared["prompt_masks"],
            finished=shared["finished"],
            post_edit_steps=shared["post_edit_steps"],
            mask_id=self.mask_id,
            blk=self.block_size,
            threshold=self.threshold,
            edit_threshold=self.edit_threshold,
            max_post_edit_steps=self.max_post_edit_steps,
            penalty_lambda=self.penalty_lambda,
        )
        return shared["finished"].tolist()

    def _step_vectorized_fdfo(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:


        device = forward_batch.input_ids.device
        prompt_masks = torch.stack([state["prompt_mask"] for state in states])
        finished = torch.tensor(
            [state["finished"] for state in states], dtype=torch.bool, device=device
        )
        post_edit_steps = torch.tensor(
            [state["post_edit_steps"] for state in states],
            dtype=torch.int32,
            device=device,
        )

        joint_threshold_update_step_vectorized(
            input_ids_1d=forward_batch.input_ids,
            full_logits_2d=full_logits,
            prompt_masks=prompt_masks,
            finished=finished,
            post_edit_steps=post_edit_steps,
            mask_id=self.mask_id,
            blk=self.block_size,
            threshold=self.threshold,
            edit_threshold=self.edit_threshold,
            max_post_edit_steps=self.max_post_edit_steps,
            penalty_lambda=self.penalty_lambda,
        )

        done = finished.tolist()
        new_post_edit_steps = post_edit_steps.tolist()
        for i, state in enumerate(states):
            state["finished"] = done[i]
            state["post_edit_steps"] = new_post_edit_steps[i]
        return done

    def _step_per_row(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        done: List[bool] = []

        for i in range(batch_size):
            state = states[i]
            if state["finished"]:
                done.append(True)
                continue

            block_start = i * self.block_size
            block_end = block_start + self.block_size
            curr_input_ids = forward_batch.input_ids[block_start:block_end]
            curr_logits = full_logits[block_start:block_end]
            curr_prompt_mask = state["prompt_mask"]

            if self.penalty_lambda > 0:
                prev_ids = curr_input_ids[:-1]
                curr_logits[1:, :].scatter_(
                    1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                )

            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            mask_index = curr_input_ids == self.mask_id
            has_mask = mask_index.any()


            mask_transfer_index = torch.zeros_like(mask_index)
            budget_exhausted = False
            if has_mask:
                confidence = torch.where(mask_index, p, -np.inf)
                mask_transfer_index = confidence > self.threshold
                if not mask_transfer_index.any():
                    _, select_index = torch.topk(confidence, k=1)
                    mask_transfer_index[select_index] = True
            else:
                state["post_edit_steps"] += 1
                if state["post_edit_steps"] > self.max_post_edit_steps:
                    state["finished"] = True
                    budget_exhausted = True

            if not budget_exhausted:

                edit_mask = ~mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )
                transfer_index = mask_transfer_index | edit_transfer_index
                if transfer_index.any():
                    curr_input_ids[transfer_index] = x[transfer_index]
                else:
                    state["finished"] = True


            done.append(state["finished"])

        return done


Algorithm = JointThreshold
