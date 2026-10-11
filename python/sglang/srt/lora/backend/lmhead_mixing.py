from typing import List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.lora.utils import LoRABatchInfo, build_lm_head_pass_segments
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LoRABackendLmHeadMixing:
    def init_lm_head_config(self):
        self.lm_head_batch_info = None


        self.lm_head_pass_batch_infos = None


        self._lm_head_pass_idx = None

    def _get_lm_head_pass_segments(
        self,
        weight_indices: list[int],
        pruned_lens: List[int],
    ) -> Optional[List[Tuple[List[int], List[int]]]]:

        logprobs_chunk_size = envs.SGLANG_LOGPROB_CHUNK_SIZE.get()
        enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGPROB_CHUNK.get()
        pruned_total = sum(pruned_lens)

        if not enable_logprobs_chunk or pruned_total <= logprobs_chunk_size:
            return None

        return build_lm_head_pass_segments(
            weight_indices, pruned_lens, logprobs_chunk_size
        )

    def _prepare_lm_head_batch_info(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        batch_info: LoRABatchInfo,
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:


        pass

    def _build_lm_head_batch_info(
        self,
        lm_head_segments: Tuple[List[int], List[int]],
        batch_info: LoRABatchInfo,
        chunk_size: int,
        expected_tokens: int,
    ) -> LoRABatchInfo:

        pass
