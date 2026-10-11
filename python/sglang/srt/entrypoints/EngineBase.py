from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch


class EngineBase(ABC):


    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: Optional[
            Union[bool, Literal["last"], List[Union[bool, Literal["last"]]]]
        ] = None,
        stream: Optional[bool] = None,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
        rid: Optional[Union[List[str], str]] = None,
        priority: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Union[Dict, Iterator[Dict]]:

        pass

    @abstractmethod
    def flush_cache(self):

        pass

    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):

        pass

    def load_lora_adapter(self, lora_name: str, lora_path: str):

        pass

    def unload_lora_adapter(self, lora_name: str):

        pass

    @abstractmethod
    def release_memory_occupation(self):

        pass

    @abstractmethod
    def resume_memory_occupation(self):

        pass

    @abstractmethod
    def shutdown(self):

        pass
