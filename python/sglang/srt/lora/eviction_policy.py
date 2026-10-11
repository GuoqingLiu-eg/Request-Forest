# Copyright 2023-2024 SGLang Team
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


import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Set

logger = logging.getLogger(__name__)


class EvictionPolicy(ABC):


    @abstractmethod
    def mark_used(self, uid: Optional[str]) -> None:

        pass

    @abstractmethod
    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:

        pass

    @abstractmethod
    def remove(self, uid: Optional[str]) -> None:

        pass


class LRUEvictionPolicy(EvictionPolicy):


    def __init__(self):
        self.access_order = OrderedDict()
        self.total_accesses = 0
        self.eviction_count = 0

    def mark_used(self, uid: Optional[str]) -> None:
        if uid is not None:
            current_time = time.monotonic()

            self.access_order.pop(uid, None)
            self.access_order[uid] = current_time
            self.total_accesses += 1
            logger.debug(f"LoRA {uid} marked as used at {current_time}")

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:


        for uid in list(self.access_order.keys()):
            if uid in candidates:
                logger.debug(f"Selected LoRA {uid} for eviction (LRU)")
                self.eviction_count += 1
                return uid


        if None in candidates:
            logger.debug("Selected None (base model) for eviction")
            self.eviction_count += 1
            return None


        assert False, f"Failed to select LRU victim from candidates: {candidates}"

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.access_order.pop(uid, None)
            logger.debug(f"Removed LoRA {uid} from LRU tracking")


class FIFOEvictionPolicy(EvictionPolicy):


    def __init__(self):
        self.insertion_order = (
            OrderedDict()
        )
        self.eviction_count = 0

    def mark_used(self, uid: Optional[str]) -> None:

        if uid is not None and uid not in self.insertion_order:
            self.insertion_order[uid] = (
                True
            )

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:


        for uid in list(self.insertion_order.keys()):
            if uid in candidates:
                logger.debug(f"Selected LoRA {uid} for eviction (FIFO)")
                self.eviction_count += 1
                return uid


        if None in candidates:
            logger.debug("Selected None (base model) for eviction")
            self.eviction_count += 1
            return None


        assert False, f"Failed to select FIFO victim from candidates: {candidates}"

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.insertion_order.pop(uid, None)


def get_eviction_policy(policy_name: str) -> EvictionPolicy:

    policies = {
        "fifo": FIFOEvictionPolicy,
        "lru": LRUEvictionPolicy,
    }
    if policy_name not in policies:
        raise ValueError(f"Unknown eviction policy: {policy_name}")
    return policies[policy_name]()
