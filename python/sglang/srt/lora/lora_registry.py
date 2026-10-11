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


import asyncio
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from uuid import NAMESPACE_URL, uuid4, uuid5

import msgspec
from msgspec.structs import fields

from sglang.srt.utils import ConcurrentCounter
from sglang.srt.utils.aio_rwlock import RWLock


class LoRARef(msgspec.Struct, frozen=True, array_like=True):


    lora_id: str = msgspec.field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    pinned: Optional[bool] = None

    def __post_init__(self):
        if self.lora_id is None:
            raise ValueError("lora_id cannot be None")

    @staticmethod
    def deterministic_id(lora_name: str, lora_path: str) -> str:

        return uuid5(NAMESPACE_URL, f"{lora_name}\0{lora_path}").hex

    def __str__(self) -> str:
        parts = [
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class LoRARegistry:


    def __init__(self, lora_paths: Optional[List[LoRARef]] = None):
        assert lora_paths is None or all(
            isinstance(lora, LoRARef) for lora in lora_paths
        ), (
            "server_args.lora_paths should have been normalized to LoRARef objects during server initialization. "
            "Please file an issue if you see this error."
        )


        self._registry_lock = RWLock()


        self._registry: OrderedDict[str, LoRARef] = OrderedDict()

        self._counters: Dict[str, ConcurrentCounter] = {}


        if lora_paths:
            for lora_ref in lora_paths:
                self._register_adapter(lora_ref)

    async def register(self, lora_ref: LoRARef):

        async with self._registry_lock.writer_lock:
            self._register_adapter(lora_ref)

    async def unregister(self, lora_name: str) -> str:

        async with self._registry_lock.writer_lock:
            lora_ref = self._registry.get(lora_name, None)
            if lora_ref is None:
                raise ValueError(
                    f"LoRA with name {lora_name} does not exist. Loaded LoRAs: {self._registry.keys()}"
                )
            del self._registry[lora_name]

        return lora_ref.lora_id

    async def acquire(self, lora_name: Union[str, List[str]]) -> Union[str, List[str]]:


        def _lookup(name: str) -> str:
            if name is None:
                return None

            lora_ref = self._registry.get(name, None)
            if lora_ref is None:
                raise ValueError(
                    f"The following requested LoRA adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(name)
            return lora_ref.lora_id

        if isinstance(lora_name, str):
            async with self._registry_lock.writer_lock:
                lora_id = _lookup(lora_name)

            await self._counters[lora_id].increment(notify_all=False)
            return lora_id
        elif isinstance(lora_name, list):
            async with self._registry_lock.writer_lock:
                lora_ids = [_lookup(name) for name in lora_name]


            await asyncio.gather(
                *[
                    self._counters[id].increment(notify_all=False)
                    for id in lora_ids
                    if id is not None
                ]
            )
            return lora_ids
        else:
            raise TypeError("lora_name must be either a string or a list of strings.")

    async def release(self, lora_id: Union[str, List[str]]):


        async with self._registry_lock.reader_lock:
            if isinstance(lora_id, str):
                await self._counters[lora_id].decrement()
            elif isinstance(lora_id, list):
                await asyncio.gather(
                    *[
                        self._counters[id].decrement()
                        for id in lora_id
                        if id is not None
                    ]
                )
            else:
                raise TypeError("lora_id must be either a string or a list of strings.")

    async def wait_for_unload(self, lora_id: str):

        assert (
            lora_id not in self._registry
        ), "wait_for_unload should only be called after the LoRA adapter has been unregistered. "
        assert (
            lora_id in self._counters
        ), "The LoRA ID should still have a counter if it has been registered before."


        await self._counters[lora_id].wait_for_zero()
        del self._counters[lora_id]

    async def get_unregistered_loras(self, lora_name: set[str]):

        async with self._registry_lock.writer_lock:
            unregistered_loras = []

            for name in lora_name:
                if name in self._registry:

                    self._registry.move_to_end(name)
                else:
                    unregistered_loras.append(name)

            return unregistered_loras

    async def lru_lora_name(self, exclude_pinned=False):

        async with self._registry_lock.reader_lock:
            if not exclude_pinned:
                return next(iter(self._registry), None)

            for lora_name, lora_ref in self._registry.items():
                if not lora_ref.pinned:
                    return lora_name
            else:
                return None

    def _register_adapter(self, lora_ref: LoRARef):


        if lora_ref.lora_name in self._registry:
            raise ValueError(
                f"LoRA with name {lora_ref.lora_name} already exists. Loaded LoRAs: {self._registry.keys()}"
            )
        self._registry[lora_ref.lora_name] = lora_ref
        self._counters[lora_ref.lora_id] = ConcurrentCounter()
        return lora_ref

    @property
    def num_registered_loras(self) -> int:

        return len(self._registry)

    def get_all_adapters(self) -> Dict[str, LoRARef]:

        return dict(self._registry)
