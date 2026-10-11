import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncDynamicbatchTokenizer:


    def __init__(
        self,
        tokenizer,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.002,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s


        self._queue: Optional[asyncio.Queue] = None
        self._batcher_task: Optional[asyncio.Task] = None


        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False

    def _ensure_initialized(self):

        if not self._initialized:
            self._queue = asyncio.Queue()
            self._batcher_task = asyncio.create_task(self._dynamic_batch_loop())
            self._initialized = True

    async def __call__(self, prompt: str, **kwargs) -> Any:

        return await self.encode(prompt, **kwargs)

    async def encode(self, prompt: str, **kwargs) -> Any:

        self._ensure_initialized()
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((prompt, kwargs, result_future))
        return await result_future

    async def _dynamic_batch_loop(self):

        while True:
            try:

                prompt, kwargs, result_future = await self._queue.get()


                prompts = [prompt]
                kwargs_list = [kwargs]
                result_futures = [result_future]


                if self._queue.empty():

                    pass
                else:

                    start_time = asyncio.get_running_loop().time()


                    while len(prompts) < self.max_batch_size:
                        elapsed = asyncio.get_running_loop().time() - start_time
                        if elapsed >= self.batch_wait_timeout_s:
                            break

                        remaining_time = self.batch_wait_timeout_s - elapsed
                        try:
                            prompt, kwargs, result_future = await asyncio.wait_for(
                                self._queue.get(), remaining_time
                            )
                            prompts.append(prompt)
                            kwargs_list.append(kwargs)
                            result_futures.append(result_future)
                        except asyncio.TimeoutError:
                            break


                logger.debug(
                    f"AsyncDynamicbatchTokenizer: Processing dynamic batch of size {len(prompts)}"
                )


                await self._process_dynamic_batch(prompts, kwargs_list, result_futures)

            except Exception as e:
                logger.error(f"Error in dynamic batch loop: {e}")


    async def _process_dynamic_batch(
        self,
        prompts: List[str],
        kwargs_list: List[Dict],
        result_futures: List[asyncio.Future],
    ) -> None:


        first_kw = kwargs_list[0]
        can_batch = all(kw == first_kw for kw in kwargs_list[1:])
        kwargs = first_kw if can_batch else None

        try:


            if can_batch and len(prompts) > 1:
                encode_fn = partial(self.tokenizer, prompts, **kwargs)
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, encode_fn
                )

                for i, fut in enumerate(result_futures):
                    if not fut.done():
                        data = {k: v[i] for k, v in results.items()}
                        fut.set_result(data)
            else:

                if len(prompts) > 1 and not can_batch:
                    logger.warning(
                        f"AsyncDynamicbatchTokenizer: Dynamic batching disabled for batch of {len(prompts)} "
                        f"requests due to differing kwargs. This reduces performance benefits. "
                        f"Consider using consistent tokenization parameters across requests."
                    )

                encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                    self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs_list)
                ]
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, encode_fn
                )

                for fut, res in zip(result_futures, results):
                    if not fut.done():
                        fut.set_result(res)
        except Exception as e:
            logger.error(f"Error in dynamic batch processing: {e}")
            for fut in result_futures:
                if not fut.done():
                    fut.set_exception(e)

    def __del__(self):

        if hasattr(self, "_batcher_task") and self._batcher_task:
            if not self._batcher_task.done():
                self._batcher_task.cancel()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
