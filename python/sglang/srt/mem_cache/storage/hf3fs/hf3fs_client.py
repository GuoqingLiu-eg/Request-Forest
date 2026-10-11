import logging
import os
from abc import ABC, abstractmethod
from typing import List

import torch


class Hf3fsClient(ABC):


    @abstractmethod
    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):

        pass

    @abstractmethod
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:

        pass

    @abstractmethod
    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:

        pass

    @abstractmethod
    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:

        pass

    @abstractmethod
    def get_size(self) -> int:

        pass

    @abstractmethod
    def close(self) -> None:

        pass

    @abstractmethod
    def flush(self) -> None:

        pass


logger = logging.getLogger(__name__)


class Hf3fsMockClient(Hf3fsClient):


    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):

        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries


        os.makedirs(os.path.dirname(self.path), exist_ok=True)


        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.file, size)

        logger.info(
            f"Hf3fsMockClient initialized: path={path}, size={size}, "
            f"bytes_per_page={bytes_per_page}, entries={entries}"
        )

    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:

        self.check(offsets, tensors)

        results = []

        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize

            try:
                os.lseek(self.file, offset, os.SEEK_SET)
                bytes_read = os.read(self.file, size)

                if len(bytes_read) == size:

                    bytes_tensor = torch.frombuffer(bytes_read, dtype=torch.uint8)
                    typed_tensor = bytes_tensor.view(tensor.dtype).view(tensor.shape)
                    tensor.copy_(typed_tensor)
                    results.append(size)
                else:
                    logger.warning(
                        f"Short read: expected {size}, got {len(bytes_read)}"
                    )
                    results.append(len(bytes_read))

            except Exception as e:
                logger.error(f"Error reading from offset {offset}: {e}")
                results.append(0)

        return results

    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:

        self.check(offsets, tensors)

        results = []

        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize

            try:

                tensor_bytes = tensor.contiguous().view(torch.uint8).flatten()
                data = tensor_bytes.numpy().tobytes()

                os.lseek(self.file, offset, os.SEEK_SET)
                bytes_written = os.write(self.file, data)

                if bytes_written == size:
                    results.append(size)
                else:
                    logger.warning(f"Short write: expected {size}, got {bytes_written}")
                    results.append(bytes_written)

            except Exception as e:
                logger.error(f"Error writing to offset {offset}: {e}")
                results.append(0)

        return results

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:

        pass

    def get_size(self) -> int:

        return self.size

    def close(self) -> None:

        try:
            if hasattr(self, "file") and self.file >= 0:
                os.close(self.file)
                self.file = -1
            logger.info(f"MockHf3fsClient closed: {self.path}")
        except Exception as e:
            logger.error(f"Error closing MockHf3fsClient: {e}")

    def flush(self) -> None:

        try:
            os.fsync(self.file)
        except Exception as e:
            logger.error(f"Error flushing MockHf3fsClient: {e}")
