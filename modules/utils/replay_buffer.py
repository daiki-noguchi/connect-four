import os
import random
import sys
from collections import deque
from typing import Deque, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typings import Example  # noqa


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.buffer: Deque[Example] = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, data: Example) -> None:
        self.buffer.append(data)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self) -> List[Example]:
        return random.sample(self.buffer, self.batch_size)
