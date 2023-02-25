import os
import random
import sys
from collections import deque
from typing import Deque, List

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typings import Example  # noqa


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, prioritized: bool = True) -> None:
        self.buffer: Deque[Example] = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.prioritized = prioritized

    def add(self, data: Example) -> None:
        self.buffer.append(data)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self) -> List[Example]:
        if self.prioritized:
            delta_sum = sum([ex.delta for ex in self.buffer])
            prob: np.ndarray = np.array([ex.delta / delta_sum for ex in self.buffer])
            batch_np = np.random.choice(self.buffer, size=self.batch_size, p=prob)
            batch = [ex for ex in batch_np]
        else:
            batch = random.sample(self.buffer, self.batch_size)
        return batch
