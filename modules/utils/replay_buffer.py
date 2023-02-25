import os
import random
import sys
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
from torch import Tensor

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

    def get_batch(self) -> Tuple[Tensor, ...]:
        data = random.sample(self.buffer, self.batch_size)

        states = torch.tensor(np.stack([x.state for x in data]))
        actions = torch.tensor(np.array([x.action for x in data]), dtype=torch.long)
        rewards = torch.tensor(np.array([x.reward for x in data]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([x.next_state for x in data]))
        players = torch.tensor(np.stack([x.player for x in data]))
        done_list = torch.tensor(np.stack([x.done for x in data]))
        return states, actions, rewards, next_states, players, done_list