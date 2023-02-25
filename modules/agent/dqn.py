import os
import sys
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from conf.game_conf import BOARD_COLUMN  # noqa
from modules.typings import BoardType  # noqa


@dataclass
class DQNParams:
    gamma: float = 0.98
    action_size: int = BOARD_COLUMN
    lr: float = 0.0005
    epsilon: float = 0.1
    buffer_size: float = 10000
    batch_size: float = 32


class QNet(nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        out_channels = 128
        hidden_size = 64
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(out_channels * 2 * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._one_hot(x)
        batch_size = x.size()[0]
        x = self.conv_block(x.float())
        x = x.reshape(batch_size, -1)
        x = self.fc_block(x)
        x = F.softmax(x, dim=1)
        return x

    def _one_hot(self, x: Tensor) -> Tensor:
        x = torch.where(x < 0, 2, x).to(torch.long)
        x = nn.functional.one_hot(x, num_classes=3)
        if x.ndim == 3:
            return x.permute(2, 0, 1).unsqueeze(0)
        return x.permute(0, 3, 1, 2)


class DQNAgent:
    def __init__(self, params: DQNParams) -> None:
        self.gamma = params.gamma
        self.lr = params.lr
        self.epsilon = params.epsilon
        self.buffer_size = params.buffer_size
        self.batch_size = params.batch_size
        self.action_size = params.action_size

        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: BoardType) -> int:
        """e-greedy法によって行動選択する

        Args:
            state (BoardType): 現在のゲームボードの状態

        Returns:
            int: 行動
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.from_numpy(deepcopy(state))
            qs = self.qnet(state)
            return int(qs.argmax().item())

    def update(
        self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor
    ) -> None:
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done.to(torch.int)) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self) -> None:
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def play(self, state: BoardType) -> int:
        state = torch.tensor(state[np.newaxis, :])
        qs = self.qnet(state)
        return int(qs.argmax().item())
