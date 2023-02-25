import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.models.layers import DropPath
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from conf.game_conf import BOARD_COLUMN  # noqa
from modules.typings import BoardType, Example  # noqa

LayerNormDataFormat = Literal["channels_last", "channels_first"]


@dataclass
class DQNParams:
    gamma: float = 0.98
    action_size: int = BOARD_COLUMN
    lr: float = 0.0005
    epsilon: float = 0.1
    double_dqn: bool = True
    buffer_size: int = 10000
    batch_size: int = 32


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: LayerNormDataFormat = "channels_last",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class QVNet(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        action_size (int): Number of action spaces
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        action_size: int,
        drop_path_rate: float = 0.2,
        dims: Union[List[int], None] = None,
        depths: Union[List[int], None] = None,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=dims[0],
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for dim, depth in zip(dims, depths):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dim,
                        drop_path=dp_rates[cur + d],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for d in range(depth)
                ]
            )
            self.stages.append(stage)
            cur += depth
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.q_head = nn.Sequential(
            nn.Linear(dims[-1], action_size),
            nn.Softmax(dim=1),
        )
        self.v_head = nn.Sequential(
            nn.Linear(dims[-1], 1),
            nn.Tanh(),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        g_avg_pool: Tensor = self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)
        return g_avg_pool

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._one_hot(x).float()
        x = self.forward_features(x)
        q = self.q_head(x)
        v = self.v_head(x)
        return q, v

    def _one_hot(self, x: Tensor) -> Tensor:
        x = torch.where(x < 0, 2, x).to(torch.long)
        x = nn.functional.one_hot(x, num_classes=3)
        if x.ndim == 3:
            return x.permute(2, 0, 1).unsqueeze(0)  # (N, C, H, W)
        return x.permute(0, 3, 1, 2)  # (N, C, H, W)


class DQNAgent:
    def __init__(self, params: DQNParams) -> None:
        self.gamma = params.gamma
        self.lr = params.lr
        self.epsilon = params.epsilon
        self.buffer_size = params.buffer_size
        self.batch_size = params.batch_size
        self.action_size = params.action_size
        self.double_dqn = params.double_dqn

        self.qvnet = QVNet(self.action_size, dims=[96], depths=[3])
        self.qvnet_target = QVNet(self.action_size, dims=[96], depths=[3])
        self.optimizer = optim.Adam(self.qvnet.parameters(), lr=self.lr)

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
            qs, _ = self.qvnet(state)
            return int(qs.argmax().item())

    def get_target(self, reward: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        """TDターゲットを返す

        Args:
            reward (Tensor): 報酬
            next_state (Tensor): 次のゲームボードの状態
            done (Tensor): エピソードが終了したか

        Returns:
            Tensor: TDターゲット
        """
        if self.double_dqn:
            target = self.get_ddqn_target(reward, next_state, done)
        else:
            target = self.get_default_target(reward, next_state, done)
        return target

    def get_default_target(self, reward: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        """通常のDQNのTDターゲットを返す

        TDターゲット = reward + gamma * max_a qnet_target(next_state)

        Args:
            reward (Tensor): 報酬
            next_state (Tensor): 次のゲームボードの状態
            done (Tensor): エピソードが終了したか

        Returns:
            Tensor: TDターゲット
        """
        next_qs, _ = self.qvnet_target(next_state)
        next_q = next_qs.max(1)[0]
        next_q.detach()
        target: Tensor = reward + (1 - done.to(torch.int)) * self.gamma * next_q
        return target

    def get_ddqn_target(self, reward: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        """過大評価を解消するTDターゲットを返す
        qnet_target(next_state)は誤差が含まれる推定値であり、この最大値を取ると、真のQ関数より過大評価されてしまう
        Double DQNでは、真のQ関数qnet(next_state)が最大となる行動を選び、
        その行動を取ったときのqnet_target(next_state)を使ってTDターゲットを作る

        TDターゲット = reward + gamma * qnet_target(next_state, a=argmax_a qnet(next_state))

        Args:
            reward (Tensor): 報酬
            next_state (Tensor): 次のゲームボードの状態
            done (Tensor): エピソードが終了したか

        Returns:
            Tensor: TDターゲット
        """
        qs, _ = self.qvnet(next_state)
        q_max_action = qs.argmax(axis=1)
        next_qs, _ = self.qvnet_target(next_state)
        next_q = torch.tensor([qi[a] for qi, a in zip(next_qs, q_max_action)])
        next_q.detach()
        target: Tensor = reward + (1 - done.to(torch.int)) * self.gamma * next_q
        return target

    def to_tensor(self, data: Example) -> Tuple[Tensor, ...]:
        states = torch.tensor(np.stack([x.state for x in data]))
        actions = torch.tensor(np.array([x.action for x in data]), dtype=torch.long)
        rewards = torch.tensor(np.array([x.reward for x in data]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([x.next_state for x in data]))
        players = torch.tensor(np.stack([x.player for x in data]))
        winning_players = torch.tensor(
            np.array([x.winning_player for x in data]), dtype=torch.float32
        )
        done_list = torch.tensor(np.stack([x.done for x in data]))
        return states, actions, rewards, next_states, players, winning_players, done_list

    def update(self, example: Example) -> float:
        state, action, reward, next_state, player, winning_player, done = self.to_tensor(example)
        qs, v = self.qvnet(state)  # qs.size(): (N, 7) / vs.size(): (N, 1)
        q = qs[np.arange(len(action)), action]
        target = self.get_target(reward, next_state, done)

        loss_fn = nn.MSELoss()
        loss_q = loss_fn(q, target)
        loss_v = loss_fn(v.flatten(), winning_player)
        loss = loss_q + loss_v

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_qvnet(self) -> None:
        self.qvnet_target.load_state_dict(self.qvnet.state_dict())

    def play(self, state: BoardType) -> int:
        state = torch.tensor(state[np.newaxis, :])
        qs, _ = self.qvnet(state)
        return int(qs.argmax().item())


def main() -> None:
    agent = DQNAgent(DQNParams())
    state = np.zeros((1, 6, 7))
    print(state)
    action: np.ndarray = np.array([[2]])
    next_state = deepcopy(state)
    print(action[0, 0])
    next_state[0, 0, action[0, 0]] = 1
    print(next_state)

    reward: np.ndarray = np.array([[1]])
    done: np.ndarray = np.array([[True]])
    example = Example(
        state=state,
        action=action,
        next_state=next_state,
        player=1,
        done=done,
        reward=reward,
        winning_player=1,
    )
    agent.update(example)


if __name__ == "__main__":
    main()
