import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent.models.convnext import ConvNeXtBlock
from agent.models.utils import LayerNorm
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from conf.game_conf import BOARD_COLUMN  # noqa
from modules.typings import BoardType, Example, TensorExamples  # noqa
from modules.utils.model_utils import (  # noqa
    calculate_quantile_huber_loss,
    disable_gradients,
    evaluate_quantile_at_action,
    update_params,
)


@dataclass
class DQNParams:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gamma: float = 0.98
    action_size: int = BOARD_COLUMN
    lr: float = 0.0005
    epsilon: float = 0.1
    double_dqn: bool = True
    dueling: bool = True
    buffer_size: int = 10000
    batch_size: int = 32
    multi_step_td_target: int = 2  # multi step learningのステップ数。1の時、TD法
    num_taus: int = 200  # QR-DQNにおける分位の数。1の時、シンプルなQ-learning


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
        num_taus: int,
        drop_path_rate: float = 0.2,
        dims: Union[List[int], None] = None,
        depths: Union[List[int], None] = None,
        layer_scale_init_value: float = 1e-6,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.num_taus = num_taus
        self.dueling = dueling
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
        if self.dueling:
            self.dueling_qv_head = nn.Sequential(
                nn.Linear(dims[-1], dims[-1]),
                nn.Linear(dims[-1], self.num_taus),
            )
            self.dueling_qa_head = nn.Sequential(
                nn.Linear(dims[-1], dims[-1]),
                nn.Linear(dims[-1], self.action_size * self.num_taus),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(dims[-1], self.action_size * self.num_taus),
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
        if self.dueling:
            q_v = self.dueling_qv_head(x).view(-1, self.num_taus, 1)
            q_a = self.dueling_qa_head(x).view(-1, self.num_taus, self.action_size)
            mean_q_a = q_a.mean(dim=2, keepdim=True)  # 各アクションに対するAdvantageの平均
            q = q_v + q_a - mean_q_a
        else:
            q = self.q_head(x).view(-1, self.num_taus, self.action_size)
        v = self.v_head(x)
        return q, v

    def _one_hot(self, x: Tensor) -> Tensor:
        x = torch.where(x < 0, 2, x).to(torch.long)
        x = nn.functional.one_hot(x, num_classes=3)
        if x.ndim == 3:
            return x.permute(2, 0, 1).unsqueeze(0)  # (N, C, H, W)
        return x.permute(0, 3, 1, 2)  # (N, C, H, W)

    def calculate_q(self, states: Tensor) -> Tensor:
        # Calculate quantiles.
        qs, _ = self(states)

        # Calculate expectations of value distributions.
        q: Tensor = qs.mean(dim=1)

        return q


class DQNAgent:
    def __init__(self, params: DQNParams) -> None:
        self.device = params.device
        self.gamma = params.gamma
        self.lr = params.lr
        self.epsilon = params.epsilon
        self.buffer_size = params.buffer_size
        self.batch_size = params.batch_size
        self.action_size = params.action_size
        self.double_dqn = params.double_dqn
        self.dueling = params.dueling
        self.multi_step_td_target = params.multi_step_td_target
        self.num_taus = params.num_taus

        self.qvnet = QVNet(
            self.action_size, self.num_taus, dims=[96], depths=[3], dueling=self.dueling
        )
        self.qvnet_target = QVNet(
            self.action_size, self.num_taus, dims=[96], depths=[3], dueling=self.dueling
        )
        disable_gradients(self.qvnet_target)
        self.optimizer = optim.Adam(self.qvnet.parameters(), lr=self.lr)

        # Fixed fractions.
        taus = (
            torch.arange(0, self.num_taus + 1, device=self.device, dtype=torch.float32)
            / self.num_taus
        )
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.num_taus)

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
            return int(qs.sum(dim=1).argmax().item())

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
            next_q = self.get_next_q_for_ddqn(reward, next_state, done)
        else:
            next_q = self.get_next_q(reward, next_state, done)
        next_q.detach()
        target = self.get_td_target(reward, done, next_q)
        return target

    def get_td_target(self, reward: Tensor, done: Tensor, next_q: Tensor) -> Tensor:
        gamma = self.gamma**self.multi_step_td_target
        reward = reward.unsqueeze(-1).unsqueeze(-1)  # (B, ) -> (B, 1, 1)
        done = done.to(torch.int).unsqueeze(-1).unsqueeze(-1)  # (B, ) -> (B, 1, 1)
        td_target: Tensor = reward + (1 - done) * gamma * next_q
        return td_target

    def get_next_q(self, reward: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        """通常のDQNにおいて、次の状態next_stateにおけるQ関数を返す

        next_q = max_a qnet_target(next_state)

        Args:
            reward (Tensor): 報酬
            next_state (Tensor): 次のゲームボードの状態
            done (Tensor): エピソードが終了したか

        Returns:
            Tensor: next_q
        """
        next_qs = self.qvnet_target.calculate_q(next_state)
        next_q: Tensor = next_qs.max(2).unsqueeze(2).transpose(1, 2)
        assert next_q.shape == (self.batch_size, 1, self.num_taus)

        return next_q

    def get_next_q_for_ddqn(self, reward: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        """過大評価を解消するように、次の状態next_stateにおけるQ関数を返す
        qnet_target(next_state)は誤差が含まれる推定値であり、この最大値を取ると、真のQ関数より過大評価されてしまう
        Double DQNでは、真のQ関数qnet(next_state)が最大となる行動を選び、
        その行動を取ったときのqnet_target(next_state)を返す

        next_q = qnet_target(next_state, a=argmax_a qnet(next_state))

        Args:
            reward (Tensor): 報酬
            next_state (Tensor): 次のゲームボードの状態
            done (Tensor): エピソードが終了したか

        Returns:
            Tensor: TDターゲット
        """
        qs = self.qvnet.calculate_q(next_state)
        # Calculate greedy actions.
        q_max_actions = torch.argmax(qs, dim=1, keepdim=True).squeeze(1)
        assert q_max_actions.shape == (self.batch_size,)

        # Calculate quantile values of next states and actions at tau_hats.
        target_qs, _ = self.qvnet_target(next_state)
        next_sa_quantiles: Tensor = evaluate_quantile_at_action(target_qs, q_max_actions).transpose(
            1, 2
        )
        assert next_sa_quantiles.shape == (self.batch_size, 1, self.num_taus)

        return next_sa_quantiles

    def to_tensor(self, data: List[Example]) -> TensorExamples:
        states = torch.tensor(np.stack([x.state for x in data]))
        actions = torch.tensor(np.array([x.action for x in data]), dtype=torch.long)
        rewards = torch.tensor(np.array([x.reward for x in data]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([x.next_state for x in data]))
        players = torch.tensor(np.stack([x.player for x in data]))
        winning_players = torch.tensor(
            np.array([x.winning_player for x in data]), dtype=torch.float32
        )
        done_list = torch.tensor(np.stack([x.done for x in data]))
        return TensorExamples(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            player=players,
            winning_player=winning_players,
            done=done_list,
        )

    def get_loss_q(self, q: Tensor, target: Tensor) -> Tensor:
        td_errors = q - target
        assert td_errors.shape == (self.batch_size, self.num_taus, self.num_taus)
        quantile_huber_loss: Tensor = calculate_quantile_huber_loss(td_errors, self.tau_hats)
        return quantile_huber_loss

    def get_loss_v(self, v: Tensor, target: Tensor) -> Tensor:
        loss_fn = nn.MSELoss()
        loss: Tensor = loss_fn(v, target)
        return loss

    def get_loss(self, loss_q: Tensor, loss_v: Tensor) -> Tensor:
        return loss_q + loss_v

    def update(self, example: List[Example]) -> float:
        tensor_examples = self.to_tensor(example)
        qs, v = self.qvnet(
            tensor_examples.state
        )  # qs.size(): (N, num_taus, action_size) / vs.size(): (N, num_taus, 1)

        # Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = evaluate_quantile_at_action(qs, tensor_examples.action)
        assert current_sa_quantiles.shape == (self.batch_size, self.num_taus, 1)
        with torch.no_grad():
            target_sa_quantiles = self.get_target(
                tensor_examples.reward, tensor_examples.next_state, tensor_examples.done
            )
        assert target_sa_quantiles.shape == (self.batch_size, 1, self.num_taus)

        loss_q = self.get_loss_q(q=current_sa_quantiles, target=target_sa_quantiles)
        loss_v = self.get_loss_v(v.flatten(), tensor_examples.winning_player)
        loss = self.get_loss(loss_q, loss_v)

        update_params(self.optimizer, loss, retain_graph=False)
        return float(loss.item())

    def sync_qvnet(self) -> None:
        self.qvnet_target.load_state_dict(self.qvnet.state_dict())

    def play(self, state: BoardType) -> int:
        state = torch.tensor(state[np.newaxis, :])
        qs, _ = self.qvnet(state)
        return int(qs.argmax().item())


def main() -> None:
    agent = DQNAgent(DQNParams())
    state = np.zeros((6, 7))
    print(state)
    action = 2
    print(action)
    next_state = deepcopy(state)
    next_state[0, action] = 1
    print(next_state)

    reward = 1
    done = True
    example = Example(
        state=state,
        action=action,
        next_state=next_state,
        player=1,
        done=done,
        reward=reward,
        winning_player=1,
    )
    loss = agent.update([example])
    print(loss)


if __name__ == "__main__":
    main()
