from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from agent.dqn import DQNAgent, DQNParams
from env.env import ConnectFour
from typings import STEP_OUTPUT, BoardType, Example
from utils.replay_buffer import ReplayBuffer
from utils.utils import plot_mean_loss, plot_total_reward


def get_key_from_value(d: Dict[int, DQNAgent], val: DQNAgent) -> int:
    return [k for k, v in d.items() if v == val][0]


def env_step(
    env: ConnectFour, agent: DQNAgent, state: BoardType, player: int
) -> Tuple[int, STEP_OUTPUT]:
    """agentが行動して、envによって、その行動が実現可能かをを判断して、
    可能であれば、一つ先の盤面に進めて次のプレイヤーにバトンパス。不可であれば、再試行する

    Args:
        env (ConnectFour): 環境
        agent (DQNAgent): エージェント
        state (BoardType): 状態
        player (int): プレイヤー

    Returns:
        : STEP_OUTPUT
    """
    is_valid_location = False
    while not is_valid_location:
        action = agent.get_action(state)
        # playerが既に埋まっている列を選択したら(is_valid_location=False)、もう一回試行する
        is_valid_location = env.board.check_is_valid_location(col=action)
    output = env.step(action, player)
    # playerが正しく列を選択したら(is_valid_location=True)、ボードの状態が変わって次のプレイヤーにバトンパス
    return action, output


class Coach:
    def __init__(
        self, env: ConnectFour, agent: DQNAgent, replay_buffer: ReplayBuffer, sync_interval: int
    ) -> None:
        self.env = env
        self.agent = agent
        self.gamma = 0.9
        self.replay_buffer = replay_buffer
        self.sync_interval = sync_interval

    def get_delta(self, example: Example) -> float:
        """予測誤差(TDターゲット - self.agent.qvnet(state))を求める

        Args:
            example (Example): 現在の状態、プレイヤー、行動、次の状態、などを保持

        Returns:
            float: 予測誤差. prioritized_experience_replayで、予測誤差が大きいほど優先されてバッチとして選ばれて学習される
        """
        tensor_examples = self.agent.to_tensor([example])
        t = (
            self.agent.get_target(
                tensor_examples.reward, tensor_examples.next_state, tensor_examples.done
            )
            .detach()
            .item()
        )
        qs, _ = self.agent.qvnet(tensor_examples.state)
        q = qs[np.arange(len(tensor_examples.action)), tensor_examples.action].detach().item()
        return float(abs(t - q))

    def execute_one_episode(self) -> List[Example]:
        state = deepcopy(self.env.reset())
        player = 1

        state_action_list: Dict[int, List[Tuple[BoardType, int]]] = {
            1: [],
            -1: [],
        }
        while True:
            action, output = env_step(self.env, self.agent, state, player)
            if player == 1:
                state_action_list[1].append((state, action))
            else:
                state_action_list[-1].append((state, action))
            if output.done:
                winning_player = player
                examples_by_player: Dict[int, List[Example]] = {
                    1: [],
                    -1: [],
                }
                for player, _list in state_action_list.items():
                    for i, (state, action) in enumerate(_list):
                        if i == len(_list) - 1:  # 最後の状態
                            # next_stateの注意点
                            # done=Trueであれば、TDターゲットの計算に使われないため、勝者の最後の状態のnext_stateはどんな値を入れてもいい
                            # 敗者の最後の状態のnext_stateは勝敗が決着した状態（connect4では、4つのコインが揃った状態）
                            # よって、ここでは最後の状態のnext_stateは勝者でも敗者でも同じ値を入れる
                            next_state = output.next_state
                            # 試合終了フラグ: 勝つ一手を打った瞬間に試合終了
                            done = winning_player == player
                            # reward: 各プレイヤーの最後のプレイに報酬を与える
                            # 勝者の報酬は+1、敗者の報酬は-1、ドローの場合は両プレイヤーともに0
                            reward = output.reward if player == winning_player else -output.reward
                        else:
                            next_state = _list[i + 1][0]  # 一つ目の要素がstate
                            done = False
                            reward = 0
                        examples_by_player[player].append(
                            Example(
                                state=state,
                                action=action,
                                next_state=next_state,
                                player=player,
                                done=done,
                                reward=reward,
                                winning_player=winning_player,
                            )
                        )
                    # get reward and next_state for multi-step learning
                    for i, ex in enumerate(
                        examples_by_player[player][: -self.agent.multi_step_td_target + 1]
                    ):
                        next_i = i + self.agent.multi_step_td_target - 1
                        ex.next_state = examples_by_player[player][next_i].next_state  # n個先の状態
                        total_reward = 0.0  # n個先まで行動したときに実際に得られた収益（割引付き報酬合計）
                        for power in range(self.agent.multi_step_td_target):
                            total_reward += (
                                self.agent.gamma**power
                                * examples_by_player[player][i + power].reward
                            )
                        ex.reward = total_reward
                        ex.delta = self.get_delta(ex)
                return examples_by_player[1] + examples_by_player[-1]

            state = deepcopy(output.next_state)
            player = output.next_player

    def update(self, episode: int) -> Union[float, None]:
        examples_one_episode = self.execute_one_episode()
        for ex in examples_one_episode:
            self.replay_buffer.add(ex)

        if len(self.replay_buffer) < self.agent.batch_size:
            return None

        examples = self.replay_buffer.get_batch()
        loss = self.agent.update(examples)

        if episode % self.sync_interval == 0:
            self.agent.sync_qvnet()

        return loss


def vs(latest: DQNAgent, past: DQNAgent, num_games: int) -> int:
    """最新のエージェントと過去のエージェントを戦わせて、前者の勝った試合数を返す

    Args:
        latest (DQNAgent): 最新のエージェント
        past (DQNAgent): 過去のエージェント
        num_games (int): 試合数

    Returns:
        int: 最新のエージェントの勝った試合数
    """
    env = ConnectFour()
    num_win = 0
    agent_dict = {
        1: latest,
        -1: past,
    }
    for _ in range(num_games):
        state = deepcopy(env.reset())
        cur_player = 1
        done = False
        while not done:
            cur_agent = agent_dict[cur_player]
            action, output = env_step(env, cur_agent, state, cur_player)
            if output.done:
                num_win += 1 if cur_player == get_key_from_value(agent_dict, latest) else 0

            state = deepcopy(output.next_state)
            cur_player = output.next_player
            done = output.done

    return num_win


def main() -> None:
    mean_win_rates = []
    mean_loss = []
    output_dir = Path(f"outputs/{datetime.now().strftime('%Y%m%d/%H%M%S')}")
    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(10):
        (output_dir / f"{i}").mkdir(exist_ok=True, parents=True)
        win_rate_list = []
        loss_list = []

        num_episodes = 100000
        sync_interval = 100
        num_games_for_eval = 100
        vs_interval = 1000

        env = ConnectFour()
        agent = DQNAgent(DQNParams())
        replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=agent.batch_size)
        coach = Coach(env, agent, replay_buffer, sync_interval=sync_interval)
        past_agent = deepcopy(coach.agent)

        print(f"=============={i}: {num_episodes} episode==================")
        for episode in range(num_episodes):
            loss = coach.update(episode)

            if episode % vs_interval == 0:
                loss_list.append(loss)
                # current vs before
                win = vs(latest=coach.agent, past=past_agent, num_games=num_games_for_eval)
                win_rate = win / num_games_for_eval
                print(
                    f"episode :{episode}, winning rate : {win_rate:.3f} ({win} / {num_games_for_eval})"
                )
                win_rate_list.append(win_rate)

                torch.save(
                    coach.agent.qvnet.state_dict(),
                    str(output_dir / f"{i}" / f"q_net_{episode}.pth"),
                )

        mean_win_rates.append(win_rate_list)
        mean_loss.append(loss_list)
        plot_total_reward(np.mean(mean_win_rates, axis=0), output_dir)
        plot_mean_loss(np.mean(mean_loss, axis=0), output_dir)


if __name__ == "__main__":
    main()
