import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from agent.dqn import DQNAgent, DQNParams
from env.env import ConnectFour
from typings import STEP_OUTPUT, BoardType, Example
from utils.replay_buffer import ReplayBuffer
from utils.utils import plot_total_reward


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

    def execute_one_episode(self) -> List[Example]:
        examples: List[Example] = []
        state = deepcopy(self.env.reset())
        player = 1

        while True:
            action, output = env_step(self.env, self.agent, state, player)
            ex = Example(
                state=state,
                action=action,
                next_state=deepcopy(output.next_state),
                player=player,
                done=output.done,
            )
            examples.append(ex)

            if output.done:
                # save (state, action, reward (v), next_state, player)
                for i, ex in enumerate(
                    reversed(examples)
                ):  # The longer the match, the less rewarding.
                    # reward is +1 if player is winner, otherwise -1
                    # if it is draw, the reward of both player is 0
                    r = output.reward if ex.player == player else -output.reward
                    ex.reward = r * self.gamma**i
                return examples

            state = deepcopy(output.next_state)
            player = output.next_player

    def update(self, episode: int) -> None:
        examples_one_episode = self.execute_one_episode()
        for ex in examples_one_episode:
            self.replay_buffer.add(ex)

        if len(self.replay_buffer) < self.agent.batch_size:
            return

        state, action, reward, next_state, player, done = self.replay_buffer.get_batch()
        self.agent.update(state, action, reward, next_state, done)

        if episode % self.sync_interval == 0:
            self.agent.sync_qnet()


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
    output_dir = Path(f"outputs/{datetime.now().strftime('%Y%m%d/%H%M%S')}")
    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(200):
        (output_dir / f"{i}").mkdir(exist_ok=True, parents=True)
        win_rate_list = []

        num_episodes = 100000
        sync_interval = 100
        num_games_for_eval = 100

        env = ConnectFour()
        agent = DQNAgent(DQNParams())
        replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)
        coach = Coach(env, agent, replay_buffer, sync_interval=sync_interval)
        past_agent = deepcopy(coach.agent)

        for episode in range(num_episodes):
            coach.update(episode)

            if episode % 1000 == 0:
                # current vs before
                win = vs(latest=coach.agent, past=past_agent, num_games=num_games_for_eval)
                win_rate = win / num_games_for_eval
                print(
                    f"episode :{episode}, winning rate : {win_rate:.3f} ({win} / {num_games_for_eval})"
                )
                win_rate_list.append(win_rate)

                torch.save(
                    coach.agent.qnet.state_dict(), str(output_dir / f"{i}" / f"q_net_{episode}.pth")
                )

        mean_win_rates.append(win_rate_list)

        plot_total_reward(np.mean(mean_win_rates, axis=0), output_dir)

        with open(str(output_dir / "mean_win_rates.pkl"), "w") as f:
            pickle.dump(mean_win_rates, f)  # type: ignore


if __name__ == "__main__":
    main()
