import os
import sys
from copy import deepcopy
from typing import Generator, List, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from conf.game_conf import BOARD_COLUMN, BOARD_ROW  # noqa
from modules.env.game import Board  # noqa
from modules.env.typings import STEP_OUTPUT  # noqa
from modules.typings import BoardType  # noqa


class ConnectFour:
    def __init__(self) -> None:
        self.action_space = [c for c in range(BOARD_COLUMN)]
        self.board = Board(row=BOARD_ROW, col=BOARD_COLUMN)

    @property
    def height(self) -> int:
        return int(self.board.row)

    @property
    def width(self) -> int:
        return int(self.board.col)

    @property
    def shape(self) -> Tuple[int, int]:
        h, w = self.board.state.shape
        return h, w

    def actions(self) -> List[int]:
        return self.action_space

    def states(self) -> Generator[Tuple[int, int], None, None]:
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, action: int, player: int) -> Tuple[BoardType, int]:
        """
        どちらのプレイヤー`player`が
        どんな行動`action`をするかを受け取って、
        次のゲームボード状態とプレイヤーを返す

        Args:
            action (int): どちらのプレイヤー?
            player (int): どんな行動?具体的にはどの列にピースを落とすか? ex. 2

        Returns:
            BoardType: 次のゲームボード状態
            int: 次のプレイヤー
        """
        is_valid_location = self.board.drop_piece(col=action, player=player)
        if is_valid_location:  # playerが正しく列を選択したら、ボードの状態が変わってバトンパス
            return self.board.state, -player
        else:  # playerが既に埋まっている列を選択したら、もう一回試行する
            return self.board.state, player

    def reward(
        self, state: BoardType, action: int, next_state: BoardType, player: int
    ) -> Tuple[bool, int]:
        do_wins, _ = self.board.check_winning_move(player)
        if do_wins:
            return do_wins, int(do_wins)
        if self.board.check_draw():  # draw
            return True, 0
        if np.allclose(state, next_state):
            return False, 0
        return False, 0

    def reset(self) -> BoardType:
        self.board.reset()
        return self.board.state

    def step(self, action: int, player: int) -> STEP_OUTPUT:
        state = deepcopy(self.board.state)
        next_state, next_player = self.next_state(action, player)
        done, reward = self.reward(state, action, next_state, player)

        return STEP_OUTPUT(next_state, reward, player, next_player, done)
