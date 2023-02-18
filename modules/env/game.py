import os
import sys
from typing import List, Tuple, Union

import numpy as np
from typings import Location

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from conf.game_conf import BOARD_COLUMN, BOARD_ROW  # noqa


class Board:
    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col
        self.board = self.create_board()

    def create_board(self) -> np.ndarray:
        board = np.zeros((self.row, self.col))
        return board

    def drop_piece(self, row: int, col: int, player: int) -> None:
        self.board[row, col] = player

    def get_next_open_row(self, col: int) -> int:
        """
        ボードがピースで溜まっていく
        その中で、プレイヤーがピースを入れる列を選択したときに、ピースを入れることができる行を返す

        Args:
            col (int): プレイヤーが選択した列
        """
        for r in range(BOARD_ROW):
            if self.board[r, col] == 0:
                return r
        return 99  # 全ての行がplayer=1or2で埋まっている状態

    def check_done_player(self, state: int) -> bool:
        """
        get_next_open_rowの返り値が-1
        (すなわち、全ての行がplayer=1or2で埋まっているカラムをplayerが選択した状態)
        であれば、警告を促してFalseを返す

        Args:
            col (int): プレイヤーが選択した列
        """
        if state == 99:
            print(
                "[WARNING] That column is already filled with pieces. Please select another column."
            )
            return False
        else:
            return True

    def check_winning_move(self, player: int) -> Tuple[bool, Union[List[Location], None]]:
        # Check horizontal locations for win
        for c in range(BOARD_COLUMN - 3):
            for r in range(BOARD_ROW):
                if all(
                    [
                        self.board[r, c] == player,
                        self.board[r, c + 1] == player,
                        self.board[r, c + 2] == player,
                        self.board[r, c + 3] == player,
                    ]
                ):
                    return True, [
                        Location(r, c),
                        Location(r, c + 1),
                        Location(r, c + 2),
                        Location(r, c + 3),
                    ]

        # Check vertical locations for win
        for c in range(BOARD_COLUMN):
            for r in range(BOARD_ROW - 3):
                if all(
                    [
                        self.board[r, c] == player,
                        self.board[r + 1, c] == player,
                        self.board[r + 2, c] == player,
                        self.board[r + 3, c] == player,
                    ]
                ):
                    return True, [
                        Location(r, c),
                        Location(r + 1, c),
                        Location(r + 2, c),
                        Location(r + 3, c),
                    ]

        # Check positively sloped diaganols
        for c in range(BOARD_COLUMN - 3):
            for r in range(BOARD_ROW - 3):
                if all(
                    [
                        self.board[r, c] == player,
                        self.board[r + 1, c + 1] == player,
                        self.board[r + 2, c + 2] == player,
                        self.board[r + 3, c + 3] == player,
                    ]
                ):
                    return True, [
                        Location(r, c),
                        Location(r + 1, c + 1),
                        Location(r + 2, c + 2),
                        Location(r + 3, c + 3),
                    ]

        # Check negatively sloped diaganols
        for c in range(BOARD_COLUMN - 3):
            for r in range(3, BOARD_ROW):
                if all(
                    [
                        self.board[r, c] == player,
                        self.board[r - 1, c + 1] == player,
                        self.board[r - 2, c + 2] == player,
                        self.board[r - 3, c + 3] == player,
                    ]
                ):
                    return True, [
                        Location(r, c),
                        Location(r - 1, c + 1),
                        Location(r - 2, c + 2),
                        Location(r - 3, c + 3),
                    ]

        return False, None

    def print_board(self) -> None:
        print(np.flip(self.board, 0))


def main() -> None:
    board = Board(row=BOARD_ROW, col=BOARD_COLUMN)
    print(board)
    game_over = False
    turn = 0
    while not game_over:
        if turn % 2 == 0:
            player = 1
        else:
            player = -1
        done_playing = False
        while not done_playing:
            # Ask for player input
            col = int(input(f"Player {player}, Make your Selection(0-6):"))
            row = board.get_next_open_row(col)
            done_playing = board.check_done_player(row)
        # Player 1 will drop a piece on the board
        board.drop_piece(row, col, player=player)
        do_wins, win_list = board.check_winning_move(player=player)
        if do_wins:
            print(f"Player {player} Wins !!")
            print(f"Location is {win_list}")
            game_over = True

        board.print_board()
        turn += 1


if __name__ == "__main__":
    main()
