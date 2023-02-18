import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
from conf.game_conf import BOARD_ROW, BOARD_COLUMN


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
    
    def is_valid_location(self, col: int) -> None:
        """
        その列が既にピースで埋まっていないかをチェック

        Args:
            col (int): プレイヤーが選択した列
        """
        return self.board[BOARD_ROW - 1, col] == 0
    
    def get_next_open_row(self, col: int) -> None:
        """
        ボードがピースで溜まっていく
        その中で、プレイヤーがピースを入れる列を選択したときに、ピースを入れることができる行を返す

        Args:
            col (int): プレイヤーが選択した列
        """
        for r in range(BOARD_ROW):
            if self.board[r, col] == 0:
                return r
        
    def print_board(self) -> None:
        print(np.flip(self.board, 0))


def main() -> None:
    board = Board(row=BOARD_ROW, col=BOARD_COLUMN)
    print(board)
    game_over = False
    turn = 0
    while not game_over:
        # Ask for player 1 input
        if turn % 2 == 0:
            col = int(input("Player 1, Make your Selection(0-6):"))
            # Player 1 will drop a piece on the board
            if board.is_valid_location(col):
                row = board.get_next_open_row(col)
                board.drop_piece(row, col, 1)
            
        # Ask for player 2 input
        else:
            col = int(input("Player 2, Make your Selection(0-6):"))
            # Player 2 will drop a piece on the board
            if board.is_valid_location(col):
                row = board.get_next_open_row(col)
                board.drop_piece(row, col, 2)
    
        board.print_board()
        turn += 1


if __name__ == "__main__":
    main()
