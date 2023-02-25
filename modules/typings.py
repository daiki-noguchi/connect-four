from dataclasses import dataclass

import numpy as np

BoardType = np.ndarray  # shape: BOARD_ROW, BOARD_COLUMN / dtype: np.float32


@dataclass
class Example:
    state: BoardType
    action: int
    next_state: BoardType
    player: int
    done: bool
    reward: float = 0


@dataclass
class Location:
    row: int
    col: int


@dataclass
class STEP_OUTPUT:
    next_state: BoardType
    reward: int
    player: int
    next_player: int
    done: bool
