from dataclasses import dataclass

import numpy as np
import torch

BoardType = np.ndarray  # shape: BOARD_ROW, BOARD_COLUMN / dtype: np.float32


@dataclass
class Example:
    state: BoardType
    action: int
    next_state: BoardType
    player: int
    done: bool
    reward: float
    winning_player: int
    delta: float = 0


@dataclass
class TensorExamples:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    player: torch.Tensor
    winning_player: torch.Tensor
    done: torch.Tensor


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
