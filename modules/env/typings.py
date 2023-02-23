import numpy as np
from dataclasses import dataclass


@dataclass
class Location:
    row: int
    col: int


@dataclass
class STEP_OUTPUT:
    next_state: np.ndarray
    reward: int
    player: int
    next_player: int
    done: bool
