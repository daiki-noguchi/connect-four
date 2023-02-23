from dataclasses import dataclass
import numpy as np


@dataclass
class Example:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    player: int
    done: bool
    reward: float = 0
