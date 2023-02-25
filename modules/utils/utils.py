from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_total_reward(reward_history: np.ndarray, output_dir: Path) -> None:
    plt.xlabel("Episode")
    plt.ylabel("Total winning rate")
    plt.plot(range(len(reward_history)), reward_history)
    plt.savefig(str(output_dir / "reward_history.png"))
