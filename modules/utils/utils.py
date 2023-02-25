from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_total_reward(reward_history: np.ndarray, output_dir: Path) -> None:
    plt.clf()
    plt.xlabel("Episode")
    plt.ylabel("Total winning rate")
    plt.plot(range(len(reward_history)), reward_history)
    plt.savefig(str(output_dir / "reward_history.png"))


def plot_mean_loss(loss: np.ndarray, output_dir: Path) -> None:
    plt.clf()
    plt.xlabel("Episode")
    plt.ylabel("loss")
    plt.plot(range(len(loss)), loss)
    plt.savefig(str(output_dir / "loss.png"))
