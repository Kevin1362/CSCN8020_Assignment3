from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import read_csv, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate required assignment plots.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def save_training_plots(results_root: Path, config_name: str) -> None:
    rows = read_csv(results_root / config_name / "training_metrics.csv")
    episodes = np.asarray([int(r["episode"]) for r in rows])
    rewards = np.asarray([float(r["cumulative_reward"]) for r in rows])
    successes = np.asarray([float(r["success"]) for r in rows])
    epsilons = np.asarray([float(r["epsilon"]) for r in rows])
    losses = np.asarray([
        float(r["mean_loss"]) if r["mean_loss"].lower() != "nan" else np.nan
        for r in rows
    ])

    plot_dir = results_root / config_name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.45, label="Raw reward")
    plt.plot(episodes, moving_average(rewards, 20), label="20-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title(f"{config_name}: Training Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "training_reward.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, moving_average(successes, 50))
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Episode")
    plt.ylabel("Rolling success rate (50 episodes)")
    plt.title(f"{config_name}: Training Success Rate")
    plt.tight_layout()
    plt.savefig(plot_dir / "training_success_rate.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title(f"{config_name}: Epsilon Decay")
    plt.tight_layout()
    plt.savefig(plot_dir / "epsilon.png", dpi=180)
    plt.close()

    valid = np.isfinite(losses)
    plt.figure(figsize=(9, 5))
    plt.plot(episodes[valid], losses[valid])
    plt.xlabel("Episode")
    plt.ylabel("Mean Huber loss")
    plt.title(f"{config_name}: Training Loss")
    plt.tight_layout()
    plt.savefig(plot_dir / "loss.png", dpi=180)
    plt.close()


def save_config_comparison(results_root: Path) -> None:
    selection = read_json(results_root / "comparison" / "selection.json")
    rows = read_csv(results_root / "comparison" / "epsilon_decay_comparison.csv")
    labels = [r["configuration"] for r in rows]
    success = [100.0 * float(r["evaluation_success_rate"]) for r in rows]

    plot_dir = results_root / "comparison" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(labels, success)
    plt.ylim(0, 100)
    plt.ylabel("Greedy evaluation success rate (%)")
    plt.title(
        "Epsilon-Decay Configuration Comparison\n"
        f"Selected: {selection['selected_configuration']}"
    )
    plt.tight_layout()
    plt.savefig(plot_dir / "config_evaluation_success.png", dpi=180)
    plt.close()


def save_evaluation_by_goal(results_root: Path) -> None:
    selection = read_json(results_root / "comparison" / "selection.json")
    selected = str(selection["selected_configuration"])
    rows = read_csv(results_root / selected / "evaluation_summary.csv")
    rows = [r for r in rows if r["goal"] != "Overall"]
    goals = [r["goal"] for r in rows]
    rates = [100.0 * float(r["success_rate"]) for r in rows]

    plot_dir = results_root / "comparison" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(goals, rates)
    plt.ylim(0, 100)
    plt.xlabel("Target angle (rad)")
    plt.ylabel("Success rate (%)")
    plt.title(f"Selected DQN ({selected}): Evaluation Success by Target")
    plt.tight_layout()
    plt.savefig(plot_dir / "evaluation_success_by_target.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    for config_name in ("config_a", "config_b"):
        save_training_plots(args.results_root, config_name)
    save_config_comparison(args.results_root)
    save_evaluation_by_goal(args.results_root)
    print("Plots generated under results/*/plots and results/comparison/plots")


if __name__ == "__main__":
    main()
