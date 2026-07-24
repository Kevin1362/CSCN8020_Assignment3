from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based vs selected-DQN table.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = read_json(args.results_root / "comparison" / "selection.json")
    selected = str(selection["selected_configuration"])
    dqn = read_json(args.results_root / selected / "evaluation_summary.json")
    baseline = read_json(args.results_root / "rule_based" / "evaluation_summary.json")

    rows = [
        {
            "metric": "Successes / 20",
            "rule_based_policy": baseline["successes"],
            "selected_dqn": dqn["successes"],
        },
        {
            "metric": "Success rate",
            "rule_based_policy": baseline["success_rate"],
            "selected_dqn": dqn["success_rate"],
        },
        {
            "metric": "Mean cumulative reward",
            "rule_based_policy": baseline["mean_reward"],
            "selected_dqn": dqn["mean_reward"],
        },
        {
            "metric": "Mean episode length",
            "rule_based_policy": baseline["mean_episode_length"],
            "selected_dqn": dqn["mean_episode_length"],
        },
        {
            "metric": "Mean final absolute error",
            "rule_based_policy": baseline["mean_final_absolute_error"],
            "selected_dqn": dqn["mean_final_absolute_error"],
        },
    ]
    output_dir = args.results_root / "comparison"
    write_csv(output_dir / "rule_based_vs_dqn.csv", rows)
    write_json(
        output_dir / "rule_based_vs_dqn.json",
        {
            "selected_configuration": selected,
            "rule_based": baseline,
            "selected_dqn": dqn,
        },
    )
    print("Wrote rule-based vs DQN comparison for", selected)


if __name__ == "__main__":
    main()
