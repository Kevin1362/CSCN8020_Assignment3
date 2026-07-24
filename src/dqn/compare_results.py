from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .common import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare A/B experiments and select DQN.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    return parser.parse_args()


def _choose_checkpoint(models_root: Path, config_name: str) -> Path:
    best = models_root / config_name / "best.pt"
    final = models_root / config_name / "final.pt"
    return best if best.is_file() else final


def main() -> None:
    args = parse_args()
    configs = ["config_a", "config_b"]
    rows = []
    data = {}

    for name in configs:
        training = read_json(args.results_root / name / "training_summary.json")
        evaluation = read_json(args.results_root / name / "evaluation_summary.json")
        data[name] = {"training": training, "evaluation": evaluation}
        rows.append(
            {
                "configuration": name,
                "epsilon_decay": training["epsilon_decay"],
                "training_episodes": training["episodes_completed"],
                "training_minutes": training["training_minutes"],
                "final_epsilon": training["final_epsilon"],
                "mean_reward_final_20": training["mean_reward_final_20"],
                "training_success_rate_final_50": training[
                    "training_success_rate_final_50"
                ],
                "evaluation_success_rate": evaluation["success_rate"],
                "evaluation_mean_reward": evaluation["mean_reward"],
            }
        )

    # Evidence-based automatic tie-breaker:
    # 1) final greedy success rate, 2) training stability (final-50 success),
    # 3) mean evaluation reward, 4) shorter training time.
    def score(name: str) -> tuple[float, float, float, float]:
        training = data[name]["training"]
        evaluation = data[name]["evaluation"]
        return (
            float(evaluation["success_rate"]),
            float(training["training_success_rate_final_50"]),
            float(evaluation["mean_reward"]),
            -float(training["training_seconds"]),
        )

    selected = max(configs, key=score)
    selected_checkpoint = _choose_checkpoint(args.models_root, selected)
    destination = args.models_root / "selected_dqn.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, destination)

    comparison_dir = args.results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_dir / "epsilon_decay_comparison.csv", rows)
    selection = {
        "selected_configuration": selected,
        "source_checkpoint": str(selected_checkpoint),
        "selected_checkpoint": str(destination),
        "selection_rule": (
            "Highest greedy evaluation success rate; ties broken by final-50 training "
            "success rate, then mean evaluation reward, then shorter training time."
        ),
    }
    write_json(comparison_dir / "selection.json", selection)

    print("Selected configuration:", selected)
    print("Copied checkpoint to:", destination)


if __name__ == "__main__":
    main()
