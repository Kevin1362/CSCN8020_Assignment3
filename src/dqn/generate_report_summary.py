from __future__ import annotations

from pathlib import Path

from .common import read_csv, read_json


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    root = Path("results")
    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)

    a_train = read_json(root / "config_a" / "training_summary.json")
    b_train = read_json(root / "config_b" / "training_summary.json")
    a_eval = read_json(root / "config_a" / "evaluation_summary.json")
    b_eval = read_json(root / "config_b" / "evaluation_summary.json")
    selection = read_json(root / "comparison" / "selection.json")
    selected = str(selection["selected_configuration"])
    selected_eval_rows = read_csv(root / selected / "evaluation_summary.csv")
    baseline = read_json(root / "rule_based" / "evaluation_summary.json")
    selected_eval = read_json(root / selected / "evaluation_summary.json")

    lines = [
        "# Generated Results Summary",
        "",
        "> This file contains measured values produced by your own run. Copy these values into the technical report and add your own interpretation of the plots and robot behaviour.",
        "",
        "## Exploration-Decay Comparison",
        "",
        "| Metric | Configuration A (0.995) | Configuration B (0.985) |",
        "|---|---:|---:|",
        f"| Training episodes | {a_train['episodes_completed']} | {b_train['episodes_completed']} |",
        f"| Wall-clock training time (min) | {float(a_train['training_minutes']):.2f} | {float(b_train['training_minutes']):.2f} |",
        f"| Final epsilon | {float(a_train['final_epsilon']):.4f} | {float(b_train['final_epsilon']):.4f} |",
        f"| Mean reward, final 20 training episodes | {float(a_train['mean_reward_final_20']):.4f} | {float(b_train['mean_reward_final_20']):.4f} |",
        f"| Training success rate, final 50 | {pct(float(a_train['training_success_rate_final_50']))} | {pct(float(b_train['training_success_rate_final_50']))} |",
        f"| Final greedy evaluation success | {pct(float(a_eval['success_rate']))} | {pct(float(b_eval['success_rate']))} |",
        f"| Mean evaluation reward | {float(a_eval['mean_reward']):.4f} | {float(b_eval['mean_reward']):.4f} |",
        "",
        f"**Automatically selected configuration:** `{selected}`",
        "",
        f"Selection rule used by the helper script: {selection['selection_rule']}",
        "",
        "## Selected DQN Final Evaluation",
        "",
        "| Goal | Episodes | Successes | Success rate | Mean reward |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in selected_eval_rows:
        lines.append(
            f"| {row['goal']} | {row['episodes']} | {row['successes']} | "
            f"{pct(float(row['success_rate']))} | {float(row['mean_reward']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Rule-Based Baseline vs Selected DQN",
            "",
            "| Metric | Rule-based | Selected DQN |",
            "|---|---:|---:|",
            f"| Successes / 20 | {baseline['successes']} | {selected_eval['successes']} |",
            f"| Success rate | {pct(float(baseline['success_rate']))} | {pct(float(selected_eval['success_rate']))} |",
            f"| Mean cumulative reward | {float(baseline['mean_reward']):.4f} | {float(selected_eval['mean_reward']):.4f} |",
            f"| Mean episode length | {float(baseline['mean_episode_length']):.2f} | {float(selected_eval['mean_episode_length']):.2f} |",
            f"| Mean final absolute error | {float(baseline['mean_final_absolute_error']):.5f} | {float(selected_eval['mean_final_absolute_error']):.5f} |",
            "",
            "## Interpretation Prompts",
            "",
            "Write your own observations after reviewing the plots and rendered demo:",
            "",
            "1. Which epsilon decay produced more stable learning, and what evidence supports that conclusion?",
            "2. Did the selected DQN meet the required 80% (16/20) success threshold?",
            "3. Which target angle was easiest and which was hardest?",
            "4. Did the DQN use HOLD near the target, or did it oscillate between increase/decrease actions?",
            "5. Was the rule-based policy more sample efficient? Explain that it begins with task knowledge while DQN must learn from interactions.",
            "6. Compare stability near the goal using episode length, final error, reward, and what you observed in the viewer.",
        ]
    )

    output = report_dir / "Generated_Results_Summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Generated", output)


if __name__ == "__main__":
    main()
