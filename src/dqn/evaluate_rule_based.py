from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from g1_rl import G1ElbowTargetEnv

from .common import set_global_seeds, write_csv, write_json
from .evaluate_dqn import BENCHMARK_GOALS


def choose_rule_based_action(
    observation: np.ndarray,
    controller_target: float,
    action_increment: float,
) -> int:
    """Same decision rule used by the primer environment validation script."""
    goal_angle = float(observation[2])
    target_error = goal_angle - controller_target
    target_tolerance = action_increment / 2.0
    if target_error < -target_tolerance:
        return G1ElbowTargetEnv.ACTION_DECREASE
    if target_error > target_tolerance:
        return G1ElbowTargetEnv.ACTION_INCREASE
    return G1ElbowTargetEnv.ACTION_HOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based baseline on 20 episodes.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rule_based"))
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def evaluate(output_dir: Path, seed: int = 1000) -> dict[str, float | int | str]:
    set_global_seeds(seed)
    env = G1ElbowTargetEnv(
        render_mode=None,
        goal_angle=None,
        goal_range=(-0.8, 0.8),
        maximum_episode_steps=150,
    )

    rows: list[dict[str, float | int]] = []
    episode_counter = 0
    try:
        for goal in BENCHMARK_GOALS:
            for repetition in range(1, 6):
                episode_counter += 1
                observation, info = env.reset(
                    seed=seed + episode_counter,
                    options={"goal_angle": goal},
                )
                cumulative_reward = 0.0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action = choose_rule_based_action(
                        observation=observation,
                        controller_target=float(info["controller_target"]),
                        action_increment=float(env.action_increment),
                    )
                    observation, reward, terminated, truncated, info = env.step(action)
                    cumulative_reward += float(reward)

                rows.append(
                    {
                        "episode": episode_counter,
                        "goal_angle": goal,
                        "repetition": repetition,
                        "success": int(bool(info.get("is_success", terminated))),
                        "cumulative_reward": cumulative_reward,
                        "episode_length": int(info["episode_step"]),
                        "final_absolute_error": float(
                            info.get("absolute_error", abs(info["angle_error"]))
                        ),
                    }
                )
    finally:
        env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_episodes.csv", rows)

    grouped: dict[float, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["goal_angle"])].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for goal in BENCHMARK_GOALS:
        goal_rows = grouped[goal]
        successes = sum(int(r["success"]) for r in goal_rows)
        summary_rows.append(
            {
                "goal": f"{goal:+.1f}",
                "episodes": len(goal_rows),
                "successes": successes,
                "success_rate": successes / len(goal_rows),
                "mean_reward": float(
                    np.mean([float(r["cumulative_reward"]) for r in goal_rows])
                ),
                "mean_episode_length": float(
                    np.mean([int(r["episode_length"]) for r in goal_rows])
                ),
                "mean_final_absolute_error": float(
                    np.mean([float(r["final_absolute_error"]) for r in goal_rows])
                ),
            }
        )

    successes = sum(int(r["success"]) for r in rows)
    overall = {
        "goal": "Overall",
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_reward": float(np.mean([float(r["cumulative_reward"]) for r in rows])),
        "mean_episode_length": float(np.mean([int(r["episode_length"]) for r in rows])),
        "mean_final_absolute_error": float(
            np.mean([float(r["final_absolute_error"]) for r in rows])
        ),
    }
    summary_rows.append(overall)
    write_csv(output_dir / "evaluation_summary.csv", summary_rows)
    write_json(output_dir / "evaluation_summary.json", overall)

    print("\nRule-based evaluation complete")
    print(f"Successes: {successes}/20")
    print(f"Success rate: {100.0 * float(overall['success_rate']):.1f}%")
    print(f"Mean reward: {float(overall['mean_reward']):.4f}")
    return overall


def main() -> None:
    args = parse_args()
    evaluate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
