from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from g1_rl import G1ElbowTargetEnv

from .agent import DQNAgent
from .common import get_device, set_global_seeds, write_csv, write_json


BENCHMARK_GOALS = (-0.8, -0.4, 0.4, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy 20-episode DQN evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def evaluate(
    checkpoint: Path,
    output_dir: Path,
    seed: int = 1000,
    device_name: str = "cpu",
) -> dict[str, float | int | str]:
    set_global_seeds(seed)
    device = get_device(device_name)
    agent = DQNAgent.load_checkpoint(checkpoint, device=device)
    agent.online_network.eval()

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
                    action = agent.select_action(observation, epsilon=0.0)
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
                        "terminated": int(terminated),
                        "truncated": int(truncated),
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

    overall_successes = sum(int(r["success"]) for r in rows)
    overall = {
        "goal": "Overall",
        "episodes": len(rows),
        "successes": overall_successes,
        "success_rate": overall_successes / len(rows),
        "mean_reward": float(np.mean([float(r["cumulative_reward"]) for r in rows])),
        "mean_episode_length": float(np.mean([int(r["episode_length"]) for r in rows])),
        "mean_final_absolute_error": float(
            np.mean([float(r["final_absolute_error"]) for r in rows])
        ),
    }
    summary_rows.append(overall)
    write_csv(output_dir / "evaluation_summary.csv", summary_rows)

    json_summary: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint),
        "evaluation_epsilon": 0.0,
        "episodes": len(rows),
        "successes": overall_successes,
        "success_rate": float(overall["success_rate"]),
        "mean_reward": float(overall["mean_reward"]),
        "mean_episode_length": float(overall["mean_episode_length"]),
        "mean_final_absolute_error": float(overall["mean_final_absolute_error"]),
    }
    write_json(output_dir / "evaluation_summary.json", json_summary)

    print("\nGreedy evaluation complete")
    print(f"Checkpoint: {checkpoint}")
    print(f"Successes: {overall_successes}/20")
    print(f"Success rate: {100.0 * float(overall['success_rate']):.1f}%")
    print(f"Mean reward: {float(overall['mean_reward']):.4f}")
    return json_summary


def main() -> None:
    args = parse_args()
    evaluate(args.checkpoint, args.output_dir, args.seed, args.device)


if __name__ == "__main__":
    main()
