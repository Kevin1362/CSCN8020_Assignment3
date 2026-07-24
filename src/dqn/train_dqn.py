from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import statistics
import time

import numpy as np

from g1_rl import G1ElbowTargetEnv

from .agent import DQNAgent, DQNConfig
from .common import get_device, set_global_seeds, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on the Unitree G1 elbow task.")
    parser.add_argument("--config-name", required=True, choices=["config_a", "config_b"])
    parser.add_argument("--epsilon-decay", required=True, type=float)
    parser.add_argument("--episodes", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-hours", type=float, default=2.30)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    set_global_seeds(args.seed)
    device = get_device(args.device)

    output_dir = args.output_dir or Path("results") / args.config_name
    checkpoint_dir = args.checkpoint_dir or Path("models") / args.config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = DQNConfig(epsilon_decay=float(args.epsilon_decay))
    agent = DQNAgent(config=config, device=device)

    env = G1ElbowTargetEnv(
        render_mode=None,
        goal_angle=None,
        goal_range=(-0.8, 0.8),
        maximum_episode_steps=150,
    )
    env.action_space.seed(args.seed)

    episode_rows: list[dict[str, float | int | bool]] = []
    loss_rows: list[dict[str, float | int]] = []
    recent_successes: deque[int] = deque(maxlen=50)
    recent_rewards: deque[float] = deque(maxlen=20)

    best_score = (-1.0, float("-inf"))
    start_time = time.perf_counter()
    stop_reason = "episode_cap"

    try:
        for episode in range(1, args.episodes + 1):
            elapsed_hours = (time.perf_counter() - start_time) / 3600.0
            if elapsed_hours >= args.max_hours:
                stop_reason = "time_limit"
                print(
                    f"Stopping {args.config_name}: reached {args.max_hours:.2f} hour limit."
                )
                break

            observation, info = env.reset(seed=args.seed + episode)
            cumulative_reward = 0.0
            losses: list[float] = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(observation)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                agent.replay_buffer.push(
                    state=observation,
                    action=action,
                    reward=reward,
                    next_state=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )

                loss = agent.optimize_model()
                if loss is not None:
                    losses.append(loss)
                    loss_rows.append(
                        {
                            "optimization_step": agent.optimization_steps,
                            "episode": episode,
                            "loss": loss,
                        }
                    )

                observation = next_observation
                cumulative_reward += float(reward)

            is_success = bool(info.get("is_success", terminated))
            final_abs_error = float(info.get("absolute_error", abs(info["angle_error"])))
            episode_length = int(info["episode_step"])
            goal_angle = float(info["goal_angle"])
            mean_loss = float(statistics.mean(losses)) if losses else float("nan")

            recent_successes.append(int(is_success))
            recent_rewards.append(cumulative_reward)
            rolling_success = float(np.mean(recent_successes))
            rolling_reward = float(np.mean(recent_rewards))

            row = {
                "episode": episode,
                "cumulative_reward": cumulative_reward,
                "success": int(is_success),
                "episode_length": episode_length,
                "final_absolute_error": final_abs_error,
                "goal_angle": goal_angle,
                "epsilon": agent.epsilon,
                "mean_loss": mean_loss,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            episode_rows.append(row)

            score = (rolling_success, rolling_reward)
            if len(recent_successes) == recent_successes.maxlen and score > best_score:
                best_score = score
                agent.save_checkpoint(
                    checkpoint_dir / "best.pt",
                    extra={
                        "config_name": args.config_name,
                        "episode": episode,
                        "rolling_50_success_rate": rolling_success,
                        "rolling_20_mean_reward": rolling_reward,
                        "seed": args.seed,
                    },
                )

            agent.decay_epsilon()

            if episode == 1 or episode % 25 == 0:
                print(
                    f"{args.config_name} | episode={episode:4d} | "
                    f"reward={cumulative_reward:+8.3f} | success={int(is_success)} | "
                    f"epsilon={agent.epsilon:.4f} | "
                    f"rolling50_success={rolling_success:.3f} | "
                    f"buffer={len(agent.replay_buffer)}"
                )
    finally:
        env.close()

    training_seconds = time.perf_counter() - start_time
    if not episode_rows:
        raise RuntimeError("Training ended before any episode completed.")

    agent.save_checkpoint(
        checkpoint_dir / "final.pt",
        extra={
            "config_name": args.config_name,
            "episodes_completed": len(episode_rows),
            "training_seconds": training_seconds,
            "seed": args.seed,
            "stop_reason": stop_reason,
        },
    )

    write_csv(output_dir / "training_metrics.csv", episode_rows)
    write_csv(output_dir / "loss_metrics.csv", loss_rows)

    final_20_rewards = [float(row["cumulative_reward"]) for row in episode_rows[-20:]]
    final_50_successes = [int(row["success"]) for row in episode_rows[-50:]]
    summary: dict[str, float | int | str] = {
        "config_name": args.config_name,
        "epsilon_decay": float(args.epsilon_decay),
        "seed": int(args.seed),
        "device": str(device),
        "episodes_requested": int(args.episodes),
        "episodes_completed": len(episode_rows),
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60.0,
        "final_epsilon": float(agent.epsilon),
        "mean_reward_final_20": float(np.mean(final_20_rewards)),
        "training_success_rate_final_50": float(np.mean(final_50_successes)),
        "optimization_steps": int(agent.optimization_steps),
        "stop_reason": stop_reason,
    }
    write_json(output_dir / "training_summary.json", summary)

    print("\nTraining complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
