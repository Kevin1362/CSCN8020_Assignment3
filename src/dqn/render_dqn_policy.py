from __future__ import annotations

import argparse
from pathlib import Path
import time

from g1_rl import G1ElbowTargetEnv

from .agent import DQNAgent
from .common import get_device, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render saved DQN policy in MuJoCo viewer.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/selected_dqn.pt"))
    parser.add_argument(
        "--goals",
        nargs="+",
        type=float,
        default=[-0.8, -0.4, 0.4, 0.8],
        help="Target angles to demonstrate.",
    )
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    device = get_device(args.device)
    agent = DQNAgent.load_checkpoint(args.checkpoint, device=device)
    agent.online_network.eval()

    env = G1ElbowTargetEnv(
        render_mode="human",
        goal_angle=None,
        goal_range=(-0.8, 0.8),
        maximum_episode_steps=150,
    )

    try:
        for index, goal in enumerate(args.goals, start=1):
            observation, info = env.reset(
                seed=args.seed + index,
                options={"goal_angle": goal},
            )
            cumulative_reward = 0.0
            action_counts = {0: 0, 1: 0, 2: 0}
            terminated = False
            truncated = False

            print(f"\n=== DQN DEMO {index}: target={goal:+.2f} rad ===")
            while not (terminated or truncated):
                action = agent.select_action(observation, epsilon=0.0)
                action_counts[action] += 1
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += float(reward)

            print(f"Success: {bool(info.get('is_success', terminated))}")
            print(f"Episode steps: {info['episode_step']}")
            print(f"Final angle: {info['elbow_angle']:+.4f} rad")
            print(f"Final absolute error: {abs(info['angle_error']):.4f} rad")
            print(f"Cumulative reward: {cumulative_reward:.4f}")
            print(
                "Action counts: "
                f"DECREASE={action_counts[0]}, HOLD={action_counts[1]}, "
                f"INCREASE={action_counts[2]}"
            )
            time.sleep(max(0.0, args.pause_seconds))

        print("\nDemonstration finished. Close the MuJoCo viewer window.")
        while env.viewer is not None and env.viewer.is_running():
            time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
