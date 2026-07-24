from __future__ import annotations

import numpy as np
import torch

from g1_rl import G1ElbowTargetEnv

from .agent import DQNAgent, DQNConfig
from .common import set_global_seeds


def main() -> None:
    seed = 42
    set_global_seeds(seed)
    device = torch.device("cpu")

    env = G1ElbowTargetEnv(
        render_mode=None,
        goal_angle=None,
        goal_range=(-0.8, 0.8),
        maximum_episode_steps=150,
    )
    agent = DQNAgent(config=DQNConfig(warmup_transitions=64), device=device)

    try:
        observation, _ = env.reset(seed=seed)
        for i in range(80):
            action = agent.select_action(observation, epsilon=1.0)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.replay_buffer.push(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                truncated,
            )
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset(seed=seed + i + 1)

        batch = agent.replay_buffer.sample(64, device)
        assert batch.states.shape == (64, 4)
        assert batch.actions.shape == (64,)
        assert batch.next_states.shape == (64, 4)

        greedy_action = agent.select_action(np.zeros(4, dtype=np.float32), epsilon=0.0)
        assert greedy_action in {0, 1, 2}

        loss = agent.optimize_model()
        assert loss is not None and np.isfinite(loss)

        print("Smoke test PASSED")
        print(f"Replay size: {len(agent.replay_buffer)}")
        print(f"Sample states shape: {tuple(batch.states.shape)}")
        print(f"Greedy action: {greedy_action}")
        print(f"One optimization loss: {loss:.6f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
