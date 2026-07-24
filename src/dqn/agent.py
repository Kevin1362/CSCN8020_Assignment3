from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.95
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 50_000
    epsilon_start: float = 1.00
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    target_update_interval: int = 250
    warmup_transitions: int = 500
    gradient_clip_norm: float = 10.0


class DQNAgent:
    """Student-written DQN agent with online/target networks and replay."""

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        config: DQNConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.device = device or torch.device("cpu")

        self.online_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity)
        self.epsilon = float(self.config.epsilon_start)
        self.optimization_steps = 0

    def select_action(self, observation: np.ndarray, epsilon: float | None = None) -> int:
        """Epsilon-greedy action selection.

        During evaluation call with epsilon=0.0 for a deterministic greedy policy.
        """
        eps = self.epsilon if epsilon is None else float(epsilon)
        if random.random() < eps:
            return random.randrange(self.action_dim)

        state = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state)
        return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(self) -> float | None:
        """Perform one DQN update and return scalar loss.

        Time-limit truncation treatment:
        - an episode ends when either terminated or truncated is true;
        - Bellman bootstrapping is blocked only for true terminated states;
        - truncated states are allowed to bootstrap because the environment ended due
          to the time limit rather than a terminal MDP state.
        """
        minimum_samples = max(
            self.config.batch_size, self.config.warmup_transitions
        )
        if len(self.replay_buffer) < minimum_samples:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        selected_q_values = self.online_network(batch.states).gather(
            1, batch.actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(batch.next_states).max(dim=1).values
            non_terminal_mask = (~batch.terminated).float()
            bellman_targets = (
                batch.rewards
                + self.config.gamma * non_terminal_mask * next_q_values
            )

        loss = F.smooth_l1_loss(selected_q_values, bellman_targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_interval == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        return self.epsilon

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "online_network_state_dict": self.online_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "optimization_steps": self.optimization_steps,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, checkpoint_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | None = None,
        load_optimizer: bool = False,
    ) -> "DQNAgent":
        resolved_device = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
        config = DQNConfig(**checkpoint["config"])
        agent = cls(
            observation_dim=int(checkpoint.get("observation_dim", 4)),
            action_dim=int(checkpoint.get("action_dim", 3)),
            config=config,
            device=resolved_device,
        )
        agent.online_network.load_state_dict(checkpoint["online_network_state_dict"])
        target_state = checkpoint.get(
            "target_network_state_dict", checkpoint["online_network_state_dict"]
        )
        agent.target_network.load_state_dict(target_state)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = float(checkpoint.get("epsilon", config.epsilon_min))
        agent.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        agent.online_network.eval()
        agent.target_network.eval()
        return agent
