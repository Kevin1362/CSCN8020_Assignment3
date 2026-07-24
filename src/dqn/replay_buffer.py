from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Bounded replay memory with uniform random mini-batch sampling."""

    def __init__(self, capacity: int = 50_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._memory: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._memory)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._memory.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._memory) < batch_size:
            raise ValueError(
                f"Not enough samples: have {len(self._memory)}, need {batch_size}."
            )

        batch = random.sample(self._memory, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        terminated = np.asarray([t.terminated for t in batch], dtype=np.bool_)
        truncated = np.asarray([t.truncated for t in batch], dtype=np.bool_)

        return TransitionBatch(
            states=torch.as_tensor(states, dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.long, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            next_states=torch.as_tensor(
                next_states, dtype=torch.float32, device=device
            ),
            terminated=torch.as_tensor(terminated, dtype=torch.bool, device=device),
            truncated=torch.as_tensor(truncated, dtype=torch.bool, device=device),
        )
