from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Simple feed-forward Q-network required by the assignment.

    Input:  4 values from G1ElbowTargetEnv observation.
    Hidden: 64 ReLU -> 64 ReLU.
    Output: 3 unconstrained Q-values, one for each discrete action.
    """

    def __init__(self, observation_dim: int = 4, action_dim: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
