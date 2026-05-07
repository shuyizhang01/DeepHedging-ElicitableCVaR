"""
src/agents/StaticRiskModel.py

Deterministic MLP actor for static-risk hedging.
"""

import numpy as np
import torch
import torch.nn as nn


class MLPActorStatic(nn.Module):
    """
    Deterministic MLP actor for static-risk hedging.

    Maps state → tanh-squashed action in [-action_high, action_high].
    A learnable quantile parameter q is attached for the CVaR dual optimisation
    in TrainingStatic.py.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, action_high: float = 2.0):
        super().__init__()
        self.action_high = action_high

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(hidden_dim, action_dim),
        )

        # Learnable VaR quantile for CVaR dual formulation
        self.q = nn.Parameter(torch.tensor(10.0))

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.action_high * torch.tanh(self.fc(s))

    def sample(self, s: torch.Tensor, deterministic: bool = True):
        """API-compatible with Actor — always deterministic, zero log-prob."""
        action   = self.forward(s)
        log_prob = torch.zeros(s.shape[0], device=s.device)
        return action, log_prob
