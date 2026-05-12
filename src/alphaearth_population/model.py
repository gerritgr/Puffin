"""Neural network modules for fine-cell Gaussian prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Small numerical stabilizer to prevent zero variance
EPS: float = 1e-6


class FineCellGaussianNN(nn.Module):
    """Predicts a Gaussian distribution (mean, variance) for each fine cell.

    Parameters
    ----------
    in_dim:
        Dimensionality of the AlphaEarth feature vector (default ``64``).
    hidden_dim:
        Width of hidden layers (default ``128``).
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input features of shape ``[..., in_dim]``.

        Returns
        -------
        mu:
            Predicted fine-cell means of shape ``[...]``.
        var:
            Predicted fine-cell variances (positive) of shape ``[...]``.
        """
        out = self.net(x)

        mu = out[..., 0]

        # softplus guarantees positive variance
        raw_var = out[..., 1]
        var = torch.nn.functional.softplus(raw_var) + EPS

        return mu, var
