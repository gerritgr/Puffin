"""Per-cell head: d-dim AE embedding -> (mu, sigma2).

A small MLP -- `Linear(d -> 32) + GELU + Linear(32 -> 2)` -- deliberately kept
small: a large network over-fits the coarse block sums and generalises worse, so
this compact head is the single architecture used throughout. sigma2 > 0 via softplus.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

EPS: float = 1e-6


class MLPHead(nn.Module):
    """Predicts (mu, sigma2) for each fine cell from its d-dim AE embedding.

    Parameters
    ----------
    in_dim      : input feature dimension (64 for full AEF).
    dropout_p   : dropout after the hidden activation (0 = disabled).
    input_mask_p: randomly drop this fraction of input dims each forward pass
                  (train only) -- a regularizer that stops the head leaning on a few dimensions.
    """

    def __init__(self, in_dim: int = 64, dropout_p: float = 0.0,
                 input_mask_p: float = 0.0) -> None:
        super().__init__()
        self.input_mask_p = float(input_mask_p)
        layers: list[nn.Module] = [nn.Linear(in_dim, 32), nn.GELU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(32, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mu, sigma2) of shape [...] for input [..., in_dim]."""
        if self.input_mask_p > 0.0:
            x = torch.nn.functional.dropout(x, p=self.input_mask_p, training=self.training)
        out = self.net(x)
        mu = out[..., 0]
        sigma2 = torch.nn.functional.softplus(out[..., 1]) + EPS
        return mu, sigma2
