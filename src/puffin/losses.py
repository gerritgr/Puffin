"""Training losses for weakly-supervised disaggregation.

Total loss:  L = L_NLL  +  lambda_s * L_smooth  +  lambda_v * L_var_reg
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

EPS: float = 1e-6


def aggregate_nll(
    mu_fine: Tensor,
    sigma2_fine: Tensor,
    y_coarse: Tensor,
    tau2_B: Tensor | float = 0.0,
) -> Tensor:
    """Mean Gaussian aggregate NLL over a batch of coarse blocks.

    L_B = 0.5 * [(Y_B - M_B)² / V_B  +  log V_B]
    V_B = Σ σ²_i + τ_B²  (τ_B² already included - do not add again elsewhere).

    Parameters
    ----------
    mu_fine    : [batch, n_cells]  per-cell prior means
    sigma2_fine: [batch, n_cells]  per-cell prior variances (> 0)
    y_coarse   : [batch]           observed coarse block totals
    tau2_B     : [batch] or float  per-block observation noise τ_B²
    """
    M_B = mu_fine.sum(dim=1)
    V_B = torch.clamp(sigma2_fine.sum(dim=1) + tau2_B, min=EPS)
    nll = 0.5 * ((y_coarse - M_B) ** 2 / V_B + torch.log(V_B))
    return nll.mean()


def poisson_aggregate_nll(
    rate_fine: Tensor,
    y_coarse: Tensor,
    block_ids: Tensor,
    n_blocks: int,
    train_mask: Tensor | None = None,
) -> Tensor:
    """Mean Poisson aggregate NLL over coarse blocks (count targets).

    The block total is a sum of independent Poisson cell counts, so it is itself
    Poisson with rate Λ_B = Σ_{i∈B} rate_i:

        L_B = Λ_B − Y_B · log Λ_B        (constant log Y_B! dropped)

    Parameters
    ----------
    rate_fine  : [N]   per-cell expected count (> 0; e.g. softplus of the head)
    y_coarse   : [B]   observed coarse block totals
    block_ids  : [N]   block index per cell (long)
    n_blocks   : int
    train_mask : [B] bool or None - restrict the mean to training blocks
    """
    lam = torch.zeros(n_blocks, device=rate_fine.device).scatter_add(
        0, block_ids, rate_fine).clamp(min=EPS)
    nll = lam - y_coarse * torch.log(lam)
    return nll[train_mask].mean() if train_mask is not None else nll.mean()


def build_neighbor_index(
    x: np.ndarray,
    y: np.ndarray,
    step: int = 100,
    k: int = 4,
) -> np.ndarray:
    """Build a [N, k] neighbor index for a sparse 100m INSPIRE grid.

    Finds the k=4 cardinal neighbors (N/S/E/W at distance `step` metres).
    Cells without a neighbor in that direction use themselves as a stand-in
    so the contribution is (μ_i - μ_i)² = 0.

    Parameters
    ----------
    x, y  : EPSG:3035 cell centroid coordinates (integer metres)
    step  : grid resolution in metres (default 100)
    k     : number of neighbors (must be 4)

    Returns
    -------
    idx : np.ndarray [N, 4], dtype int64
        idx[i] = [north, south, east, west] cell indices.
        Self-index used for missing neighbors.
    """
    coord_to_idx = {(int(xi), int(yi)): i for i, (xi, yi) in enumerate(zip(x, y))}
    N = len(x)
    idx = np.empty((N, k), dtype=np.int64)
    offsets = [(0, step), (0, -step), (step, 0), (-step, 0)]  # N, S, E, W
    for i, (xi, yi) in enumerate(zip(x, y)):
        for j, (dx, dy) in enumerate(offsets):
            idx[i, j] = coord_to_idx.get((int(xi) + dx, int(yi) + dy), i)
    return idx


def smoothness_loss(
    mu_fine: Tensor,
    ae_emb: Tensor,
    neighbor_idx: Tensor,
    gamma: float = 1.0,
) -> Tensor:
    """AE-similarity-weighted spatial smoothness penalty.

    w_ij = exp(-γ · ||ẽ_i - ẽ_j||²)   where ẽ = L2-normalized embedding
    L_smooth = Σ_ij w_ij · (μ_i - μ_j)² / Σ_ij w_ij

    Parameters
    ----------
    mu_fine     : [N] predicted means
    ae_emb      : [N, 64] raw AE embeddings (normalized internally)
    neighbor_idx: [N, k] long tensor of neighbor indices; self-index = no-op
    gamma       : bandwidth; default 1.0 (calibrated for L2-normalized embeddings)
    """
    e_norm = F.normalize(ae_emb, dim=-1)              # [N, 64]
    e_j    = e_norm[neighbor_idx]                     # [N, k, 64]
    e_i    = e_norm.unsqueeze(1).expand_as(e_j)       # [N, k, 64]
    dist2  = ((e_i - e_j) ** 2).sum(dim=-1)           # [N, k]
    w      = torch.exp(-gamma * dist2)                # [N, k]  in (0,1]

    mu_j   = mu_fine[neighbor_idx]                    # [N, k]
    mu_i   = mu_fine.unsqueeze(1).expand_as(mu_j)     # [N, k]
    diff2  = (mu_i - mu_j) ** 2                       # [N, k]

    return (w * diff2).sum() / (w.sum() + EPS)


def variance_reg(sigma2_fine: Tensor) -> Tensor:
    """Penalty on extreme per-cell variance: L_var = mean((log sigma2_i)^2)."""
    return (torch.log(sigma2_fine) ** 2).mean()