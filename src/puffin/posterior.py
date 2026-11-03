"""Closed-form posterior updates after observing coarse block aggregates.

Two families, picked by target type:

COUNT targets  → `poisson_allocation` (Poisson observation model, DEFAULT).
  p_i      = rate_i / Λ_B,      Λ_B = Σ_{j∈B} rate_j
  μ_post_i = Y_B · p_i                       (exact conservation, always ≥ 0)
  σ²_post_i = Y_B · p_i · (1 − p_i)          (multinomial marginal variance)
  The block total Y_B is a sum of Poisson cell counts, so allocating it
  proportional to the predicted rate is the exact posterior mean. Unlike the
  Gaussian Kalman update it can never produce negative cell values, and it
  splits the total by predicted magnitude (not by uncertainty).

RATE targets   → `gaussian_posterior` (weighted-mean Kalman update).
  Y_B = Σ(w_i μ_i) / W_B + noise,  W_B = Σ w_i
  V_B   = Σ(w_i² σ²_i) / W_B² + τ²
  K_i   = (w_i σ²_i / W_B) / V_B
  μ_post = μ_i + K_i · (Y_B − M_B)
  σ²_post = σ²_i − (w_i² σ⁴_i / W_B²) / V_B

`gaussian_posterior` also still supports the plain sum mode (weights=None) for
ablations / MSE runs; τ_B²=0 → noiseless, τ_B²>0 → Kalman shrinkage.
"""

from __future__ import annotations

import numpy as np

EPS: float = 1e-6


def poisson_allocation(
    rate: np.ndarray,
    y_coarse: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count-target posterior: multinomial allocation of each block total.

    Distributes the observed block total Y_B across its cells proportional to
    the predicted (non-negative) Poisson rate. Exact conservation, never
    negative. This is the default posterior for COUNT targets.

    Parameters
    ----------
    rate      : (N,)  predicted per-cell expected count (≥ 0; e.g. softplus output)
    y_coarse  : (B,)  observed coarse block totals
    block_ids : (N,)  maps each fine cell to its block index in [0, B)
    n_blocks  : int

    Returns
    -------
    (mu_post, var_post) : each (N,).  mu_post ≥ 0 and sums to Y_B per block;
                          var_post is the multinomial marginal variance.
    """
    rate = np.maximum(np.asarray(rate, float), 0.0)
    lam  = np.zeros(n_blocks)
    np.add.at(lam, block_ids, rate)
    lam  = np.maximum(lam, EPS)
    p    = rate / lam[block_ids]
    mu_post  = y_coarse[block_ids] * p
    var_post = np.maximum(y_coarse[block_ids] * p * (1.0 - p), EPS)
    return mu_post, var_post


def variance_weighted_posterior(
    mu_prior: np.ndarray,
    var_prior: np.ndarray,
    y_coarse: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uncertainty-aware split, shared by the gaussian and gamma methods.

    Distributes each block's residual (Y_B - Σμ_i) across its cells in proportion
    to the per-cell variance, so confident cells stay near their prior mean and
    uncertain cells absorb most of the correction:

        mu_post_i = mu_prior_i + (var_i / Σ_{j∈B} var_j) · (Y_B - Σ_{j∈B} mu_j)

    This linear (Gaussian-conditional-mean) redistribution can go negative, so we
    then CLIP to zero and RESCALE each block to restore exact conservation. If a
    whole block clips to ~0 (e.g. a large negative residual), we fall back to the
    uniform split Y_B / n_B. The returned variance is the Gaussian conditional
    marginal var_i - var_i²/V_B, used only for interval width.

    Parameters
    ----------
    mu_prior  : (N,)  per-cell prior mean (may be signed; e.g. z-scored Gaussian)
    var_prior : (N,)  per-cell prior variance (> 0)
    y_coarse  : (B,)  observed coarse block totals
    block_ids : (N,)  block index per cell in [0, B)
    n_blocks  : int
    """
    mu_prior  = np.asarray(mu_prior, float)
    var_prior = np.maximum(np.asarray(var_prior, float), EPS)

    M_B = np.zeros(n_blocks); np.add.at(M_B, block_ids, mu_prior)     # Σ μ_i
    V_B = np.zeros(n_blocks); np.add.at(V_B, block_ids, var_prior)    # Σ var_i
    V_B = np.maximum(V_B, EPS)

    resid   = (y_coarse - M_B)[block_ids]
    mu_post = mu_prior + (var_prior / V_B[block_ids]) * resid

    # Enforce non-negativity, then rescale each block back to its exact total.
    mu_post = np.maximum(mu_post, 0.0)
    S_B = np.zeros(n_blocks); np.add.at(S_B, block_ids, mu_post)
    n_B = np.zeros(n_blocks); np.add.at(n_B, block_ids, 1.0)
    scale = np.where(S_B > EPS, y_coarse / np.maximum(S_B, EPS), 0.0)
    mu_post = mu_post * scale[block_ids]
    # Blocks that fully clipped to ~0: fall back to the uniform split Y_B / n_B.
    dead = (S_B <= EPS)[block_ids]
    mu_post[dead] = (y_coarse / np.maximum(n_B, 1.0))[block_ids][dead]

    var_post = np.maximum(var_prior - var_prior ** 2 / V_B[block_ids], EPS)
    return mu_post, var_post


def gaussian_posterior(
    mu_prior: np.ndarray,
    var_prior: np.ndarray,
    y_coarse: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int,
    tau2_B: np.ndarray | float = 0.0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior fine-cell (mean, variance) for all blocks.

    Parameters
    ----------
    mu_prior  : (N,)    prior means from trained network
    var_prior : (N,)    prior variances (must be > 0)
    y_coarse  : (B,)    observed coarse-block aggregates
    block_ids : (N,)    maps each fine cell to its block index in [0, B)
    n_blocks  : int
    tau2_B    : (B,) or float  per-block observation noise
    weights   : (N,) or None   per-cell aggregation weights.
                None → sum aggregation (count targets).
                Array → weighted-mean aggregation (rate targets).
                Must use the same weights as training block_stats.
    """
    tau2 = np.broadcast_to(tau2_B, (n_blocks,)) if np.isscalar(tau2_B) else np.asarray(tau2_B)
    rate_mode = weights is not None

    mu_post  = np.empty_like(mu_prior)
    var_post = np.empty_like(var_prior)

    for b in range(n_blocks):
        idx   = np.where(block_ids == b)[0]
        mu_b  = mu_prior[idx]
        var_b = var_prior[idx]

        if rate_mode:
            w_b = weights[idx]
            W_B = w_b.sum()
            if W_B < EPS:
                mu_post[idx]  = mu_b
                var_post[idx] = var_b
                continue
            M_B = (w_b * mu_b).sum() / W_B
            V_B = (w_b ** 2 * var_b).sum() / (W_B ** 2) + tau2[b]
            K_i = (w_b * var_b / W_B) / max(V_B, EPS)
            mu_post[idx]  = mu_b + K_i * (y_coarse[b] - M_B)
            var_post[idx] = var_b - (w_b ** 2 * var_b ** 2 / W_B ** 2) / max(V_B, EPS)
        else:
            V_B = var_b.sum() + tau2[b]
            mu_post[idx]  = mu_b + (var_b / V_B) * (y_coarse[b] - mu_b.sum())
            var_post[idx] = var_b - (var_b ** 2) / V_B

    return mu_post, np.maximum(var_post, EPS)
