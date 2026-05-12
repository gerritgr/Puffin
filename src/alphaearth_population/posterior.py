"""Gaussian conditioning utilities for fine-cell posterior updates."""

from __future__ import annotations

import numpy as np


def gaussian_posterior(
    mu_prior: np.ndarray,
    var_prior: np.ndarray,
    y_coarse: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int,
    tau2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the posterior fine-cell means and variances.

    After observing each coarse-block total ``Y_B``, the posterior for each
    fine cell ``i`` is updated analytically:

    .. math::

        \\mu_i^{\\text{post}} = \\mu_i + \\frac{\\sigma_i^2}{V_B}(Y_B - M_B)

        (\\sigma_i^{\\text{post}})^2 = \\sigma_i^2 - \\frac{\\sigma_i^4}{V_B}

    where :math:`M_B = \\sum_j \\mu_j` and
    :math:`V_B = \\sum_j \\sigma_j^2 + \\tau^2`.

    Parameters
    ----------
    mu_prior:
        Fine-cell prior means of shape ``(N,)``.
    var_prior:
        Fine-cell prior variances of shape ``(N,)``.
    y_coarse:
        Observed coarse-cell totals of shape ``(B,)``.
    block_ids:
        Integer array of shape ``(N,)`` mapping fine cells to coarse blocks.
    n_blocks:
        Total number of coarse blocks ``B``.
    tau2:
        Optional observation noise variance for the coarse ground truth.

    Returns
    -------
    mu_post:
        Posterior fine-cell means of shape ``(N,)``.
    var_post:
        Posterior fine-cell variances of shape ``(N,)`` (non-negative).
    """
    mu_post = np.zeros_like(mu_prior)
    var_post = np.zeros_like(var_prior)

    for b in range(n_blocks):
        idx = np.where(block_ids == b)[0]

        mu_b = mu_prior[idx]
        var_b = var_prior[idx]
        y_b = y_coarse[b]

        mean_agg = mu_b.sum()
        var_agg = var_b.sum() + tau2

        residual = y_b - mean_agg

        mu_post[idx] = mu_b + (var_b / var_agg) * residual
        var_post[idx] = var_b - (var_b ** 2) / var_agg

    var_post = np.maximum(var_post, 0.0)

    return mu_post, var_post
