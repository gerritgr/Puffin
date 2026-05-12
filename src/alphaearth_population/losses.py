"""Aggregate Gaussian likelihood loss for population disaggregation."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Small numerical stabilizer
EPS: float = 1e-6


def aggregate_gaussian_nll(
    mu_fine: Tensor,
    var_fine: Tensor,
    y_coarse: Tensor,
    tau2: float = 0.0,
) -> Tensor:
    """Compute the aggregate Gaussian negative log-likelihood.

    For each coarse block ``B``, the predictive aggregate distribution is:

    .. math::

        S_B \\sim \\mathcal{N}\\!\\left(\\sum_i \\mu_i,\\,
        \\sum_i \\sigma_i^2 + \\tau^2\\right)

    Parameters
    ----------
    mu_fine:
        Fine-cell predicted means of shape ``[batch, fine_per_coarse]``.
    var_fine:
        Fine-cell predicted variances of shape ``[batch, fine_per_coarse]``.
    y_coarse:
        Observed coarse-cell totals of shape ``[batch]``.
    tau2:
        Optional observation noise variance for the coarse ground truth.

    Returns
    -------
    Tensor
        Scalar mean NLL over the batch.
    """
    mean_agg = mu_fine.sum(dim=1)
    var_agg = var_fine.sum(dim=1) + tau2
    var_agg = torch.clamp(var_agg, min=EPS)

    nll = 0.5 * (
        ((y_coarse - mean_agg) ** 2) / var_agg
        + torch.log(var_agg)
        + np.log(2 * np.pi)
    )

    return nll.mean()
