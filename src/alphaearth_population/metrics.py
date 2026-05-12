"""Calibration and scoring metrics for aggregate Gaussian predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal


def coverage_metrics(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    y_true: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Compute predictive interval coverage at multiple nominal levels.

    Parameters
    ----------
    mean_pred:
        Predicted aggregate means of shape ``(B,)``.
    std_pred:
        Predicted aggregate standard deviations of shape ``(B,)``.
    y_true:
        Observed coarse values of shape ``(B,)``.
    levels:
        Nominal coverage levels to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ``nominal_coverage``, ``empirical_coverage``,
        ``avg_interval_width``.
    """
    mean_t = torch.tensor(mean_pred, dtype=torch.float32)
    std_t = torch.tensor(std_pred, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)

    dist = Normal(mean_t, std_t)

    rows = []
    for level in levels:
        alpha = 1.0 - level
        lower = dist.icdf(torch.tensor(alpha / 2))
        upper = dist.icdf(torch.tensor(1.0 - alpha / 2))

        inside = ((y_t >= lower) & (y_t <= upper)).float()
        width = upper - lower

        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": inside.mean().item(),
                "avg_interval_width": width.mean().item(),
            }
        )

    return pd.DataFrame(rows)


def pit_values(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Compute probability integral transform (PIT) values.

    Parameters
    ----------
    mean_pred:
        Predicted aggregate means of shape ``(B,)``.
    std_pred:
        Predicted aggregate standard deviations of shape ``(B,)``.
    y_true:
        Observed coarse values of shape ``(B,)``.

    Returns
    -------
    np.ndarray
        PIT values in ``[0, 1]`` of shape ``(B,)``.
    """
    mean_t = torch.tensor(mean_pred, dtype=torch.float32)
    std_t = torch.tensor(std_pred, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)

    dist = Normal(mean_t, std_t)
    return dist.cdf(y_t).numpy()


def standardized_residuals(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Compute standardized residuals ``(y - mean) / std``.

    Parameters
    ----------
    mean_pred:
        Predicted aggregate means of shape ``(B,)``.
    std_pred:
        Predicted aggregate standard deviations of shape ``(B,)``.
    y_true:
        Observed coarse values of shape ``(B,)``.

    Returns
    -------
    np.ndarray
        Standardized residuals of shape ``(B,)``.
    """
    return (y_true - mean_pred) / std_pred


def aggregate_nll_score(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute the mean aggregate Gaussian NLL.

    Parameters
    ----------
    mean_pred:
        Predicted aggregate means of shape ``(B,)``.
    std_pred:
        Predicted aggregate standard deviations of shape ``(B,)``.
    y_true:
        Observed coarse values of shape ``(B,)``.

    Returns
    -------
    float
        Mean NLL over all blocks.
    """
    mean_t = torch.tensor(mean_pred, dtype=torch.float32)
    std_t = torch.tensor(std_pred, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)

    dist = Normal(mean_t, std_t)
    return -dist.log_prob(y_t).mean().item()
