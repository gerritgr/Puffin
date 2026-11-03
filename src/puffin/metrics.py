"""Evaluation metrics: calibration, scoring rules, and fine-scale accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def coverage_metrics(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    y_true: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Coverage and interval width at multiple nominal levels.

    Returns DataFrame with: nominal_coverage, empirical_coverage, avg_interval_width.
    """
    dist = Normal(torch.tensor(mean_pred, dtype=torch.float32),
                  torch.tensor(std_pred,  dtype=torch.float32))
    y_t  = torch.tensor(y_true, dtype=torch.float32)
    rows = []
    for p in levels:
        lo = dist.icdf(torch.tensor((1 - p) / 2))
        hi = dist.icdf(torch.tensor((1 + p) / 2))
        rows.append({
            "nominal_coverage":   p,
            "empirical_coverage": ((y_t >= lo) & (y_t <= hi)).float().mean().item(),
            "avg_interval_width": (hi - lo).mean().item(),
        })
    return pd.DataFrame(rows)


def pit_values(mean_pred: np.ndarray, std_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """PIT values - should be Uniform(0,1) if calibrated."""
    dist = Normal(torch.tensor(mean_pred, dtype=torch.float32),
                  torch.tensor(std_pred,  dtype=torch.float32))
    return dist.cdf(torch.tensor(y_true, dtype=torch.float32)).numpy()


def standardized_residuals(mean_pred: np.ndarray, std_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """(y − mean) / std - should be ~N(0,1) if calibrated."""
    return (y_true - mean_pred) / std_pred


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def aggregate_nll_score(mean_pred: np.ndarray, std_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Gaussian NLL (including 0.5·log(2π) constant)."""
    z = (y_true - mean_pred) / std_pred
    return float(np.mean(0.5 * (z**2 + np.log(std_pred**2) + np.log(2 * np.pi))))


def crps(mean_pred: np.ndarray, std_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-observation CRPS (Gaussian closed form, Logic Module §1.8.6).

    CRPS = σ · [z·(2Φ(z)−1) + 2φ(z) − 1/√π]   where z = (y−μ)/σ
    Lower is better.
    """
    z = (y_true - mean_pred) / std_pred
    return std_pred * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def mean_crps(mean_pred: np.ndarray, std_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(crps(mean_pred, std_pred, y_true).mean())


# ---------------------------------------------------------------------------
# Fine-scale (Phase 6+, requires Zensus 100m ground truth)
# ---------------------------------------------------------------------------

def fine_scale_r2(y_true_fine: np.ndarray, y_pred_fine: np.ndarray) -> float:
    """R² between posterior means and held-out Zensus 100m values."""
    ss_res = ((y_true_fine - y_pred_fine) ** 2).sum()
    ss_tot = ((y_true_fine - y_true_fine.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def fine_scale_mae(y_true_fine: np.ndarray, y_pred_fine: np.ndarray) -> float:
    return float(np.abs(y_true_fine - y_pred_fine).mean())
