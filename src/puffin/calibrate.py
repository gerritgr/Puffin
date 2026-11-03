"""Region-aware conformal calibration of per-cell predictive uncertainty.

Per-cell variances from aggregate-only training are (a) globally overconfident
and (b) spatially heterogeneous (the needed scale varies ~3x across regions,
tied to spatial non-stationarity). A single global temperature cannot fix both.

This module fixes it in two stages, both leak-free and (stage 1) deployable:

  Stage 1 - REGION scale from COARSE residuals (observable everywhere):
     for each spatial tile, c²_tile = Σ_b (Y_B - M_B)² / Σ_b Σσ²_i
     σ²_region(i) = c²_{tile(i)} · σ²_i
     This equalises coverage across regions using only the coarse aggregates
     (no fine labels), so it works where no fine truth exists.

  Stage 2 - global conformal level q on the standardised scores:
     z_i = (y_i - μ_i) / sqrt(σ²_region(i)),  q = Quantile_{val}(|z|, p)
     interval_i = μ_i ± q · sqrt(σ²_region(i))
     A single constant (the within-block residual-correlation factor) fit on a
     held-out calibration fold; transfers across regions because stage 1 already
     removed the spatial variation.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln


def block_dispersion_conformal(
    lam, y, cell_fold_block, val_fold, test_fold, levels=(0.5, 0.8, 0.9, 0.95),
):
    """Calibrate block-total predictive intervals for COUNT targets.

    Predicted block totals Λ_B (= Σ predicted rate) are heavily overdispersed vs
    the pure-Poisson variance Λ_B: real census totals carry model prediction error
    plus disclosure noise, so (Y_B − Λ_B)² is ~40–150× Λ_B. Pure-Poisson intervals
    therefore collapse (coverage ~0.25–0.40). Two leak-free, deployable stages
    (coarse totals only, no fine labels) - mirrors `region_conformal` at the block
    level:

      Stage 1 - negative-binomial dispersion r (method of moments on the VAL fold):
        Var(Y_B) = Λ_B + Λ_B² / r ;   r = mean(Λ²) / (mean((Y−Λ)²) − mean(Λ))
      Stage 2 - conformal level q on VAL standardised residuals:
        z_B = (Y_B − Λ_B) / sqrt(Var(Λ_B)),   q = Quantile_val(|z_B|, p)

    Parameters
    ----------
    lam            : (B,) predicted block rate Λ_B
    y              : (B,) observed block total Y_B
    cell_fold_block: (B,) fold id per block
    val_fold, test_fold : ints

    Returns
    -------
    (r, out) where out = {"r", "q", "coverage_test", "nll_test"}.
    """
    lam = np.asarray(lam, float); y = np.asarray(y, float)
    fold = np.asarray(cell_fold_block)
    val = fold == val_fold; test = fold == test_fold
    if val.sum() < 2:
        return float("inf"), {"r": float("inf"), "q": {}, "coverage_test": {}, "nll_test": float("nan")}

    Lv, Yv = lam[val], y[val]
    den = float(((Yv - Lv) ** 2).mean() - Lv.mean())
    r = max((Lv ** 2).mean() / den, 1e-3) if den > 1e-9 else 1e12   # large r → ~Poisson
    var = lambda L: np.maximum(L + L ** 2 / r, 1e-9)

    zv = np.abs(Yv - Lv) / np.sqrt(var(Lv))
    Lt, Yt = lam[test], y[test]
    zt = np.abs(Yt - Lt) / np.sqrt(var(Lt))
    out = {"r": float(r), "q": {}, "coverage_test": {}}
    for p in levels:
        q = float(np.quantile(zv, p)) if val.sum() else float("nan")
        out["q"][p] = q
        out["coverage_test"][p] = float(np.mean(zt <= q)) if test.sum() else float("nan")

    # NB2 negative log-likelihood on the test blocks (proper block-level score)
    if test.sum() and np.isfinite(r) and r < 1e11:
        p_nb = r / (r + Lt)
        nll = -(gammaln(Yt + r) - gammaln(r) - gammaln(Yt + 1.0)
                + r * np.log(p_nb) + Yt * np.log1p(-p_nb))
        out["nll_test"] = float(np.mean(nll))
    else:
        out["nll_test"] = float("nan")
    return float(r), out


def region_scale(block_x, block_y, resid2, pred_var, tile_m=10000.0):
    """c²_tile per block from coarse block residuals; falls back to the global
    ratio for tiles with negligible predicted variance. Deployable (coarse-only)."""
    tx = (np.asarray(block_x) // tile_m * tile_m).astype(np.int64)
    ty = (np.asarray(block_y) // tile_m * tile_m).astype(np.int64)
    key = tx.astype(np.int64) * 1_000_000_007 + ty.astype(np.int64)
    resid2 = np.asarray(resid2, float); pred_var = np.maximum(np.asarray(pred_var, float), 1e-12)
    global_c2 = resid2.sum() / pred_var.sum()
    c2 = np.full(len(key), global_c2, float)
    for k in np.unique(key):
        m = key == k
        pv = pred_var[m].sum()
        if pv > 1e-9:
            c2[m] = resid2[m].sum() / pv
    return np.maximum(c2, 1e-6)


def region_conformal(
    y, mu, sig2, block_idx, cell_fold,
    block_x, block_y, block_resid2, block_predvar,
    val_fold, test_fold, tile_m=10000.0, levels=(0.5, 0.8, 0.9, 0.95),
):
    """Return calibrated per-cell sigma², the conformal quantiles, and coverage.

    Parameters are per-cell arrays (y, mu, sig2, block_idx, cell_fold) and
    per-block arrays (block_x, block_y, block_resid2=(Y_B-M_B)², block_predvar=Σσ²_i).
    Stage-1 scale uses ALL blocks (coarse only); stage-2 q uses VAL-fold cells.
    """
    c2_blk = region_scale(block_x, block_y, block_resid2, block_predvar, tile_m)
    sig2_cal = sig2 * c2_blk[block_idx]
    z = (y - mu) / np.sqrt(np.maximum(sig2_cal, 1e-12))

    val = cell_fold == val_fold
    test = cell_fold == test_fold
    out = {"tile_m": tile_m, "q": {}, "coverage_test": {}, "coverage_by_fold": {}}
    folds = sorted(set(int(f) for f in np.unique(cell_fold) if f >= 0))
    for p in levels:
        q = float(np.quantile(np.abs(z[val]), p)) if val.sum() else float("nan")
        out["q"][p] = q
        out["coverage_test"][p] = float(np.mean(np.abs(z[test]) <= q)) if test.sum() else float("nan")
        out["coverage_by_fold"][p] = {f: float(np.mean(np.abs(z[cell_fold == f]) <= q)) for f in folds}
    # report a single calibrated sigma at the 90% level (Gaussian-equivalent)
    q90 = out["q"].get(0.9, 1.645)
    z90 = 1.6448536269514722
    sig2_report = sig2_cal * (q90 / z90) ** 2
    return sig2_report, out