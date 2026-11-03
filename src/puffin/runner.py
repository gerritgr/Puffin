"""Core Puffin training and evaluation: fit the prior on regional totals, then
split each observed total among its subregions."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

from puffin.config import (
    AE_COLS, COUNT_TARGETS, DATA_PROCESSED, DEVICE, EXPERIMENTS_DIR,
)
from puffin.data import build_coarse_blocks, make_spatial_splits, aggregate_to_subregions
from puffin.model import MLPHead
from puffin.posterior import gaussian_posterior, poisson_allocation, variance_weighted_posterior
from puffin.calibrate import region_conformal, block_dispersion_conformal


# ---------------------------------------------------------------------------

@dataclass
class PuffinConfig:
    region:      str   = "berlin"
    year:        int   = 2022
    target:      str   = "population"
    region_m:    int   = 400      # region (coarse block with a known total) size in metres
    subregion_m: int   = 100      # subregion (predicted unit) size in metres; >=100, must divide region_m
    epochs:      int   = 2000     # cosine anneals LR over this horizon (runs to the end)
    lr:          float = 3e-3     # peak learning rate; the cosine schedule anneals it to 0
    lr_schedule: str   = "cosine" # "cosine" (anneal 3e-3->0; converges fast, no overtrain) | "none"
    tau2_train:  float = 5_000.0
    loss:        str   = "nll"    # "nll" | "mse"
    method:      str   = "poisson"  # count targets only:
                                  # "poisson"  Poisson aggregate NLL, multinomial split
                                  # "gaussian" Gaussian aggregate NLL, variance-weighted split
                                  # "gamma"    moment-matched Gamma aggregate NLL
                                  # gaussian and gamma clip at zero and rescale to conserve.
    mask_p:      float = 0.0      # fraction of AEF dims zeroed each train step (0 = off)
    normalize:   str   = "none"   # "none" | "R1" (per-cell L2 unit norm)
    sim_dim:     int   = 0        # 0 = use full 64-dim AEF; K>0 = K-means fallback (ignored if sim_anchors set)
    sim_anchors: list  = field(default_factory=list)  # [(lat, lon), ...] semantic anchor coords
    patience:    int   = 50       # early stopping: stop if val doesn't improve for this many epochs
    val_fold:    int   = 3
    test_fold:   int   = 4
    seed:        int   = 42        # RNG for MODEL INIT / dropout only
    split_seed:  int   = 0         # RNG for the SPATIAL SPLIT. Fixed by default so a
                                   # multi-seed sweep varies only init (clean init-variance
                                   # band, one fixed test fold) instead of also reshuffling
                                   # which region is held out. Set per-run for spatial CV.
    standardize: bool  = True      # per-dim z-score the network input (stats from TRAIN
                                   # cells; saved to feat_stats.npz so a later run can reuse them)
    super_block_m: int = 5000      # spatial-CV super-block size (m) for fold assignment.
                                   # >= region size -> contiguous held-out patches (spatial
                                   # EXTRAPOLATION); < region size -> ~random per-region
                                   # (INTERPOLATION). Berlin default 5km.
    batch_blocks: int = 0          # >0: block-wise minibatch -- sample this many whole
                                   # TRAIN blocks per step (weak-sup needs whole blocks).
                                   # 0 = full-batch. For country scale (germany_all 100m).
                                   # Count/Poisson targets only; ignored otherwise.
    eval_every:  int = 1           # val-NLL eval + checkpoint every N epochs (1 = every
                                   # epoch); >1 speeds huge runs. Last epoch always evaluated.
    save_predictions: bool = True  # write per-cell predictions.npz (for fig_disagg/prob).
                                   # Set False at country scale to avoid huge dumps.
    fixed_region: bool = False     # True -> use a data column (cfg.region_col) as the coarse
                                   # unit (e.g. us_voting county/state -> precinct); region_m/
                                   # subregion_m are ignored. False -> grid mode (parametric
                                   # region_m x subregion_m sweep, e.g. population).
    region_col:  str = "region_id" # column holding the coarse-region id in fixed mode
                                   # (us_voting: "county_fips" or "state_fips").
    exposure_col: str = ""         # count targets only: a per-unit AREA/exposure column, so
                                   # the Poisson rate is  area_i * softplus(f(AE_i)) = count.
                                   # Needed for IRREGULAR units (US census blocks); leave ""
                                   # for a uniform grid (Germany), where area is constant.
    exposure_winsor_q: float = 0.0    # optional: cap exposure at this quantile (0/>=1 = off).
                                   # off by default.
    grad_clip:   float = 0.0       # optional: max gradient norm (0 = off).
    area_variance: bool = True     # gaussian + exposure_col only: scale the PRIOR VARIANCE by
                                   # the subregion area, so a large block absorbs a larger share
                                   # of its region's correction. False = ablation (mean still
                                   # scales with area; only the variance term is dropped). No
                                   # effect on a uniform grid, where the area is constant.
    curve_points: int = 0          # >0: diagnostic learning curve. Evaluate the full
                                   # posterior split on the TEST subregions at this many
                                   # log-spaced epochs and write learning_curve.csv
                                   # (used by run_learning_curves.py). 0 = off; it costs
                                   # one extra forward pass + posterior per point and
                                   # never feeds back into training or checkpointing.
    run_dir:     Optional[Path] = None  # auto-computed from config if None

    def auto_run_dir(self) -> Path:
        # Layout: <region>_<target>_<region_m>_<subregion_m>[_mask..][_norm..][_sim..]/<loss>/
        # e.g. berlin_population_2000_500  = 2000m region, 500m subregion.
        # Method is always in the tag (poisson included) so all methods are consistent
        # and never collide: <region>_<target>_<R>_<sub>_<method>.
        tag = f"{self.region}_{self.target}_{self.region_m}_{self.subregion_m}_{self.method}"
        if self.mask_p > 0:
            tag += f"_mask{int(self.mask_p * 100)}"
        if self.normalize != "none":
            tag += f"_{self.normalize}"
        if self.sim_anchors:
            tag += f"_anchor{len(self.sim_anchors)}"
        elif self.sim_dim > 0:
            tag += f"_sim{self.sim_dim}"
        return EXPERIMENTS_DIR / tag / self.loss


# ---------------------------------------------------------------------------
# Internal helpers

class _Tee:
    def __init__(self, path: Path):
        self._file   = open(path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s: str):
        try:
            self._stdout.write(s)
        except UnicodeEncodeError:
            self._stdout.write(s.encode("ascii", "replace").decode("ascii"))
        self._file.write(s)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def _block_stats(mu, sig2, tau2, bid_t, n_blks):
    """Scatter-add per-cell (mu, sig2) into block-level (M, V)."""
    M = torch.zeros(n_blks, device=mu.device).scatter_add(0, bid_t, mu)
    V = torch.zeros(n_blks, device=mu.device).scatter_add(0, bid_t, sig2)
    return M, (V + tau2).clamp(min=1e-6)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else float("nan")


def _nll(y: np.ndarray, mu: np.ndarray, sig2: np.ndarray) -> float:
    """Gaussian NLL (nats), ignoring non-finite entries."""
    mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sig2) & (sig2 > 0)
    s    = sig2[mask].clip(min=1e-6)
    return float(0.5 * np.mean((y[mask] - mu[mask]) ** 2 / s + np.log(s)))


def _sep(char: str = "-", width: int = 60) -> str:
    return char * width


def _lgamma(x: torch.Tensor) -> torch.Tensor:
    """log Gamma via a shifted Stirling series (uses only log/mul).

    torch.lgamma on CUDA goes through the "jiterator", which runtime-compiles a kernel
    with nvrtc; on some cluster nodes that fails ("nvrtc: invalid value for
    --gpu-architecture"). This avoids it entirely. Shift by 6 so the Stirling argument
    is >=6 (accurate to ~1e-8), then subtract the log terms. Autograd-friendly.
    """
    x = x.clamp(min=1e-6)
    shift = 6
    z = x + shift
    lg = ((z - 0.5) * torch.log(z) - z + 0.9189385332046727        # 0.5*log(2*pi)
          + 1.0 / (12.0 * z) - 1.0 / (360.0 * z ** 3) + 1.0 / (1260.0 * z ** 5))
    corr = torch.zeros_like(x)
    for k in range(shift):
        corr = corr + torch.log(x + k)
    return lg - corr


# ---------------------------------------------------------------------------

def run(cfg: PuffinConfig) -> dict:
    """Train and evaluate one Puffin experiment.

    Returns a dict with the predictions and the metrics of this run.
    Also writes run.out, training_history.csv, results.json, model.pt to run_dir.
    """
    t0       = time.time()
    is_count = cfg.target in COUNT_TARGETS
    # Count targets pick a posterior family via cfg.method:
    #   "poisson"  -> Poisson aggregate NLL + multinomial allocation.
    #   "gaussian" -> z-scored-target Gaussian aggregate NLL + variance-weighted split.
    #   "gamma"    -> moment-matched Gamma aggregate NLL + variance-weighted split.
    # Rate targets ignore cfg.method and keep the Gaussian weighted-mean Kalman update.
    # loss="mse" still forces the Gaussian-sum ablation path.
    count_method = cfg.method if is_count else "rate"
    use_poisson  = is_count and count_method == "poisson" and cfg.loss == "nll"
    use_gaussian = is_count and count_method == "gaussian" and cfg.loss == "nll"
    use_gamma    = is_count and count_method == "gamma"    and cfg.loss == "nll"
    if is_count and count_method not in ("poisson", "gaussian", "gamma"):
        raise ValueError(f"unknown method '{cfg.method}' (use poisson, gaussian or gamma)")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    run_dir = cfg.run_dir if cfg.run_dir is not None else cfg.auto_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(run_dir / "run.out")
    sys.stdout = tee

    print(_sep("="))
    print(f"  Puffin  |  {cfg.region}  |  {cfg.target}  |  "
          f"region={cfg.region_m}m  subregion={cfg.subregion_m}m")
    print(f"  device={DEVICE}  epochs={cfg.epochs}  lr={cfg.lr}  loss={cfg.loss}")
    print(f"  tau2={cfg.tau2_train}")
    print(f"  mask_p={cfg.mask_p}  normalize={cfg.normalize}  seed={cfg.seed}")
    print(f"  val_fold={cfg.val_fold}  test_fold={cfg.test_fold}")
    print(_sep("="))

    # Data
    cells_path = DATA_PROCESSED / cfg.region / "cells.parquet"
    if not cells_path.exists():
        print(f"\nERROR: {cells_path} not found.")
        print(f"  Run first:  python setup_germany.py / setup_us_voting.py  (region={cfg.region})")
        tee.close()
        raise FileNotFoundError(cells_path)

    print(f"\nData: loading {cells_path} ...")
    cells = pd.read_parquet(cells_path)
    print(f"  {len(cells):,} fine cells")
    if len(cells) == 0:
        raise ValueError(f"{cells_path} has 0 rows -- rebuild it (setup_us_pop.py / setup_us_voting.py).")
    if cfg.target not in cells.columns:
        raise KeyError(f"target '{cfg.target}' not in {cells_path} (has {list(cells.columns[:8])}...). "
                       f"Rebuild with that target enabled in setup_us_pop.py.")
    if cfg.fixed_region and cfg.region_col not in cells.columns:
        # Silent fall-through to GRID mode on US (EPSG:5070) data matches zero blocks and
        # empties `cells`, later crashing cryptically. Fail loudly instead.
        raise KeyError(f"fixed_region requested but region_col '{cfg.region_col}' not in "
                       f"{cells_path} (has {list(cells.columns[:8])}...). Rebuild the dataset.")

    if cfg.fixed_region and cfg.region_col in cells.columns:
        # ---- FIXED region mode (e.g. us_voting): cfg.region_col (county/state) is
        # the coarse unit with the observed total; each row is a subregion.
        # region_m/subregion_m are ignored (regions come from the data, not the grid).
        reg                 = cells[cfg.region_col].astype(str).values
        blk_ids, cell_blk_idx = np.unique(reg, return_inverse=True)
        n_blks              = len(blk_ids)
        tgt                 = cells[cfg.target].values.astype(np.float64)
        Y_blk               = np.bincount(cell_blk_idx, tgt, minlength=n_blks).astype(np.float32)
        N_blk               = np.bincount(cell_blk_idx, minlength=n_blks).astype(np.int32)
        bx = np.bincount(cell_blk_idx, cells["x_mp"].values.astype(float), minlength=n_blks) / np.clip(N_blk, 1, None)
        by = np.bincount(cell_blk_idx, cells["y_mp"].values.astype(float), minlength=n_blks) / np.clip(N_blk, 1, None)
        blocks = pd.DataFrame({
            "block_id": blk_ids, "n_cells": N_blk, f"Y_{cfg.target}": Y_blk,
            "block_x": bx.astype(np.int64), "block_y": by.astype(np.int64),
            "tau2_B": np.maximum(1.0, 0.01 * Y_blk),
        })
        print(f"  FIXED region mode: {len(cells):,} subregions in {n_blks:,} regions "
              f"(by {cfg.region_col})  target={cfg.target}")
    else:
        # ---- GRID mode (population): region/subregion from the 100m grid.
        # Aggregate 100m cells -> subregions; region has (region_m/subregion_m)^2 of them.
        if cfg.subregion_m > 100:
            if cfg.region_m % cfg.subregion_m != 0:
                raise ValueError(f"region_m ({cfg.region_m}) must be a multiple of subregion_m ({cfg.subregion_m})")
            cells = aggregate_to_subregions(cells, cfg.subregion_m)
            n_side = cfg.region_m // cfg.subregion_m
            print(f"  region={cfg.region_m}m  subregion={cfg.subregion_m}m  "
                  f"-> {n_side}x{n_side} subregions/region")

        blocks       = build_coarse_blocks(cells, region_m=cfg.region_m, target=cfg.target)
        n_blks       = len(blocks)
        Y_blk        = blocks[f"Y_{cfg.target}"].values.astype(np.float32)
        N_blk        = blocks["n_cells"].values.astype(np.int32)

        block_size   = cfg.region_m
        cell_bx      = (cells["x_mp"].values // block_size) * block_size
        cell_by      = (cells["y_mp"].values // block_size) * block_size
        cell_bid     = np.char.add(np.char.add(cell_bx.astype(np.int64).astype(str), "_"),
                                   cell_by.astype(np.int64).astype(str))   # np.char.add: works on numpy 1.x + 2.x
        bid_to_idx   = {bid: i for i, bid in enumerate(blocks["block_id"])}
        cell_blk_idx = np.array([bid_to_idx.get(bid, -1) for bid in cell_bid])

        ok           = cell_blk_idx >= 0
        cells        = cells[ok].reset_index(drop=True)
        cell_blk_idx = cell_blk_idx[ok]

    splits = make_spatial_splits(blocks, n_folds=5, seed=cfg.split_seed,
                                 super_block_size=cfg.super_block_m, scheme=f"{cfg.region}_{cfg.region_col if cfg.fixed_region else 'grid'}_5fold")
    fold_of_block = {}
    for fold_id, bid_list in splits.items():
        for bid in bid_list:
            fold_of_block[bid] = fold_id
    folds      = np.array([fold_of_block[bid] for bid in blocks["block_id"]])
    train_mask = (folds != cfg.val_fold) & (folds != cfg.test_fold)
    val_mask   = folds == cfg.val_fold
    test_mask  = folds == cfg.test_fold

    print(f"  {n_blks:,} blocks  "
          f"(train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()})")
    print(_sep())

    # Tensors
    X_raw     = torch.tensor(cells[AE_COLS].values, dtype=torch.float32, device=DEVICE)
    if cfg.normalize == "R1":
        X_raw = X_raw / X_raw.norm(dim=1, keepdim=True).clamp(min=1e-9)

    bid_t     = torch.tensor(cell_blk_idx, dtype=torch.long,    device=DEVICE)
    y_blk_t   = torch.tensor(Y_blk,        dtype=torch.float32, device=DEVICE)
    tr_t      = torch.tensor(train_mask,    dtype=torch.bool,    device=DEVICE)
    val_t     = torch.tensor(val_mask,      dtype=torch.bool,    device=DEVICE)
    tau2_t    = float(cfg.tau2_train)
    ae_norm_t = X_raw.norm(dim=1).detach()   # always from original AEF (dasymetric baseline)

    # Poisson exposure (area). rate_i = exposure_i * softplus(f(AE_i)); the net predicts a
    # DENSITY, area scales it to a count. Constant (=1) for a uniform grid, so this leaves
    # Germany unchanged; essential for irregular units (US blocks) where area varies.
    if cfg.exposure_col and cfg.exposure_col in cells.columns:
        exposure_np = np.clip(cells[cfg.exposure_col].values.astype(np.float32), 1e-6, None)
        # Winsorize the extreme upper tail. US block areas span ~6 orders of magnitude
        # (max ~600k cell-units); those giant blocks scale the Poisson gradient by ~exposure
        # and destabilize full-scale training (rate = exposure*softplus). Capping the top
        # quantile tames them; giant blocks are near-empty so their count is ~unaffected.
        # Constant exposure (=1, Germany grid) is unchanged. 0 disables.
        if cfg.exposure_winsor_q and cfg.exposure_winsor_q < 1.0:
            cap = float(np.quantile(exposure_np, cfg.exposure_winsor_q))
            n_cap = int((exposure_np > cap).sum())
            exposure_np = np.minimum(exposure_np, cap)
            print(f"  exposure winsorized at q={cfg.exposure_winsor_q}: cap={cap:.1f} "
                  f"({n_cap:,} blocks capped)")
        print(f"  Poisson exposure: column '{cfg.exposure_col}'  "
              f"(min {exposure_np.min():.2f}  median {np.median(exposure_np):.2f}  max {exposure_np.max():.1f})")
    else:
        exposure_np = np.ones(len(cells), dtype=np.float32)
    exp_t = torch.tensor(exposure_np, dtype=torch.float32, device=DEVICE)

    # Cell-level train mask - needed here for anchor selection
    train_cell_mask = np.isin(cell_blk_idx, np.where(train_mask)[0])

    # Gaussian method: z-score by the mean DENSITY (count per exposure unit) so the
    # network's per-cell mean mu starts at O(1) and M_B = Σ exp·mu matches Y_B/g_scale
    # at init. Scaling by the block-total std instead leaves mu ~100x too large and the
    # aggregate NLL never recovers. Predictions are scaled back to counts before the
    # posterior. Unused by poisson/gamma (g_scale=1).
    if use_gaussian:
        _exp_tr = float(max(exposure_np[train_cell_mask].sum(), 1e-6))
        g_scale = float(max(Y_blk[train_mask].sum() / _exp_tr, 1e-6))
    else:
        g_scale = 1.0
    y_blk_z_t = y_blk_t / g_scale

    # --- Cosine-similarity ablation ------------------------------------------
    # Replace 64-dim AEF input with K-dim cosine similarities to K anchor embeddings.
    # Anchors are either semantic locations (sim_anchors) or K-means fallback (sim_dim).
    # ae_norm_t stays from original AEF so the dasymetric baseline is unchanged.
    effective_k = len(cfg.sim_anchors) if cfg.sim_anchors else cfg.sim_dim
    if effective_k > 0:
        if cfg.sim_anchors:
            from pyproj import Transformer
            tf = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
            xs_cells = cells["x_mp"].values
            ys_cells = cells["y_mp"].values
            anchor_embs = []
            for lat, lon in cfg.sim_anchors:
                xe, ye = tf.transform(lon, lat)
                dists   = np.sqrt((xs_cells - xe) ** 2 + (ys_cells - ye) ** 2)
                nearest = int(np.argmin(dists))
                anchor_embs.append(X_raw[nearest].cpu().numpy())
                print(f"  anchor ({lat:.4f}N, {lon:.4f}E) -> cell {nearest}  "
                      f"dist={dists[nearest]:.0f}m")
            centers = torch.tensor(np.array(anchor_embs), dtype=torch.float32, device=DEVICE)
            print(f"  sim_anchors={effective_k}: coordinate-based semantic anchors")
        else:
            from sklearn.cluster import KMeans
            X_train_np = X_raw[train_cell_mask].cpu().numpy()
            km = KMeans(n_clusters=cfg.sim_dim, random_state=cfg.seed, n_init="auto")
            km.fit(X_train_np)
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE)
            print(f"  sim_dim={cfg.sim_dim}: K-means anchors on {train_cell_mask.sum():,} train cells")

        centers = centers / centers.norm(dim=1, keepdim=True).clamp(min=1e-9)
        np.save(run_dir / "sim_anchors.npy", centers.cpu().numpy())
        X_norm  = X_raw / X_raw.norm(dim=1, keepdim=True).clamp(min=1e-9)
        X_raw   = X_norm @ centers.T          # (N, effective_k) cosine similarities
        print(f"  input: 64-dim AEF -> {effective_k}-dim cosine sim")
        print(_sep())

    # --- Per-dim standardization of the NETWORK input -----------------------
    # Improves first-layer conditioning (AEF dims have means up to +/-65 and 8x
    # scale spread). Stats from TRAIN cells only (leak-free: feature statistics
    # are label-independent) and saved so that a later run applies the
    # SAME stats at inference. The heuristic baselines below keep the ORIGINAL
    # features (X_orig), so their definitions are unchanged.
    X_orig = X_raw
    feat_mean = feat_std = None
    if cfg.standardize:
        tcm_t     = torch.tensor(train_cell_mask, dtype=torch.bool, device=DEVICE)
        feat_mean = X_raw[tcm_t].mean(dim=0)
        feat_std  = X_raw[tcm_t].std(dim=0).clamp(min=1e-6)
        X_raw     = (X_raw - feat_mean) / feat_std
        np.savez(run_dir / "feat_stats.npz",
                 mean=feat_mean.detach().cpu().numpy(),
                 std=feat_std.detach().cpu().numpy())

    in_dim = effective_k if effective_k > 0 else len(AE_COLS)
    model  = MLPHead(in_dim=in_dim).to(DEVICE)
    opt      = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched    = (optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
                if cfg.lr_schedule == "cosine" else None)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MLPHead  params={n_params:,}  loss={cfg.loss}  method={count_method}  "
          f"mask_p={cfg.mask_p}  normalize={cfg.normalize}")
    print(_sep())

    def _scatter(z):
        return torch.zeros(n_blks, device=z.device).scatter_add(0, bid_t, z)

    def _agg_loss(mu, sig2, mask):
        """Method-aware aggregate NLL over `mask` blocks (train or val)."""
        if use_poisson:
            rate = exp_t * (F.softplus(mu) + 1e-6)
            M    = _scatter(rate).clamp(min=1e-6)
            return (M - y_blk_t * torch.log(M))[mask].mean()
        if use_gaussian:                         # z-scored Gaussian aggregate NLL
            M = _scatter(exp_t * mu)             # signed mean, in z-units (target /g_scale)
            V = _scatter(exp_t * sig2).clamp(min=1e-6)   # variance density x area (adds over area)
            return (0.5 * ((y_blk_z_t - M) ** 2 / V + torch.log(V)))[mask].mean()
        if use_gamma:                            # moment-matched Gamma aggregate NLL
            M  = _scatter(exp_t * (F.softplus(mu) + 1e-6)).clamp(min=1e-6)   # Σ mean
            S2 = _scatter(sig2).clamp(min=1e-6)                              # Σ var
            K  = (M * M / S2).clamp(min=1e-6)     # shape
            R  = (M / S2).clamp(min=1e-9)         # rate
            Yc = y_blk_t.clamp(min=1e-6)
            nll = R * Yc - (K - 1.0) * torch.log(Yc) - K * torch.log(R) + _lgamma(K)
            return nll[mask].mean()
        # rate targets (Gaussian weighted-mean) or MSE ablation
        M, V = _block_stats(mu, sig2, tau2_t, bid_t, n_blks)
        if cfg.loss == "nll":
            return (0.5 * ((y_blk_t - M) ** 2 / V + torch.log(V)))[mask].mean()
        return ((y_blk_t - M) ** 2)[mask].mean()

    def _pred_M(mu, sig2):
        """Predicted per-block total in COUNT units (for R^2 reporting)."""
        if use_poisson:
            return _scatter(exp_t * (F.softplus(mu) + 1e-6))
        if use_gaussian:
            return g_scale * _scatter(exp_t * mu)
        if use_gamma:
            return _scatter(exp_t * (F.softplus(mu) + 1e-6))
        M, _ = _block_stats(mu, sig2, tau2_t, bid_t, n_blks)
        return M

    # --- Block-wise minibatch index (country scale) -------------------------
    # Weak supervision needs WHOLE blocks per step, so we sample block indices and
    # gather their cells (never random cells). CSR-style layout: tc_sorted holds
    # train-cell indices grouped by block, ptr marks each block's span.
    use_minibatch = cfg.batch_blocks > 0 and (use_poisson or use_gaussian or use_gamma)
    if cfg.batch_blocks > 0 and not (use_poisson or use_gaussian or use_gamma):
        print("  [WARN] batch_blocks ignored (minibatch supports count methods only)")
    if use_minibatch:
        _tc          = np.where(train_cell_mask)[0]
        _order       = np.argsort(cell_blk_idx[_tc], kind="stable")
        tc_sorted    = _tc[_order]
        uniq_blocks, _first = np.unique(cell_blk_idx[tc_sorted], return_index=True)
        ptr          = np.append(_first, len(tc_sorted)).astype(np.int64)
        n_tr_blocks  = len(uniq_blocks)
        mb_rng       = np.random.default_rng(cfg.seed)
        print(f"  block-minibatch: {cfg.batch_blocks} blocks/step over "
              f"{n_tr_blocks:,} train blocks "
              f"(~{int(np.diff(ptr).mean() * cfg.batch_blocks):,} cells/step)  "
              f"eval_every={cfg.eval_every}")
        print(_sep())

    def _posterior(mu_raw, sig2_raw):
        """Head output -> (prior mean, prior variance, posterior mean, posterior variance).

        Shared by the final evaluation and the `curve_points` diagnostic, so the
        learning curve and the reported numbers are produced by the same code.
        """
        if use_poisson:
            # COUNT: multinomial allocation of Y_B by predicted Poisson rate (area * density).
            # Exact conservation, always non-negative; no tau2 knob.
            rate_np = exposure_np * (F.softplus(mu_raw) + 1e-6).cpu().numpy().flatten()
            mu_np   = rate_np          # prior per-cell expected count
            sig2_np = rate_np          # Poisson prior variance (var = mean)
            mu_post, sig2_post = poisson_allocation(rate_np, Y_blk, cell_blk_idx, n_blks)
        elif use_gaussian:
            # z-scored Gaussian: the head predicts a per-unit-area mean and variance DENSITY, so
            # both scale linearly with the subregion area (variance adds over area, as in Poisson).
            # prior mean (counts) = g_scale · area · μ ; prior var = g_scale² · area · σ².
            # The area factor on the VARIANCE is what lets a large irregular block absorb a
            # correspondingly larger share of the region's correction; area_variance=False
            # drops it, which is the ablation reported in the paper. (area == 1 on a uniform
            # grid, so Germany is unaffected either way.)
            # Variance-weighted split + clip + rescale.
            mu_z    = mu_raw.cpu().numpy().flatten()
            sig2_z  = sig2_raw.cpu().numpy().flatten()
            mu_np   = g_scale * exposure_np * mu_z
            sig2_np = (g_scale ** 2) * sig2_z
            if cfg.area_variance:
                sig2_np = sig2_np * exposure_np
            mu_post, sig2_post = variance_weighted_posterior(mu_np, sig2_np, Y_blk, cell_blk_idx, n_blks)
        elif use_gamma:
            # Gamma: positive prior mean = area · softplus(μ); independent variance = head σ².
            # Variance-weighted split + clip + rescale (linear approximation to the Gamma conditional).
            mu_np   = exposure_np * (F.softplus(mu_raw) + 1e-6).cpu().numpy().flatten()
            sig2_np = sig2_raw.cpu().numpy().flatten()
            mu_post, sig2_post = variance_weighted_posterior(mu_np, sig2_np, Y_blk, cell_blk_idx, n_blks)
        else:
            # RATE (Gaussian weighted-mean Kalman) or MSE ablation (Gaussian sum).
            mu_np   = mu_raw.cpu().numpy().flatten()
            sig2_np = sig2_raw.cpu().numpy().flatten()
            # MSE training gives no gradient to sigma², so it stays near random init;
            # use constant sigma²=1 (with tau2_B=0 -> uniform correction per cell).
            if cfg.loss == "mse":
                sig2_np = np.ones_like(mu_np)
            weights = None
            if not is_count:
                weights = cells["population"].fillna(1).clip(lower=1).values.astype(np.float32)
            mu_post, sig2_post = gaussian_posterior(
                mu_prior  = mu_np,
                var_prior = sig2_np,
                y_coarse  = Y_blk,
                block_ids = cell_blk_idx,
                n_blocks  = n_blks,
                tau2_B    = 0.0,
                weights   = weights,
            )

        return mu_np, sig2_np, mu_post, sig2_post

    # Training loop
    best_val       = float("inf")
    best_state     = None
    best_epoch     = 0
    epochs_no_imp  = 0
    loss_label     = "NLL" if cfg.loss == "nll" else "MSE"

    # One-off reuse: if PUFFIN_REUSE_MODEL is set and this run already has a saved
    # model.pt, load it and skip training entirely (inference only). Lets us regenerate
    # an aggregation JSON from already-trained seeds without a 2h retrain. Deterministic:
    # standardization/splits rebuild identically from split_seed, so the OOF matches.
    import os as _os
    _reuse = bool(_os.environ.get("PUFFIN_REUSE_MODEL")) and (run_dir / "model.pt").exists()
    if _reuse:
        best_state = torch.load(run_dir / "model.pt", map_location=DEVICE)
        best_epoch = -1
        print("  [PUFFIN_REUSE_MODEL] loaded model.pt -> skipping training", flush=True)

    # Diagnostic learning curve: the aggregate loss the model is trained on against the
    # fine-scale error its posterior split makes on the TEST subregions, at log-spaced
    # epochs. Read-only -- it never touches the checkpoint or the gradient, so a run with
    # curve_points>0 produces the same model as one without.
    curve_rows, curve_at = [], set()
    if cfg.curve_points > 0 and not _reuse:
        Y_fine_curve = cells[cfg.target].values.astype(np.float32)
        test_cells_c = np.isin(cell_blk_idx, np.where(test_mask)[0])
        blk_sum_c    = np.bincount(cell_blk_idx, Y_fine_curve, minlength=n_blks)
        blk_mean_c   = blk_sum_c / N_blk.clip(min=1)
        y_within_c   = (Y_fine_curve - blk_mean_c[cell_blk_idx])[test_cells_c]
        curve_at     = set(np.unique(np.geomspace(1, cfg.epochs, cfg.curve_points)
                                     .round().astype(int)).tolist())

    print(f"{'Epoch':>7}  {f'train_{loss_label}':>12}  {f'val_{loss_label}':>11}  "
          f"{'train_R2':>9}  {'val_R2':>7}  {'test_R2':>7}")
    print(_sep())

    for epoch in range(1, (0 if _reuse else cfg.epochs) + 1):
        model.train()

        if use_minibatch:
            # Sample whole TRAIN blocks; forward only their cells (bounded memory).
            sel  = mb_rng.choice(n_tr_blocks, size=min(cfg.batch_blocks, n_tr_blocks),
                                 replace=False)
            cidx = np.concatenate([tc_sorted[ptr[k]:ptr[k + 1]] for k in sel])
            lbid = np.concatenate([np.full(ptr[k + 1] - ptr[k], j, np.int64)
                                   for j, k in enumerate(sel)])
            cidx_t = torch.as_tensor(cidx, dtype=torch.long, device=DEVICE)
            lbid_t = torch.as_tensor(lbid, dtype=torch.long, device=DEVICE)
            K      = len(sel)
            Xb = X_raw.index_select(0, cidx_t)
            if cfg.mask_p > 0:
                Xb = Xb.masked_fill(torch.rand_like(Xb) < cfg.mask_p, 0.0)
            mu_b, sig2_b = model(Xb)
            exp_b = exp_t.index_select(0, cidx_t)
            Yb = y_blk_t.index_select(0, torch.as_tensor(uniq_blocks[sel], dtype=torch.long, device=DEVICE))
            if use_poisson:
                rate_b = exp_b * (F.softplus(mu_b) + 1e-6)
                Mb = torch.zeros(K, device=DEVICE).scatter_add(0, lbid_t, rate_b).clamp(min=1e-6)
                loss = (Mb - Yb * torch.log(Mb)).mean()
            elif use_gaussian:                       # z-scored Gaussian aggregate NLL
                Mb = torch.zeros(K, device=DEVICE).scatter_add(0, lbid_t, exp_b * mu_b)
                Vb = torch.zeros(K, device=DEVICE).scatter_add(0, lbid_t, exp_b * sig2_b).clamp(min=1e-6)  # var density x area
                loss = (0.5 * ((Yb / g_scale - Mb) ** 2 / Vb + torch.log(Vb))).mean()
            else:                                    # use_gamma: moment-matched Gamma NLL
                Mb  = torch.zeros(K, device=DEVICE).scatter_add(0, lbid_t, exp_b * (F.softplus(mu_b) + 1e-6)).clamp(min=1e-6)
                S2b = torch.zeros(K, device=DEVICE).scatter_add(0, lbid_t, sig2_b).clamp(min=1e-6)
                Kk = (Mb * Mb / S2b).clamp(min=1e-6)
                Rk = (Mb / S2b).clamp(min=1e-9)
                Yc = Yb.clamp(min=1e-6)
                loss = (Rk * Yc - (Kk - 1.0) * torch.log(Yc) - Kk * torch.log(Rk) + _lgamma(Kk)).mean()
        else:
            if cfg.mask_p > 0:
                X_in = X_raw.masked_fill(torch.rand_like(X_raw) < cfg.mask_p, 0.0)
            else:
                X_in = X_raw

            mu, sig2 = model(X_in)
            loss = _agg_loss(mu, sig2, tr_t)

        opt.zero_grad(); loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()

        if epoch in curve_at:
            model.eval()
            with torch.no_grad():
                _mu_c, _s2_c = model(X_raw)
            _, _, mu_post_c, _ = _posterior(_mu_c, _s2_c)
            err = mu_post_c[test_cells_c] - Y_fine_curve[test_cells_c]
            curve_rows.append({
                "epoch":          epoch,
                "train_nll":      float(loss.item()),
                "post_mae":       float(np.abs(err).mean()),
                "post_within_r2": _r2(y_within_c,
                                      mu_post_c[test_cells_c] - blk_mean_c[cell_blk_idx][test_cells_c]),
            })
            model.train()

        # Val eval + checkpoint (gated by eval_every; last epoch always evaluated)
        if not ((epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)):
            continue

        model.eval()
        with torch.no_grad():
            mu_e, sig2_e = model(X_raw)
            val_metric = float(_agg_loss(mu_e, sig2_e, val_t))
            M_e = _pred_M(mu_e, sig2_e)

        M_np = M_e.cpu().numpy()

        tr_r2  = _r2(Y_blk[train_mask], M_np[train_mask])
        val_r2 = _r2(Y_blk[val_mask],   M_np[val_mask])
        te_r2  = _r2(Y_blk[test_mask],  M_np[test_mask])

        if val_metric < best_val:
            best_val      = val_metric
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += cfg.eval_every

        print(f"{epoch:>7}  {float(loss.item()):>12.4f}  {val_metric:>11.4f}  "
              f"{tr_r2:>9.3f}  {val_r2:>7.3f}  {te_r2:>7.3f}")

        # With cosine, run the full schedule (early-stopping mid-anneal would leave
        # LR high and land short of the optimum). Patience only applies to "none".
        if sched is None and epochs_no_imp >= cfg.patience:
            print(f"\n  Early stop at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
            break


    # Posterior with best model (exact block conservation)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mu_raw, sig2_raw = model(X_raw)
    mu_np, sig2_np, mu_post, sig2_post = _posterior(mu_raw, sig2_raw)

    Y_fine         = cells[cfg.target].values.astype(np.float32)
    test_cell_mask = np.isin(cell_blk_idx, np.where(test_mask)[0])
    # train_cell_mask already computed above (used for anchor selection)

    # Baselines
    bc_fine      = Y_blk[cell_blk_idx] / N_blk[cell_blk_idx].clip(min=1)
    ae_norm_np   = X_orig.norm(dim=1).cpu().numpy()   # baselines use ORIGINAL AEF
    blk_norm_sum = np.zeros(n_blks)
    np.add.at(blk_norm_sum, cell_blk_idx, ae_norm_np)
    dasy_fine = (ae_norm_np / blk_norm_sum[cell_blk_idx].clip(min=1e-9)
                 * Y_blk[cell_blk_idx])

    X_np         = X_orig.cpu().numpy()               # baselines use ORIGINAL AEF
    blk_emb      = np.zeros((n_blks, X_np.shape[1]))
    np.add.at(blk_emb, cell_blk_idx, X_np)
    blk_emb     /= N_blk[:, None].clip(min=1)
    cell_mean    = blk_emb[cell_blk_idx]
    dot          = (X_np * cell_mean).sum(axis=1)
    sim          = (dot / (ae_norm_np.clip(min=1e-9)
                           * np.linalg.norm(cell_mean, axis=1).clip(min=1e-9))).clip(min=0)
    blk_sim_sum  = np.zeros(n_blks)
    np.add.at(blk_sim_sum, cell_blk_idx, sim)
    sim_fine     = sim / blk_sim_sum[cell_blk_idx].clip(min=1e-9) * Y_blk[cell_blk_idx]

    bc_r2    = _r2(Y_fine[test_cell_mask], bc_fine[test_cell_mask])
    dasy_r2  = _r2(Y_fine[test_cell_mask], dasy_fine[test_cell_mask])
    sim_r2   = _r2(Y_fine[test_cell_mask], sim_fine[test_cell_mask])
    prior_r2 = _r2(Y_fine[test_cell_mask], mu_np[test_cell_mask])
    post_r2  = _r2(Y_fine[test_cell_mask], mu_post[test_cell_mask])

    # Train-fold fine R² - Y_fine is held out even for training blocks
    # (model only ever saw block totals Y_blk, never individual cell labels)
    prior_r2_train = _r2(Y_fine[train_cell_mask], mu_np[train_cell_mask])
    post_r2_train  = _r2(Y_fine[train_cell_mask], mu_post[train_cell_mask])

    # Within-block R² - the honest disaggregation metric. Cell-level R² is
    # dominated by the between-region signal that both the region average and Puffin
    # get for free from the observed block total; subtracting each block's mean
    # isolates the part the model actually contributes (the region average scores 0).
    blk_sum_true  = np.zeros(n_blks)
    np.add.at(blk_sum_true, cell_blk_idx, Y_fine)
    blk_mean_true = blk_sum_true / N_blk.clip(min=1)
    y_within      = Y_fine       - blk_mean_true[cell_blk_idx]
    post_within   = mu_post      - blk_mean_true[cell_blk_idx]
    within_post_r2 = _r2(y_within[test_cell_mask], post_within[test_cell_mask])
    # prior within-region (non-conserving: subtract the prior's OWN region mean)
    blk_mean_prior = (np.bincount(cell_blk_idx, mu_np, minlength=n_blks)
                      / N_blk.clip(min=1))
    prior_within   = mu_np - blk_mean_prior[cell_blk_idx]
    within_prior_r2 = _r2(y_within[test_cell_mask], prior_within[test_cell_mask])

    # NLL on test blocks (prior predictions, before exact posterior).
    # Poisson counts: Λ_B rate, var=mean, no tau2. Gaussian/rate: V=Σσ²+tau2.
    M_prior_blk = np.zeros(n_blks)
    V_prior_blk = np.zeros(n_blks)
    np.add.at(M_prior_blk, cell_blk_idx, mu_np)
    np.add.at(V_prior_blk, cell_blk_idx, sig2_np)
    nb_r = None
    if use_poisson:
        # Block totals are heavily overdispersed vs pure Poisson; calibrate the
        # block predictive interval with an NB dispersion + conformal quantile fit
        # on the VAL fold (coarse-only, leak-free). Report NB block NLL + coverage.
        nb_r, blk_cal = block_dispersion_conformal(
            M_prior_blk, Y_blk, folds, cfg.val_fold, cfg.test_fold)
        cov_blk_90    = blk_cal["coverage_test"].get(0.9, float("nan"))
        nll_blk_prior = blk_cal["nll_test"]
    else:
        nll_blk_prior = _nll(Y_blk[test_mask], M_prior_blk[test_mask],
                             (V_prior_blk + cfg.tau2_train)[test_mask])
        V_blk_total   = (V_prior_blk + cfg.tau2_train).clip(min=0)
        z90 = 1.6449
        lo  = M_prior_blk - z90 * np.sqrt(V_blk_total)
        hi  = M_prior_blk + z90 * np.sqrt(V_blk_total)
        cov_blk_90 = float(((Y_blk >= lo) & (Y_blk <= hi))[test_mask].mean())

    # NLL on test cells (posterior predictions, uncalibrated)
    nll_cell_post = _nll(Y_fine[test_cell_mask],
                         mu_post[test_cell_mask], sig2_post[test_cell_mask])

    # Per-cell 90% coverage (calibrated, via region_conformal)
    _, cal = region_conformal(
        y             = Y_fine,
        mu            = mu_post,
        sig2          = sig2_post,
        block_idx     = cell_blk_idx,
        cell_fold     = folds[cell_blk_idx],
        block_x       = blocks["block_x"].values.astype(float),
        block_y       = blocks["block_y"].values.astype(float),
        block_resid2  = (Y_blk - M_prior_blk) ** 2,
        block_predvar = V_prior_blk.clip(min=1e-12),
        val_fold      = cfg.val_fold,
        test_fold     = cfg.test_fold,
    )
    cov_cell_90 = cal.get("coverage_test", {}).get(0.9, float("nan"))

    elapsed = time.time() - t0
    delta   = post_r2 - bc_r2
    sign    = "+" if delta >= 0 else ""

    print(_sep())
    print(_sep("="))
    print("  RESULTS")
    print(_sep("-"))
    if use_poisson:      post_kind, post_tag = "Poisson multinomial allocation", "poisson"
    elif use_gaussian:   post_kind, post_tag = "Gaussian z-scored (variance-weighted split)", "gaussian"
    elif use_gamma:      post_kind, post_tag = "Gamma moment-matched (variance-weighted split)", "gamma"
    elif not is_count:   post_kind, post_tag = "Gaussian weighted-mean Kalman", "gaussian_wmean"
    else:                post_kind, post_tag = "Gaussian sum (MSE ablation)", "gaussian_sum"
    print(f"  best epoch       : {best_epoch} / {cfg.epochs}")
    print(f"  posterior        : {post_kind}")
    print(f"  region average   : subregion-R2 = {bc_r2:.4f}  (test subregions)")
    print(f"  dasymetric       : subregion-R2 = {dasy_r2:.4f}  (test subregions)")
    print(f"  sim-projection   : subregion-R2 = {sim_r2:.4f}  (test subregions)")
    print(f"  Puffin prior     : R2 = {prior_r2:.4f} (test)   {prior_r2_train:.4f} (train)   gap={prior_r2_train - prior_r2:+.4f}")
    print(f"  Puffin posterior : R2 = {post_r2:.4f} (test)   {post_r2_train:.4f} (train)   gap={post_r2_train - post_r2:+.4f}  ({sign}{delta:.4f} vs region average)")
    print(f"  within-region R2 : {within_post_r2:.4f}  (test subregions; the region average scores 0)")
    blk_note = f"NB (r={nb_r:.1f})" if nb_r is not None and np.isfinite(nb_r) else "Gaussian"
    print(f"  NLL region prior : {nll_blk_prior:.4f}  nats (test regions, {blk_note})")
    print(f"  NLL subregion    : {nll_cell_post:.4f}  nats (test subregions, uncalibrated)")
    print(f"  90% region cov   : {cov_blk_90:.3f}  (nominal 0.90, test regions, calibrated)")
    print(f"  90% subregion cov: {cov_cell_90:.3f}  (nominal 0.90, test subregions)")
    print(f"  elapsed          : {elapsed / 60:.1f} min")
    print(_sep("="))
    print(f"\n  Saved -> {run_dir}/\n")

    torch.save(best_state, run_dir / "model.pt")
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(run_dir / "learning_curve.csv", index=False)
        print(f"  learning curve ({len(curve_rows)} points) -> {run_dir}/learning_curve.csv")
    if cfg.save_predictions:
        np.savez(
            run_dir / "predictions.npz",
            mu_prior       = mu_np,
            sig2_prior     = sig2_np,       # stage-1 calibration uses the PRIOR variance
            mu_post        = mu_post,
            sig2_post      = sig2_post,
            cell_blk_idx   = cell_blk_idx,
            Y_fine         = Y_fine,
            Y_blk          = Y_blk,
            test_cell_mask = test_cell_mask,   # per-fold OOF mask -> viz stitches crossfit maps
            folds          = folds,            # fold of each region, for post-hoc calibration
            block_x        = blocks["block_x"].values.astype(float),
            block_y        = blocks["block_y"].values.astype(float),
        )

    results = {
        "run_id":        run_dir.name,
        "region":        cfg.region,
        "year":          cfg.year,
        "target":        cfg.target,
        "region_m":      cfg.region_m,
        "subregion_m":   cfg.subregion_m,
        "loss":          cfg.loss,
        "mask_p":        cfg.mask_p,
        "normalize":     cfg.normalize,
        "sim_dim":       effective_k,
        "sim_anchors":   cfg.sim_anchors if cfg.sim_anchors else None,
        "epochs":        cfg.epochs,
        "best_epoch":    best_epoch,
        "lr":            cfg.lr,
        "lr_schedule":   cfg.lr_schedule,
        "tau2_train":    cfg.tau2_train,
        "seed":          cfg.seed,
        "split_seed":    cfg.split_seed,
        "standardize":   cfg.standardize,
        "super_block_m": cfg.super_block_m,
        "n_cells":       len(cells),
        "n_blocks":      n_blks,
        "n_train":       int(train_mask.sum()),
        "n_val":         int(val_mask.sum()),
        "n_test":        int(test_mask.sum()),
        "bc_r2":         round(bc_r2,         4),
        "dasy_r2":       round(dasy_r2,        4),
        "sim_r2":        round(sim_r2,         4),
        "method":              count_method,
        "posterior":           post_tag,
        "prior_r2":            round(prior_r2,                     4),
        "post_r2":             round(post_r2,                      4),
        "within_post_r2":      round(within_post_r2,               4),
        "within_prior_r2":     round(within_prior_r2,              4),
        "prior_r2_train":      round(prior_r2_train,               4),
        "post_r2_train":       round(post_r2_train,                4),
        "train_test_gap_prior": round(prior_r2_train - prior_r2,   4),
        "train_test_gap_post":  round(post_r2_train  - post_r2,    4),
        "delta_vs_bc":         round(delta,                        4),
        "nll_blk_prior": round(nll_blk_prior,  4),
        "nll_cell_post": round(nll_cell_post,  4),
        "nb_dispersion_r": round(nb_r, 3) if nb_r is not None and np.isfinite(nb_r) else None,
        "cal_blk_90":    round(cov_blk_90,     4),
        "cal_cell_90":   round(cov_cell_90, 4) if not np.isnan(cov_cell_90) else None,
        "elapsed_min":   round(elapsed / 60, 1),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    tee.close()

    return {
        "mu_prior":       mu_np,
        "sig2_prior":     sig2_np,
        "mu_post":        mu_post,
        "sig2_post":      sig2_post,
        "Y_fine":         Y_fine,
        "Y_blk":          Y_blk,
        "test_cell_mask": test_cell_mask,
        "test_mask":      test_mask,
        "cell_blk_idx":   cell_blk_idx,
        # the split and the region geometry, so a caller can stitch several folds
        # together without re-deriving them
        "folds":          folds,
        "N_blk":          N_blk,
        "block_x":        blocks["block_x"].values.astype(float),
        "block_y":        blocks["block_y"].values.astype(float),
        "val_fold":       cfg.val_fold,
        "test_fold":      cfg.test_fold,
        "results":        results,
    }
