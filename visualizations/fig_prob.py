# ============================================================================
# FIGURE : fig_prob  -- Puffin probabilistic output
# TYPE   : two raster maps over a basemap -- posterior mean | per-cell uncertainty (std)
# ANSWERS: how confident Puffin is per cell (the calibrated uncertainty that WorldPop
#          etc. do not provide), alongside the posterior mean
# INPUT   : experiments/runs/<tag>/nll/seed_<s>/crossfit_test<t>/predictions.npz
#           (crossfit OOF output of run_berlin_all_crossfit.py). Per seed the 5 folds
#           are stitched into one OOF map (via the saved test_cell_mask); across seeds
#           the posterior mean is averaged and the per-cell variance is aleatoric
#           (mean sig2_post) + epistemic (variance of the seed means).
# OUTPUT  : experiments/figures/fig_prob.png
# ============================================================================
"""Self-contained. Edit CONFIG, then: python fig_prob.py"""
from __future__ import annotations

import sys
from pathlib import Path

import contextily as cx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).parent.parent.resolve()   # repo root (this script lives in visualizations/)
sys.path.insert(0, str(ROOT / "src"))
from puffin.config import DATA_PROCESSED, EXPERIMENTS_DIR, FIGURES_DIR
from puffin.data import build_coarse_blocks, aggregate_to_subregions

# ============================ CONFIG ============================
REGION          = "berlin_all"   # crossfit is run on the all-cells build (Berlin bbox)
YEAR            = 2022
TARGET          = "population"
MAP_REGION_M    = 1000       # region (coarse block) size in metres
MAP_SUBREGION_M = 100        # subregion (predicted unit) size in metres (100 = native grid)
MAP_SEEDS       = [0, 1, 2, 3, 4]   # init seeds (single value ok: [0])
MAP_LOSS        = "nll"
MAP_METHOD      = "gaussian"   # which run directory to read: poisson | gaussian | gamma
ZOOM_CENTER     = (52.544090842706204, 13.205431903612446)   # (lat, lon); None = whole city
ZOOM_RADIUS     = 20000      # metres (matches fig_disagg so the two figures show the same view)
CMAP_POP        = "viridis"
CMAP_UNC        = "Blues"
VMAX_PCT        = 98
ALPHA           = 0.5
BASEMAP         = cx.providers.OpenStreetMap.HOT
CRS             = "EPSG:3035"   # native LAEA grid: rasterises the 100m lattice exactly
                                # (reprojecting to UTM skews the axis-aligned raster -> tilt)
DPI             = 300
OUT             = FIGURES_DIR / "fig_prob.png"
# ===============================================================

_laea_to_crs  = Transformer.from_crs("EPSG:3035", CRS, always_xy=True)
_wgs84_to_crs = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


def _zoom_centre_crs():
    if ZOOM_CENTER is None:
        return None
    lat, lon = ZOOM_CENTER
    return _wgs84_to_crs.transform(lon, lat)


def _cells_to_grid(x, y, values, cell_size=100):
    x0, y0 = x.min(), y.min()
    w = int(round((x.max() - x0) / cell_size)) + 1
    h = int(round((y.max() - y0) / cell_size)) + 1
    img = np.full((h, w), np.nan)
    img[((y - y0) / cell_size).round().astype(int), ((x - x0) / cell_size).round().astype(int)] = values
    ext = [x0 - cell_size / 2, x0 + w * cell_size - cell_size / 2,
           y0 - cell_size / 2, y0 + h * cell_size - cell_size / 2]
    return img, ext


def _crop(img, ext, cx_m, cy_m, r, cell_size=100):
    x0, x1, y0, y1 = ext
    cl = max(0, int((cx_m - r - x0) / cell_size)); ch = min(img.shape[1], int((cx_m + r - x0) / cell_size) + 1)
    rl = max(0, int((cy_m - r - y0) / cell_size)); rh = min(img.shape[0], int((cy_m + r - y0) / cell_size) + 1)
    if ch <= cl or rh <= rl:   # ZOOM_CENTER outside the data -> no valid crop; show full region
        print("[WARN] ZOOM_CENTER is outside the data extent -- showing the full region instead")
        return img, ext
    return img[rl:rh, cl:ch], [x0 + cl * cell_size, x0 + ch * cell_size, y0 + rl * cell_size, y0 + rh * cell_size]


def _imshow(ax, img, ext, cmap, vmin, vmax, title):
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", alpha=ALPHA, zorder=2)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    try:
        cx.add_basemap(ax, crs=CRS, source=BASEMAP, zoom="auto", attribution=False, reset_extent=False)
    except Exception as e:
        print(f"[WARN] basemap failed ({e})")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11.6)
    return im


def _stitch_seed_oof(base, seed):
    """Stitch one seed's 5 crossfit folds into OOF (mu_post, sig2_post) vectors.
    Returns (oof_post, oof_sig2, Y_fine, cbi) or None if no folds present."""
    oof_post = oof_sig2 = Y_fine = cbi = None
    for t in range(5):
        _sub = "crossfit_test"
        p = base / f"seed_{seed}" / f"{_sub}{t}" / "predictions.npz"
        if not p.exists():
            continue
        npz = np.load(p)
        if oof_post is None:
            oof_post = np.full(len(npz["mu_post"]),  np.nan)
            oof_sig2 = np.full(len(npz["sig2_post"]), np.nan)
            Y_fine = npz["Y_fine"].astype(np.float64); cbi = npz["cell_blk_idx"].astype(int)
        tcm = npz["test_cell_mask"]
        oof_post[tcm] = npz["mu_post"][tcm].astype(np.float64)
        oof_sig2[tcm] = npz["sig2_post"][tcm].astype(np.float64)
    return None if oof_post is None else (oof_post, oof_sig2, Y_fine, cbi)


def load_map_data():
    seeds = MAP_SEEDS if isinstance(MAP_SEEDS, list) else [MAP_SEEDS]
    pred_tag = f"{REGION}_{TARGET}_{MAP_REGION_M}_{MAP_SUBREGION_M}_{MAP_METHOD}"
    base = EXPERIMENTS_DIR / pred_tag / MAP_LOSS
    all_post, all_sig2 = [], []
    Y_fine = cell_blk_idx = None
    for seed in seeds:
        got = _stitch_seed_oof(base, seed)
        if got is None:
            print(f"[WARN] no crossfit folds for seed {seed} under {base}/seed_{seed}/crossfit_test*/"); continue
        oof_post, oof_sig2, Y_fine, cell_blk_idx = got
        all_post.append(oof_post); all_sig2.append(oof_sig2)
    if not all_post:
        print(f"[WARN] no crossfit predictions for {pred_tag} {MAP_LOSS} "
              f"(run run_berlin_all_crossfit.py first)"); return None
    mu_post = np.nanmean(all_post, 0)
    sig2_post = np.nanmean(all_sig2, 0) + np.nanvar(all_post, 0)   # aleatoric + epistemic
    cells_full = pd.read_parquet(DATA_PROCESSED / REGION / "cells.parquet")
    # predictions are per SUBREGION when subregion > 100m (runner mean-pools first);
    # mirror that so x/y align 1:1 with the prediction arrays.
    if MAP_SUBREGION_M > 100:
        cells_full = aggregate_to_subregions(cells_full, MAP_SUBREGION_M)
    blocks = build_coarse_blocks(cells_full, region_m=MAP_REGION_M, target=TARGET)
    bs = MAP_REGION_M
    cbx = np.floor_divide(cells_full["x_mp"].values, bs) * bs
    cby = np.floor_divide(cells_full["y_mp"].values, bs) * bs
    cbid = np.char.add(np.char.add(cbx.astype(np.int64).astype(str), "_"),
                       cby.astype(np.int64).astype(str))   # np.char.add: works on numpy 1.x + 2.x
    b2i = {b: i for i, b in enumerate(blocks["block_id"])}
    ok = np.array([b2i.get(b, -1) for b in cbid]) >= 0
    cells = cells_full[ok].reset_index(drop=True)
    x, y = _laea_to_crs.transform(cells["x_mp"].values.astype(float), cells["y_mp"].values.astype(float))
    return dict(x=np.asarray(x), y=np.asarray(y), mu_post=mu_post, sig2_post=sig2_post, Y_fine=Y_fine)


def main():
    d = load_map_data()
    if d is None:
        print("[SKIP] fig_prob -- no predictions.npz"); return
    gt = d["Y_fine"][np.isfinite(d["Y_fine"])]
    vmax_pop = float(np.percentile(gt, VMAX_PCT)) if len(gt) else 1.0
    std = np.sqrt(np.clip(d["sig2_post"], 0, None))
    unc = std[np.isfinite(std)]
    vmax_unc = float(np.percentile(unc, VMAX_PCT)) if len(unc) else 1.0
    # Canvas as in the published version: large map panels, labels small relative to them.
    # Sizing the canvas to the printed width instead makes LaTeX scale it up and the text
    # then swamps the maps.
    pw, cw = 3.1, 0.208
    fig = plt.figure(figsize=(8.87, 4.30))
    gs = matplotlib.gridspec.GridSpec(1, 4, width_ratios=[pw, cw, pw, cw],
                                      wspace=0.30, left=0.02, right=0.94, top=0.86, bottom=0.03)
    ax1, cax1, ax2, cax2 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ns = len(MAP_SEEDS) if isinstance(MAP_SEEDS, list) else 1
    _lab = TARGET.replace('_', ' ').title()
    fig.suptitle(f"{_lab}: estimate and calibrated uncertainty", fontsize=13.3, y=0.97)
    cs = MAP_SUBREGION_M   # raster/crop cell size = the predicted unit (subregion)
    zoom = _zoom_centre_crs()
    for ax, cax, vals, cmap, vmax, title, label in [
        (ax1, cax1, d["mu_post"], CMAP_POP, vmax_pop, "Puffin estimate", "People per cell"),
        (ax2, cax2, std,          CMAP_UNC, vmax_unc, "Uncertainty",   "Std. dev. per cell"),
    ]:
        img, ext = _cells_to_grid(d["x"], d["y"], vals, cell_size=cs)
        if zoom: img, ext = _crop(img, ext, *zoom, ZOOM_RADIUS, cell_size=cs)
        im = _imshow(ax, img, ext, cmap, 0, vmax, title)
        cb = fig.colorbar(im, cax=cax); cb.set_label(label, fontsize=10.4); cb.ax.tick_params(labelsize=9.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
    fig.savefig(Path(OUT).with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")   # vector text/lines, full-res rasters
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
