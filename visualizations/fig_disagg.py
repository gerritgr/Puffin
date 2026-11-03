# ============================================================================
# FIGURE : fig_disagg  -- Puffin disaggregation map panels
# TYPE   : side-by-side raster maps over a basemap: AEF-PCA / baselines /
#          Puffin prior+posterior / ground truth (pick via MAP_PANELS)
# ANSWERS: what Puffin's fine-cell disaggregation looks like vs the baselines
#          and the ground truth, for one region/subregion size
# INPUT   : experiments/runs/<tag>/nll/seed_<s>/crossfit_test<t>/predictions.npz
#           (the crossfit OOF output of run_berlin_all_crossfit.py). Per seed the 5
#           folds are stitched into one out-of-fold map (each cell from the fold that
#           held it out, via the saved test_cell_mask), then averaged across seeds.
# OUTPUT  : experiments/figures/fig_disagg.png
# ============================================================================
"""Self-contained. Edit CONFIG, then: python fig_disagg.py"""
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
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent.resolve()   # repo root (this script lives in visualizations/)
sys.path.insert(0, str(ROOT / "src"))
from puffin.config import AE_COLS, DATA_PROCESSED, EXPERIMENTS_DIR, FIGURES_DIR
from puffin.data import build_coarse_blocks, aggregate_to_subregions

# ============================ CONFIG ============================
REGION          = "berlin_all"   # crossfit is run on the all-cells build (Berlin bbox)
YEAR            = 2022
TARGET          = "population"
MAP_REGION_M    = 1000       # region (coarse block) size in metres
MAP_SUBREGION_M = 100        # subregion (predicted unit) size in metres (100 = native grid)
# 1000/500  2000/400  2000/1000  4000/400  4000/800  4000/2000 10000/500  10000/1000  10000/2000  10000/5000
# this configs available preds for germany_all
MAP_SEEDS       = [0, 1, 2, 3, 4]   # init seeds to average (single value ok: [0])
MAP_LOSS        = "nll"
MAP_METHOD      = "gaussian"   # which run directory to read: poisson | gaussian | gamma
# Panels (any subset, any order): aef_pca | bc | dasy | sim | prior | posterior | truth
MAP_PANELS      = ["bc", "posterior", "truth"]
MAP_METRIC      = "cell"     # "cell" (raw counts) | "within" (per-cell deviation from its block mean)
MAP_POP         = "all"      # "all" (every cell) | "inhab" (only ground-truth inhabited, pop>0)
# ZOOM_CENTER must be INSIDE the REGION above (else the crop guard shows the full region).
ZOOM_CENTER     = (52.544090842706204, 13.205431903612446)   # Berlin (lat, lon) -- matches berlin_all
ZOOM_RADIUS     = 20000      # metres, half-width of the square crop
CMAP_POP        = "viridis"
CMAP_WITHIN     = "RdBu_r"   # diverging, centred at 0 (used when MAP_METRIC="within")
VMAX_PCT        = 98
ALPHA           = 0.5
BASEMAP         = cx.providers.OpenStreetMap.HOT
CRS             = "EPSG:3035"   # native LAEA grid: rasterises the 100m lattice exactly
                                # (reprojecting to UTM skews the axis-aligned raster -> tilt)
DPI             = 300
OUT             = FIGURES_DIR / "fig_disagg.png"
# ===============================================================

_laea_to_crs  = Transformer.from_crs("EPSG:3035", CRS, always_xy=True)
_wgs84_to_crs = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

_PANEL_DEFS = {
    "aef_pca":   ("AEF embeddings (PCA)", lambda d: (d["aef_pca_rgb"], "rgb")),
    "bc":        ("Region average",       lambda d: (d["bc"],       "pop")),
    "dasy":      ("Dasymetric",           lambda d: (d["dasy"],     "pop")),
    "sim":       ("Sim-projection",       lambda d: (d["sim"],      "pop")),
    "prior":     ("Puffin prior",         lambda d: (d["mu_prior"], "pop")),
    "posterior": ("Puffin estimate",     lambda d: (d["mu_post"],  "pop")),
    "truth":     ("Ground truth",         lambda d: (d["Y_fine"],   "pop")),
}

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


def _cells_to_grid_rgb(x, y, rgb, cell_size=100):
    x0, y0 = x.min(), y.min()
    w = int(round((x.max() - x0) / cell_size)) + 1
    h = int(round((y.max() - y0) / cell_size)) + 1
    img = np.full((h, w, 3), np.nan, dtype=np.float32)
    img[((y - y0) / cell_size).round().astype(int), ((x - x0) / cell_size).round().astype(int)] = rgb.astype(np.float32)
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


def _basemap(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    try:
        cx.add_basemap(ax, crs=CRS, source=BASEMAP, zoom="auto", attribution=False, reset_extent=False)
    except Exception as e:
        print(f"[WARN] basemap failed ({e})")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])


def _imshow_rgb(ax, img_rgb, ext, title):
    has_nan = np.isnan(img_rgb).any(axis=2)
    rgba = np.zeros((*img_rgb.shape[:2], 4), dtype=np.float32)
    rgba[..., :3] = np.nan_to_num(img_rgb); rgba[..., 3] = np.where(has_nan, 0.0, ALPHA)
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest", zorder=2)
    _basemap(ax, ext); ax.set_title(title, fontsize=17.5)


def _imshow(ax, img, ext, cmap, vmin, vmax, title):
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", alpha=ALPHA, zorder=2)
    _basemap(ax, ext); ax.set_title(title, fontsize=17.5)
    return im


def _stitch_seed_oof(base, seed):
    """Stitch one seed's 5 crossfit folds into a single OOF vector (each cell filled
    by the fold that held it out). Returns (oof_prior, oof_post, Y_fine, Y_blk, cbi)
    or None if no fold files are present for this seed."""
    oof_prior = oof_post = Y_fine = Y_blk = cbi = None
    for t in range(5):
        _sub = "crossfit_test"
        p = base / f"seed_{seed}" / f"{_sub}{t}" / "predictions.npz"
        if not p.exists():
            continue
        npz = np.load(p)
        if oof_post is None:
            oof_prior = np.full(len(npz["mu_prior"]), np.nan)
            oof_post  = np.full(len(npz["mu_post"]),  np.nan)
            Y_fine = npz["Y_fine"].astype(np.float64); Y_blk = npz["Y_blk"].astype(np.float64)
            cbi = npz["cell_blk_idx"].astype(int)
        tcm = npz["test_cell_mask"]
        oof_prior[tcm] = npz["mu_prior"][tcm].astype(np.float64)
        oof_post[tcm]  = npz["mu_post"][tcm].astype(np.float64)
    return None if oof_post is None else (oof_prior, oof_post, Y_fine, Y_blk, cbi)


def load_map_data():
    seeds = MAP_SEEDS if isinstance(MAP_SEEDS, list) else [MAP_SEEDS]
    pred_tag = f"{REGION}_{TARGET}_{MAP_REGION_M}_{MAP_SUBREGION_M}_{MAP_METHOD}"
    base = EXPERIMENTS_DIR / pred_tag / MAP_LOSS
    all_prior, all_post = [], []
    Y_fine = Y_blk = cell_blk_idx = None
    for seed in seeds:
        got = _stitch_seed_oof(base, seed)
        if got is None:
            print(f"[WARN] no crossfit folds for seed {seed} under {base}/seed_{seed}/crossfit_test*/"); continue
        oof_prior, oof_post, Y_fine, Y_blk, cell_blk_idx = got
        all_prior.append(oof_prior); all_post.append(oof_post)
    if not all_prior:
        print(f"[WARN] no crossfit predictions for {pred_tag} {MAP_LOSS} "
              f"(run run_berlin_all_crossfit.py first)"); return None
    mu_prior = np.nanmean(all_prior, 0); mu_post = np.nanmean(all_post, 0)
    cells_full = pd.read_parquet(DATA_PROCESSED / REGION / "cells.parquet")
    # The crossfit predicts SUBREGIONS: when subregion > 100m the runner mean-pools
    # the 100m cells first, so predictions (mu_post, Y_fine, cell_blk_idx) are per
    # subregion. Mirror that here so x/y/PCA align 1:1 with the prediction arrays.
    if MAP_SUBREGION_M > 100:
        cells_full = aggregate_to_subregions(cells_full, MAP_SUBREGION_M)
    blocks = build_coarse_blocks(cells_full, region_m=MAP_REGION_M, target=TARGET)
    N_blk = blocks["n_cells"].values.astype(np.float64)
    bs = MAP_REGION_M
    cbx = np.floor_divide(cells_full["x_mp"].values, bs) * bs
    cby = np.floor_divide(cells_full["y_mp"].values, bs) * bs
    cbid = np.char.add(np.char.add(cbx.astype(np.int64).astype(str), "_"),
                       cby.astype(np.int64).astype(str))   # np.char.add: works on numpy 1.x + 2.x
    b2i = {b: i for i, b in enumerate(blocks["block_id"])}
    ok = np.array([b2i.get(b, -1) for b in cbid]) >= 0
    cells = cells_full[ok].reset_index(drop=True)
    x, y = _laea_to_crs.transform(cells["x_mp"].values.astype(float), cells["y_mp"].values.astype(float))
    bc = Y_blk[cell_blk_idx] / N_blk[cell_blk_idx].clip(min=1)
    X = np.asarray(cells[AE_COLS], dtype=np.float64)
    pca = PCA(n_components=3).fit_transform(X)
    rgb = ((pca - pca.min(0)) / (pca.max(0) - pca.min(0) + 1e-9)).astype(np.float32)
    return dict(x=np.asarray(x), y=np.asarray(y), bc=bc, mu_prior=mu_prior, mu_post=mu_post,
                Y_fine=Y_fine, aef_pca_rgb=rgb, dasy=bc, sim=bc,
                cell_blk_idx=np.asarray(cell_blk_idx, int))


def main():
    d = load_map_data()
    if d is None:
        print("[SKIP] fig_disagg -- no predictions.npz"); return
    panels = []
    for name in MAP_PANELS:
        if name not in _PANEL_DEFS:
            print(f"[WARN] unknown panel {name!r}"); continue
        title, fn = _PANEL_DEFS[name]; vals, key = fn(d)
        panels.append((title, vals, key))
    if not panels:
        return

    # POP mask + METRIC transform applied to every pop-type panel (rgb panel is left
    # as-is: it's the input embedding, not a count).
    cbi = d["cell_blk_idx"]; Yf = np.asarray(d["Y_fine"], float)
    active = (Yf > 0) if MAP_POP == "inhab" else np.isfinite(Yf)
    nb = (int(cbi.max()) + 1) if cbi.size else 1

    def _prep(v):
        vv = np.asarray(v, float).copy()
        if MAP_METRIC == "within":               # subtract each cell's block mean (over active cells)
            m = active & np.isfinite(vv)
            s = np.bincount(cbi[m], vv[m], minlength=nb)
            c = np.bincount(cbi[m], minlength=nb)
            vv = vv - (s / np.clip(c, 1, None))[cbi]
        if MAP_POP == "inhab":                    # hide uninhabited cells (basemap shows through)
            vv = np.where(active, vv, np.nan)
        return vv

    if MAP_METRIC == "within":                    # diverging scale, symmetric about 0
        pv = [np.abs(_prep(vals)) for (_t, vals, key) in panels if key == "pop"]
        V = float(np.nanpercentile(np.concatenate(pv), VMAX_PCT)) if pv else 1.0
        cmap, vmin, vmax = CMAP_WITHIN, -V, V
        cb_label = f"{TARGET} - within-block deviation\n(count/cell vs block mean)"
    else:                                         # sequential 0..vmax on the ground truth
        g = _prep(d["Y_fine"]); g = g[np.isfinite(g)]
        vmax = float(np.percentile(g, VMAX_PCT)) if len(g) else 1.0
        cmap, vmin, cb_label = CMAP_POP, 0.0, f"{TARGET.replace('_',' ').title()}\n(count per cell)"

    # Panels drawn large, as in the published version; see fig_prob for why sizing the
    # canvas to the printed width makes the labels dominate.
    n = len(panels); pw, cw = 4.0, 0.16
    fig = plt.figure(figsize=(12.63, 4.32))
    gs = matplotlib.gridspec.GridSpec(1, n + 1, width_ratios=[pw] * n + [cw],
                                      wspace=0.04, left=0.01, right=0.99, top=0.84, bottom=0.02)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]; cax = fig.add_subplot(gs[0, n])
    ns = len(MAP_SEEDS) if isinstance(MAP_SEEDS, list) else 1
    # at the printed size the suptitle sits on the panel titles unless lifted clear
    fig.suptitle(f"{TARGET.replace('_',' ').title()} disaggregation", fontsize=21, y=0.99)
    cs = MAP_SUBREGION_M   # raster/crop cell size = the predicted unit (subregion)
    zoom = _zoom_centre_crs(); last = None
    for ax, (title, vals, key) in zip(axes, panels):
        if key == "rgb":
            img, ext = _cells_to_grid_rgb(d["x"], d["y"], vals, cell_size=cs)
            if zoom: img, ext = _crop(img, ext, *zoom, ZOOM_RADIUS, cell_size=cs)
            _imshow_rgb(ax, img, ext, title)
        else:
            img, ext = _cells_to_grid(d["x"], d["y"], _prep(vals), cell_size=cs)
            if zoom: img, ext = _crop(img, ext, *zoom, ZOOM_RADIUS, cell_size=cs)
            last = _imshow(ax, img, ext, cmap, vmin, vmax, title)
    if last is not None:
        cb = fig.colorbar(last, cax=cax); cb.set_label(cb_label, fontsize=16); cb.ax.tick_params(labelsize=14)
    else:
        cax.set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
    fig.savefig(Path(OUT).with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")   # vector text/lines, full-res rasters
    plt.close(fig)
    print(f"saved -> {OUT}  (metric={MAP_METRIC}, pop={MAP_POP}, {ns} seeds)")


if __name__ == "__main__":
    main()
