"""US block-level disaggregation choropleth (any block target):
   region-average baseline | Puffin posterior | ground truth.

Self-contained (like fig_disagg): reads the per-fold predictions.npz directly, stitches
each seed's 5-fold out-of-fold map, averages across seeds, joins block polygons from the
Census TABBLOCK20 shapefile by GEOID -- no separate export step.

Edit the CONFIG block (TARGET / LEVEL / view window), then run:
    python visualizations/fig_us_pop_map.py

TARGET  : population | housing | jobs | gun_violence   (uses run dir us_population_<TARGET>_<LEVEL>)
LEVEL   : tract | county            METRIC : value (counts, log) | within (dev. from region mean)
View    : COUNTY=[fips,..] (metro) | STATE="NY" (whole state) | NATIONAL=True (5 km raster)
A national 8M-block choropleth is unrenderable, so NATIONAL uses a coarse raster instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

BASEMAP = cx.providers.OpenStreetMap.HOT
ALPHA   = 0.5          # choropleth opacity so the basemap shows through

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
from puffin.config import AE_RAW_DIR, DATA_PROCESSED, EXPERIMENTS_DIR, FIGURES_DIR
from puffin.datasets.us_population import STATE_FIPS

# ============================ CONFIG (edit these) ============================
TARGET   = __import__("os").environ.get("FIG_TARGET", "population")   # "population" | "housing" | "jobs" | "gun_violence"
LEVEL    = "tract"          # coarse level the run used: "tract" | "county"
SEEDS    = [0, 1, 2, 3, 4]                     # averaged into the map
METRIC   = "value"                           # "value" (counts, log) | "within" (dev. from region mean)

METHOD     = "gaussian"     # which method's run dir to read: poisson | gaussian
CFG        = f"us_population_{TARGET}_{LEVEL}_{METHOD}"          # experiments/runs/<CFG>/nll/seed_*/crossfit_test*
REGION_COL = "tract_fips" if LEVEL == "tract" else "county_fips"
# Title-case display name + colorbar unit per target.
LABEL = {"population": "Population", "housing": "Housing Units",
         "jobs": "Jobs", "gun_violence": "Gun-Violence Incidents"}.get(TARGET, TARGET.title())
UNIT  = {"population": "people", "housing": "housing units",
         "jobs": "jobs", "gun_violence": "incidents"}.get(TARGET, "count")

# --- view window (which blocks to draw) ---
NATIONAL = False              # True  -> whole-US 5 km raster (ignores STATE / COUNTY)
STATE    = None               # None/"" -> use COUNTY;  "PA" | "NY" | "42" -> that whole state
COUNTY   = ["36061", "36047", "36081", "36005", "36085"]   # county FIPS (when STATE is None)
                              #   ["42101"]                                      Philadelphia
                              #   ["36061", "36047", "36081", "36005", "36085"]  NYC 5 boroughs
                              #   ["06037"]                                      Los Angeles

# Human-readable name for the view (keyed by the county-FIPS set); falls back to the FIPS tag.
REGION_NAMES = {
    "36061,36047,36081,36005,36085": "New York City",
    "06037": "Los Angeles",
    "17031": "Chicago",
    "42101": "Philadelphia",
}
REGION_NAME = ""              # optional manual override for the title

PANELS   = [("baseline", "Region average"),
            ("puffin", "Puffin estimate"), ("gt", "Ground truth")]
DPI      = 300
# =============================================================================


def _stitch_predictions():
    """Seed-averaged out-of-fold posterior per block, with ground truth and the region average.
    Returns (puffin, gt, baseline) as arrays aligned with cells.parquet row order."""
    base = EXPERIMENTS_DIR / CFG / "nll"
    oof_seeds, Yf, cbi = [], None, None
    for s in SEEDS:
        sdir = base / f"seed_{s}"
        if not (sdir / "crossfit_test0" / "predictions.npz").exists():
            continue
        oof = None
        for t in range(5):
            d = np.load(sdir / f"crossfit_test{t}" / "predictions.npz")
            if oof is None:
                oof = np.full(len(d["mu_post"]), np.nan)
                if Yf is None:
                    Yf, cbi = d["Y_fine"], d["cell_blk_idx"]
            oof[d["test_cell_mask"]] = d["mu_post"][d["test_cell_mask"]]
        oof_seeds.append(oof)
    if not oof_seeds:
        raise SystemExit(f"no predictions under {base} -- run the tract crossfit with SAVE_PREDS=True")
    puffin = np.nanmean(np.stack(oof_seeds, 0), 0)
    N = np.bincount(cbi); Yblk = np.bincount(cbi, Yf)
    baseline = Yblk[cbi] / np.clip(N[cbi], 1, None)
    print(f"  stitched {len(oof_seeds)} seeds x 5 folds", flush=True)
    return puffin, Yf.astype(np.float64), baseline


def _plot_national(puffin, gt, baseline, res_m: int = 5000) -> None:
    """Whole-US view: bin block predictions into a coarse grid (default 5 km) and imshow.
    A block choropleth is unrenderable at 8M polygons, and block detail is sub-pixel
    nationally anyway. NOTE: at this zoom baseline/Puffin/GT look alike -- disaggregation
    quality is a within-tract property, only visible in a metro zoom (FIG_COUNTY=<fips>)."""
    cells = pd.read_parquet(DATA_PROCESSED / "us_population" / "cells.parquet",
                            columns=["x_mp", "y_mp"])
    x = np.asarray(cells["x_mp"].values, dtype=float); y = np.asarray(cells["y_mp"].values, dtype=float)
    x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    nx = int((x1 - x0) / res_m) + 1; ny = int((y1 - y0) / res_m) + 1
    flat = (np.clip(((y - y0) / res_m).astype(int), 0, ny - 1) * nx +
            np.clip(((x - x0) / res_m).astype(int), 0, nx - 1))
    def grid(v):
        g = np.bincount(flat, weights=np.nan_to_num(v), minlength=nx * ny).reshape(ny, nx)
        g[g <= 0] = np.nan
        return g
    imgs = {c: grid(v) for c, v in (("baseline", baseline), ("puffin", puffin), ("gt", gt))}
    allv = np.concatenate([np.ravel(im) for im in imgs.values()])
    vmin = max(1.0, float(np.nanpercentile(allv, 50))); vmax = float(np.nanpercentile(allv, 99.5))
    norm = LogNorm(vmin=vmin, vmax=vmax); cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, (col, title) in zip(axes, PANELS):
        ax.imshow(imgs[col], origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=15); ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, label=f"{LABEL} per {res_m//1000} km cell (log)")
    fig.suptitle(f"United States: {LABEL} at {res_m//1000}\\,km resolution", fontsize=21, y=0.99)
    out = FIGURES_DIR / f"fig_us_{TARGET}_national_{res_m//1000}km.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)


def main() -> None:
    puffin, gt, baseline = _stitch_predictions()
    if NATIONAL:
        _plot_national(puffin, gt, baseline)
        return
    cells = pd.read_parquet(DATA_PROCESSED / "us_population" / "cells.parquet",
                            columns=["unit_id", "county_fips", "tract_fips"])

    # resolve the view window -> a list of county FIPS + an output tag
    if STATE:
        st = STATE_FIPS.get(str(STATE).upper(), str(STATE))             # postal or fips
        counties = sorted(cells.loc[cells["county_fips"].str[:2] == st, "county_fips"].unique())
        tag = f"state_{st}"
    else:
        counties = [str(c).strip() for c in COUNTY if str(c).strip()]
        tag = counties[0] if len(counties) == 1 else f"{counties[0]}+{len(counties)-1}"
    if not counties:
        raise SystemExit("no counties resolved for the requested view")

    # human-readable region name for the figure title (falls back to the FIPS tag)
    region_name = REGION_NAME or REGION_NAMES.get(",".join(counties),
                    (f"State {st}" if STATE else tag))

    sel = cells["county_fips"].isin(counties).values
    if not sel.any():
        raise SystemExit(f"no blocks for {counties[:3]}{'...' if len(counties)>3 else ''}")
    preds = pd.DataFrame({
        "GEOID20":     cells["unit_id"].values[sel].astype(str),
        "county_fips": cells["county_fips"].values[sel].astype(str),
        "tract_fips":  cells["tract_fips"].values[sel].astype(str),
        "gt":          gt[sel], "baseline": baseline[sel], "puffin": puffin[sel],
    })
    print(f"view {tag}: {len(counties)} counties, {len(preds):,} blocks", flush=True)

    # read block polygons per state the view touches, then filter to the target counties in
    # pandas. We deliberately do NOT use a pyogrio `where`/attribute pushdown: that drives a
    # record-by-record seek via the .shx index, which fails hard ("fread() at offset ...") if
    # the .shx/.dbf sidecar is corrupt. A plain full read of one state is robust and cheap
    # enough for a metro figure (one or few states).
    keep = set(counties)
    parts = []
    for st in sorted({c[:2] for c in counties}):
        shp = AE_RAW_DIR.parent / "census_us" / f"tl_2020_{st}_tabblock20" / f"tl_2020_{st}_tabblock20.shp"
        print(f"  reading polygons for state {st} ...", flush=True)
        g = gpd.read_file(shp, columns=["GEOID20"], engine="pyogrio")
        g["GEOID20"] = g["GEOID20"].astype(str)
        g = g[g["GEOID20"].str[:2].eq(st)] if STATE else g[g["GEOID20"].str[:5].isin(keep)]
        parts.append(g)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    gdf["GEOID20"] = gdf["GEOID20"].astype(str)
    # display in Web Mercator (EPSG:3857): north-up (Albers/5070 tilts a metro away from
    # vertical) and native to the OSM basemap so tiles stay aligned.
    gdf = gdf.merge(preds, on="GEOID20", how="inner").to_crs(3857)
    print(f"joined {len(gdf):,} block polygons", flush=True)

    cols = [c for c, _ in PANELS]
    if METRIC == "within":
        tmean = gdf.groupby(REGION_COL)[cols].transform("mean")
        for c in cols:
            gdf[c] = gdf[c] - tmean[c]
        lim = np.nanpercentile(np.abs(np.concatenate([gdf[c].values for c in cols])), 98)
        norm = Normalize(vmin=-lim, vmax=lim); cmap = "RdBu_r"
        clabel = f"{LABEL} relative to {LEVEL} average ({UNIT})"
        suptitle = f"{region_name}: {LABEL} disaggregated to census blocks (deviation from {LEVEL} average)"
    else:
        # clip to vmin so 0-count blocks (log can't show 0) render at the DARK end in all three
        # panels instead of blank in GT -> panels become visually comparable.
        allv = np.concatenate([gdf[c].values for c in cols])
        vmin = max(1.0, np.nanpercentile(allv, 2)); vmax = np.nanpercentile(allv, 98)
        for c in cols:
            gdf[c] = np.clip(np.asarray(gdf[c].values, dtype=float), vmin, None)
        norm = LogNorm(vmin=vmin, vmax=vmax); cmap = "viridis"
        clabel = f"{UNIT.capitalize()} per block (log)"   # short: a long rotated label runs off the canvas
        suptitle = f"{region_name}: {LABEL} disaggregated to census blocks"

    # height set close to the metro's aspect so the square maps do not leave tall vertical
    # margins; the title then sits right above the panels (no big title-to-content gap).
    b = gdf.total_bounds
    aspect = (b[3] - b[1]) / (b[2] - b[0])            # metro height/width
    # Panels drawn large, as in the published version. Sizing the canvas to the printed
    # width instead leaves the maps tiny and the labels oversized once LaTeX scales the
    # figure to \linewidth.
    panel_w = 3.42
    fig, axes = plt.subplots(1, 3, figsize=(3 * panel_w + 0.9, panel_w * aspect + 0.6))
    for ax, (col, title) in zip(axes, PANELS):
        # OSM basemap FIRST as the full-rectangle background (zorder 0); the semi-transparent
        # values go ON TOP (zorder 2), so the map shows through everywhere, not just the gaps.
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
        try:
            cx.add_basemap(ax, crs=gdf.crs, source=BASEMAP, attribution=False,
                           reset_extent=False, zorder=0)
        except Exception as e:
            # affine >= 2.4 slots Affine, which breaks rasterio's cached_property and
            # takes the basemap down with it -- pin affine==2.3.1 when this fires.
            print(f"  basemap unavailable ({type(e).__name__}: {e}); plotting without it", flush=True)
        gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax, linewidth=0,
                 antialiased=False, alpha=ALPHA, zorder=2,
                 rasterized=True)   # blocks as a 300-dpi raster in the PDF; text stays vector
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
        ax.set_title(title, fontsize=12); ax.set_axis_off()
    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.01, right=0.90, wspace=0.03)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.012)
    cbar.set_label(clabel, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    # Equal-aspect maps shrink inside their axes boxes, so a title pinned to the figure top leaves
    # a gap. Place the title just above the panel titles as actually drawn instead.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    title_top = max(to_fig.transform(ax.title.get_window_extent(rend))[1][1] for ax in axes)
    fig.suptitle(suptitle, fontsize=14, y=title_top + 0.02, va="bottom")
    out = FIGURES_DIR / f"fig_us_{TARGET}_{tag}_{METRIC}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (+ .pdf)", flush=True)


if __name__ == "__main__":
    main()
