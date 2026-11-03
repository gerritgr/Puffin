"""
Baselines for Puffin's disaggregation benchmark: same data, same folds, same metric,
and no fine labels.

  region_avg : each region total Y_i is split uniformly over its M_i subregions
               (within-region R^2 = 0 by construction; doubles as a sanity check).
  bicubic    : cubic interpolation of the coarse density field Y_i / A_i, evaluated at
               every subregion centroid, clipped at 0, times the subregion area, then
               rescaled within each region so the region total is reproduced exactly.
               Regular grids (Berlin, Germany): bicubic B-spline interpolation of the
               region grid (region centres carry the region densities).
               Irregular units (US): Clough-Tocher cubic interpolation over the
               area-weighted region centroids, the scattered-data analogue.
  goodman    : Goodman's ecological regression with embedding covariates. The region
               density Y_i / A_i is regressed (ridge) on the area-weighted mean of the
               standardized subregion embeddings, over TRAINING regions only. The ridge
               strength is chosen on the VALIDATION fold's region totals only (no fine
               labels anywhere). The fitted density is applied to every test subregion,
               times its area, clipped at 0, then rescaled to reproduce each region total.

Data build, folds, and scoring mirror runner.run() and run_*_crossfit.py:
  - grid: aggregate_to_subregions -> build_coarse_blocks; fixed: region_col ids
  - folds: 5 km super-blocks, split_seed 0, val = (t-1)%5, test = t
  - OOF over the 5 test folds, scored with _within / _r2 on all and inhabited cells
All baselines are deterministic, so there are no seeds.

Usage:
  python baselines.py --suite berlin                 # 18 grid configs
  python baselines.py --suite germany [--configs 1000/100,10000/5000]
  python baselines.py --suite us [--targets population,jobs,gun_violence]
  python baselines.py --suite voting
Writes experiments/baselines_<suite>_summary.json (merged, resumable per config).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import r2_score

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
from puffin.config import DATA_PROCESSED, EXPERIMENTS_DIR   # noqa: E402
from puffin.data import aggregate_to_subregions, build_coarse_blocks   # noqa: E402

N_FOLDS       = 5
SPLIT_SEED    = 0
SUPER_BLOCK_M = 5000
EPS           = 1e-9
METHODS       = ["region_avg", "bicubic", "goodman"]

REGION_SIZES = [200, 1000, 2000, 4000, 10000]
FACTORS      = [2, 5, 10, 20, 50, 100]
GRID = [(R, R // f) for R in REGION_SIZES for f in FACTORS
        if R % f == 0 and R // f >= 100 and (R // f) % 100 == 0]

SUITES = {
    "berlin":  dict(region="berlin_all",    mode="grid",  target="population"),
    "germany": dict(region="germany_all",   mode="grid",  target="population"),
    "us":      dict(region="us_population", mode="fixed", region_col="tract_fips",
                    level="tract", exposure_col="exposure",
                    targets=["population", "jobs", "gun_violence"]),
    # Precincts hold similar numbers of voters, so the area term makes the split
    # worse; us_voting/cells.parquet carries no exposure column for that reason.
    "voting":  dict(region="us_voting",     mode="fixed", region_col="county_fips",
                    level="county", exposure_col="",
                    targets=["votes_rep", "votes_dem", "votes_other", "votes_total"]),
}


# ----------------------------------------------------------------------------- scoring
# Identical to runner._r2 and run_*_crossfit._within.
def _r2(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else float("nan")


def _within(y, p, cbi, mask):
    m = mask & np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return float("nan")
    _, inv = np.unique(cbi[m], return_inverse=True)
    ym = np.bincount(inv, y[m]) / np.bincount(inv)
    pm = np.bincount(inv, p[m]) / np.bincount(inv)
    yw, pw = y[m] - ym[inv], p[m] - pm[inv]
    return float(1 - ((yw - pw) ** 2).sum() / (((yw - yw.mean()) ** 2).sum()))


def _score(Yf, oof, cbi):
    scored = np.isfinite(oof)
    m_all, m_inh = scored, scored & (Yf > 0)
    return {
        "within_all":    round(_within(Yf, oof, cbi, m_all), 4),
        "within_inhab":  round(_within(Yf, oof, cbi, m_inh), 4),
        "cell_r2_all":   round(_r2(Yf[m_all], oof[m_all]), 4),
        "cell_r2_inhab": round(_r2(Yf[m_inh], oof[m_inh]), 4),
    }


# ----------------------------------------------------------------------------- data
def _load(region, cols):
    path = DATA_PROCESSED / region / "cells.parquet"
    names = pq.ParquetFile(path).schema.names
    ae = [c for c in names if c.startswith("ae_")]
    return pd.read_parquet(path, columns=list(dict.fromkeys(cols + ae))), ae


def build_grid(region, target, region_m, subregion_m):
    cells, ae = _load(region, ["x_mp", "y_mp", target])
    if subregion_m > 100:
        cells = aggregate_to_subregions(cells, subregion_m)
    blocks = build_coarse_blocks(cells, region_m=region_m, target=target)
    bx = (cells["x_mp"].values // region_m) * region_m
    by = (cells["y_mp"].values // region_m) * region_m
    key = pd.MultiIndex.from_arrays([blocks["block_x"].values.astype(np.int64),
                                     blocks["block_y"].values.astype(np.int64)])
    cbi = key.get_indexer(pd.MultiIndex.from_arrays([bx.astype(np.int64), by.astype(np.int64)]))
    ok = cbi >= 0
    cells, cbi = cells[ok].reset_index(drop=True), cbi[ok]
    d = dict(cbi=cbi,
             Yf=cells[target].values.astype(np.float64),
             Yb=blocks[f"Y_{target}"].values.astype(np.float64),
             Nb=blocks["n_cells"].values.astype(np.float64),
             bx=blocks["block_x"].values.astype(np.int64),
             by=blocks["block_y"].values.astype(np.int64),
             x=cells["x_mp"].values.astype(np.float64),
             y=cells["y_mp"].values.astype(np.float64),
             a=np.ones(len(cells)),
             X=cells[ae].to_numpy(dtype=np.float32),
             region_m=region_m)
    del cells
    return d


def build_fixed(region, target, region_col, exposure_col):
    cols = ["x_mp", "y_mp", target, region_col] + ([exposure_col] if exposure_col else [])
    cells, ae = _load(region, cols)
    reg = cells[region_col].astype(str).values
    _, cbi = np.unique(reg, return_inverse=True)
    nb = int(cbi.max()) + 1
    Nb = np.bincount(cbi, minlength=nb).astype(np.float64)
    x = cells["x_mp"].values.astype(np.float64)
    y = cells["y_mp"].values.astype(np.float64)
    a = (np.clip(cells[exposure_col].values.astype(np.float64), 1e-6, None)
         if exposure_col else np.ones(len(cells)))
    d = dict(cbi=cbi,
             Yf=cells[target].values.astype(np.float64),
             Yb=np.bincount(cbi, cells[target].values.astype(np.float64), minlength=nb),
             Nb=Nb,
             # runner uses the UNWEIGHTED mean subregion centroid for the fold assignment
             bx=(np.bincount(cbi, x, minlength=nb) / np.clip(Nb, 1, None)).astype(np.int64),
             by=(np.bincount(cbi, y, minlength=nb) / np.clip(Nb, 1, None)).astype(np.int64),
             x=x, y=y, a=a,
             X=cells[ae].to_numpy(dtype=np.float32),
             region_m=None)
    del cells
    return d


def folds_of(d):
    """Vectorized make_spatial_splits: identical super-block order, RNG, and fold rule."""
    S = SUPER_BLOCK_M
    sid = pd.Series((d["bx"] // S) * S).astype(str) + "_" + pd.Series((d["by"] // S) * S).astype(str)
    uniq = sid.unique()
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(uniq))
    fold_of_super = dict(zip(uniq, (perm % N_FOLDS).astype(int)))
    return sid.map(fold_of_super).values.astype(int)


# ----------------------------------------------------------------------------- baselines
def _conserve(raw, d):
    """Clip at 0 and rescale so every region reproduces its observed total exactly."""
    cbi, nb = d["cbi"], len(d["Yb"])
    raw = np.clip(np.where(np.isfinite(raw), raw, 0.0), 0.0, None)
    S = np.bincount(cbi, raw, minlength=nb)
    share = np.where(S[cbi] > EPS, raw / np.maximum(S[cbi], EPS), 1.0 / d["Nb"][cbi])
    return d["Yb"][cbi] * share


def region_avg(d):
    return d["Yb"][d["cbi"]] / d["Nb"][d["cbi"]]


def bicubic_grid(d):
    from scipy.ndimage import map_coordinates
    R = d["region_m"]
    x0, y0 = d["bx"].min(), d["by"].min()
    ci = ((d["bx"] - x0) // R).astype(int)
    ri = ((d["by"] - y0) // R).astype(int)
    img = np.zeros((ri.max() + 1, ci.max() + 1))           # regions with no land stay 0
    img[ri, ci] = d["Yb"] / d["Nb"]                          # density per subregion
    fr = (d["y"] - (y0 + R / 2)) / R                         # region centre -> integer index
    fc = (d["x"] - (x0 + R / 2)) / R
    dens = map_coordinates(img, [fr, fc], order=3, mode="nearest", prefilter=True)
    return _conserve(d["a"] * dens, d)


def bicubic_scattered(d):
    from scipy.interpolate import CloughTocher2DInterpolator
    cbi, nb = d["cbi"], len(d["Yb"])
    A = np.bincount(cbi, d["a"], minlength=nb)
    rho = d["Yb"] / np.maximum(A, EPS)
    cx = np.bincount(cbi, d["a"] * d["x"], minlength=nb) / np.maximum(A, EPS)
    cy = np.bincount(cbi, d["a"] * d["y"], minlength=nb) / np.maximum(A, EPS)
    interp = CloughTocher2DInterpolator(np.c_[cx, cy], rho)
    dens = np.empty(len(cbi))
    for s in range(0, len(cbi), 2_000_000):
        e = min(s + 2_000_000, len(cbi))
        dens[s:e] = interp(d["x"][s:e], d["y"][s:e])
    dens = np.where(np.isfinite(dens), dens, rho[cbi])       # outside the hull: own region
    return _conserve(d["a"] * dens, d)


def _region_sums(d):
    """Per-region cell count, area, and (area-weighted) embedding sums; fold-independent."""
    cbi, nb, X, a = d["cbi"], len(d["Yb"]), d["X"], d["a"]
    D = X.shape[1]
    Sx, Sx2, SAx = np.zeros((nb, D)), np.zeros((nb, D)), np.zeros((nb, D))
    for k in range(D):
        col = X[:, k].astype(np.float64)
        Sx[:, k]  = np.bincount(cbi, col, minlength=nb)
        Sx2[:, k] = np.bincount(cbi, col * col, minlength=nb)
        SAx[:, k] = np.bincount(cbi, a * col, minlength=nb)
    A = np.bincount(cbi, a, minlength=nb)
    return dict(Sx=Sx, Sx2=Sx2, SAx=SAx, A=A)


def goodman(d, folds, rs):
    cbi, X, a = d["cbi"], d["X"], d["a"]
    rho = d["Yb"] / np.maximum(rs["A"], EPS)
    oof = np.full(len(cbi), np.nan)
    info = []
    for t in range(N_FOLDS):
        v = (t - 1) % N_FOLDS
        tr, va, te = (folds != t) & (folds != v), folds == v, folds == t
        # standardization from TRAIN subregions (as the Puffin runner does)
        n = d["Nb"][tr].sum()
        mu = rs["Sx"][tr].sum(0) / n
        sd = np.sqrt(np.clip(rs["Sx2"][tr].sum(0) / n - mu ** 2, 1e-12, None))
        Z = (rs["SAx"] / np.maximum(rs["A"], EPS)[:, None] - mu) / sd   # area-wtd mean std. emb.
        Ztr, rtr = Z[tr], rho[tr]
        zm, rm = Ztr.mean(0), rtr.mean()
        U, s, Vt = np.linalg.svd(Ztr - zm, full_matrices=False)
        Utr = U.T @ (rtr - rm)
        lams = (s.max() ** 2 if s.size else 1.0) * np.logspace(-8, 2, 31)
        best = None
        for lam in lams:
            beta = Vt.T @ (s / (s ** 2 + lam) * Utr)
            b0 = rm - zm @ beta
            mse = float(np.mean((Z[va] @ beta + b0 - rho[va]) ** 2)) if va.any() else 0.0
            if best is None or mse < best[0]:
                best = (mse, lam, beta, b0)
        _, lam, beta, b0 = best
        w, bias = beta / sd, b0 - (mu / sd) @ beta               # rho_hat = X @ w + bias
        te_c = np.where(te[cbi])[0]
        dens = np.empty(len(te_c))
        for s0 in range(0, len(te_c), 2_000_000):
            idx = te_c[s0:s0 + 2_000_000]
            dens[s0:s0 + len(idx)] = X[idx].astype(np.float64) @ w + bias
        raw = np.zeros(len(cbi))
        raw[te_c] = a[te_c] * dens
        full = _conserve(raw, d)            # regions outside te are overwritten below anyway
        oof[te_c] = full[te_c]
        info.append({"fold": t, "lambda_rel": float(lam / (s.max() ** 2 if s.size else 1.0)),
                     "n_train_regions": int(tr.sum()), "n_val_regions": int(va.sum())})
    return oof, info


# ----------------------------------------------------------------------------- driver
def run_one(d, label):
    t0 = time.time()
    folds = folds_of(d)
    out = {"n_regions": int(len(d["Yb"])), "n_cells": int(len(d["cbi"])),
           "n_inhab": int((d["Yf"] > 0).sum()), "methods": {}}
    Yf, cbi = d["Yf"], d["cbi"]

    out["methods"]["region_avg"] = _score(Yf, region_avg(d), cbi)
    tb = time.time()
    bic = bicubic_grid(d) if d["region_m"] is not None else bicubic_scattered(d)
    out["methods"]["bicubic"] = _score(Yf, bic, cbi)
    out["methods"]["bicubic"]["conserves"] = bool(np.allclose(
        np.bincount(cbi, bic, minlength=len(d["Yb"])), d["Yb"], rtol=1e-6, atol=1e-6))
    tg = time.time()
    rs = _region_sums(d)
    gm, info = goodman(d, folds, rs)
    out["methods"]["goodman"] = _score(Yf, gm, cbi)
    out["methods"]["goodman"]["folds"] = info
    print(f"  {label}: regions={out['n_regions']:,} cells={out['n_cells']:,} "
          f"| bicubic {tg - tb:.0f}s goodman {time.time() - tg:.0f}s total {time.time() - t0:.0f}s")
    for m in METHODS:
        r = out["methods"][m]
        print(f"     {m:>11}: within all {r['within_all']:+.4f} inhab {r['within_inhab']:+.4f} "
              f"| subregion-R2 all {r['cell_r2_all']:.4f} inhab {r['cell_r2_inhab']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--configs", default="", help="grid only, e.g. 1000/100,10000/5000")
    ap.add_argument("--targets", default="", help="fixed only, e.g. population,jobs")
    ap.add_argument("--force", action="store_true", help="recompute cached rows")
    args = ap.parse_args()
    cfg = SUITES[args.suite]

    path = EXPERIMENTS_DIR.parent / f"baselines_{args.suite}_summary.json"
    rows = json.loads(path.read_text()) if path.exists() else []

    def save():
        path.write_text(json.dumps(rows, indent=2))

    if cfg["mode"] == "grid":
        todo = ([tuple(int(v) for v in c.split("/")) for c in args.configs.split(",")]
                if args.configs else GRID)
        for (R, S) in todo:
            if not args.force and any(r.get("region_m") == R and r.get("subregion_m") == S for r in rows):
                print(f"  {R}/{S}: cached"); continue
            d = build_grid(cfg["region"], cfg["target"], R, S)
            res = {"region_m": R, "subregion_m": S, "n_side": R // S, **run_one(d, f"{R}/{S}")}
            del d
            rows = [r for r in rows if not (r.get("region_m") == R and r.get("subregion_m") == S)] + [res]
            rows.sort(key=lambda r: (r["region_m"], r["subregion_m"])); save()
    else:
        targets = args.targets.split(",") if args.targets else cfg["targets"]
        for tg in targets:
            if not args.force and any(r.get("target") == tg and r.get("level") == cfg["level"] for r in rows):
                print(f"  {tg}: cached"); continue
            d = build_fixed(cfg["region"], tg, cfg["region_col"], cfg["exposure_col"])
            res = {"target": tg, "level": cfg["level"], **run_one(d, f"{tg}@{cfg['level']}")}
            del d
            rows = [r for r in rows if not (r.get("target") == tg and r.get("level") == cfg["level"])] + [res]
            rows.sort(key=lambda r: (r["target"], r["level"])); save()
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
