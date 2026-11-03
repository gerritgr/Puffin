"""Does the conformal factor kappa transfer between countries?

Stage 1 of the calibration rescales the posterior variances per 10 km tile using
coarse residuals only, so it needs no fine labels. Stage 2 fits one scalar kappa
as a quantile of standardised fine-scale errors, and that scalar is the only place
fine labels enter the method. The question this script answers is whether a kappa
fitted on one country gives the nominal coverage in another, because if it does,
the intervals can be deployed where no fine labels exist.

Nothing is retrained: kappa is computed from predictions that the runs already
saved. For each source and target we

  1. rebuild the stage-1 scaled residuals z = (y - mu) / sqrt(c2_tile * sigma2),
  2. take kappa from the SOURCE validation subregions,
  3. measure coverage of that kappa on the TARGET test subregions,

and compare against the target's own kappa, which is the number the paper reports.

    python kappa_transfer.py
Writes experiments/kappa_transfer.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from puffin.calibrate import region_scale
from puffin.config import EXPERIMENTS_DIR

LEVELS = (0.5, 0.8, 0.9, 0.95)

# (name, run directory, the five cross-fit folds live underneath)
SOURCES = {
    "germany_10km_1km": "germany_all_population_10000_1000_gaussian",
    "berlin_1km_100m":  "berlin_all_population_1000_100_gaussian",
    "us_pop_tract":     "us_population_population_tract_gaussian",
}
SEED = 0


def load_fold(run_dir: Path, test_fold: int):
    """Per-subregion arrays for one cross-fit fold, as the runner saved them."""
    f = run_dir / "nll" / f"seed_{SEED}" / f"crossfit_test{test_fold}" / "predictions.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    need = ("Y_fine", "mu_prior", "sig2_prior", "mu_post", "sig2_post", "cell_blk_idx")
    if any(k not in d for k in need):
        print("   %s lacks %s" % (f.parent.name, [k for k in need if k not in d]))
        return None
    return d


def standardised_scores(d):
    """z = (y - mu) / sqrt(stage-1 scaled variance), plus the fold of each subregion.

    Exactly as `region_conformal` does it inside the runner: the per-tile scale c2
    comes from the PRIOR residuals and the PRIOR variance, and it then rescales the
    POSTERIOR variance. Using the posterior for the coarse residual would be wrong,
    since the posterior reproduces every total and that residual is zero by
    construction.
    """
    y, mu, sig2 = d["Y_fine"], d["mu_post"], d["sig2_post"]
    bi = d["cell_blk_idx"].astype(int)
    n_blk = int(bi.max()) + 1
    Y_blk = np.bincount(bi, y, minlength=n_blk)
    M_blk = np.bincount(bi, d["mu_prior"], minlength=n_blk)
    V_blk = np.bincount(bi, d["sig2_prior"], minlength=n_blk).clip(min=1e-12)
    bx = d["block_x"] if "block_x" in d else np.zeros(n_blk)
    by = d["block_y"] if "block_y" in d else np.zeros(n_blk)
    c2 = region_scale(bx, by, (Y_blk - M_blk) ** 2, V_blk, 10000.0)
    sig2_cal = sig2 * c2[bi]
    z = (y - mu) / np.sqrt(np.maximum(sig2_cal, 1e-12))
    folds = d["folds"][bi] if "folds" in d else np.full(len(y), -1)
    return z, folds


def kappa_and_coverage(name, run_dir):
    """kappa from each fold's validation subregions, coverage on its test subregions."""
    per_fold = []
    for t in range(5):
        d = load_fold(run_dir, t)
        if d is None:
            continue
        z, folds = standardised_scores(d)
        val = folds == (t - 1) % 5
        test = folds == t
        if not val.any() or not test.any():
            continue
        row = {"test_fold": t, "n_val": int(val.sum()), "n_test": int(test.sum())}
        for p in LEVELS:
            k = float(np.quantile(np.abs(z[val]), p))
            row["kappa_%g" % p] = k
            row["cov_own_%g" % p] = float(np.mean(np.abs(z[test]) <= k))
        per_fold.append({"row": row, "z_test": z[test]})
    return per_fold


def main():
    runs = {n: EXPERIMENTS_DIR / d for n, d in SOURCES.items()}
    have = {n: p for n, p in runs.items() if p.exists()}
    print("run directories found:", ", ".join(have) or "none")
    data = {n: kappa_and_coverage(n, p) for n, p in have.items()}
    data = {n: v for n, v in data.items() if v}

    out = {"own": {}, "transfer": {}}
    for n, folds in data.items():
        out["own"][n] = {"kappa_0.9": float(np.mean([f["row"]["kappa_0.9"] for f in folds])),
                         "coverage_0.9": float(np.mean([f["row"]["cov_own_0.9"] for f in folds]))}
        print("  %-18s own kappa %.3f -> coverage %.3f"
              % (n, out["own"][n]["kappa_0.9"], out["own"][n]["coverage_0.9"]))

    print("\ntransfer (kappa from source, applied to target):")
    for src, sf in data.items():
        k_src = float(np.mean([f["row"]["kappa_0.9"] for f in sf]))
        for tgt, tf in data.items():
            if tgt == src:
                continue
            cov = float(np.mean([np.mean(np.abs(f["z_test"]) <= k_src) for f in tf]))
            out["transfer"]["%s -> %s" % (src, tgt)] = {"kappa": k_src, "coverage_0.9": cov}
            print("  %-18s -> %-18s kappa %.3f  coverage %.3f (nominal 0.90)"
                  % (src, tgt, k_src, cov))

    p = EXPERIMENTS_DIR.parent / "kappa_transfer.json"
    p.write_text(json.dumps(out, indent=2))
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
