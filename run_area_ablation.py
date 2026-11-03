"""Ablate the area term on the prior variance (US population and jobs, tract -> block).

The method scales both the prior mean and the prior variance of a subregion by its area.
On a uniform grid the area is constant and the factor cancels, so the ablation only has an
effect on irregular units. It answers one question: how much of the US result comes from
letting a large block absorb a larger share of its region's correction, rather than from the
predicted mean alone.

Both arms use the same architecture, loss, folds, seeds and epochs, and differ only in
`area_variance`. Full 5-fold spatial cross-validation, so the numbers are comparable with
Table 2 of the paper.

    python run_area_ablation.py                 # population and jobs, seed 0
    python run_area_ablation.py --targets jobs --seeds 0 1 2

Writes experiments/area_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from puffin.runner import PuffinConfig, run
from puffin.config import EXPERIMENTS_DIR

COMMON = dict(region="us_population", fixed_region=True, region_col="tract_fips",
              exposure_col="exposure", loss="nll", method="gaussian", epochs=2000,
              split_seed=0, super_block_m=5000, save_predictions=False)


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float("nan") if ss <= 0 else 1.0 - ((y - p) ** 2).sum() / ss


def crossfit(target: str, seed: int, area_variance: bool) -> dict:
    """5-fold out-of-fold within-region R^2, stitched over the folds as the paper does."""
    y_all, p_all, b_all = [], [], []
    for test_fold in range(5):
        tag = f"us_area_abl_{target}_{'on' if area_variance else 'off'}"
        run_dir = EXPERIMENTS_DIR / tag / f"seed_{seed}" / f"crossfit_test{test_fold}"
        out = run(PuffinConfig(**COMMON, target=target, seed=seed,
                               val_fold=(test_fold - 1) % 5, test_fold=test_fold,
                               area_variance=area_variance, run_dir=run_dir))
        sys.stdout = sys.__stdout__
        m = out["test_cell_mask"]
        y_all.append(out["Y_fine"][m]); p_all.append(out["mu_post"][m])
        b_all.append(out["cell_blk_idx"][m])
        print(f"  [{target} area_variance={area_variance}] fold {test_fold} done", flush=True)

    y = np.concatenate(y_all); p = np.concatenate(p_all)
    b = np.concatenate(b_all, dtype=np.int64)
    _, inv = np.unique(b, return_inverse=True)
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).clip(min=1)
    mean = np.bincount(inv, y, minlength=n) / cnt
    return {"within_all": _r2(y - mean[inv], p - mean[inv]), "n_cells": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["population", "jobs"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    out = {}
    for target in args.targets:
        for area_variance in (True, False):
            rows = [crossfit(target, s, area_variance) for s in args.seeds]
            key = f"{target}_area_variance_{'on' if area_variance else 'off'}"
            out[key] = {"within_all": float(np.mean([r["within_all"] for r in rows])),
                        "within_all_std": float(np.std([r["within_all"] for r in rows])),
                        "seeds": args.seeds, "n_cells": rows[0]["n_cells"]}
            print(f"{key}: within-region R2 = {out[key]['within_all']:+.4f}", flush=True)

        on = out[f"{target}_area_variance_on"]["within_all"]
        off = out[f"{target}_area_variance_off"]["within_all"]
        print(f"=> {target}: {on:+.4f} with the area term, {off:+.4f} without "
              f"(delta {on - off:+.4f})", flush=True)

    p = EXPERIMENTS_DIR.parent / "area_ablation.json"
    p.write_text(json.dumps(out, indent=2))
    print("saved ->", p)


if __name__ == "__main__":
    main()
