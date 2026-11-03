"""
Learning curves in the style of CLAM (Grossmann et al., 2026, Fig. 3): does the fine-scale
error of the split fall as the aggregate training loss falls?

For each setting and seed, trains the Gaussian model exactly as in the paper (same folds,
epochs, schedule), with the runner's opt-in `curve_points` diagnostic switched on. At
log-spaced epochs it writes the aggregate NLL (train / val / test regions) next to the
fine-scale error of the posterior split on the TEST subregions (within-region R2, MAE),
and of the prior alone. The diagnostic never feeds checkpoint selection.

    python run_learning_curves.py --suite berlin     # two Berlin configurations, 5 seeds
    python run_learning_curves.py --suite germany    # two Germany configurations, 5 seeds
    python run_learning_curves.py --suite us         # US population, tract -> block

Output: experiments/learning_curves/<setting>/seed_<s>/learning_curve.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from puffin.config import EXPERIMENTS_DIR, FIGURES_DIR
from puffin.runner import PuffinConfig, run as runner_run

OUT = EXPERIMENTS_DIR / "learning_curves"
TEST_FOLD = 4                      # one fixed spatial fold; its predecessor validates
COMMON = dict(target="population", loss="nll", method="gaussian", epochs=2000,
              split_seed=0, super_block_m=5000, val_fold=(TEST_FOLD - 1) % 5,
              test_fold=TEST_FOLD, curve_points=60, save_predictions=False)

SUITES = {
    "berlin": [
        ("berlin_1km_100m",  dict(region="berlin_all", region_m=1000,  subregion_m=100)),
        ("berlin_10km_1km",  dict(region="berlin_all", region_m=10000, subregion_m=1000)),
    ],
    "germany": [   # both fit full-batch (<= 3M prediction units), as in run_germany_all_crossfit.py
        ("germany_10km_1km",  dict(region="germany_all", region_m=10000, subregion_m=1000)),
        ("germany_4km_400m",  dict(region="germany_all", region_m=4000,  subregion_m=400)),
    ],
    "us": [
        ("us_population_tract_block", dict(region="us_population", fixed_region=True,
                                           region_col="tract_fips", exposure_col="exposure")),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    for name, kw in SUITES[args.suite]:
        for seed in args.seeds:
            run_dir = OUT / name / f"seed_{seed}"
            if (run_dir / "learning_curve.csv").exists():
                print(f"[skip] {name} seed {seed}: already done", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = PuffinConfig(**COMMON, **kw, seed=seed, run_dir=run_dir)
            print(f"[run] {name} seed {seed} -> {run_dir}", flush=True)
            runner_run(cfg)
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
