"""Run Puffin on the German census for ONE configuration.

Set the configuration below and run:

    python main_germany.py

Output goes to experiments/runs/<tag>/, with one subdirectory per seed and a
summary in seeds.json. To reproduce a whole table instead of a single
configuration, use the sweep wrapper run_berlin_all_crossfit.py or
run_germany_all_crossfit.py, which call this same runner for every cell of the
grid.
"""

from __future__ import annotations

# =============================================================================
# CONFIGURATION
# =============================================================================

REGION = "berlin_all"   # a directory under data/processed/ holding cells.parquet
YEAR = 2022
TARGET = "population"   # column of cells.parquet to disaggregate

# A region is the coarse unit whose total is observed; a subregion is the unit
# we predict. Each region is split into (REGION_M / SUBREGION_M)^2 subregions.
REGION_M = 1000         # region size in metres
SUBREGION_M = 100       # subregion size in metres; must divide REGION_M

# Training
EPOCHS = 2000           # the cosine schedule anneals the learning rate over this horizon
LR = 3e-3               # peak learning rate
LOSS = "nll"            # "nll" (aggregate likelihood) | "mse" (squared error ablation)
METHOD = "gaussian"     # "gaussian" (the paper) | "poisson" | "gamma"
TAU2_TRAIN = 5_000.0    # region-level noise variance, used by rate targets only
STANDARDIZE = True      # per-dimension z-score of the network input

# Evaluation
SPLIT_SEED = 0          # the spatial split is fixed across seeds, so several seeds
SEEDS = [0, 1, 2, 3, 4] # differ only in initialisation
VAL_FOLD = 3
TEST_FOLD = 4
SUPER_BLOCK_M = 5000    # size of the spatial groups assigned to folds as a whole
CROSSFIT = True         # rotate the test fold over all 5 folds, scoring every
                        # subregion once out of fold (the protocol used in the paper)

# =============================================================================

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from puffin.runner import PuffinConfig, _r2, run


def _base_cfg() -> PuffinConfig:
    return PuffinConfig(
        region=REGION, year=YEAR, target=TARGET,
        region_m=REGION_M, subregion_m=SUBREGION_M,
        epochs=EPOCHS, lr=LR, tau2_train=TAU2_TRAIN, loss=LOSS, method=METHOD,
        val_fold=VAL_FOLD, test_fold=TEST_FOLD, split_seed=SPLIT_SEED,
        standardize=STANDARDIZE, super_block_m=SUPER_BLOCK_M,
    )


def _crossfit(base: PuffinConfig) -> None:
    """Rotate the test fold over all five folds so every subregion is scored once
    out of fold, then report the pooled scores."""
    base_dir = base.auto_run_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    seed = SEEDS[0]
    oof_mu = Yf = cbi = Yblk = None

    for t in range(5):
        v = (t - 1) % 5
        print(f"\n{'='*60}\n  Cross-fit  test_fold={t}  val_fold={v}  ({t+1}/5)\n{'='*60}")
        cfg = PuffinConfig(
            **{k: val for k, val in base.__dict__.items()
               if k not in ("run_dir", "seed", "val_fold", "test_fold")},
            seed=seed, val_fold=v, test_fold=t,
            run_dir=base_dir / f"crossfit_test{t}",
        )
        out = run(cfg)
        sys.stdout = sys.__stdout__
        if oof_mu is None:
            oof_mu = np.full(len(out["mu_post"]), np.nan)
            Yf, cbi, Yblk = out["Y_fine"], out["cell_blk_idx"], out["Y_blk"]
        tcm = out["test_cell_mask"]
        oof_mu[tcm] = out["mu_post"][tcm]

    n_missing = int(np.isnan(oof_mu).sum())
    N = np.bincount(cbi).astype(float)
    region_mean = (np.bincount(cbi, Yf) / np.clip(N, 1, None))[cbi]
    region_avg = Yblk[cbi] / np.clip(N[cbi], 1, None)
    cell_r2 = _r2(Yf, oof_mu)
    within = _r2(Yf - region_mean, oof_mu - region_mean)

    summary = {
        "region": REGION, "target": TARGET,
        "region_m": REGION_M, "subregion_m": SUBREGION_M, "seed": seed,
        "scheme": "5-fold cross-fit, every subregion scored once out of fold",
        "n_subregions_scored": int((~np.isnan(oof_mu)).sum()), "n_missing": n_missing,
        "region_average_subregion_r2": round(_r2(Yf, region_avg), 4),
        "puffin_subregion_r2": round(cell_r2, 4),
        "puffin_within_region_r2": round(within, 4),
    }
    (base_dir / "crossfit.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}\n  CROSS-FIT  {REGION} | {TARGET} | {REGION_M}m/{SUBREGION_M}m")
    print("-" * 60)
    print(f"  region average   subregion-R2 : {summary['region_average_subregion_r2']:.4f}   within-R2 : 0.000")
    print(f"  Puffin           subregion-R2 : {cell_r2:.4f}   within-R2 : {within:.4f}")
    print(f"  subregions scored: {summary['n_subregions_scored']:,}  (missing: {n_missing})")
    print(f"{'='*60}\n  Saved -> {base_dir}/crossfit.json")


def _multi_seed(base: PuffinConfig) -> None:
    """Train one model per seed on the same split and report mean +/- std."""
    base_dir = base.auto_run_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}\n  Seed {seed}  ({i+1}/{len(SEEDS)})\n{'='*60}")
        cfg = PuffinConfig(
            **{k: v for k, v in base.__dict__.items() if k not in ("run_dir", "seed")},
            seed=seed, run_dir=base_dir / f"seed_{seed}",
        )
        results.append(run(cfg)["results"])
        sys.stdout = sys.__stdout__

    def stat(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4)

    keys = ("prior_r2", "post_r2", "nll_blk_prior", "nll_cell_post", "cal_blk_90", "cal_cell_90")
    summary = {
        "region": REGION, "year": YEAR, "target": TARGET,
        "region_m": REGION_M, "subregion_m": SUBREGION_M,
        "loss": LOSS, "method": METHOD, "seeds": SEEDS,
        **{f"{k}_{s}": v for k in keys for s, v in zip(("mean", "std"), stat(k))},
        "per_seed": [{"seed": SEEDS[i], **{k: results[i][k] for k in keys}}
                     for i in range(len(SEEDS))],
    }
    (base_dir / "seeds.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}\n  RESULTS over {len(SEEDS)} seeds")
    print(f"  {REGION} | {TARGET} | {REGION_M}m/{SUBREGION_M}m | {METHOD}")
    print("-" * 60)
    for k in keys:
        m, s = stat(k)
        print(f"  {k:<16}: {m:.4f} +/- {s:.4f}")
    print(f"{'='*60}\n  Saved -> {base_dir}/seeds.json")


def main() -> None:
    base = _base_cfg()
    if CROSSFIT:
        _crossfit(base)
    elif len(SEEDS) == 1:
        run(PuffinConfig(**{**base.__dict__, "seed": SEEDS[0]}))
    else:
        _multi_seed(base)


if __name__ == "__main__":
    main()
