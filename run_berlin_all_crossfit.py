"""Run every configuration of the German grid on the Berlin data.

This is the sweep version of main_germany.py: it calls the same runner once per
(region size, subregion size) pair, so use main_germany.py to run a single
configuration and this script to reproduce a whole table.

The grid has region sizes as rows and the number of subregions per side as
columns, keeping the pairs whose subregion is at least 100 m and a multiple of
100 m, so that every subregion tiles the census lattice exactly. That leaves 18
configurations. Each one is evaluated by 5-fold cross-fitting, so every
subregion is scored once out of fold, and repeated over several seeds that
differ only in initialisation, which gives the reported mean and standard
deviation.

    python run_berlin_all_crossfit.py [--dry-run]

Results are written to experiments/berlin_allcells_<method>_summary.json.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from puffin.config import EXPERIMENTS_DIR
from puffin.runner import PuffinConfig, _r2, run as runner_run

REGION = "berlin_all"
TARGET = "population"
LOSS   = "nll"
METHOD = "gaussian"  # "gaussian" (the paper) | "poisson" | "gamma";
                     # each method writes its own run dirs and summary
EPOCHS = 2000

# The spatial split is fixed, so seeds differ only in model initialisation and
# the reported standard deviation is the spread over initialisations.
SEEDS      = [0, 1, 2, 3, 4]
SPLIT_SEED = 0

# region size (rows) x factor (subregions per side, columns). subregion = R / factor.
# Regions chosen so the triangle is HOLE-FREE: every factor <= m (where m = R/100)
# divides m, so R/f is always a clean multiple of 100m. The 1-2-5 offenders 500m and
# 5km break this (500/2 = 5km/20 = 250m, not a multiple of the 100m lattice), so they
# are dropped and 5km is replaced by 4km. Any in-triangle blank would then mean an
# infeasible <100m subregion. => 18 configs.
REGION_SIZES = [200, 1000, 2000, 4000, 10000]
FACTORS      = [2, 5, 10, 20, 50, 100]
GRID = [(R, R // f) for R in REGION_SIZES for f in FACTORS
        if R % f == 0 and R // f >= 100 and (R // f) % 100 == 0]

# metrics computed per seed (per-config), aggregated to mean +/- std across seeds
SEED_METRICS = (
    "cell_r2_post_all", "cell_r2_post_inhab", "cell_r2_prior_all", "cell_r2_prior_inhab",
    "within_post_all", "within_post_inhab", "within_prior_all", "within_prior_inhab",
    "cal_cell_90", "cal_blk_90", "nb_dispersion_r", "nll_blk_prior", "nll_cell_post",
    "mean_pred_on_zero_cells",
)


def _within(y, p, cbi, mask):
    """within-region R2 over `mask` cells; region mean taken over the SAME mask."""
    m = mask & np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return float("nan")
    idx = cbi[m]
    _, inv = np.unique(idx, return_inverse=True)
    ym = np.bincount(inv, y[m]) / np.bincount(inv)
    pm = np.bincount(inv, p[m]) / np.bincount(inv)
    yw, pw = y[m] - ym[inv], p[m] - pm[inv]
    return float(1 - ((yw - pw) ** 2).sum() / (((yw - yw.mean()) ** 2).sum()))


def _num(v):
    """results.json value -> float (None -> nan)."""
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _wavg(vals, weights):
    """weighted mean over finite entries; nan if none."""
    v = np.array(vals, float); w = np.array(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")


def _oof_one_seed(region_m, sub_m, seed, base):
    """One 5-fold OOF pass at a given init seed. Returns (oof_post, oof_prior, Yf,
    cbi, per_fold) where per_fold carries each fold's results.json + test sizes."""
    oof_post = oof_prior = Yf = cbi = None
    per_fold = []
    for t in range(5):
        cfg = PuffinConfig(region=REGION, target=TARGET, region_m=region_m, subregion_m=sub_m,
                           epochs=EPOCHS, loss=LOSS, method=METHOD, seed=seed, split_seed=SPLIT_SEED,
                           val_fold=(t - 1) % 5, test_fold=t,
                           run_dir=base / f"seed_{seed}" / f"crossfit_test{t}")
        out = runner_run(cfg)
        mu_post_t, mu_prior_t, r = out["mu_post"], out["mu_prior"], out["results"]
        sys.stdout = sys.__stdout__
        if oof_post is None:
            oof_post  = np.full(len(mu_post_t),  np.nan)
            oof_prior = np.full(len(mu_prior_t), np.nan)
            Yf, cbi   = out["Y_fine"], out["cell_blk_idx"]
        tcm = out["test_cell_mask"]
        oof_post[tcm]  = mu_post_t[tcm]
        oof_prior[tcm] = mu_prior_t[tcm]
        per_fold.append({"res": r, "n_cells": int(tcm.sum()), "n_blocks": int(r["n_test"])})
    return oof_post, oof_prior, Yf, cbi, per_fold


def _seed_metrics(oof_post, oof_prior, Yf, cbi, per_fold):
    """All seed-dependent scalars for one OOF pass. R2s pooled over the OOF vector;
    calibration/NLL are the (test-count-weighted) pooled per-fold conformal values."""
    scored = np.isfinite(oof_post)
    inhab  = Yf > 0
    m_all, m_inh = scored, scored & inhab
    zero = scored & ~inhab
    nc = [f["n_cells"] for f in per_fold]; nb = [f["n_blocks"] for f in per_fold]
    R  = [f["res"] for f in per_fold]
    return {
        "cell_r2_post_all":     _r2(Yf[m_all], oof_post[m_all]),
        "cell_r2_post_inhab":   _r2(Yf[m_inh], oof_post[m_inh]),
        "cell_r2_prior_all":    _r2(Yf[m_all], oof_prior[m_all]),
        "cell_r2_prior_inhab":  _r2(Yf[m_inh], oof_prior[m_inh]),
        "within_post_all":      _within(Yf, oof_post,  cbi, m_all),
        "within_post_inhab":    _within(Yf, oof_post,  cbi, m_inh),
        "within_prior_all":     _within(Yf, oof_prior, cbi, m_all),
        "within_prior_inhab":   _within(Yf, oof_prior, cbi, m_inh),
        "cal_cell_90":     _wavg([_num(r.get("cal_cell_90")) for r in R], nc),
        "cal_blk_90":      _wavg([_num(r.get("cal_blk_90"))  for r in R], nb),
        "nb_dispersion_r": _wavg([_num(r.get("nb_dispersion_r")) for r in R], nb),
        "nll_blk_prior":   _wavg([_num(r.get("nll_blk_prior")) for r in R], nb),
        "nll_cell_post":   _wavg([_num(r.get("nll_cell_post")) for r in R], nc),
        "mean_pred_on_zero_cells": float(np.nanmean(oof_post[zero])) if zero.sum() else float("nan"),
    }


def _ms(vals):
    """(mean, std) over seeds; std uses ddof=1 when K>1, else 0.0."""
    a = np.array([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def crossfit_cell(region_m, sub_m):
    n = region_m // sub_m
    tag = f"{REGION}_{TARGET}_{region_m}_{sub_m}_{METHOD}"   # method always in tag (matches auto_run_dir)
    base = EXPERIMENTS_DIR / tag / LOSS
    cf = base / "crossfit_allcells.json"
    if cf.exists():
        print(f"  {region_m}m/{sub_m}m ({n}x{n}): skip")
        return json.loads(cf.read_text())
    seeds_used = SEEDS
    print(f"  {region_m}m/{sub_m}m ({n}x{n}): 5-fold OOF x {len(seeds_used)} seed(s) ...", flush=True)

    per_seed, Yf0, cbi0 = [], None, None
    for seed in seeds_used:
        oof_post, oof_prior, Yf, cbi, per_fold = _oof_one_seed(region_m, sub_m, seed, base)
        per_seed.append({"seed": seed, **_seed_metrics(oof_post, oof_prior, Yf, cbi, per_fold)})
        Yf0, cbi0 = Yf, cbi

    # The region-average baseline: each subregion gets the regional total over the count.
    scored = np.isfinite(Yf0); inhab = Yf0 > 0
    N   = np.bincount(cbi0); Yblk = np.bincount(cbi0, Yf0)
    bc  = Yblk[cbi0] / np.clip(N[cbi0], 1, None)
    agg = {k: _ms([s[k] for s in per_seed]) for k in SEED_METRICS}

    res = {
        "region_m": region_m, "subregion_m": sub_m, "n_side": n, "factor": n,
        "n_cells": int(len(Yf0)), "n_inhabited": int(inhab.sum()),
        "frac_inhabited": round(float(inhab.mean()), 4),
        "seeds": seeds_used, "K": len(seeds_used),
        # region-average baseline, which scores 0 within-region by construction
        "bc_cell_all":   round(_r2(Yf0[scored], bc[scored]), 4),
        "bc_cell_inhab": round(_r2(Yf0[scored & inhab], bc[scored & inhab]), 4),
        # posterior subregion-R2, mean +/- std over seeds
        "puffin_cell_all": round(agg["cell_r2_post_all"][0], 4),   "puffin_cell_all_std": round(agg["cell_r2_post_all"][1], 4),
        "puffin_cell_inhab": round(agg["cell_r2_post_inhab"][0], 4), "puffin_cell_inhab_std": round(agg["cell_r2_post_inhab"][1], 4),
        # prior subregion-R2
        "prior_cell_all": round(agg["cell_r2_prior_all"][0], 4),   "prior_cell_all_std": round(agg["cell_r2_prior_all"][1], 4),
        "prior_cell_inhab": round(agg["cell_r2_prior_inhab"][0], 4), "prior_cell_inhab_std": round(agg["cell_r2_prior_inhab"][1], 4),
        # within-region R2 of the posterior
        "within_all": round(agg["within_post_all"][0], 4),   "within_all_std": round(agg["within_post_all"][1], 4),
        "within_inhab": round(agg["within_post_inhab"][0], 4), "within_inhab_std": round(agg["within_post_inhab"][1], 4),
        # within-region R2 (prior)
        "within_prior_all": round(agg["within_prior_all"][0], 4),   "within_prior_all_std": round(agg["within_prior_all"][1], 4),
        "within_prior_inhab": round(agg["within_prior_inhab"][0], 4), "within_prior_inhab_std": round(agg["within_prior_inhab"][1], 4),
        # conformal / calibration (posterior) + NLL
        "cal_cell_90": round(agg["cal_cell_90"][0], 4),   "cal_cell_90_std": round(agg["cal_cell_90"][1], 4),
        "cal_blk_90": round(agg["cal_blk_90"][0], 4),     "cal_blk_90_std": round(agg["cal_blk_90"][1], 4),
        "nb_dispersion_r": round(agg["nb_dispersion_r"][0], 3), "nb_dispersion_r_std": round(agg["nb_dispersion_r"][1], 3),
        "nll_blk_prior": round(agg["nll_blk_prior"][0], 4), "nll_blk_prior_std": round(agg["nll_blk_prior"][1], 4),
        "nll_cell_post": round(agg["nll_cell_post"][0], 4), "nll_cell_post_std": round(agg["nll_cell_post"][1], 4),
        "mean_pred_on_zero_cells": round(agg["mean_pred_on_zero_cells"][0], 4),
        "mean_pred_on_zero_cells_std": round(agg["mean_pred_on_zero_cells"][1], 4),
        "per_seed": per_seed,
    }
    base.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(res, indent=2))
    print(f"    -> within all={res['within_all']:+.3f}+-{res['within_all_std']:.3f}  "
          f"inhab={res['within_inhab']:+.3f}+-{res['within_inhab_std']:.3f}  "
          f"prior(all)={res['within_prior_all']:+.3f}  "
          f"cell(all)={res['puffin_cell_all']:.3f}  "
          f"cov90={res['cal_cell_90']:.2f}  pred@0={res['mean_pred_on_zero_cells']:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(f"Berlin ALL-CELLS crossfit: {len(GRID)} configs x 5 folds x {len(SEEDS)} seed(s)\n")
    if a.dry_run:
        for rm, sm in GRID:
            print(f"  {rm}m/{sm}m ({rm//sm}x{rm//sm})  factor={rm//sm}")
        return
    rows = [crossfit_cell(rm, sm) for rm, sm in GRID]
    print(f"\n{'='*104}\n  BERLIN ALL-CELLS CROSSFIT  (within-region R2, mean+-std over {len(SEEDS)} init seeds)\n{'-'*104}")
    print(f"  {'region':>7} {'subreg':>7} {'NxN':>7} {'%inhab':>7} "
          f"{'within(all)':>16} {'within(inhab)':>16} {'prior(all)':>12} {'cell(all)':>10} {'cov90':>6}")
    for e in rows:
        nxn = f"{e['n_side']}x{e['n_side']}"
        print(f"  {str(e['region_m'])+'m':>7} {str(e['subregion_m'])+'m':>7} {nxn:>7} "
              f"{e['frac_inhabited']*100:>6.0f}% "
              f"{e['within_all']:>+8.3f}+-{e['within_all_std']:<5.3f} "
              f"{e['within_inhab']:>+8.3f}+-{e['within_inhab_std']:<5.3f} "
              f"{e['within_prior_all']:>+12.3f} {e['puffin_cell_all']:>10.3f} {e['cal_cell_90']:>6.2f}")
    _msuf = METHOD
    (EXPERIMENTS_DIR.parent / f"berlin_allcells_{_msuf}_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  saved -> experiments/berlin_allcells_{_msuf}_summary.json\nDONE_BERLIN_ALL")


if __name__ == "__main__":
    main()
