"""Run Puffin on the US data for ONE configuration.

The fine unit is fixed by the dataset (a census block, or a precinct for the
voting data), so a configuration is a (dataset, target, coarse level) triple
rather than a grid. Set it below and run:

    python main_us.py

Build the data first with setup_us_pop.py or setup_us_voting.py. Results are
merged into experiments/<dataset>_<method>_summary.json, one row per
(target, level), and each (target, seed) is cached so a re-run resumes.
To reproduce a whole table, use the sweep wrapper run_voting.py, which calls
this same runner for every target.

    DATASET = "us_population"   fine unit = census block
        TARGETS  population | housing | jobs | gun_violence
        LEVELS   {"tract": "tract_fips"} | {"county": "county_fips"}
        EXPOSURE_COL = "exposure"   irregular blocks need the area term

    DATASET = "us_voting"       fine unit = precinct
        TARGETS  votes_rep | votes_dem | votes_other | votes_total
        LEVELS   {"county": "county_fips"} | {"state": "state_fips"}
        EXPOSURE_COL = ""           precincts hold similar numbers of voters,
                                    so the area term makes the split worse
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from puffin.config import EXPERIMENTS_DIR
from puffin.runner import PuffinConfig, _r2, run as runner_run

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET = "us_population"           # "us_population" | "us_voting"
TARGETS = ["population", "jobs", "housing", "gun_violence"]
LEVELS = {"tract": "tract_fips"}    # coarse level(s) to disaggregate from
EXPOSURE_COL = "exposure"           # "exposure" for us_population, "" for us_voting

LOSS = "nll"
METHOD = "gaussian"                 # "gaussian" (the paper) | "poisson" | "gamma"
EPOCHS = 2000
SEEDS = [0, 1, 2, 3, 4]
SUPER_BLOCK_M = 5000                # size of the spatial groups assigned to folds
SAVE_PREDS = True                   # write per-subregion predictions for the figures

# =============================================================================


def _within(y, p, cbi, mask):
    """Within-region R^2: truth and prediction are each centred on their own
    region mean, so a constant split scores exactly zero."""
    m = mask & np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return float("nan")
    _, inv = np.unique(cbi[m], return_inverse=True)
    ym = np.bincount(inv, y[m]) / np.bincount(inv)
    pm = np.bincount(inv, p[m]) / np.bincount(inv)
    yw, pw = y[m] - ym[inv], p[m] - pm[inv]
    return float(1 - ((yw - pw) ** 2).sum() / (((yw - yw.mean()) ** 2).sum()))


def _num(v):
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _wavg(vals, w):
    v, w = np.array(vals, float), np.array(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")


def _mean_std(vals):
    a = np.array([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


METRICS = ["within_all", "within_inhab", "cell_r2_all", "cell_r2_inhab", "cov",
           "within_prior_all", "within_prior_inhab",
           "cell_r2_prior_all", "cell_r2_prior_inhab"]


def _one_seed(target, level_col, seed, base):
    """One 5-fold out-of-fold pass at a given seed. The prior and the posterior
    are stitched across folds the same way, so the two are comparable."""
    oof = oof_prior = Yf = cbi = None
    per_fold = []
    for t in range(5):
        cfg = PuffinConfig(
            region=DATASET, target=target, loss=LOSS, epochs=EPOCHS, seed=seed,
            split_seed=0, fixed_region=True, region_col=level_col,
            exposure_col=EXPOSURE_COL, super_block_m=SUPER_BLOCK_M,
            save_predictions=SAVE_PREDS, method=METHOD,
            val_fold=(t - 1) % 5, test_fold=t,
            run_dir=base / f"seed_{seed}" / f"crossfit_test{t}",
        )
        out = runner_run(cfg)
        sys.stdout = sys.__stdout__
        mu, mu_prior = out["mu_post"], out["mu_prior"]
        if oof is None:
            oof = np.full(len(mu), np.nan)
            oof_prior = np.full(len(mu), np.nan)
            Yf, cbi = out["Y_fine"], out["cell_blk_idx"]
        tcm = out["test_cell_mask"]
        oof[tcm], oof_prior[tcm] = mu[tcm], mu_prior[tcm]
        per_fold.append({"cov": _num(out["results"].get("cal_cell_90")), "n": int(tcm.sum())})

    scored = np.isfinite(oof)
    inhabited = Yf > 0
    m_inh = scored & inhabited
    N = np.bincount(cbi)
    region_avg = np.bincount(cbi, Yf)[cbi] / np.clip(N[cbi], 1, None)
    return {
        "within_all": _within(Yf, oof, cbi, scored),
        "within_inhab": _within(Yf, oof, cbi, m_inh),
        "within_prior_all": _within(Yf, oof_prior, cbi, scored),
        "within_prior_inhab": _within(Yf, oof_prior, cbi, m_inh),
        "cell_r2_all": _r2(Yf[scored], oof[scored]),
        "cell_r2_inhab": _r2(Yf[m_inh], oof[m_inh]),
        "cell_r2_prior_all": _r2(Yf[scored], oof_prior[scored]),
        "cell_r2_prior_inhab": _r2(Yf[m_inh], oof_prior[m_inh]),
        "bc_cell_all": _r2(Yf[scored], region_avg[scored]),
        "bc_cell_inhab": _r2(Yf[m_inh], region_avg[m_inh]),
        "cov": _wavg([f["cov"] for f in per_fold], [f["n"] for f in per_fold]),
        "n_blocks": int(N.size), "n_cells": int(len(Yf)), "n_inhab": int(inhabited.sum()),
    }


def main() -> None:
    summary_path = EXPERIMENTS_DIR.parent / f"{DATASET}_{METHOD}_summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    rows = {(r["target"], r["level"]): r for r in existing if "target" in r}

    for target in TARGETS:
        for level_name, level_col in LEVELS.items():
            base = EXPERIMENTS_DIR / f"{DATASET}_{target}_{level_name}_{METHOD}" / LOSS
            print(f"\n{DATASET} | {target} | {level_name} -> fine | "
                  f"5-fold out-of-fold x {len(SEEDS)} seeds", flush=True)
            per_seed = []
            for s in SEEDS:
                cache = base / f"seed_{s}" / "seed_summary.json"
                if cache.exists():
                    cached = json.loads(cache.read_text())
                    if "within_prior_all" in cached:
                        print(f"  seed {s}: cached", flush=True)
                        per_seed.append(cached)
                        continue
                    print(f"  seed {s}: cached without the prior, recomputing", flush=True)
                r = _one_seed(target, level_col, s, base)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(r))
                per_seed.append(r)

            agg = {k: _mean_std([p.get(k, float("nan")) for p in per_seed]) for k in METRICS}
            row = {
                "target": target, "level": level_name,
                "n_regions": per_seed[0]["n_blocks"], "n_cells": per_seed[0]["n_cells"],
                "n_inhab": per_seed[0]["n_inhab"],
                "within_all": round(agg["within_all"][0], 4),
                "within_all_std": round(agg["within_all"][1], 4),
                "within_inhab": round(agg["within_inhab"][0], 4),
                "within_inhab_std": round(agg["within_inhab"][1], 4),
                "cell_r2_all": round(agg["cell_r2_all"][0], 4),
                "cell_r2_inhab": round(agg["cell_r2_inhab"][0], 4),
                "bc_cell_all": round(per_seed[0]["bc_cell_all"], 4),
                "bc_cell_inhab": round(per_seed[0]["bc_cell_inhab"], 4),
                "cal_cell_90": round(agg["cov"][0], 4),
                "cal_cell_90_std": round(agg["cov"][1], 4),
                "within_prior_all": round(agg["within_prior_all"][0], 4),
                "within_prior_all_std": round(agg["within_prior_all"][1], 4),
                "within_prior_inhab": round(agg["within_prior_inhab"][0], 4),
                "cell_r2_prior_all": round(agg["cell_r2_prior_all"][0], 4),
                "cell_r2_prior_inhab": round(agg["cell_r2_prior_inhab"][0], 4),
                "seeds": SEEDS,
            }
            rows[(target, level_name)] = row
            print(f"  -> within all={row['within_all']:+.3f}+-{row['within_all_std']:.3f}  "
                  f"inhab={row['within_inhab']:+.3f}  cov {row['cal_cell_90']:.2f}  "
                  f"prior(all)={row['within_prior_all']:+.3f}", flush=True)

    out = sorted(rows.values(), key=lambda r: (r["target"], r["level"]))
    print(f"\n{'='*84}\n  {DATASET.upper()}  within-region R2, mean +/- std over "
          f"{len(SEEDS)} seeds\n{'-'*84}")
    print(f"  {'target':>14} {'level':>7} {'regions':>9} {'within(all)':>14} "
          f"{'within(inhab)':>14} {'cov':>5}")
    for e in out:
        print(f"  {e['target']:>14} {e['level']:>7} {e['n_regions']:>9,} "
              f"{e['within_all']:>+8.3f}+-{e['within_all_std']:<4.3f} "
              f"{e['within_inhab']:>+8.3f}+-{e['within_inhab_std']:<4.3f} "
              f"{e['cal_cell_90']:>5.2f}")
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved -> {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
