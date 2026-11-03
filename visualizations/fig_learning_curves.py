"""
CLAM-style learning curves (cf. Grossmann et al., 2026, Fig. 3 "Loss Curve and MSE"):
the aggregate NLL the model is trained on (left axis) against the fine-scale error of
its posterior split on held-out TEST subregions (right axis), over log-scaled epochs.
Lines are the mean over seeds, bands the min-max range.

    python visualizations/fig_learning_curves.py
Reads  experiments/runs/learning_curves/<setting>/seed_*/learning_curve.csv
Writes experiments/runs/figures/
    fig_lc_<setting>.{png,pdf}  one panel per setting
    fig_lc_main.{png,pdf}       the two main-text settings, side by side
    fig_lc_appendix.{png,pdf}   the remaining settings, one row (appendix)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from puffin.config import EXPERIMENTS_DIR, FIGURES_DIR

IN = EXPERIMENTS_DIR / "learning_curves"
OUT = FIGURES_DIR
SETTINGS = [
    ("berlin_1km_100m",           "Berlin, 1 km $\\to$ 100 m"),
    ("berlin_10km_1km",           "Berlin, 10 km $\\to$ 1 km"),
    ("germany_4km_400m",          "Germany, 4 km $\\to$ 400 m"),
    ("germany_10km_1km",          "Germany, 10 km $\\to$ 1 km"),
    ("us_population_tract_block", "US population"),
]
LOSS_C, ERR_C = "#1f5fa8", "#c0392b"
MAIN = ["us_population_tract_block", "berlin_10km_1km"]   # main-text figure; the appendix holds the rest


def load(name):
    runs = [pd.read_csv(f) for f in sorted((IN / name).glob("seed_*/learning_curve.csv"))]
    if not runs:
        return None, 0
    df = pd.concat(runs, keys=range(len(runs)), names=["seed"]).reset_index(level=0)
    return df.groupby("epoch").agg(["mean", "min", "max"]), len(runs)


def panel(ax, st, k, title, legend, fs=1.0):
    ep = st.index.values
    ax.plot(ep, st[("train_nll", "mean")], color=LOSS_C, lw=1.6, label="training loss (aggregate NLL)")
    ax.fill_between(ep, st[("train_nll", "min")], st[("train_nll", "max")], color=LOSS_C, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch", fontsize=9 * fs)
    ax.set_ylabel("Training loss", color=LOSS_C, fontsize=9 * fs)
    ax.tick_params(axis="y", colors=LOSS_C, labelsize=7.5 * fs)
    ax.tick_params(axis="x", labelsize=7.5 * fs)
    ax2 = ax.twinx()
    ax2.plot(ep, st[("post_mae", "mean")], color=ERR_C, lw=1.6, ls="--", label="fine-scale MAE (test)")
    ax2.fill_between(ep, st[("post_mae", "min")], st[("post_mae", "max")], color=ERR_C, alpha=0.15, lw=0)
    ax2.set_ylabel("Fine-scale MAE", color=ERR_C, fontsize=8 * fs)
    ax2.tick_params(axis="y", colors=ERR_C, labelsize=7.5 * fs)
    ax.set_title(title, fontsize=9 * fs, pad=6)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if legend:
        ax.legend(h1 + h2, l1 + l2, fontsize=7 * fs, loc="upper right", frameon=False)
    return h1 + h2, l1 + l2


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved ->", OUT / f"{stem}.png")


def main():
    avail = [(n, t, *load(n)) for n, t in SETTINGS]
    avail = [a for a in avail if a[2] is not None]
    if not avail:
        sys.exit(f"no learning curves under {IN}")

    for name, title, st, k in avail:                       # one panel each
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        panel(ax, st, k, title, legend=True)
        save(fig, f"fig_lc_{name}")

    main = [x for x in avail if x[0] in MAIN]            # main text: side by side, no wasted width
    if main:
        fig, axes = plt.subplots(1, len(main), figsize=(3.6 * len(main), 2.4), squeeze=False)
        for i, (name, title, st, k) in enumerate(sorted(main, key=lambda x: MAIN.index(x[0]))):
            panel(axes.flat[i], st, k, title, legend=(i == 0))
        fig.tight_layout(w_pad=2.0)
        save(fig, "fig_lc_main")

    # appendix: everything else in ONE row. A grid would leave an empty cell and, because
    # the figure is printed at a fixed \linewidth, a squarer canvas is scaled up more and
    # the labels come out larger than in the main-text figure.
    rest = [x for x in avail if x[0] not in MAIN]
    n = len(rest)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.5), squeeze=False)
    handles = labels = None
    for i, (name, title, st, k) in enumerate(rest):
        handles, labels = panel(axes.flat[i], st, k, title, legend=False, fs=1.13)
    # one shared legend under the row: the two entries together are wider than a single
    # panel, so inside a panel it would sit on the curves and the tick labels
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7 * 1.13,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.06, 1, 1))
    save(fig, "fig_lc_appendix")

    for name, _, st, k in avail:
        a, b = st[("train_nll", "mean")].values, st[("post_mae", "mean")].values
        w = st[("post_within_r2", "mean")]
        print(f"{name:28s} seeds={k}  corr(loss, test MAE)={np.corrcoef(a, b)[0, 1]:+.3f}  "
              f"within first/last/best = {w.iloc[0]:+.3f} / {w.iloc[-1]:+.3f} / {w.max():+.3f} (epoch {w.idxmax()})")


if __name__ == "__main__":
    main()
