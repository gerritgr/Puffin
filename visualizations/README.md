# Figures

Each script draws one map and writes it to `experiments/figures/` as `.png` and
`.pdf`. Every script is self-contained, with a `CONFIG` block at the top (dataset,
target, region and subregion size, view window) that you edit before running it:

```bash
python visualizations/fig_disagg.py
```

| Script | What it shows | Output |
|---|---|---|
| `fig_disagg.py` | The region average, the Puffin estimate and the ground truth side by side, on the German grid | `fig_disagg.{png,pdf}` |
| `fig_prob.py` | The Puffin estimate beside its calibrated per-subregion standard deviation | `fig_prob.{png,pdf}` |
| `fig_us_pop_map.py` | The same comparison on US census blocks, for the target named by `FIG_TARGET` (`population`, `jobs`, ...) | `fig_us_<target>_<view>_value.{png,pdf}` |

The maps stitch the out-of-fold predictions, taking each subregion from the fold
that held it out and averaging over seeds, so they show exactly what the tables
score. They read `experiments/runs/<tag>/nll/seed_<s>/crossfit_test<t>/predictions.npz`,
so run the experiment first: `run_berlin_all_crossfit.py` for the German maps and
`main_us.py` for the US maps, with `save_predictions` left on.

The scripts import the package from `../src`, so they can be run from anywhere in
the repository.
