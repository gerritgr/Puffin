<p align="center">
  <img src="logo_color.jpg" alt="Puffin logo" width="100">
</p>

# Puffin: Probabilistic Learning of Spatial Detail From Coarse Observations

Probabilistic spatial disaggregation of socioeconomic counts (population, jobs, votes, ...) from coarse administrative totals to fine subregions, using AlphaEarth Foundation embeddings as the only input. No fine-scale labels are used for training. For every subregion Puffin returns a mean and a calibrated variance; the estimates are non-negative and add back up to each observed total exactly.

---

## What This Project Does

Most socioeconomic data (population, jobs, votes) is published only as one total per coarse region, such as a census tract or county. Satellite embeddings exist everywhere at 10 m. Puffin turns each coarse total into a fine map:

1. **Input:** a 64-dim AlphaEarth embedding for every subregion (mean-pooled over a 100 m grid cell or over a census-block polygon).
2. **Supervision:** only the observed total of each coarse region; the fine values are never seen.
3. **Model:** a small per-subregion network predicts a Gaussian over each subregion's count (a mean and a variance), trained so the implied region totals match the observed ones.
4. **Split:** at test time the observed total is split in closed form by conditioning that Gaussian on the total, giving estimates that are non-negative, conserve the total exactly, and come with calibrated per-subregion intervals.

The network is deliberately compact (`64 -> 32 -> 2`, about 2,000 parameters); a larger one over-fits the coarse totals. All results are scored out of fold with **spatial cross-validation**.

**Entry points.** `main_germany.py` and `main_us.py` each run a single configuration, which you set in the CONFIG block at the top of the file. The `run_*.py` scripts are sweeps over many configurations that call the same runner, and are what reproduce the tables below.

---

## Quickstart

```bash
git clone https://github.com/gerritgr/Puffin.git && cd Puffin
uv sync

# 1. Build cells.parquet (one row per subregion: 64 AlphaEarth dims + labels)
python setup_germany.py        # Zensus 2022 100 m grid (Berlin or all of Germany)
python setup_us_pop.py         # US 2020 census blocks: population, jobs, gun violence
python setup_us_voting.py      # US 2024 presidential precincts

# 2. Run Puffin on ONE configuration: set the CONFIG block at the top of the file, then
python main_germany.py         # German grid: pick REGION, REGION_M, SUBREGION_M
python main_us.py              # US: pick DATASET, TARGETS, LEVELS

# 3. Baselines: region average, bicubic, Goodman (same folds and metric, no fine labels)
python baselines.py --suite berlin | germany | us | voting
```

To reproduce a whole table rather than one configuration, use the sweep wrappers.
They call the same runner once per configuration and write one summary file:

```bash
python run_berlin_all_crossfit.py                         # Berlin, all 18 configurations
python run_germany_all_crossfit.py [--nshard N --shard k] # Germany, shardable across GPUs
python run_voting.py                                      # US county -> precinct, per party
```

Two further experiments reproduce the paper's appendices, and the learning-curve figure:

```bash
python run_area_ablation.py    # ablate the area term on the prior variance (US pop + jobs)
python kappa_transfer.py       # does the conformal factor kappa transfer between settings?
python run_learning_curves.py  # aggregate NLL vs held-out fine-scale error, over epochs
```

Summaries are written to `experiments/<dataset>_<method>_summary.json` (Puffin) and `experiments/baselines_<suite>_summary.json` (baselines); per-run logs, checkpoints, and per-subregion predictions go to `experiments/runs/`, and the figure scripts write to `experiments/figures/`. Only the summaries are tracked, since everything else is reproducible from them.

**Region vs subregion.** The *region* is the coarse unit whose total is observed; the *subregion* is the unit predicted. On the German grid a region of `REGION_M` metres is split into `(REGION_M / SUBREGION_M)²` subregions (e.g. `1000/200` = a 1 km region split into 5×5 subregions of 200 m); subregion labels are sums of the 100 m Zensus cells, and embeddings are mean-pooled. In the US the regions and subregions are fixed census units (tract → block, county → precinct).

---

## Project Layout

```
Puffin/
  setup_germany.py             # Zensus 2022 100 m grid + AlphaEarth -> cells.parquet
  setup_us_pop.py              # US census blocks (population, jobs, gun violence) -> cells.parquet
  setup_us_voting.py           # US 2024 precincts (votes per party) -> cells.parquet
  main_germany.py              # ENTRY POINT: one configuration on the German grid
  main_us.py                   # ENTRY POINT: one configuration on the US data
  run_berlin_all_crossfit.py   # sweep: every configuration on Berlin
  run_germany_all_crossfit.py  # sweep: every configuration on Germany, shardable across GPUs
  run_voting.py                # sweep: every party, US county -> precinct
  baselines.py                 # region average, bicubic, Goodman on identical folds
  run_area_ablation.py         # ablation: area term on the prior variance
  kappa_transfer.py            # ablation: does the conformal factor transfer?
  run_learning_curves.py       # diagnostic: training loss vs held-out fine-scale error
  visualizations/              # maps of the disaggregation, its uncertainty, learning curves

  src/puffin/
    config.py                  # paths, AlphaEarth columns, COUNT_TARGETS
    data.py                    # cells, regions, subregion pooling, spatial CV splits
    model.py                   # per-subregion head: 64-dim embedding -> (mean, variance)
    runner.py                  # training, closed-form split, calibration, evaluation
    posterior.py               # Gaussian conditional split + non-negativity rescaling
    calibrate.py               # two-stage (tile scale + conformal) interval calibration
    losses.py, metrics.py      # aggregate likelihoods; R² and coverage
    datasets/                  # US dataset builders

  data/processed/<dataset>/cells.parquet
  experiments/                 # *_summary.json, runs/<experiment>/..., figures/
```

---

## Probabilistic Model

Notation follows the paper: a coarse **region** *i* has an observed total `Yhat_i`; its **subregions** *j = 1..M_i* each carry a 64-dim embedding `c_ij` and an area `a_ij` (constant on a uniform grid, the block area in the US). The fine values `y_ij` are never observed during training.

### Per-subregion prediction

A small head `f_theta = Linear(64 -> 32) -> GELU -> Linear(32 -> 2)` (input standardized with training statistics) predicts a Gaussian over each subregion's count. Both the mean and the variance are densities, so both scale with area:

```
(u_ij, v_ij)  = f_theta(c_ij)                        mean and variance per unit area
mu_ij         = a_ij * u_ij                          subregion mean
sigma2_ij     = a_ij * (softplus(v_ij) + eps)        subregion variance (> 0), eps = 1e-6
```

On a uniform grid the area is constant and cancels. On irregular units it is what
makes the split work: scaling the variance with the area lets a large block absorb
a correspondingly larger share of the region's correction, and without it large
blocks take far too little of it.

### Training on totals

Counts add, so under independent subregion Gaussians each region total is itself Gaussian. The network is fit by the Gaussian negative log-likelihood of the observed totals over the training regions `T`:

```
mu_i = sum_j mu_ij           sigma2_i = sum_j sigma2_ij
L    = (1/|T|) sum_{i in T} [ (Yhat_i - mu_i)^2 / sigma2_i + log sigma2_i ]
```

up to an additive constant and an overall positive factor. Only the prior is fitted
here; the posterior below is never trained through.

Totals are standardized by the mean training density for optimization, and predictions are mapped back to counts. AdamW with a cosine-annealed learning rate (peak `3e-3`, 2,000 epochs); the checkpoint is chosen by the aggregate NLL on the validation fold. Large datasets train on minibatches of regions.

### Splitting the total

At test time the observed total is split by conditioning the region's joint Gaussian on it:

```
y_ij_MAP     = mu_ij + (sigma2_ij / sigma2_i) * (Yhat_i - mu_i)     posterior mean
sigma2~_ij   = sigma2_ij - sigma2_ij^2 / sigma2_i                   posterior variance
```

Each subregion keeps its predicted mean and absorbs a share of the region's residual in proportion to its variance, so confident subregions barely move and uncertain ones absorb most of the correction. These values already sum to `Yhat_i`; to also keep them non-negative, negative ones are clipped to 0 and the region is rescaled back to `Yhat_i` (an even split if a whole region vanishes). The result is written `mu~_ij` and called the **Puffin estimate**, since after clipping it is no longer exactly the posterior mean. It is what every map and table reports, while the posterior variances give the intervals.

### Calibration

The posterior variances are too small on their own and vary across space, so intervals are calibrated in two leak-free stages:

```
Stage 1 (coarse only):  gamma2_t = sum_{i in t} (Yhat_i - mu_i)^2 / sum_{i in t} sigma2_i   per 10 km tile t
                        sigma2bar_ij = gamma2_t(i) * sigma2~_ij
Stage 2 (val fold):     kappa = 90% quantile of |y_ij - mu~_ij| / sigmabar_ij
Interval:               mu~_ij +/- kappa * sigmabar_ij
```

Stage 1 uses only region totals, so it adapts to a new area without fine labels. Stage 2 is split conformal: `kappa` is a single global factor and is the one part that does use fine labels, on validation regions only, never on the test fold.

### Metrics

- **Within-region R²** (headline): each region's mean is removed from both the truth and the estimate before computing R², so it measures only how well each total is split. The region-average baseline scores exactly 0 by construction.
- **Subregion-R²**: plain R² over all subregions against the global mean, for comparison with other products. It includes the between-region signal that the observed total gives every method for free, so even the region average scores high.
- **All subregions** includes empty land (forest, water) labelled 0; **inhabited** keeps subregions with a true count above 0.

---

## Results

**Protocol.** 5-fold spatial cross-validation: regions are grouped into 5 km super-blocks, whole super-blocks go to one fold, and for test fold *t* fold *t−1* is used for validation and the remaining three for training. Every subregion is scored exactly once, out of fold. Puffin is reported as a **5-seed mean** (the split is fixed; only initialization varies). The baselines are deterministic and use the identical subregions, regions, and folds. **Best value per group in bold.**

**Methods compared.** All four see the same inputs, and all reproduce every region total exactly. No point estimate uses fine labels; only Puffin's interval calibration factor uses them, on the validation fold.

- **Region-average:** each total split uniformly over the region's subregions.
- **Bicubic interpolation** (spatial smoothness, no embeddings): the region-density field is interpolated with a cubic spline at every subregion centroid, clipped at 0, multiplied by area, and rescaled to each total. Bicubic B-splines over region centres on the grids; Clough–Tocher cubic interpolation over region centroids for irregular US units.
- **Goodman-style ecological regression** (a linear model on the same embeddings and the same totals): ridge regression of region density on the area-weighted mean embedding, fit on training regions, with the ridge strength chosen from validation-region totals; the fitted density times area is clipped at 0 and rescaled to each total.
- **Puffin:** the probabilistic model above.

**Checks.** The baselines' folds are identical to Puffin's, and the region-average scores within-region R² = 0.000 while reproducing Puffin's own region-average subregion-R² in all 43 dataset/config rows. Goodman is not leaking: refitting it with the training-region totals shuffled drops its within-region R² to ≤ 0 (checked on three Berlin configurations).

### Berlin

Every 100 m cell in the Berlin bounding box (102 × 96 km): 979,200 cells, of which 92,724 are inhabited and 886,476 hold a count of 0. No land/water mask is applied - every cell with an AlphaEarth embedding is kept. 18 configurations: regions of 200 m to 10 km, split into subregions of 100 m and up.

#### Within-region R²

| region / subregion | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| 200 m / 100 m | +0.000 | +0.114 | **+0.173** | +0.123 | | +0.000 | +0.059 | **+0.101** | +0.088 |
| 1,000 m / 100 m | +0.000 | +0.062 | **+0.284** | +0.280 | | +0.000 | +0.032 | +0.162 | **+0.163** |
| 1,000 m / 200 m | +0.000 | +0.110 | **+0.405** | +0.384 | | +0.000 | +0.075 | **+0.332** | +0.313 |
| 1,000 m / 500 m | +0.000 | +0.212 | **+0.494** | +0.409 | | +0.000 | +0.191 | **+0.472** | +0.387 |
| 2,000 m / 100 m | +0.000 | +0.042 | +0.279 | **+0.299** | | +0.000 | +0.026 | +0.182 | **+0.189** |
| 2,000 m / 200 m | +0.000 | +0.070 | +0.384 | **+0.415** | | +0.000 | +0.051 | +0.322 | **+0.341** |
| 2,000 m / 400 m | +0.000 | +0.109 | +0.463 | **+0.485** | | +0.000 | +0.088 | +0.428 | **+0.443** |
| 2,000 m / 1,000 m | +0.000 | +0.170 | **+0.528** | +0.492 | | +0.000 | +0.164 | **+0.520** | +0.479 |
| 4,000 m / 200 m | +0.000 | +0.050 | +0.360 | **+0.392** | | +0.000 | +0.042 | +0.311 | **+0.334** |
| 4,000 m / 400 m | +0.000 | +0.075 | +0.438 | **+0.471** | | +0.000 | +0.069 | +0.406 | **+0.435** |
| 4,000 m / 800 m | +0.000 | +0.114 | +0.501 | **+0.507** | | +0.000 | +0.110 | +0.482 | **+0.489** |
| 4,000 m / 2,000 m | +0.000 | +0.195 | **+0.572** | +0.536 | | +0.000 | +0.192 | **+0.568** | +0.531 |
| 10,000 m / 100 m | +0.000 | +0.053 | **+0.291** | +0.282 | | +0.000 | +0.073 | **+0.261** | +0.250 |
| 10,000 m / 200 m | +0.000 | +0.079 | **+0.393** | +0.379 | | +0.000 | +0.092 | **+0.367** | +0.350 |
| 10,000 m / 500 m | +0.000 | +0.124 | **+0.501** | +0.453 | | +0.000 | +0.126 | **+0.472** | +0.415 |
| 10,000 m / 1,000 m | +0.000 | +0.178 | **+0.588** | +0.515 | | +0.000 | +0.176 | **+0.573** | +0.493 |
| 10,000 m / 2,000 m | +0.000 | +0.271 | **+0.695** | +0.583 | | +0.000 | +0.274 | **+0.694** | +0.581 |
| 10,000 m / 5,000 m | +0.000 | +0.536 | **+0.710** | +0.682 | | +0.000 | +0.536 | **+0.710** | +0.682 |

Mean over the 18 configs (all cells): region-average 0.000, bicubic 0.142, Goodman **0.448**, Puffin 0.427 (Puffin beats bicubic on 18/18 and Goodman on 6/18). Puffin std over seeds ≤ 0.024.

#### Subregion-R²

| region / subregion | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| 200 m / 100 m | 0.753 | 0.781 | **0.796** | 0.784 | | 0.715 | 0.741 | **0.756** | 0.745 |
| 1,000 m / 100 m | 0.468 | 0.501 | **0.619** | 0.617 | | 0.410 | 0.444 | **0.557** | 0.557 |
| 1,000 m / 200 m | 0.622 | 0.663 | **0.775** | 0.767 | | 0.581 | 0.621 | **0.734** | 0.727 |
| 1,000 m / 500 m | 0.823 | 0.861 | **0.911** | 0.896 | | 0.801 | 0.841 | **0.897** | 0.880 |
| 2,000 m / 100 m | 0.385 | 0.411 | 0.557 | **0.569** | | 0.310 | 0.339 | 0.499 | **0.511** |
| 2,000 m / 200 m | 0.512 | 0.546 | 0.699 | **0.714** | | 0.466 | 0.500 | 0.661 | **0.675** |
| 2,000 m / 400 m | 0.638 | 0.678 | 0.806 | **0.814** | | 0.607 | 0.645 | 0.782 | **0.789** |
| 2,000 m / 1,000 m | 0.823 | 0.853 | **0.916** | 0.910 | | 0.810 | 0.842 | **0.909** | 0.902 |
| 4,000 m / 200 m | 0.421 | 0.450 | 0.630 | **0.648** | | 0.367 | 0.399 | 0.597 | **0.614** |
| 4,000 m / 400 m | 0.525 | 0.561 | 0.733 | **0.749** | | 0.490 | 0.527 | 0.707 | **0.722** |
| 4,000 m / 800 m | 0.637 | 0.679 | 0.819 | **0.821** | | 0.612 | 0.656 | 0.801 | **0.804** |
| 4,000 m / 2,000 m | 0.823 | 0.858 | **0.924** | 0.918 | | 0.819 | 0.854 | **0.922** | 0.915 |
| 10,000 m / 100 m | 0.247 | 0.287 | **0.466** | 0.460 | | 0.103 | 0.168 | **0.409** | 0.401 |
| 10,000 m / 200 m | 0.328 | 0.381 | **0.592** | 0.582 | | 0.246 | 0.317 | **0.561** | 0.551 |
| 10,000 m / 500 m | 0.434 | 0.504 | **0.718** | 0.690 | | 0.396 | 0.474 | **0.692** | 0.659 |
| 10,000 m / 1,000 m | 0.527 | 0.611 | **0.805** | 0.771 | | 0.506 | 0.594 | **0.791** | 0.752 |
| 10,000 m / 2,000 m | 0.640 | 0.738 | **0.890** | 0.850 | | 0.632 | 0.733 | **0.887** | 0.846 |
| 10,000 m / 5,000 m | 0.829 | 0.920 | **0.950** | 0.946 | | 0.828 | 0.920 | **0.950** | 0.945 |

Mean over the 18 configs (all cells): region-average 0.580, bicubic 0.627, Goodman **0.756**, Puffin 0.750. Puffin std over seeds ≤ 0.011.

### Germany

Every 100 m cell in Germany, clipped to the national border: 35,722,205 cells, of which 3,081,336 are inhabited. Same 18 configurations as Berlin, about 36× more training data.

#### Within-region R²

| region / subregion | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| 200 m / 100 m | +0.000 | +0.141 | **+0.187** | +0.179 | | +0.000 | +0.086 | **+0.115** | +0.112 |
| 1,000 m / 100 m | +0.000 | +0.081 | +0.337 | **+0.346** | | +0.000 | +0.032 | **+0.163** | +0.163 |
| 1,000 m / 200 m | +0.000 | +0.132 | +0.479 | **+0.479** | | +0.000 | +0.084 | **+0.373** | +0.366 |
| 1,000 m / 500 m | +0.000 | +0.226 | **+0.622** | +0.580 | | +0.000 | +0.204 | **+0.591** | +0.540 |
| 2,000 m / 100 m | +0.000 | +0.046 | +0.358 | **+0.373** | | +0.000 | +0.025 | +0.177 | **+0.180** |
| 2,000 m / 200 m | +0.000 | +0.070 | +0.488 | **+0.514** | | +0.000 | +0.047 | +0.370 | **+0.389** |
| 2,000 m / 400 m | +0.000 | +0.103 | +0.590 | **+0.608** | | +0.000 | +0.084 | +0.531 | **+0.542** |
| 2,000 m / 1,000 m | +0.000 | +0.162 | +0.688 | **+0.703** | | +0.000 | +0.157 | +0.674 | **+0.685** |
| 4,000 m / 200 m | +0.000 | +0.041 | +0.478 | **+0.499** | | +0.000 | +0.034 | +0.368 | **+0.388** |
| 4,000 m / 400 m | +0.000 | +0.059 | +0.576 | **+0.606** | | +0.000 | +0.051 | +0.517 | **+0.547** |
| 4,000 m / 800 m | +0.000 | +0.089 | +0.653 | **+0.689** | | +0.000 | +0.083 | +0.631 | **+0.664** |
| 4,000 m / 2,000 m | +0.000 | +0.157 | +0.733 | **+0.767** | | +0.000 | +0.156 | +0.733 | **+0.765** |
| 10,000 m / 100 m | +0.000 | +0.022 | +0.354 | **+0.364** | | +0.000 | +0.026 | +0.200 | **+0.216** |
| 10,000 m / 200 m | +0.000 | +0.031 | +0.468 | **+0.483** | | +0.000 | +0.033 | +0.364 | **+0.395** |
| 10,000 m / 500 m | +0.000 | +0.048 | +0.586 | **+0.617** | | +0.000 | +0.046 | +0.542 | **+0.582** |
| 10,000 m / 1,000 m | +0.000 | +0.071 | +0.664 | **+0.698** | | +0.000 | +0.068 | +0.650 | **+0.686** |
| 10,000 m / 2,000 m | +0.000 | +0.116 | +0.737 | **+0.746** | | +0.000 | +0.115 | +0.735 | **+0.742** |
| 10,000 m / 5,000 m | +0.000 | +0.216 | +0.800 | **+0.837** | | +0.000 | +0.215 | +0.800 | **+0.837** |

Mean over the 18 configs (all cells): region-average 0.000, bicubic 0.100, Goodman 0.544, Puffin **0.561** (Puffin beats bicubic on 18/18 and Goodman on 16/18). Puffin std over seeds ≤ 0.016.

#### Subregion-R²

| region / subregion | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| 200 m / 100 m | 0.745 | 0.781 | **0.793** | 0.791 | | 0.685 | 0.722 | **0.734** | 0.733 |
| 1,000 m / 100 m | 0.392 | 0.441 | 0.597 | **0.603** | | 0.272 | 0.326 | 0.491 | **0.500** |
| 1,000 m / 200 m | 0.526 | 0.589 | 0.753 | **0.753** | | 0.486 | 0.545 | 0.710 | **0.711** |
| 1,000 m / 500 m | 0.744 | 0.802 | **0.903** | 0.892 | | 0.732 | 0.789 | **0.894** | 0.882 |
| 2,000 m / 100 m | 0.279 | 0.312 | 0.537 | **0.548** | | 0.111 | 0.154 | 0.421 | **0.438** |
| 2,000 m / 200 m | 0.375 | 0.418 | 0.680 | **0.696** | | 0.320 | 0.365 | 0.635 | **0.653** |
| 2,000 m / 400 m | 0.486 | 0.539 | 0.789 | **0.799** | | 0.476 | 0.526 | 0.772 | **0.780** |
| 2,000 m / 1,000 m | 0.711 | 0.758 | 0.910 | **0.914** | | 0.707 | 0.753 | 0.906 | **0.910** |
| 4,000 m / 200 m | 0.267 | 0.297 | 0.618 | **0.632** | | 0.183 | 0.220 | 0.571 | **0.593** |
| 4,000 m / 400 m | 0.346 | 0.385 | 0.723 | **0.743** | | 0.333 | 0.371 | 0.706 | **0.728** |
| 4,000 m / 800 m | 0.458 | 0.506 | 0.812 | **0.832** | | 0.459 | 0.506 | 0.806 | **0.825** |
| 4,000 m / 2,000 m | 0.711 | 0.757 | 0.923 | **0.933** | | 0.710 | 0.755 | 0.923 | **0.932** |
| 10,000 m / 100 m | 0.128 | 0.147 | 0.437 | **0.445** | | -0.165 | -0.131 | 0.289 | **0.323** |
| 10,000 m / 200 m | 0.172 | 0.197 | 0.559 | **0.572** | | 0.043 | 0.079 | 0.506 | **0.536** |
| 10,000 m / 500 m | 0.243 | 0.279 | 0.686 | **0.710** | | 0.229 | 0.268 | 0.675 | **0.704** |
| 10,000 m / 1,000 m | 0.326 | 0.374 | 0.774 | **0.796** | | 0.331 | 0.378 | 0.771 | **0.795** |
| 10,000 m / 2,000 m | 0.458 | 0.521 | 0.858 | **0.862** | | 0.461 | 0.523 | 0.857 | **0.861** |
| 10,000 m / 5,000 m | 0.710 | 0.773 | 0.942 | **0.953** | | 0.710 | 0.772 | 0.942 | **0.953** |

Mean over the 18 configs (all cells): region-average 0.449, bicubic 0.493, Goodman 0.739, Puffin **0.749**. Puffin std over seeds ≤ 0.011.

### United States

#### Census counts, tract → census block

83,774 census tracts split into their 7,980,473 blocks (2020 census; jobs from LEHD LODES; gun-violence incidents geocoded to blocks). Blocks are irregular polygons whose areas span about six orders of magnitude, so Puffin, bicubic, and Goodman all use the area term (density × block area).

Coverage is the 48 contiguous states plus DC (49 state codes, 325.1M residents). Alaska and Hawaii fall outside the UTM zones the AlphaEarth sampler is run over and are dropped. A fourth target, `housing`, is produced by the same sweep (within-region R² +0.355 all / +0.421 inhabited) but is omitted from the table below because the baselines were not run for it.

##### Within-region R²

| target | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| population | +0.000 | -0.407 | -1.037 | **+0.377** | | +0.000 | -0.297 | -0.901 | **+0.429** |
| jobs | +0.000 | +0.057 | +0.050 | **+0.209** | | +0.000 | +0.068 | +0.073 | **+0.183** |
| gun violence | +0.000 | -0.079 | -0.075 | **+0.014** | | **+0.000** | -0.117 | -0.097 | -0.034 |

Puffin std over seeds ≤ 0.022.

The prior on its own, before conditioning on the totals, is negative on every US
target (population -0.559, jobs -0.048, gun violence -0.089, turnout -0.459), so on
irregular units the result comes from the variance-weighted split rather than from
the prior being accurate by itself.

##### Subregion-R²

| target | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| population | 0.222 | -0.095 | -0.584 | **0.515** | | 0.231 | 0.035 | -0.406 | **0.568** |
| jobs | 0.087 | 0.139 | 0.133 | **0.277** | | 0.094 | 0.174 | 0.179 | **0.278** |
| gun violence | 0.156 | 0.089 | 0.092 | **0.168** | | -0.302 | -0.278 | -0.267 | **-0.216** |



#### Voting per party, county → precinct

2024 presidential votes: 3,096 counties split into their 163,690 precincts, each party disaggregated independently (`run_voting.py`). Precincts are drawn to hold roughly equal populations, so there is no area term.

##### Within-region R²

| party | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| Republican (Trump) | **+0.000** | -0.075 | -0.606 | -0.694 | | **+0.000** | -0.075 | -0.607 | -0.693 |
| Democratic (Harris) | **+0.000** | -0.087 | -0.303 | -0.554 | | **+0.000** | -0.087 | -0.305 | -0.555 |
| other | **+0.000** | -0.059 | -0.249 | -0.434 | | **+0.000** | -0.062 | -0.264 | -0.448 |
| total | **+0.000** | -0.104 | -0.557 | -0.785 | | **+0.000** | -0.104 | -0.557 | -0.785 |

Puffin std over seeds ≤ 0.053.

##### Subregion-R²

| party | Region-avg. | Bicubic | Goodman | Puffin | | Region-avg. | Bicubic | Goodman | Puffin |
|---|---|---|---|---|---|---|---|---|---|
| | *all cells* | | | | | *inhabited* | | | |
| Republican (Trump) | **0.510** | 0.473 | 0.213 | 0.170 | | **0.510** | 0.473 | 0.213 | 0.171 |
| Democratic (Harris) | **0.501** | 0.458 | 0.350 | 0.225 | | **0.502** | 0.459 | 0.350 | 0.225 |
| other | **0.464** | 0.433 | 0.331 | 0.232 | | **0.471** | 0.438 | 0.332 | 0.235 |
| total | **0.513** | 0.462 | 0.241 | 0.130 | | **0.513** | 0.462 | 0.241 | 0.130 |

Puffin's 90% intervals cover 0.90–0.91 of the truth on every party.

### What the results show

- **Spatial smoothness does not disaggregate.** Bicubic interpolation stays well below the learned methods on every grid config (at best +0.54 in Berlin and +0.23 in Germany, both at the coarsest subregions), so the gains are not just neighboring totals.
- **A linear model on the embeddings is a strong baseline on one city.** Goodman beats Puffin on 12/18 Berlin configs. With about 36× more data (Germany), Puffin pulls ahead on 16/18 configs, by small but consistent margins.
- **Irregular units are where Puffin clearly wins.** On US blocks neither baseline beats the region average on population, while Puffin reaches a clearly positive within-region R²; on jobs it is about 3.7× the best baseline. The likely reason is that a linear or smooth density cannot fall to near zero on huge empty blocks, whereas the nonlinear head can.
- **Voting is a boundary case for every method.** Precincts are drawn to hold equal populations, so there is no within-county count signal: all three non-trivial methods fall below the region average, and Puffin falls furthest. Its 90% intervals still cover ≈ 0.90.
- **Uncertainty is calibrated throughout.** Puffin's 90% per-subregion intervals cover 0.88–0.92 of the truth across resolutions, countries, and targets, including the negative cases.

---

## Data

No data is tracked in this repository. The `setup_*.py` scripts download everything
except the gun-violence incidents, which must be placed by hand.

| Source | Used for | Access |
|---|---|---|
| AlphaEarth Foundations embeddings | the 64-dim input, all datasets | `data.source.coop` (COG tiles; `.env.example` shows the optional S3 credentials) |
| Zensus 2022 100 m grid (Destatis) | German population and the rate targets | `destatis.de/static/DE/zensus/gitterdaten/` |
| TIGER/Line 2020 blocks + 2020 Census | US block geometry and population | `www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/` |
| LEHD LODES8 WAC | US jobs | `lehd.ces.census.gov/data/lodes/LODES8/` |
| NYT 2024 precinct results | US votes per party | `int.nyt.com/newsgraphics/elections/map-data/2024/` |
| Gun Violence Archive | US gun-violence incidents | **not redistributed.** Obtain from the Gun Violence Archive and save as `data/raw/Gun Violence/Gun Violence Archive - DFKI.csv`. If absent, the target is skipped and scored as all zeros. |

Both censuses apply statistical disclosure control to small counts (the US 2020 count
uses differential privacy; Zensus 2022 uses the cell-key method), so the fine-scale
ground truth is itself perturbed. Exact zeros are published unchanged in Zensus 2022,
which is why absent cells are read as 0.

## Citation

```bibtex
@article{jobanputra2026puffin,
  title   = {Puffin: Probabilistic Learning of Spatial Detail From Coarse Observations},
  author  = {Jobanputra, Chaitanya and Vollmer, Sebastian and Gro{\ss}mann, Gerrit},
  journal = {arXiv preprint arXiv:2610.11914},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.11914}
}
```

## License

MIT, see [LICENSE](LICENSE). The datasets listed above keep their own licenses and are
not redistributed here.
