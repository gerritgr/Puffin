# Puffin

We introduce **PUFFIN**: **P**robabilistic **U**ncertainty for **F**ine-scale **F**eature **I**nference **N**etworks.

PUFFIN is a probabilistic neural downscaling method that uses high-resolution AlphaEarth embeddings as covariates to infer fine-scale spatial patterns from coarse aggregate observations. We investigate how well this framework can predict a range of sociological, demographic, economic, and political variables of interest, including population density, demographic composition, political preferences, income, wealth, and related indicators.

Unlike standard downscaling approaches, PUFFIN does not predict only point estimates. Instead, it predicts full probability distributions over fine-scale quantities for principled uncertainty quantification. The model is trained through a constraint-satisfaction objective: high-resolution AlphaEarth inputs induce fine-scale predictive distributions, which are aggregated and matched against low-resolution target observations. We evaluate uncertainty calibration using, for instance, calibration curves, PIT histograms, and related diagnostics.


## Project Layout

```text
alphaearth_population/
├── config.py                         <- Shared paths, grid constants, loaders
├── data/
│   ├── raw/                          <- Raw AlphaEarth and population files (not in git)
│   ├── processed/                    <- Aligned tensors/arrays (not in git)
│   └── preprocess_data.ipynb         <- Align AE features and coarse population grid
├── notebooks/
│   ├── 01_probabilistic_disaggregation.ipynb
│   │                                  <- Main model: Gaussian aggregate NLL, posterior update
│   ├── 02_calibration_analysis.ipynb  <- Coverage, PIT histogram, CRPS, residual diagnostics
│   └── 03_posterior_maps.ipynb        <- Fine-grid posterior means/uncertainty maps
├── references/                       <- Papers, notes, data documentation
├── reports/
│   ├── figures/                      <- Generated plots
│   └── tables/                       <- Generated metrics
├── src/
│   └── alphaearth_population/
│       ├── __init__.py
│       ├── data.py                   <- Dataset utilities
│       ├── model.py                  <- Neural network modules
│       ├── losses.py                 <- Aggregate Gaussian likelihood loss
│       ├── posterior.py              <- Gaussian conditioning utilities
│       └── metrics.py                <- Calibration and scoring metrics
├── pyproject.toml
└── README.md
```

## Goal

The observed population data is available only on a coarse grid:

```text
Ground truth population: 10 × 10 grid
AlphaEarth features:    100 × 100 grid
```

Each coarse population cell contains `10 × 10 = 100` fine AlphaEarth cells.

For each fine cell `i`, the model receives a 64-dimensional AlphaEarth feature vector:

```text
ae_i ∈ R^64
```

and predicts a Gaussian distribution over the latent fine-scale population value:

```text
x_i ~ Normal(mu_i, sigma_i²)
```

The fine-scale values are not directly observed. Instead, for each coarse cell `B`, the observed target is the aggregate:

```text
Y_B = sum_{i in B} x_i
```

The model is therefore trained by comparing the observed coarse population value to the aggregated predictive distribution.

## Probabilistic Model

For each fine cell:

```text
x_i ~ Normal(mu_i, sigma_i²)
```

where

```text
(mu_i, sigma_i²) = f_theta(ae_i)
```

For a coarse cell `B`, assuming conditional independence between fine cells:

```text
S_B = sum_{i in B} x_i
```

is also Gaussian:

```text
S_B ~ Normal(M_B, V_B)
```

with

```text
M_B = sum_{i in B} mu_i
V_B = sum_{i in B} sigma_i² + tau²
```

where `tau²` is optional observation noise for the coarse ground truth.

The training loss is the negative log-likelihood of the observed coarse value:

```text
L_B = 0.5 * [ (Y_B - M_B)² / V_B + log(V_B) ]
```

This encourages the model to be both accurate and calibrated. Large uncertainty is penalized by the `log(V_B)` term, while overconfident wrong predictions are penalized by the residual term.

## Posterior Fine-Cell Update

After predicting fine-cell priors and observing the true coarse total `Y_B`, the posterior distribution of each fine cell can be computed analytically.

Let

```text
M_B = sum_j mu_j
V_B = sum_j sigma_j² + tau²
```

Then the posterior mean for fine cell `i` is

```text
mu_i_post = mu_i + sigma_i² / V_B * (Y_B - M_B)
```

and the posterior variance is

```text
sigma_i_post² = sigma_i² - sigma_i⁴ / V_B
```

The observed coarse total redistributes the aggregate residual across fine cells in proportion to their prior uncertainty.

## Dataset

### AlphaEarth Features

Expected input:

```text
data/raw/alphaearth_features.npy
```

Shape:

```text
100 × 100 × 64
```

or flattened as:

```text
10000 × 64
```

Each fine cell should contain a 64-dimensional AlphaEarth embedding.

### Coarse Population Ground Truth

Expected input:

```text
data/raw/population_coarse.npy
```

Shape:

```text
10 × 10
```

The population values may be raw counts, densities, or z-scored values. If z-scoring is used, the aggregation logic should be applied consistently.

For raw population counts, aggregation is physically meaningful:

```text
coarse population = sum of fine-cell populations
```

For z-scored values, be careful: sums in z-space are not necessarily equivalent to sums in population space unless the transformation is handled consistently.

## Prerequisites

| Tool          | Notes                                   |
| ------------- | --------------------------------------- |
| `uv`          | Fast Python package and project manager |
| Python ≥ 3.10 | Managed automatically by `uv`           |

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/gerritgr/alphaearth_population
cd alphaearth_population

# 2. Install dependencies
uv sync

# 3. Place input files in data/raw/
#    data/raw/alphaearth_features.npy
#    data/raw/population_coarse.npy

# 4. Launch JupyterLab
uv run jupyter lab

# 5. Run the main notebook
#    notebooks/01_probabilistic_disaggregation.ipynb
```

## Example Data Flow

```text
data/raw/alphaearth_features.npy      100 × 100 × 64
data/raw/population_coarse.npy        10 × 10
        │
        ▼
data/preprocess_data.ipynb
        │
        ▼
data/processed/ae_fine.npy            10000 × 64
data/processed/y_coarse.npy           100
data/processed/block_ids.npy          10000
        │
        ▼
01_probabilistic_disaggregation.ipynb
        │
        ├─▶ train neural network using aggregate Gaussian NLL
        ├─▶ predict fine-cell prior distributions
        ├─▶ condition on observed coarse totals
        └─▶ save posterior fine-grid maps
```

## Notebook Descriptions

| Notebook                                | Contents                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------- |
| `01_probabilistic_disaggregation.ipynb` | Main probabilistic model, aggregate Gaussian NLL training, posterior fine-cell update |
| `02_calibration_analysis.ipynb`         | Predictive interval coverage, PIT histogram, standardized residuals, NLL, CRPS        |
| `03_posterior_maps.ipynb`               | Fine-grid posterior mean map, posterior uncertainty map, aggregate residual maps      |

## Evaluation

Because the true `100 × 100` fine population grid is not observed, uncertainty can be directly evaluated only at the observed coarse scale.

The main diagnostics are:

| Metric                 | Purpose                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------- |
| Aggregate NLL          | Measures likelihood assigned to observed coarse values                                 |
| Coverage curve         | Checks whether 50%, 80%, 90%, 95% intervals contain the truth at the correct frequency |
| PIT histogram          | Checks full predictive distribution calibration                                        |
| Standardized residuals | Checks whether residuals behave like `Normal(0, 1)`                                    |
| CRPS                   | Proper scoring rule for probabilistic regression                                       |
| Average interval width | Measures sharpness of predictive intervals                                             |

For each coarse cell `B`, the model predicts:

```text
S_B ~ Normal(M_B, V_B)
```

where `S_B` is the predicted aggregate population. Calibration is evaluated by comparing observed coarse values `Y_B` against these predictive distributions.

## Configuration (`config.py`)

Shared paths and constants should live in `config.py`.

Suggested constants:

| Constant               | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `AE_RAW`               | Path to raw AlphaEarth feature array            |
| `POP_RAW`              | Path to raw coarse population grid              |
| `AE_PROCESSED`         | Path to processed flattened AlphaEarth features |
| `POP_PROCESSED`        | Path to processed coarse targets                |
| `FINE_H`               | Fine-grid height, default `100`                 |
| `FINE_W`               | Fine-grid width, default `100`                  |
| `COARSE_H`             | Coarse-grid height, default `10`                |
| `COARSE_W`             | Coarse-grid width, default `10`                 |
| `AE_DIM`               | AlphaEarth feature dimension, default `64`      |
| `FINE_PER_COARSE_SIDE` | Default `10`                                    |
| `DEVICE`               | `"cuda"` if available, otherwise `"cpu"`        |

## Dependencies

Declared in `pyproject.toml`, managed by `uv sync`.

| Package        | Purpose                                |
| -------------- | -------------------------------------- |
| `numpy`        | Array handling                         |
| `pandas`       | Metrics tables                         |
| `torch`        | Neural network training                |
| `matplotlib`   | Plots and diagnostics                  |
| `scipy`        | Normal CDF/PPF, CRPS utilities         |
| `scikit-learn` | Train/test splitting and preprocessing |
| `jupyterlab`   | Interactive notebooks                  |
| `tqdm`         | Progress bars                          |

## Notes on Z-Scoring

If population values are z-scored, the Gaussian model is convenient because sums of Gaussian variables remain Gaussian.

However, if the final object of interest is actual population count, it is often better to train or evaluate the aggregate likelihood in the original population scale.

If

```text
z_i = (p_i - mean_pop) / std_pop
```

then

```text
p_i = mean_pop + std_pop * z_i
```

and the aggregate distribution can still be computed analytically because this transformation is affine.

## Limitations

The fine-scale population values are latent. Therefore, fine-cell uncertainty cannot be directly validated unless independent high-resolution population labels are available.

The model can be calibrated at the coarse observation scale while still producing fine-scale maps that depend strongly on modeling assumptions, such as:

* independence between fine cells,
* spatial smoothness,
* Gaussianity,
* quality of AlphaEarth features,
* treatment of population zeros,
* use of z-scored versus raw population values.

The recommended interpretation is:

> The model is empirically calibrated at the observed aggregate scale. Fine-scale posterior uncertainty follows from the Gaussian disaggregation assumptions.

## License

MIT License.
