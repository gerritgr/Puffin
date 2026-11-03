"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent.resolve()

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
AE_RAW_DIR = DATA_RAW / "aef"          # AlphaEarth embedding tiles, by year and UTM zone
ZENSUS_RAW_DIR = DATA_RAW / "zensus"
EXPERIMENTS_DIR = ROOT / "experiments" / "runs"
FIGURES_DIR = ROOT / "experiments" / "figures"

AE_DIM: int = 64                       # dimensions of an AlphaEarth embedding
FINE_RES_M: int = 100                  # native resolution of the German census grid
EPSG_INSPIRE: int = 3035               # the CRS the German grid is defined in

AE_COLS: list[str] = [f"ae_{i}" for i in range(AE_DIM)]

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Counts add up over subregions, so they use an aggregate count likelihood and a
# conserving split. Rates are averages, and use the weighted-mean formulation.
COUNT_TARGETS = {"population", "citizens_18plus", "housing", "jobs", "gun_violence",
                 "votes_rep", "votes_dem", "votes_total", "votes_other"}
RATE_TARGETS = {"rent_per_m2", "avg_age", "vacancy_rate", "foreigner_share"}
