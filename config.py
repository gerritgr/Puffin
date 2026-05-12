"""Shared paths, grid constants, and device configuration."""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

AE_RAW = ROOT / "data" / "raw" / "alphaearth_features.npy"
POP_RAW = ROOT / "data" / "raw" / "population_coarse.npy"

AE_PROCESSED = ROOT / "data" / "processed" / "ae_fine.npy"
POP_PROCESSED = ROOT / "data" / "processed" / "y_coarse.npy"
BLOCK_IDS_PROCESSED = ROOT / "data" / "processed" / "block_ids.npy"

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

FINE_H: int = 100
FINE_W: int = 100
COARSE_H: int = 10
COARSE_W: int = 10

AE_DIM: int = 64

FINE_PER_COARSE_SIDE: int = FINE_H // COARSE_H
FINE_PER_COARSE: int = FINE_PER_COARSE_SIDE ** 2

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
