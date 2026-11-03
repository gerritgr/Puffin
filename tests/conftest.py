"""Shared pytest fixtures."""
import numpy as np
import pytest


@pytest.fixture
def small_block_batch():
    """Tiny synthetic batch: 4 blocks, 16 fine cells each, 64-dim AE features."""
    rng = np.random.default_rng(42)
    n_blocks, n_cells, ae_dim = 4, 16, 64
    x_fine   = rng.standard_normal((n_blocks * n_cells, ae_dim)).astype("float32")
    y_coarse = rng.uniform(10, 200, size=n_blocks).astype("float32")
    block_ids = np.repeat(np.arange(n_blocks), n_cells)
    tau2_B   = np.maximum(1.0, 0.01 * y_coarse)
    return x_fine, y_coarse, block_ids, tau2_B
