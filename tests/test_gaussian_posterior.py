"""Unit tests for the closed-form Gaussian posterior update.

These are the most critical math tests in the project.
All assertions come directly from Logic Module §1.5.
"""

import numpy as np
import pytest
from puffin.posterior import gaussian_posterior, poisson_allocation
from puffin.calibrate import block_dispersion_conformal


# ---------------------------------------------------------------------------
# Block-level dispersion calibration (count targets)
# ---------------------------------------------------------------------------

def test_block_dispersion_conformal_calibrates_overdispersed_totals():
    """Overdispersed block totals: naive Poisson under-covers; the NB+conformal
    fit recovers ~90% coverage on a held-out fold."""
    rng = np.random.default_rng(0)
    B = 4000
    lam = rng.uniform(50, 500, size=B)              # predicted block rates
    r_true = 8.0                                     # heavy overdispersion
    # NB2 sample: Var = lam + lam^2/r_true  (gamma-Poisson mixture)
    gam = rng.gamma(shape=r_true, scale=lam / r_true)
    y = rng.poisson(gam).astype(float)
    fold = rng.integers(0, 5, size=B)

    # pure-Poisson 90% interval badly under-covers
    z = 1.6449
    test = fold == 4
    pois_cov = np.mean(np.abs(y[test] - lam[test]) <= z * np.sqrt(lam[test]))
    assert pois_cov < 0.75

    r, out = block_dispersion_conformal(lam, y, fold, val_fold=3, test_fold=4)
    assert np.isfinite(r) and r > 0
    assert 0.85 <= out["coverage_test"][0.9] <= 0.95


# ---------------------------------------------------------------------------
# Poisson multinomial allocation (count-target posterior)
# ---------------------------------------------------------------------------

def test_poisson_allocation_conserves_block_total():
    """Allocated cell counts sum exactly to the observed block total Y_B."""
    rng = np.random.default_rng(0)
    n_blocks, n_cells = 8, 16
    rate = rng.uniform(0.01, 20.0, size=n_blocks * n_cells)
    y    = rng.uniform(50, 500, size=n_blocks)
    ids  = np.repeat(np.arange(n_blocks), n_cells)

    mu_post, _ = poisson_allocation(rate, y, ids, n_blocks)
    for b in range(n_blocks):
        np.testing.assert_allclose(mu_post[ids == b].sum(), y[b], rtol=1e-6)


def test_poisson_allocation_non_negative():
    """Allocation never produces negative cell values (unlike Gaussian Kalman)."""
    rng = np.random.default_rng(1)
    n_blocks, n_cells = 5, 9
    rate = rng.uniform(0.0, 5.0, size=n_blocks * n_cells)   # some exactly ~0
    y    = rng.uniform(1, 40, size=n_blocks)
    ids  = np.repeat(np.arange(n_blocks), n_cells)

    mu_post, var_post = poisson_allocation(rate, y, ids, n_blocks)
    assert (mu_post >= 0).all()
    assert (var_post >= 0).all()


def test_poisson_allocation_proportional_to_rate():
    """Cells with double the rate receive double the allocated count."""
    rate = np.array([1.0, 2.0, 1.0, 2.0])   # one block of 4 cells
    y    = np.array([60.0])
    ids  = np.zeros(4, dtype=int)
    mu_post, _ = poisson_allocation(rate, y, ids, 1)
    np.testing.assert_allclose(mu_post, [10.0, 20.0, 10.0, 20.0], rtol=1e-6)


def test_posterior_sums_to_coarse_total_when_noiseless():
    """When tau_B²=0, posterior means sum exactly to Y_B (noiseless Kalman update)."""
    rng = np.random.default_rng(0)
    n_blocks, n_cells = 10, 16
    mu   = rng.uniform(0, 10, size=n_blocks * n_cells)
    var  = rng.uniform(0.1, 2.0, size=n_blocks * n_cells)
    y    = rng.uniform(50, 200, size=n_blocks)
    ids  = np.repeat(np.arange(n_blocks), n_cells)

    mu_post, _ = gaussian_posterior(mu, var, y, ids, n_blocks, tau2_B=0.0)

    for b in range(n_blocks):
        idx = ids == b
        np.testing.assert_allclose(mu_post[idx].sum(), y[b], rtol=1e-5)


def test_posterior_sum_shrinks_toward_prior_when_noisy():
    """When tau_B²>0, posterior sum lies strictly between M_B and Y_B (Kalman shrinkage)."""
    rng = np.random.default_rng(3)
    n_blocks, n_cells = 4, 16
    mu   = rng.uniform(0, 5, size=n_blocks * n_cells)
    var  = rng.uniform(0.5, 2.0, size=n_blocks * n_cells)
    ids  = np.repeat(np.arange(n_blocks), n_cells)
    y    = rng.uniform(30, 80, size=n_blocks)  # different from M_B
    tau2 = np.full(n_blocks, 5.0)              # non-trivial noise

    mu_post, _ = gaussian_posterior(mu, var, y, ids, n_blocks, tau2_B=tau2)

    for b in range(n_blocks):
        idx = ids == b
        M_B = mu[idx].sum()
        V_B = var[idx].sum() + tau2[b]
        # Expected: M_B + (V_B - tau2[b]) / V_B * (Y_B - M_B)
        expected_sum = M_B + (V_B - tau2[b]) / V_B * (y[b] - M_B)
        np.testing.assert_allclose(mu_post[idx].sum(), expected_sum, rtol=1e-5)


def test_posterior_variance_non_negative():
    """Posterior variance must be >= eps (clamped, never negative)."""
    rng = np.random.default_rng(1)
    n_blocks, n_cells = 5, 9
    mu   = rng.standard_normal(n_blocks * n_cells)
    var  = rng.uniform(0.01, 1.0, size=n_blocks * n_cells)
    y    = rng.uniform(-5, 5, size=n_blocks)
    ids  = np.repeat(np.arange(n_blocks), n_cells)

    _, var_post = gaussian_posterior(mu, var, y, ids, n_blocks)
    assert (var_post >= 0).all(), "Posterior variance must be non-negative"


def test_posterior_variance_shrinks():
    """Posterior variance must be <= prior variance (conditioning reduces uncertainty)."""
    rng = np.random.default_rng(2)
    n_blocks, n_cells = 4, 16
    mu   = rng.uniform(0, 5, size=n_blocks * n_cells)
    var  = rng.uniform(0.5, 3.0, size=n_blocks * n_cells)
    y    = rng.uniform(10, 50, size=n_blocks)
    ids  = np.repeat(np.arange(n_blocks), n_cells)
    tau2 = np.full(n_blocks, 0.5)

    _, var_post = gaussian_posterior(mu, var, y, ids, n_blocks, tau2_B=tau2)
    assert (var_post <= var + 1e-9).all(), "Posterior variance must not exceed prior"


def test_zero_residual_no_mean_update():
    """If Y_B == M_B exactly, posterior mean equals prior mean."""
    n_blocks, n_cells = 3, 4
    mu  = np.ones(n_blocks * n_cells) * 2.0
    var = np.ones(n_blocks * n_cells) * 1.0
    ids = np.repeat(np.arange(n_blocks), n_cells)
    y   = np.array([n_cells * 2.0] * n_blocks)   # Y_B = M_B

    mu_post, _ = gaussian_posterior(mu, var, y, ids, n_blocks)
    np.testing.assert_allclose(mu_post, mu, atol=1e-6)
