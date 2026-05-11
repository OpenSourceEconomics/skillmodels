"""Tests for AF initialization strategies."""

import numpy as np
import pytest

from skillmodels.af.types import AFEstimationOptions
from skillmodels.amn.moments import spearman_factor_moments


def test_default_initialization_strategy_is_moment_based():
    """Default initialization is moment-based (Spearman cross-cov seeds)."""
    opts = AFEstimationOptions()

    assert opts.initialization_strategy == "moment_based"


def test_initialization_strategy_can_be_set_to_constant():
    """Legacy constant init remains available for regression testing."""
    opts = AFEstimationOptions(
        initialization_strategy="constant",
    )

    assert opts.initialization_strategy == "constant"


def test_spearman_seed_closer_to_truth_than_constant_default():
    """Moment-based seed is closer to truth than the static 0.5 default.

    Synthetic data with known sigma_meas and Var(latent) — assert that the
    Spearman residual variance gives a starting sigma_meas closer to truth
    than the legacy ``obs_sd * 0.5`` heuristic.
    """
    rng = np.random.default_rng(0)
    n = 1000
    truth_loadings = np.array([1.0, 1.2, 0.9])
    truth_meas_sds = np.array([0.3, 0.4, 0.3])
    truth_factor_sd = 1.5
    factor = rng.normal(0.0, truth_factor_sd, size=n)
    eps = rng.normal(0.0, 1.0, size=(n, 3)) * truth_meas_sds
    measurements = truth_loadings * factor[:, None] + eps

    spearman = spearman_factor_moments(measurements, anchor_idx=0)

    # Spearman recovers sigma_meas within 30% of truth.
    for k in range(3):
        assert spearman.meas_sds[k] == pytest.approx(truth_meas_sds[k], rel=0.30)

    # Legacy default is obs_sd * 0.5; for sigma_meas truth=0.3 with anchor
    # variance λ²·Var(F)+sigma_meas² ≈ 1²·2.25+0.09 ≈ 2.34, obs_sd ≈ 1.53,
    # default seed ≈ 0.76 — way off truth 0.3. Spearman should be closer.
    obs_sds = np.nanstd(measurements, axis=0)
    legacy_seeds = np.maximum(obs_sds * 0.5, 0.01)
    spearman_dist = np.abs(spearman.meas_sds - truth_meas_sds).sum()
    legacy_dist = np.abs(legacy_seeds - truth_meas_sds).sum()
    assert spearman_dist < legacy_dist


def test_spearman_falls_back_for_single_measurement_factor():
    """`valid=False` → moment-init returns the same fallback values."""
    measurements = np.random.default_rng(0).normal(size=(100, 1))

    result = spearman_factor_moments(measurements, anchor_idx=0)

    assert not result.valid
    # Fallback values are constant; downstream code should keep using
    # the static defaults instead of overriding from these.
    assert result.loadings.shape == (1,)
    assert result.meas_sds.shape == (1,)


def test_initialization_strategy_other_options_unchanged():
    """Other AFEstimationOptions fields remain at their existing defaults."""
    opts = AFEstimationOptions()

    assert opts.n_halton_points == 50
    assert opts.n_halton_points_shock == 30
    assert opts.n_mixture_components == 2
    assert opts.optimizer_algorithm == "fides"
    assert opts.two_stage is False
    assert opts.coarse_fraction == 0.5
    assert opts.stability_floor == 1e-217
    assert opts.n_obs_per_batch is None


def test_moment_init_handles_pinned_anchor_loading():
    """When user pins loading to a non-1.0 value, anchor_loading respects it."""
    rng = np.random.default_rng(0)
    n = 800
    loadings = np.array([2.0, 0.6, 1.2])  # anchor=2.0 (user normalization)
    factor = rng.normal(0.0, 1.0, size=n)
    eps = rng.normal(0.0, 0.4, size=(n, 3))
    measurements = loadings * factor[:, None] + eps

    result = spearman_factor_moments(measurements, anchor_idx=0, anchor_loading=2.0)

    assert result.loadings[0] == pytest.approx(2.0, abs=1e-12)
    # Other loadings should be on the same scale.
    assert result.loadings[1] == pytest.approx(0.6, rel=0.30)
    assert result.loadings[2] == pytest.approx(1.2, rel=0.30)
