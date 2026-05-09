"""Unit tests for `skillmodels.af.moment_init` Spearman estimators."""

import numpy as np
import pytest

from skillmodels.af.moment_init import (
    SpearmanResult,
    derive_unexplained_sd,
    seed_beta_from_ols,
    spearman_factor_moments,
)


def _simulate_three_indicators(
    *,
    n: int,
    loadings: np.ndarray,
    meas_sds: np.ndarray,
    factor_var: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, np.sqrt(factor_var), size=n)
    eps = rng.normal(0.0, 1.0, size=(n, len(loadings))) * meas_sds
    return loadings * factor[:, None] + eps


def test_spearman_recovers_loadings_within_30pct():
    truth_loadings = np.array([1.0, 1.3, 0.8])
    truth_meas_sds = np.array([0.4, 0.5, 0.3])
    truth_factor_var = 1.5
    measurements = _simulate_three_indicators(
        n=2000,
        loadings=truth_loadings,
        meas_sds=truth_meas_sds,
        factor_var=truth_factor_var,
        seed=42,
    )

    result = spearman_factor_moments(measurements, anchor_idx=0)

    assert result.valid
    assert result.loadings[0] == pytest.approx(1.0, abs=1e-12)
    assert result.loadings[1] == pytest.approx(truth_loadings[1], rel=0.30)
    assert result.loadings[2] == pytest.approx(truth_loadings[2], rel=0.30)
    assert result.latent_var == pytest.approx(truth_factor_var, rel=0.30)
    for k in range(3):
        assert result.meas_sds[k] == pytest.approx(truth_meas_sds[k], rel=0.30)


def test_spearman_anchor_fallback_on_zero_cov():
    rng = np.random.default_rng(0)
    n = 1500
    factor = rng.normal(0.0, 1.0, size=n)
    # First measurement is independent noise; the next two share the factor.
    indep = rng.normal(0.0, 1.0, size=n)
    measurements = np.column_stack(
        [
            indep,
            1.2 * factor + 0.4 * rng.normal(size=n),
            0.9 * factor + 0.3 * rng.normal(size=n),
        ]
    )

    result = spearman_factor_moments(measurements, anchor_idx=0)

    # Anchor candidate 0 is uncorrelated with the others — but the routine
    # rotates to a different anchor and still returns a valid result, with
    # the user-requested anchor (idx 0) reported on a 1.0 loading scale.
    assert result.valid
    assert result.loadings[0] == pytest.approx(1.0, abs=1e-12)
    # The loading on idx 0 is on a degenerate scale; what matters is that
    # the routine didn't NaN out and returned finite values everywhere.
    assert np.all(np.isfinite(result.loadings))
    assert np.all(np.isfinite(result.meas_sds))
    assert np.isfinite(result.latent_var)


def test_spearman_handles_negative_residual_variance():
    # Tiny n forces sample noise where S_kk < λ_k² Var(F) is possible.
    truth_loadings = np.array([1.0, 0.9, 1.1])
    truth_meas_sds = np.array([0.05, 0.05, 0.05])
    measurements = _simulate_three_indicators(
        n=20,
        loadings=truth_loadings,
        meas_sds=truth_meas_sds,
        factor_var=1.0,
        seed=7,
    )

    result = spearman_factor_moments(measurements, sd_floor=1e-3)

    assert np.all(np.isfinite(result.meas_sds))
    assert np.all(result.meas_sds >= 1e-3 - 1e-12)
    assert np.isfinite(result.latent_var)


def test_spearman_below_two_measurements_returns_invalid():
    measurements = np.random.default_rng(0).normal(size=(100, 1))

    result = spearman_factor_moments(measurements)

    assert not result.valid
    assert result.loadings.shape == (1,)


def test_spearman_pairwise_complete_handles_nan():
    truth_loadings = np.array([1.0, 1.2, 0.8])
    truth_meas_sds = np.array([0.3, 0.3, 0.3])
    truth_factor_var = 1.0
    measurements = _simulate_three_indicators(
        n=3000,
        loadings=truth_loadings,
        meas_sds=truth_meas_sds,
        factor_var=truth_factor_var,
        seed=1,
    )
    # Punch a few NaNs into different columns so listwise-complete would
    # discard most rows.
    rng = np.random.default_rng(2)
    for col in range(3):
        idx = rng.choice(3000, size=400, replace=False)
        measurements[idx, col] = np.nan

    result = spearman_factor_moments(measurements)

    assert result.valid
    assert result.loadings[1] == pytest.approx(truth_loadings[1], rel=0.30)
    assert result.loadings[2] == pytest.approx(truth_loadings[2], rel=0.30)


def test_derive_unexplained_sd_clamped():
    # β'Σβ > latent_var → clamped to floor, not NaN.
    sd = derive_unexplained_sd(
        latent_var=0.5,
        beta=np.array([2.0]),
        prev_state_cov=np.array([[1.0]]),
        sd_floor=1e-3,
    )

    assert sd == pytest.approx(1e-3, abs=1e-12)


def test_derive_unexplained_sd_recovers_residual():
    # latent_var = 1.0, β'Σβ = 0.36 → residual var = 0.64 → sd = 0.8.
    sd = derive_unexplained_sd(
        latent_var=1.0,
        beta=np.array([0.6]),
        prev_state_cov=np.array([[1.0]]),
    )

    assert sd == pytest.approx(0.8, rel=1e-9)


def test_derive_unexplained_sd_handles_multivariate_state():
    beta = np.array([0.3, 0.4])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    # β'Σβ = 0.09 + 2*0.3*0.4*0.2 + 0.16 = 0.298
    expected = float(np.sqrt(1.0 - 0.298))

    sd = derive_unexplained_sd(latent_var=1.0, beta=beta, prev_state_cov=cov)

    assert sd == pytest.approx(expected, rel=1e-9)


def test_seed_beta_from_ols_recovers_known_coefs():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, 2))
    y = 0.7 * x[:, 0] - 0.3 * x[:, 1] + 0.1 * rng.normal(size=n)

    beta = seed_beta_from_ols(y, x)

    assert beta.shape == (2,)
    assert beta[0] == pytest.approx(0.7, rel=0.10)
    assert beta[1] == pytest.approx(-0.3, rel=0.20)


def test_seed_beta_from_ols_handles_nan_pairwise():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, 2))
    y = 0.5 * x[:, 0] + 0.05 * rng.normal(size=n)
    y[::5] = np.nan
    x[::7, 0] = np.nan

    beta = seed_beta_from_ols(y, x)

    assert beta.shape == (2,)
    assert np.all(np.isfinite(beta))


def test_seed_beta_from_ols_returns_zeros_on_rank_deficient():
    n = 50
    x = np.zeros((n, 3))
    y = np.random.default_rng(0).normal(size=n)

    beta = seed_beta_from_ols(y, x)

    assert beta.shape == (3,)
    assert np.allclose(beta, 0.0)


def test_spearman_result_dataclass_is_frozen():
    result = SpearmanResult(
        loadings=np.zeros(2),
        meas_sds=np.zeros(2),
        latent_var=0.0,
        valid=False,
    )

    with pytest.raises(AttributeError):
        result.valid = True  # type: ignore[misc]
