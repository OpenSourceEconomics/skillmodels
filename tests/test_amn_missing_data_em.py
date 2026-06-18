"""Tests for the missing-data Gaussian-mixture EM (AMN Stage 1 fallback)."""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from skillmodels.amn.missing_data_em import _run_em, fit_gaussian_mixture_missing


def _mixture_loglik(
    data: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> float:
    """Independent complete-data Gaussian-mixture log-likelihood (an oracle)."""
    log_comp = np.column_stack(
        [
            np.log(weights[k])
            + multivariate_normal.logpdf(data, mean=means[k], cov=covs[k])
            for k in range(weights.shape[0])
        ]
    )
    return float(logsumexp(log_comp, axis=1).sum())


def _simulate_two_component(
    *,
    n: int,
    weights: tuple[float, float],
    means: tuple[np.ndarray, np.ndarray],
    chols: tuple[np.ndarray, np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], size=n, p=list(weights))
    out = np.empty((n, means[0].shape[0]))
    for k in (0, 1):
        idx = labels == k
        if idx.any():
            standard = rng.normal(size=(int(idx.sum()), means[k].shape[0]))
            out[idx] = standard @ chols[k].T + means[k]
    return out


_WEIGHTS = (0.4, 0.6)
_MEANS = (np.array([-2.0, 1.0]), np.array([2.0, -1.0]))
_CHOLS = (
    np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.2]])),
    np.linalg.cholesky(np.array([[0.8, -0.2], [-0.2, 1.0]])),
)


def _align(fitted_means: np.ndarray) -> np.ndarray:
    """Return the index order lining fitted components up to the truth by mean[0]."""
    return np.argsort(fitted_means[:, 0])


def test_missing_data_em_matches_sklearn_on_complete_data():
    """With no missingness the EM must match sklearn's GaussianMixture."""
    data = _simulate_two_component(
        n=4000, weights=_WEIGHTS, means=_MEANS, chols=_CHOLS, seed=1
    )

    mine = fit_gaussian_mixture_missing(
        data, n_components=2, max_iter=500, tol=1e-7, n_init=5, reg_covar=1e-6, seed=1
    )
    ref = GaussianMixture(
        n_components=2, covariance_type="full", n_init=5, reg_covar=1e-6, random_state=1
    ).fit(data)

    mine_order = _align(mine.means)
    ref_order = _align(ref.means_)
    np.testing.assert_allclose(
        mine.weights[mine_order], ref.weights_[ref_order], atol=0.02
    )
    np.testing.assert_allclose(mine.means[mine_order], ref.means_[ref_order], atol=0.05)


def test_missing_data_em_covariances_match_sklearn_under_nontrivial_ridge():
    """On complete data the EM must reproduce sklearn's covariances exactly.

    The fitted covariance is the observed-data MLE: observed entries carry no
    extra measurement noise. A non-trivial `reg_covar` is the ridge added once
    by the M-step (exactly as sklearn does); it must not also inflate the
    covariance used in the E-step's density and conditional moments. With a
    large ridge any double-counting is plainly visible in the covariances.
    """
    data = _simulate_two_component(
        n=6000, weights=_WEIGHTS, means=_MEANS, chols=_CHOLS, seed=11
    )
    reg = 0.1

    mine = fit_gaussian_mixture_missing(
        data, n_components=2, max_iter=800, tol=1e-10, n_init=5, reg_covar=reg, seed=11
    )
    # Match the convergence criteria so both EMs sit at the same fixed point:
    # at convergence the complete-data update equations are identical, so any
    # remaining covariance gap is a genuine math discrepancy, not early stopping.
    ref = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=5,
        reg_covar=reg,
        random_state=11,
        tol=1e-10,
        max_iter=800,
    ).fit(data)

    mine_order = _align(mine.means)
    ref_order = _align(ref.means_)
    np.testing.assert_allclose(
        mine.covariances[mine_order],
        ref.covariances_[ref_order],
        rtol=2e-3,
        atol=2e-3,
    )


def test_missing_data_em_recovers_params_under_mcar():
    """Under ~30% MCAR missingness the EM recovers the true mixture."""
    data = _simulate_two_component(
        n=8000, weights=_WEIGHTS, means=_MEANS, chols=_CHOLS, seed=2
    )
    rng = np.random.default_rng(2)
    data = data.copy()
    data[rng.uniform(size=data.shape) < 0.3] = np.nan

    fit = fit_gaussian_mixture_missing(
        data, n_components=2, max_iter=500, tol=1e-7, n_init=5, reg_covar=1e-6, seed=2
    )

    order = _align(fit.means)
    truth_order = _align(np.vstack(_MEANS))
    np.testing.assert_allclose(
        fit.weights[order], np.array(_WEIGHTS)[truth_order], atol=0.05
    )
    for fk, tk in zip(order, truth_order, strict=True):
        np.testing.assert_allclose(fit.means[fk], _MEANS[tk], atol=0.15)


def test_missing_data_em_fits_when_no_row_is_complete():
    """Fits even when every row misses an entry (the unbalanced-panel case).

    Column 0 is missing for even rows and column 1 for odd rows, so no row is
    complete -- the complete-case GaussianMixture is infeasible -- yet the
    pairwise/marginal information still identifies a single Gaussian's means.
    No row observes both columns, so the cross-covariance is unidentified: the
    fit must flag that (and warn) rather than report ordinary convergence alone.
    """
    rng = np.random.default_rng(3)
    data = rng.normal(loc=[5.0, -3.0], scale=[1.0, 1.0], size=(2000, 2))
    data[::2, 0] = np.nan  # even rows miss column 0
    data[1::2, 1] = np.nan  # odd rows miss column 1
    assert (~np.isnan(data).any(axis=1)).sum() == 0  # no complete rows

    with pytest.warns(RuntimeWarning, match="co-observation"):
        fit = fit_gaussian_mixture_missing(
            data,
            n_components=1,
            max_iter=500,
            tol=1e-7,
            n_init=3,
            reg_covar=1e-6,
            seed=3,
        )

    np.testing.assert_allclose(fit.means[0], [5.0, -3.0], atol=0.1)
    assert fit.cross_covariance_identified is False


def test_missing_data_em_raises_when_a_column_is_never_observed():
    """A column observed in no row leaves its mean and (co)variances unidentified."""
    rng = np.random.default_rng(5)
    data = rng.normal(size=(200, 3))
    data[:, 1] = np.nan  # column 1 never observed

    with pytest.raises(ValueError, match="never observed"):
        fit_gaussian_mixture_missing(
            data,
            n_components=1,
            max_iter=10,
            tol=1e-6,
            n_init=1,
            reg_covar=1e-6,
            seed=0,
        )


def test_missing_data_em_reports_identified_covariance_under_mcar():
    """Under MCAR every column pair is co-observed somewhere, so the flag is True."""
    data = _simulate_two_component(
        n=3000, weights=_WEIGHTS, means=_MEANS, chols=_CHOLS, seed=7
    )
    rng = np.random.default_rng(7)
    data[rng.uniform(size=data.shape) < 0.2] = np.nan

    fit = fit_gaussian_mixture_missing(
        data, n_components=2, max_iter=300, tol=1e-7, n_init=2, reg_covar=1e-6, seed=7
    )

    assert fit.cross_covariance_identified is True


def test_missing_data_em_reports_convergence_and_shapes():
    data = _simulate_two_component(
        n=1000, weights=_WEIGHTS, means=_MEANS, chols=_CHOLS, seed=4
    )
    data[::5, 0] = np.nan

    fit = fit_gaussian_mixture_missing(
        data, n_components=2, max_iter=500, tol=1e-7, n_init=2, reg_covar=1e-6, seed=4
    )

    assert fit.weights.shape == (2,)
    assert fit.means.shape == (2, 2)
    assert fit.covariances.shape == (2, 2, 2)
    assert isinstance(fit.converged, bool)
    assert fit.n_iter >= 1


def test_run_em_returns_loglik_of_returned_params():
    """The returned log-likelihood must score the returned parameters.

    The score is computed in the E-step at the start of each iteration, but the
    parameters are then updated by the M-step. Returning that pre-M-step score
    alongside the post-M-step parameters makes the restart ranking one step
    stale. A single EM step from a deliberately-off start moves the parameters
    a lot, so the stale score lags the true score of the returned params by a
    wide margin.
    """
    data = _simulate_two_component(
        n=2000, weights=_WEIGHTS, means=_MEANS, chols=_CHOLS, seed=21
    )
    mask = np.ones_like(data)
    weights0 = np.array([0.5, 0.5])
    means0 = np.array([[0.0, 0.0], [0.5, -0.5]])  # far from the true means at ±2
    covs0 = np.stack([np.eye(2), np.eye(2)])

    weights, means, covs, loglik, _n_iter, _converged = _run_em(
        data, mask, weights0, means0, covs0, max_iter=1, tol=0.0, reg_covar=1e-6
    )

    oracle = _mixture_loglik(data, weights, means, covs)
    np.testing.assert_allclose(loglik, oracle, rtol=1e-6)


def test_missing_data_em_raises_when_all_columns_missing_for_all_rows():
    data = np.full((10, 2), np.nan)
    with pytest.raises(ValueError, match="no observed"):
        fit_gaussian_mixture_missing(
            data,
            n_components=1,
            max_iter=10,
            tol=1e-6,
            n_init=1,
            reg_covar=1e-6,
            seed=0,
        )
