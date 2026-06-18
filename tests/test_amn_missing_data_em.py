"""Tests for the missing-data Gaussian-mixture EM (AMN Stage 1 fallback)."""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from skillmodels.amn.missing_data_em import fit_gaussian_mixture_missing


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
    """
    rng = np.random.default_rng(3)
    data = rng.normal(loc=[5.0, -3.0], scale=[1.0, 1.0], size=(2000, 2))
    data[::2, 0] = np.nan  # even rows miss column 0
    data[1::2, 1] = np.nan  # odd rows miss column 1
    assert (~np.isnan(data).any(axis=1)).sum() == 0  # no complete rows

    fit = fit_gaussian_mixture_missing(
        data, n_components=1, max_iter=500, tol=1e-7, n_init=3, reg_covar=1e-6, seed=3
    )

    np.testing.assert_allclose(fit.means[0], [5.0, -3.0], atol=0.1)


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
