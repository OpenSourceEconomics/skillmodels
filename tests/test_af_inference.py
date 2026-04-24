"""Tests for ``skillmodels.af.inference.compute_af_standard_errors``."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.af.estimate import estimate_af
from skillmodels.af.inference import (
    AFInferenceResult,
    AFPeriodInferenceResult,
    compute_af_standard_errors,
)
from skillmodels.af.types import AFEstimationOptions
from skillmodels.model_spec import (
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _simulate_linear_data(
    *,
    n_obs: int,
    n_periods: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a simple single-factor linear-transition panel."""
    rng = np.random.default_rng(seed)
    theta = np.zeros((n_obs, n_periods))
    theta[:, 0] = rng.normal(0.0, 1.0, n_obs)
    for t in range(n_periods - 1):
        theta[:, t + 1] = 0.1 + 0.7 * theta[:, t] + rng.normal(0.0, 0.3, n_obs)

    loadings = (1.0, 0.9, 1.1)
    intercepts = (0.0, 0.2, -0.1)
    sds = (0.3, 0.4, 0.35)
    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {"caseid": i, "period": t}
            for m_idx, meas in enumerate(("m1", "m2", "m3")):
                row[meas] = (
                    intercepts[m_idx]
                    + loadings[m_idx] * theta[i, t]
                    + rng.normal(0, sds[m_idx])
                )
            rows.append(row)

    return pd.DataFrame(rows).set_index(["caseid", "period"])


def _make_linear_model(n_periods: int = 2) -> ModelSpec:
    return ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("m1", "m2", "m3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"m1": 1},) * n_periods,
                    intercepts=({"m1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
        },
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )


@pytest.fixture(scope="module")
def fitted_result() -> tuple[AFInferenceResult, pd.DataFrame]:
    """Fit the AF estimator once and compute SEs; reused across tests."""
    data = _simulate_linear_data(n_obs=400, n_periods=2)
    model = _make_linear_model(n_periods=2)
    af_opts = AFEstimationOptions(
        n_halton_points=25,
        n_halton_points_shock=15,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )
    fit = estimate_af(model_spec=model, data=data, af_options=af_opts)
    inference = compute_af_standard_errors(fit, data, af_opts)
    return inference, fit.all_params


@pytest.mark.end_to_end
def test_af_inference_returns_expected_dataclass(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    assert isinstance(inference, AFInferenceResult)
    assert all(isinstance(p, AFPeriodInferenceResult) for p in inference.period_results)


@pytest.mark.end_to_end
def test_af_inference_standard_errors_align_with_params(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, all_params = fitted_result
    assert inference.standard_errors.index.equals(all_params.index)
    assert inference.vcov.index.equals(all_params.index)
    assert inference.vcov.columns.equals(all_params.index)


@pytest.mark.end_to_end
def test_af_inference_fixed_entries_have_zero_se(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    """Normalization pins (e.g. loadings[m1, skill] == 1) must have SE = 0."""
    inference, all_params = fitted_result
    se = inference.standard_errors

    pinned_loading = ("loadings", 0, "m1", "skill")
    assert pinned_loading in all_params.index
    assert se.loc[pinned_loading] == 0.0

    pinned_intercept = ("controls", 0, "m1", "constant")
    assert pinned_intercept in all_params.index
    assert se.loc[pinned_intercept] == 0.0


@pytest.mark.end_to_end
def test_af_inference_free_params_have_positive_se(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    """Free (unpinned) measurement parameters should have strictly positive SE."""
    inference, all_params = fitted_result
    se = inference.standard_errors

    free_loading = ("loadings", 0, "m2", "skill")
    assert free_loading in all_params.index
    assert se.loc[free_loading] > 0.0

    free_sd = ("meas_sds", 0, "m2", "-")
    assert free_sd in all_params.index
    assert se.loc[free_sd] > 0.0


@pytest.mark.end_to_end
def test_af_inference_vcov_is_symmetric(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    v = inference.vcov.to_numpy()
    np.testing.assert_allclose(v, v.T, atol=1e-10)


@pytest.mark.end_to_end
def test_af_inference_vcov_diagonal_nonnegative(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    diag = np.diag(inference.vcov.to_numpy())
    assert np.all(diag >= 0.0)


@pytest.mark.end_to_end
def test_af_inference_score_matrix_row_count_matches_n_obs(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    n_obs = 400
    for period_res in inference.period_results:
        assert int(period_res.score_matrix.shape[0]) == n_obs


@pytest.mark.end_to_end
def test_af_inference_se_shrinks_with_sample_size() -> None:
    """SE for a representative free parameter should shrink roughly as 1/sqrt(n)."""
    model = _make_linear_model(n_periods=2)
    af_opts = AFEstimationOptions(
        n_halton_points=25,
        n_halton_points_shock=15,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    data_small = _simulate_linear_data(n_obs=200, n_periods=2, seed=1)
    data_large = _simulate_linear_data(n_obs=800, n_periods=2, seed=1)

    fit_small = estimate_af(model_spec=model, data=data_small, af_options=af_opts)
    fit_large = estimate_af(model_spec=model, data=data_large, af_options=af_opts)

    inf_small = compute_af_standard_errors(fit_small, data_small, af_opts)
    inf_large = compute_af_standard_errors(fit_large, data_large, af_opts)

    loc = ("loadings", 0, "m2", "skill")
    se_small = float(inf_small.standard_errors.loc[loc])
    se_large = float(inf_large.standard_errors.loc[loc])

    # Sample size quadrupled: expect SE ~ halved. Tolerate a wide band
    # because the sandwich is noisy on moderate samples.
    ratio = se_large / se_small
    assert 0.25 < ratio < 0.8, (
        f"Expected SE ratio in (0.25, 0.8) under 4x sample-size bump; "
        f"got {ratio:.3f} (se_small={se_small}, se_large={se_large})"
    )
