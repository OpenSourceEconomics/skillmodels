"""Tests for ``skillmodels.af.inference.compute_af_standard_errors``.

The AF inference path is the influence-function score bootstrap of
Antweiler & Freyberger (2025) §4.2 (Armstrong-Bertanha-Hong 2014
style). A single per-observation influence matrix is computed once at
the optimum; each period block carries the earlier periods' influence
via the cross-period blocks of the full-chain Hessian, and the bootstrap
resamples the rows with ONE shared caseid index per replicate so that
cross-period covariances are non-zero and correct. There is no
analytical sandwich path: AF §4.2 explicitly notes the closed-form
variance ignores estimation error in earlier-period nuisance parameters
and is therefore incorrect for any t >= 1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from skillmodels.af.estimate import _extract_period_data, estimate_af
from skillmodels.af.inference import (
    AFInferenceResult,
    _build_period_metas,
    _compute_block_diagonal_sandwich,
    _free_positions_for_period,
    _period_t_per_obs_loglike_full,
    compute_af_standard_errors,
)
from skillmodels.af.types import AFEstimationOptions
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.process_model import process_model


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
    )


@pytest.fixture(scope="module")
def fitted_result() -> tuple[AFInferenceResult, pd.DataFrame]:
    """Fit the AF estimator once and bootstrap SEs; reused across tests."""
    data = _simulate_linear_data(n_obs=400, n_periods=2)
    model = _make_linear_model(n_periods=2)
    af_opts = AFEstimationOptions(
        n_halton_points=25,
        n_halton_points_shock=15,
        optimizer_algorithm="scipy_lbfgsb",
    )
    fit = estimate_af(model_spec=model, data=data, options=af_opts)
    inference = compute_af_standard_errors(fit, data, af_opts, n_boot=2000, seed=0)
    return inference, fit.params


@pytest.mark.end_to_end
def test_af_inference_result_is_inference_dataclass(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    assert isinstance(inference, AFInferenceResult)


@pytest.mark.end_to_end
def test_af_inference_replicate_params_shape(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, all_params = fitted_result
    assert inference.n_boot == 2000
    assert inference.n_clusters == 400
    assert inference.replicate_params.shape == (2000, len(all_params.index))
    assert list(inference.replicate_params.columns) == list(all_params.index)


@pytest.mark.end_to_end
def test_af_inference_standard_errors_index_matches_params(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, all_params = fitted_result
    assert inference.standard_errors.index.equals(all_params.index)


@pytest.mark.end_to_end
def test_af_inference_vcov_row_index_matches_params(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, all_params = fitted_result
    assert inference.vcov.index.equals(all_params.index)


@pytest.mark.end_to_end
def test_af_inference_vcov_column_index_matches_params(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, all_params = fitted_result
    assert inference.vcov.columns.equals(all_params.index)


@pytest.mark.end_to_end
def test_af_inference_vcov_diagonal_matches_se_squared(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    """SEs and vcov are computed from the same replicate distribution."""
    inference, _ = fitted_result
    diag = np.diag(inference.vcov.to_numpy())
    se_squared = inference.standard_errors.to_numpy() ** 2
    np.testing.assert_allclose(diag, se_squared, rtol=1e-10, atol=1e-12)


@pytest.mark.end_to_end
def test_af_inference_pinned_loading_has_zero_se(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    assert float(inference.standard_errors.loc[("loadings", 0, "m1", "skill")]) == (
        pytest.approx(0.0, abs=1e-12)
    )


@pytest.mark.end_to_end
def test_af_inference_pinned_intercept_has_zero_se(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    assert float(
        inference.standard_errors.loc[("controls", 0, "m1", "constant")]
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.end_to_end
def test_af_inference_free_loading_has_positive_se(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    assert inference.standard_errors.loc[("loadings", 0, "m2", "skill")] > 0.0


@pytest.mark.end_to_end
def test_af_inference_free_meas_sd_has_positive_se(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    inference, _ = fitted_result
    assert inference.standard_errors.loc[("meas_sds", 0, "m2", "-")] > 0.0


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
def test_af_inference_pinned_params_have_constant_replicates(
    fitted_result: tuple[AFInferenceResult, pd.DataFrame],
) -> None:
    """Loadings/intercepts pinned via Normalizations are constant across replicates."""
    inference, _ = fitted_result
    pinned = [("loadings", t, "m1", "skill") for t in (0, 1)] + [
        ("controls", t, "m1", "constant") for t in (0, 1)
    ]
    for loc in pinned:
        if loc in inference.replicate_params.columns:
            col = inference.replicate_params[loc].to_numpy()
            assert col.std() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.end_to_end
def test_af_inference_se_shrinks_with_sample_size() -> None:
    """SE for a representative free parameter should shrink roughly as 1/sqrt(n)."""
    model = _make_linear_model(n_periods=2)
    af_opts = AFEstimationOptions(
        n_halton_points=25,
        n_halton_points_shock=15,
        optimizer_algorithm="scipy_lbfgsb",
    )

    data_small = _simulate_linear_data(n_obs=200, n_periods=2, seed=1)
    data_large = _simulate_linear_data(n_obs=800, n_periods=2, seed=1)

    fit_small = estimate_af(model_spec=model, data=data_small, options=af_opts)
    fit_large = estimate_af(model_spec=model, data=data_large, options=af_opts)

    inf_small = compute_af_standard_errors(
        fit_small, data_small, af_opts, n_boot=2000, seed=1
    )
    inf_large = compute_af_standard_errors(
        fit_large, data_large, af_opts, n_boot=2000, seed=1
    )

    loc = ("loadings", 0, "m2", "skill")
    se_small = float(inf_small.standard_errors.loc[loc])
    se_large = float(inf_large.standard_errors.loc[loc])

    # Sample size quadrupled: expect SE ~ halved. Tolerate a wide band
    # because the bootstrap is noisy on moderate samples.
    ratio = se_large / se_small
    assert 0.25 < ratio < 0.8, (
        f"Expected SE ratio in (0.25, 0.8) under 4x sample-size bump; "
        f"got {ratio:.3f} (se_small={se_small}, se_large={se_large})"
    )


# ---------------------------------------------------------------------------
# AF-F1: influence-function score bootstrap that propagates earlier-period
# estimation uncertainty into later periods.
# ---------------------------------------------------------------------------


def _build_metas_for_test(fit, data, af_opts):
    """Rebuild the per-period inference metas (mirrors compute_af_standard_errors)."""
    model_spec = fit.model_spec
    processed_model = process_model(model_spec)
    n_periods = processed_model.dimensions.n_periods
    latent_factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    observed_factors = processed_model.labels.observed_factors
    endog_info = processed_model.endogenous_factors_info
    endogenous_factors = tuple(
        f
        for f in latent_factors
        if f in endog_info.factor_info and endog_info.factor_info[f].is_endogenous
    )
    period_data = _extract_period_data(
        data,
        n_periods,
        latent_factors,
        controls_names,
        model_spec,
        observed_factors=observed_factors,
    )
    return _build_period_metas(
        result=fit,
        period_data=period_data,
        model_spec=model_spec,
        processed_model=processed_model,
        af_options=af_opts,
        observed_factors=observed_factors,
        endogenous_factors=endogenous_factors,
    )


@pytest.fixture(scope="module")
def fit_and_metas():
    """Fit the 2-period linear model and build inference metas; reused across tests."""
    data = _simulate_linear_data(n_obs=400, n_periods=2)
    model = _make_linear_model(n_periods=2)
    af_opts = AFEstimationOptions(
        n_halton_points=25,
        n_halton_points_shock=15,
        optimizer_algorithm="scipy_lbfgsb",
    )
    fit = estimate_af(model_spec=model, data=data, options=af_opts)
    metas = _build_metas_for_test(fit, data, af_opts)
    return fit, data, af_opts, metas


@pytest.mark.end_to_end
def test_af_inference_cross_period_covariance_is_nonzero(
    fit_and_metas,
) -> None:
    """Propagation produces non-zero cross-period vcov entries.

    A period-0 free parameter that period-1 depends on (a transition
    slope and a free loading) must have non-zero covariance with a
    period-1 free parameter. The old own-block independent-resample
    bootstrap has E[cross-period cov] = 0; the influence-function fix
    shares the caseid index so psi_1 carries B_{1,0} psi_0.
    """
    fit, data, af_opts, _metas = fit_and_metas
    inference = compute_af_standard_errors(fit, data, af_opts, n_boot=4000, seed=0)
    vcov = inference.vcov

    p0_candidates = [
        ("transition", 0, "skill", "skill"),
        ("loadings", 0, "m2", "skill"),
    ]
    p1_candidates = [
        ("loadings", 1, "m2", "skill"),
        ("transition", 1, "skill", "skill"),
    ]
    p0 = next(c for c in p0_candidates if c in vcov.index)
    p1 = next(c for c in p1_candidates if c in vcov.index)

    cross_cov = float(np.asarray(vcov.loc[p0, p1]).item())
    assert abs(cross_cov) > 1e-8, (
        f"Expected non-zero cross-period covariance between {p0} and {p1}; "
        f"got {cross_cov}."
    )


@pytest.mark.end_to_end
def test_af_inference_propagation_inflates_later_period_se(
    fit_and_metas,
) -> None:
    """The propagated period-1 SE exceeds the old own-block SE.

    Replicate the OLD independent-resample own-block SE inline for a
    period-1 free parameter, then assert the NEW propagated SE is
    strictly larger (it adds the PSD term Var(A_1^{-1} B_{1,0} psi_0)).
    """
    fit, data, af_opts, metas = fit_and_metas
    inference = compute_af_standard_errors(fit, data, af_opts, n_boot=4000, seed=0)

    loc = ("loadings", 1, "m2", "skill")
    new_se = float(inference.standard_errors.loc[loc])

    # OLD own-block bootstrap SE for the same period-1 parameter.
    period_score_info = _compute_block_diagonal_sandwich(fit, metas)
    period1 = next(p for p in period_score_info if p.period == 1)
    score = np.asarray(period1.score_matrix)
    info = np.asarray(period1.information_matrix)
    a_inv = np.linalg.pinv(info)

    n_clusters = int(metas[0].loglike_kwargs["measurements"].shape[0])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_clusters, size=(4000, n_clusters))
    mean_score = score[idx].mean(axis=1)
    delta = -mean_score @ a_inv.T  # (n_boot, n_free_own)

    own_col = list(period1.free_param_locs).index(loc)
    old_se = float(np.std(delta[:, own_col], ddof=1))

    assert new_se > old_se * 1.001, (
        f"Expected propagated SE ({new_se}) to exceed own-block SE "
        f"({old_se}) by the propagation term."
    )


@pytest.mark.end_to_end
def test_af_inference_period1_information_matches_fullchain_ownblock(
    fit_and_metas,
) -> None:
    """The full-chain Hessian's period-1 own block matches the block-diagonal one.

    Validates the full-chain Hessian wiring used by the influence matrix:
    the period-1 own information sub-block recovered from
    hessian(_period_t_per_obs_loglike_full) must match the existing
    _block_diagonal_sandwich_single information matrix for period 1.
    """
    fit, _data, _af_opts, metas = fit_and_metas
    flat_super = jnp.asarray(fit.params["value"].to_numpy())

    period_score_info = _compute_block_diagonal_sandwich(fit, metas)
    period1 = next(p for p in period_score_info if p.period == 1)
    own_block_info = np.asarray(period1.information_matrix)

    meta1 = metas[1]
    pos, _locs = _free_positions_for_period(meta1.params_df)
    own_global = jnp.array([meta1.slice_start + p for p in pos], dtype=jnp.int32)

    def neg_mean(fs):
        return -jnp.mean(_period_t_per_obs_loglike_full(fs, 1, metas))

    hess_full = jax.hessian(neg_mean)(flat_super)
    fullchain_own = np.asarray(hess_full[own_global][:, own_global])

    np.testing.assert_allclose(fullchain_own, own_block_info, rtol=1e-4, atol=1e-8)
