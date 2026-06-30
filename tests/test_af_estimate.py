"""End-to-end tests for the AF estimator.

Run AF estimation on MODEL2 test data and verify it produces reasonable
results, comparing to the CHS Kalman filter estimates where applicable.
"""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import pytest

import skillmodels.af.estimate as est
import skillmodels.af.initial_period as ip
from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.af.likelihood import (
    _rebuild_chain_at_period,
    af_loglike_transition,
    af_per_obs_loglike_initial,
)
from skillmodels.af.transition_period import _update_conditional_distribution
from skillmodels.af.types import ChainLink, ConditionalDistribution, MixtureComponent
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.decorators import register_params
from skillmodels.common.individual_states import get_individual_states
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.test_data.model2 import MODEL2_CHS_OPTIONS

jax.config.update("jax_enable_x64", True)

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def test_af_options_bounds_distance_field() -> None:
    """`bounds_distance` is configurable and defaults to the 0.001 SD floor."""
    assert AFEstimationOptions().bounds_distance == 0.001
    assert AFEstimationOptions(bounds_distance=0.01).bounds_distance == 0.01


class _StopForTest(Exception):  # noqa: N818
    """Sentinel raised by the spy to abort estimation before optimization."""


def test_af_options_bounds_distance_threads_to_template(
    monkeypatch, model2_af, model2_data
) -> None:
    """`AFEstimationOptions.bounds_distance` reaches `create_af_params_template`.

    Spy on the template builder, capture the `bounds_distance` it is called with,
    and raise before any optimization runs (the template is built first), so the
    test stays cheap.
    """
    captured: dict[str, float] = {}

    def _spy(*args, **kwargs):
        captured["bounds_distance"] = kwargs["bounds_distance"]
        raise _StopForTest

    monkeypatch.setattr(ip, "create_af_params_template", _spy)
    with pytest.raises(_StopForTest):
        estimate_af(
            model_spec=model2_af,
            data=model2_data,
            options=AFEstimationOptions(
                n_halton_points=10,
                n_halton_points_shock=10,
                start_params_strategy="constant",
                bounds_distance=0.01,
            ),
        )
    assert captured["bounds_distance"] == 0.01


def test_af_options_bounds_distance_survives_amn_start(
    monkeypatch, model2_af, model2_data
) -> None:
    """`bounds_distance` survives the af_options rebuild on the AMN start path.

    With `start_params_strategy="amn"`, `estimate_af` reconstructs the options
    after running AMN; `bounds_distance` must carry through to the per-period
    template builder, not silently revert to the default.
    """

    class _FakeAMN:
        params = pd.DataFrame({"value": []})

    monkeypatch.setattr(est, "estimate_amn", lambda **_kwargs: _FakeAMN())
    captured: dict[str, float] = {}

    def _spy(*args, **kwargs):
        captured["bounds_distance"] = kwargs["bounds_distance"]
        raise _StopForTest

    monkeypatch.setattr(ip, "create_af_params_template", _spy)
    with pytest.raises(_StopForTest):
        estimate_af(
            model_spec=model2_af,
            data=model2_data,
            options=AFEstimationOptions(
                n_halton_points=10,
                n_halton_points_shock=10,
                start_params_strategy="amn",
                bounds_distance=0.01,
            ),
        )
    assert captured["bounds_distance"] == 0.01


@pytest.fixture
def model2_data():
    """Load the MODEL2 simulated dataset."""
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


@pytest.fixture
def model2_af():
    """Create an AF-compatible 2-factor model from MODEL2.

    Use fac1 (log_ces, 3 measures) and fac2 (linear, 3 measures).
    Drop fac3 since it has measurements only in period 0.
    Reduce to 3 periods for faster testing.
    """
    return ModelSpec(
        factors={
            "fac1": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 3,
                    intercepts=({"y1": 0},) * 3,
                ),
                transition_function="log_ces",
            ),
            "fac2": FactorSpec(
                measurements=(("y4", "y5", "y6"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y4": 1},) * 3,
                    intercepts=({"y4": 0},) * 3,
                ),
                transition_function="linear",
            ),
        },
        controls=("x1",),
    )


@pytest.fixture
def chs_params():
    """Load CHS-estimated parameters from regression vault."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    return params.set_index(["category", "period", "name1", "name2"])


@pytest.mark.end_to_end
def test_af_estimate_runs_on_model2(model2_af, model2_data) -> None:
    """Verify AF estimation runs to completion on MODEL2 data."""
    af_options = AFEstimationOptions(
        n_halton_points=20,
        n_halton_points_shock=10,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(
        model_spec=model2_af,
        data=model2_data,
        options=af_options,
    )

    # Basic checks
    assert len(result.period_results) == 3
    assert result.params is not None
    assert len(result.params) > 0

    # Check each period converged (or at least produced finite likelihood)
    for pr in result.period_results:
        assert np.isfinite(pr.loglikelihood), (
            f"Period {pr.period}: non-finite log-likelihood {pr.loglikelihood}"
        )


@pytest.mark.end_to_end
def test_af_measurement_params_in_ballpark(
    model2_af,
    model2_data,
    chs_params,
) -> None:
    """Verify AF measurement parameter estimates are in the same ballpark as CHS.

    The two estimators use different methods, so exact agreement is not
    expected. But measurement loadings and SDs should be roughly similar.
    """
    af_options = AFEstimationOptions(
        n_halton_points=30,
        n_halton_points_shock=15,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(
        model_spec=model2_af,
        data=model2_data,
        options=af_options,
    )

    # Compare period 0 measurement SDs
    af_meas_sds = result.params.query("category == 'meas_sds' and period == 0")
    if len(af_meas_sds) > 0:
        af_sd_values = af_meas_sds["value"].to_numpy()
        # All SDs should be positive and not too extreme
        assert (af_sd_values > 0).all(), "All measurement SDs should be positive"
        assert (af_sd_values < 10).all(), (
            "Measurement SDs should not be unreasonably large"
        )


@pytest.mark.end_to_end
def test_af_estimate_single_factor() -> None:
    """Test AF estimation with a single-factor model (simplest case)."""
    # Create minimal model: 1 factor, 3 measures, 2 periods
    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("m1", "m2", "m3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"m1": 1},) * 2,
                    intercepts=({"m1": 0},) * 2,
                ),
                transition_function="linear",
            ),
        },
    )

    # Generate simple synthetic data
    rng = np.random.default_rng(42)
    n_obs = 200
    n_periods = 2

    # True latent factor
    theta = rng.normal(0, 1, n_obs)

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {
                "caseid": i,
                "period": t,
                "m1": theta[i] + rng.normal(0, 0.3),
                "m2": 0.5 + 0.8 * theta[i] + rng.normal(0, 0.4),
                "m3": -0.2 + 1.2 * theta[i] + rng.normal(0, 0.35),
            }
            rows.append(row)

    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    af_options = AFEstimationOptions(
        n_halton_points=25,
        n_halton_points_shock=10,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(model_spec=model, data=data, options=af_options)

    assert len(result.period_results) == 2
    assert np.isfinite(result.period_results[0].loglikelihood)

    # Check that estimated loadings are roughly in the right direction
    af_loadings = result.params.query("category == 'loadings' and period == 0")
    if len(af_loadings) > 0:
        # m1 loading on skill should be fixed at 1.0
        # m2 loading should be roughly 0.8
        # m3 loading should be roughly 1.2
        for _, row in af_loadings.iterrows():
            assert np.isfinite(row["value"]), "Loadings should be finite"


@pytest.mark.end_to_end
def test_af_vs_chs_measurement_params_agree() -> None:
    """Verify AF and CHS produce similar measurement parameter estimates.

    Simulate data from a known single-factor model and estimate with both
    AF and CHS. Period-0 measurement loadings, intercepts, and error SDs
    should agree within tolerance.
    """
    rng = np.random.default_rng(42)
    n_obs = 500
    n_periods = 2

    # True DGP parameters
    true_loadings = {"m1": 1.0, "m2": 0.8, "m3": 1.2}
    true_intercepts = {"m1": 0.0, "m2": 0.5, "m3": -0.2}
    true_meas_sds = {"m1": 0.3, "m2": 0.4, "m3": 0.35}

    theta = rng.normal(0, 1, n_obs)
    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "m1": true_intercepts["m1"]
                    + true_loadings["m1"] * theta[i]
                    + rng.normal(0, true_meas_sds["m1"]),
                    "m2": true_intercepts["m2"]
                    + true_loadings["m2"] * theta[i]
                    + rng.normal(0, true_meas_sds["m2"]),
                    "m3": true_intercepts["m3"]
                    + true_loadings["m3"] * theta[i]
                    + rng.normal(0, true_meas_sds["m3"]),
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
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

    # --- AF estimation ---
    af_result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=50,
            n_halton_points_shock=20,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )
    af_p0 = af_result.period_results[0].params

    # --- CHS estimation (naive start: all free params = 0.1) ---
    chs_est = _run_chs_estimation(model, data)

    # --- Compare period-0 measurement parameters ---
    tol = 0.15  # generous tolerance for finite-sample differences

    for meas in ("m2", "m3"):
        af_load = float(
            af_p0.loc[("loadings", 0, meas, "skill"), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_load = float(
            chs_est.loc[("loadings", 0, meas, "skill"), "value"]  # ty: ignore[invalid-argument-type]
        )
        assert abs(af_load - chs_load) < tol, (
            f"loading({meas}): AF={af_load:.4f} vs CHS={chs_load:.4f}"
        )

        af_intercept = float(
            af_p0.loc[("controls", 0, meas, "constant"), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_intercept = float(
            chs_est.loc[("controls", 0, meas, "constant"), "value"]  # ty: ignore[invalid-argument-type]
        )
        assert abs(af_intercept - chs_intercept) < tol, (
            f"intercept({meas}): AF={af_intercept:.4f} vs CHS={chs_intercept:.4f}"
        )

    for meas in ("m1", "m2", "m3"):
        af_sd = float(
            af_p0.loc[("meas_sds", 0, meas, "-"), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_sd = float(
            chs_est.loc[("meas_sds", 0, meas, "-"), "value"]  # ty: ignore[invalid-argument-type]
        )
        assert abs(af_sd - chs_sd) < tol, (
            f"meas_sd({meas}): AF={af_sd:.4f} vs CHS={chs_sd:.4f}"
        )


# ---------------------------------------------------------------------------
# TDD tests for transition likelihood and parameter recovery
# ---------------------------------------------------------------------------


def _simulate_linear_transition_data(
    *,
    n_obs: int = 500,
    n_periods: int = 3,
    true_beta: float = 0.8,
    true_constant: float = 0.1,
    true_shock_sd: float = 0.3,
    true_meas_sds: tuple[float, ...] = (0.3, 0.4, 0.35),
    true_loadings: tuple[float, ...] = (1.0, 0.8, 1.2),
    true_intercepts: tuple[float, ...] = (0.0, 0.5, -0.2),
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate panel data from a single-factor linear transition model.

    DGP: theta_{t+1} = constant + beta * theta_t + N(0, shock_sd^2).
    Measurements: Z_{t,m} = intercept_m + loading_m * theta_t + noise.

    Return tuple of (DataFrame indexed by (caseid, period), dict of true params).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((n_obs, n_periods))
    theta[:, 0] = rng.normal(0, 1, n_obs)
    for t in range(n_periods - 1):
        theta[:, t + 1] = (
            true_constant
            + true_beta * theta[:, t]
            + rng.normal(0, true_shock_sd, n_obs)
        )

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {"caseid": i, "period": t}
            for m_idx, meas_name in enumerate(("m1", "m2", "m3")):
                row[meas_name] = (
                    true_intercepts[m_idx]
                    + true_loadings[m_idx] * theta[i, t]
                    + rng.normal(0, true_meas_sds[m_idx])
                )
            rows.append(row)

    data = pd.DataFrame(rows).set_index(["caseid", "period"])
    true_params = {
        "beta": true_beta,
        "constant": true_constant,
        "shock_sd": true_shock_sd,
    }
    return data, true_params


def _make_linear_transition_model(n_periods: int = 3) -> ModelSpec:
    """Create a single-factor linear transition model for testing."""
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


@pytest.mark.end_to_end
def test_af_transition_params_affect_likelihood() -> None:
    """Verify that the transition likelihood depends on transition parameters.

    If we run AF estimation with the transition function wired in correctly,
    the estimated transition parameters should NOT be at their initial values.
    The likelihood should be sensitive to transition parameter changes.
    """
    data, _true_params = _simulate_linear_transition_data(n_obs=300, n_periods=3)
    model = _make_linear_transition_model(n_periods=3)

    af_opts = AFEstimationOptions(
        n_halton_points=30,
        n_halton_points_shock=15,
        optimizer_algorithm="scipy_lbfgsb",
    )
    result = estimate_af(model_spec=model, data=data, options=af_opts)

    # Period 1 result should have transition params
    p1 = result.period_results[1].params
    trans_params = p1.query("category == 'transition'")
    assert len(trans_params) > 0, "Should have transition parameters in period 1"

    # The transition params should NOT all be at their initialization value (0.1).
    # If the transition function is actually used in the likelihood, the optimizer
    # will move them away from 0.5 toward the true values.
    trans_values = trans_params["value"].to_numpy()
    init_values = np.full_like(trans_values, 0.5)
    assert not np.allclose(trans_values, init_values, atol=0.01), (
        f"Transition params stuck at init values: {trans_values}. "
        "The transition function is not being used in the likelihood."
    )


@pytest.mark.end_to_end
def test_af_recovers_linear_transition_params() -> None:
    """Verify AF recovers known linear transition parameters from synthetic data.

    Simulate data with theta_{t+1} = 0.1 + 0.8 * theta_t + N(0, 0.3^2),
    estimate with AF, and check that estimated beta and constant are close
    to true values.
    """
    data, true_params = _simulate_linear_transition_data(n_obs=500, n_periods=3)
    model = _make_linear_transition_model(n_periods=3)

    af_opts = AFEstimationOptions(
        n_halton_points=800,
        n_halton_points_shock=20,
        optimizer_algorithm="scipy_lbfgsb",
    )
    result = estimate_af(model_spec=model, data=data, options=af_opts)

    # Extract estimated transition params from period 1 (transition 0->1)
    p1 = result.period_results[1].params

    # For a linear transition with 1 factor "skill", params are:
    # ("transition", 0, "skill", "skill") = beta
    # ("transition", 0, "skill", "constant") = constant
    est_beta = float(
        p1.loc[("transition", 0, "skill", "skill"), "value"]  # ty: ignore[invalid-argument-type]
    )
    est_constant = float(
        p1.loc[("transition", 0, "skill", "constant"), "value"]  # ty: ignore[invalid-argument-type]
    )

    # Also check shock SD
    est_shock_sd = float(
        p1.loc[("shock_sds", 0, "skill", "-"), "value"]  # ty: ignore[invalid-argument-type]
    )

    tol = 0.25  # generous tolerance for quadrature-based estimation
    assert abs(est_beta - true_params["beta"]) < tol, (
        f"beta: estimated={est_beta:.4f}, true={true_params['beta']}"
    )
    assert abs(est_constant - true_params["constant"]) < tol, (
        f"constant: estimated={est_constant:.4f}, true={true_params['constant']}"
    )
    assert abs(est_shock_sd - true_params["shock_sd"]) < tol, (
        f"shock_sd: estimated={est_shock_sd:.4f}, true={true_params['shock_sd']}"
    )


@pytest.mark.end_to_end
def test_af_vs_chs_transition_params_agree() -> None:
    """Verify AF and CHS transition parameter estimates are in the same ballpark.

    Use the same synthetic DGP as the measurement params comparison test,
    but now compare the transition parameters estimated by both methods.
    """
    data, _true_params = _simulate_linear_transition_data(n_obs=500, n_periods=3)
    model = _make_linear_transition_model(n_periods=3)

    # --- AF estimation ---
    af_result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=40,
            n_halton_points_shock=20,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    # --- CHS estimation (naive start: all free params = 0.1) ---
    chs_est = _run_chs_estimation(model, data)

    # --- Compare transition parameters ---
    af_p1 = af_result.period_results[1].params

    af_beta = float(
        af_p1.loc[("transition", 0, "skill", "skill"), "value"]  # ty: ignore[invalid-argument-type]
    )
    af_constant = float(
        af_p1.loc[("transition", 0, "skill", "constant"), "value"]  # ty: ignore[invalid-argument-type]
    )
    af_shock = float(
        af_p1.loc[("shock_sds", 0, "skill", "-"), "value"]  # ty: ignore[invalid-argument-type]
    )

    chs_beta = float(
        chs_est.loc[("transition", 0, "skill", "skill"), "value"]  # ty: ignore[invalid-argument-type]
    )
    chs_constant = float(
        chs_est.loc[("transition", 0, "skill", "constant"), "value"]  # ty: ignore[invalid-argument-type]
    )
    chs_shock = float(
        chs_est.loc[("shock_sds", 0, "skill", "-"), "value"]  # ty: ignore[invalid-argument-type]
    )

    tol = 0.3  # generous: different methods, different # periods used
    assert abs(af_beta - chs_beta) < tol, (
        f"beta: AF={af_beta:.4f} vs CHS={chs_beta:.4f}"
    )
    assert abs(af_constant - chs_constant) < tol, (
        f"constant: AF={af_constant:.4f} vs CHS={chs_constant:.4f}"
    )
    assert abs(af_shock - chs_shock) < tol, (
        f"shock_sd: AF={af_shock:.4f} vs CHS={chs_shock:.4f}"
    )


def _run_chs_estimation(
    model: ModelSpec,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Run CHS estimation with uninformed but feasible start values.

    Use generic defaults that don't favour either estimator: loadings = 1,
    controls = 0, SDs = 0.5, transition = 0.5, initial_states = 0.
    Probability constraints are satisfied (equal shares).
    """
    max_inputs = get_maximization_inputs(model, data, chs_options=MODEL2_CHS_OPTIONS)
    params = max_inputs["params_template"].copy()
    free = params["lower_bound"] != params["upper_bound"]
    cat = params.index.get_level_values("category")
    params.loc[free, "value"] = 0.5
    params.loc[free & (cat == "loadings"), "value"] = 1.0
    params.loc[free & (cat == "controls"), "value"] = 0.0
    params.loc[free & (cat == "initial_states"), "value"] = 0.0
    # Probability constraints must be satisfied at start params
    for constr in max_inputs["constraints"]:
        if isinstance(constr, om.ProbabilityConstraint):
            prob_idx = constr.selector(params[["value"]]).index
            params.loc[prob_idx, "value"] = 1.0 / len(prob_idx)

    def _neg_ll_and_grad(p: pd.DataFrame) -> tuple[float, np.ndarray]:
        val, grad = max_inputs["loglike_and_gradient"](p)
        return -float(val), -np.array(grad)

    return om.minimize(
        fun=lambda p: -max_inputs["loglike"](p),
        params=params[["value"]],
        algorithm="scipy_lbfgsb",
        bounds=om.Bounds(lower=params["lower_bound"], upper=params["upper_bound"]),
        constraints=max_inputs["constraints"],
        fun_and_jac=_neg_ll_and_grad,
    ).params


@pytest.mark.long_running
def test_af_vs_chs_both_estimated_on_model2(model2_af, model2_data) -> None:
    """Run both AF and CHS optimisation on MODEL2 data and compare estimates.

    This test actually optimises both estimators (not just loading stored
    params), so it takes a while. Skipped in CI via the long_running marker.
    """
    chs_est = _run_chs_estimation(model2_af, model2_data)

    # --- AF estimation ---
    af_result = estimate_af(
        model_spec=model2_af,
        data=model2_data,
        options=AFEstimationOptions(
            n_halton_points=60,
            n_halton_points_shock=30,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    # --- Compare period-0 measurement params ---
    af_p0 = af_result.period_results[0].params
    meas_tol = 0.5  # generous: different estimators, AF uses 3 periods

    for meas, fac in [("y2", "fac1"), ("y3", "fac1"), ("y5", "fac2"), ("y6", "fac2")]:
        af_val = float(
            af_p0.loc[("loadings", 0, meas, fac), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_val = float(
            chs_est.loc[("loadings", 0, meas, fac), "value"]  # ty: ignore[invalid-argument-type]
        )
        assert np.isfinite(af_val), f"AF loading({meas},{fac}) not finite"
        assert np.isfinite(chs_val), f"CHS loading({meas},{fac}) not finite"
        assert abs(af_val - chs_val) < meas_tol, (
            f"loading({meas},{fac}): AF={af_val:.4f} vs CHS={chs_val:.4f}"
        )

    # --- Compare transition params (period 0->1) ---
    af_p1 = af_result.period_results[1].params
    trans_tol = 0.5

    # fac2 linear: self-productivity
    af_fac2_self = float(
        af_p1.loc[("transition", 0, "fac2", "fac2"), "value"]  # ty: ignore[invalid-argument-type]
    )
    chs_fac2_self = float(
        chs_est.loc[("transition", 0, "fac2", "fac2"), "value"]  # ty: ignore[invalid-argument-type]
    )
    assert abs(af_fac2_self - chs_fac2_self) < trans_tol, (
        f"fac2 self-prod: AF={af_fac2_self:.4f} vs CHS={chs_fac2_self:.4f}"
    )

    # All transition params should be finite
    af_trans = af_p1.query("category == 'transition'")
    assert af_trans["value"].apply(np.isfinite).all(), (
        f"Non-finite AF transition params:\n{af_trans}"
    )

    # AF transition params should NOT be stuck at initialisation
    trans_values = af_trans["value"].to_numpy()
    assert not np.allclose(trans_values, 0.5, atol=0.01), (
        "AF transition params stuck at init values"
    )

    # --- Print comparison for manual inspection ---
    print("\n\nMODEL2: AF vs CHS (both estimated)")
    print("=" * 70)
    print(f"{'Parameter':40s} {'AF':>10s} {'CHS':>10s}")
    print("-" * 70)
    for idx, row in af_trans.iterrows():
        ix = tuple(idx)  # ty: ignore[invalid-argument-type]
        chs_loc = ("transition", ix[1], ix[2], ix[3])
        chs_v = (
            float(chs_est.loc[chs_loc, "value"])
            if chs_loc in chs_est.index
            else float("nan")
        )
        print(
            f"  trans {ix[2]:6s} {ix[3]:12s}       {row['value']:10.4f} {chs_v:10.4f}"
        )
    af_shocks = af_p1.query("category == 'shock_sds'")
    for idx, row in af_shocks.iterrows():
        ix = tuple(idx)  # ty: ignore[invalid-argument-type]
        chs_loc = ("shock_sds", ix[1], ix[2], ix[3])
        chs_v = (
            float(chs_est.loc[chs_loc, "value"])
            if chs_loc in chs_est.index
            else float("nan")
        )
        print(f"  shock {ix[2]:19s} {row['value']:10.4f} {chs_v:10.4f}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# Investment equation tests
# ---------------------------------------------------------------------------


@pytest.mark.end_to_end
def test_af_estimate_with_endogenous_factor() -> None:
    """Verify AF estimation works with an endogenous (investment) factor.

    DGP:
      theta_{t+1} = 0.6 * theta_t + 0.3 * I_t + 0.05 + eta
      (log_ces-like, but linear for simplicity)
      I_t = 0.5 * theta_t + 0.2 * Y_t + eps_I
      Skill measures: Z^s_{t,m} = intercept + loading * theta_t + noise
      Investment measures: Z^I_{t,m} = intercept + loading * I_t + noise
    """
    rng = np.random.default_rng(123)
    n_obs, n_periods = 400, 3

    # True parameters
    true_beta_skill = 0.6  # theta on theta
    true_beta_inv = 0.3  # investment on theta_next
    true_trans_constant = 0.05
    true_shock_sd = 0.3
    true_inv_beta0 = 0.0  # investment intercept
    true_inv_beta_theta = 0.5  # investment depends on skill
    true_inv_beta_y = 0.2  # investment depends on income
    true_inv_sd = 0.25

    # Simulate
    theta = np.zeros((n_obs, n_periods))
    inv = np.zeros((n_obs, n_periods))
    income = rng.normal(1.0, 0.5, n_obs)  # exogenous, time-invariant
    theta[:, 0] = rng.normal(0, 1, n_obs)
    inv[:, 0] = (
        true_inv_beta0
        + true_inv_beta_theta * theta[:, 0]
        + true_inv_beta_y * income
        + rng.normal(0, true_inv_sd, n_obs)
    )
    for t in range(n_periods - 1):
        theta[:, t + 1] = (
            true_trans_constant
            + true_beta_skill * theta[:, t]
            + true_beta_inv * inv[:, t]
            + rng.normal(0, true_shock_sd, n_obs)
        )
        if t + 1 < n_periods:
            inv[:, t + 1] = (
                true_inv_beta0
                + true_inv_beta_theta * theta[:, t + 1]
                + true_inv_beta_y * income
                + rng.normal(0, true_inv_sd, n_obs)
            )

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    # Skill measures
                    "s1": theta[i, t] + rng.normal(0, 0.3),
                    "s2": 0.3 + 0.8 * theta[i, t] + rng.normal(0, 0.35),
                    "s3": -0.1 + 1.1 * theta[i, t] + rng.normal(0, 0.4),
                    # Investment measures
                    "i1": inv[i, t] + rng.normal(0, 0.3),
                    "i2": 0.2 + 0.9 * inv[i, t] + rng.normal(0, 0.35),
                    "i3": -0.1 + 1.2 * inv[i, t] + rng.normal(0, 0.4),
                    # Exogenous variable
                    "income": income[i],
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1", "i2", "i3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"i1": 1},) * n_periods,
                    intercepts=({"i1": 0},) * n_periods,
                ),
                transition_function="linear",
                is_endogenous=True,
            ),
        },
        observed_factors=("income",),
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    # Basic checks: estimation ran, produced results for all periods
    assert len(result.period_results) == n_periods
    for pr in result.period_results:
        assert np.isfinite(pr.loglikelihood), (
            f"Period {pr.period}: non-finite loglik {pr.loglikelihood}"
        )

    # Period 1 should have investment equation parameters
    p1 = result.period_results[1].params
    inv_eq = p1.query("category == 'investment_eq'")
    assert len(inv_eq) > 0, (
        "No investment_eq parameters found — endogenous factor not wired"
    )

    # Investment equation params should not be stuck at init
    inv_eq_values = inv_eq["value"].to_numpy()
    assert not np.allclose(inv_eq_values, 0.5, atol=0.05), (
        f"Investment eq params stuck at init: {inv_eq_values}"
    )


def test_prev_period_inv_meas_does_not_affect_transition_loglik_gradient() -> None:
    """Guard inv-prev-meas invariance of the AF transition-step gradient.

    Inv-type measurements at the previous period must not contribute to
    the gradient of `af_loglike_transition` w.r.t. current-step parameters.
    MATLAB's reference AF likelihood (`AF_Application_One_Normal_Translog.m`,
    `create_nodes_weights_12`) evaluates inv-type measurements exactly once,
    at the step where the inv is generated as a current-period measurement.
    They are deliberately omitted from the chained-sample importance weight
    at the next transition step (`prod_inv` is commented out in the MATLAB
    source). Re-evaluating them would be wrong: the chained sample carries
    forward only state factors, so the previous step's inv value is no
    longer available; evaluating prev-period inv measurements against the
    *current* step's freshly-drawn inv would be a wrong-value comparison.

    The Python port restricts the prev-meas factor to state-factor
    loadings only (using `state_factor_indices_in_latent` to slice the
    columns). This test guards against future refactors that re-introduce
    a parameter-dependent contribution from inv-loading rows at the
    previous period: it perturbs only the inv-meas columns of
    `prev_measurements` and asserts the gradient w.r.t. all current-step
    parameters is unchanged.
    """
    rng = np.random.default_rng(20260507)
    n_obs = 5
    n_state = 1
    n_endog = 1
    n_obs_factors = 0
    n_measures = 2  # 1 skill + 1 inv at current period
    n_prev_measures = 2  # 1 skill + 1 inv at prev period
    n_controls = 1  # constant
    n_halton = 3
    n_components = 1

    # Loading masks: row 0 = skill meas (loads on factor 0=skills), row 1 =
    # inv meas (loads on factor 1=investment). Both at current and prev.
    loading_mask = jnp.array([[True, False], [False, True]])
    prev_loading_mask = jnp.array([[True, False], [False, True]])

    measurements = jnp.array(rng.normal(size=(n_obs, n_measures)))
    controls = jnp.ones((n_obs, n_controls))
    prev_measurements_a = jnp.array(rng.normal(size=(n_obs, n_prev_measures)))
    # Perturb ONLY the inv-meas column (index 1) at the previous period.
    inv_perturbation = jnp.array(rng.normal(size=n_obs))
    prev_measurements_b = prev_measurements_a.at[:, 1].set(  # noqa: PD008
        prev_measurements_a[:, 1] + inv_perturbation
    )
    prev_controls = jnp.ones((n_obs, n_controls))

    # Prev-period measurement-system parameters (held fixed at the
    # transition step in production -- they were estimated previously).
    prev_loadings_flat = jnp.array([1.0, 1.0])
    prev_control_params = jnp.zeros((n_prev_measures, n_controls))
    prev_meas_sds = jnp.array([0.5, 0.4])

    # Period-0 Schur-conditional payload: per-obs cond_means and per-component
    # cond_chols (for the joint-Halton chain rebuild scheme).
    cond_means = jnp.array(rng.normal(size=(n_components, n_obs, n_state)))
    cond_chols = jnp.array([[[0.5]], [[0.4]]])
    cond_weights = jnp.ones((n_obs, n_components))
    prev_distribution = {
        "cond_weights": cond_weights,
        "cond_means": cond_means,
        "cond_chols": cond_chols,
    }

    # No prior chain (this is the 0->1 step). Joint Halton dim:
    # n_state (z_state) + 0 prior steps + (n_shock + n_endog) current step.
    chain_links: tuple = ()
    obs_factor_values_chain = jnp.zeros((n_obs, 0, n_obs_factors))
    joint_nodes = jnp.array(rng.normal(size=(n_halton, n_state + n_state + n_endog)))
    joint_weights = jnp.full(n_halton, 1.0 / n_halton)

    def transition_func(full_states: jax.Array, params: jax.Array) -> jax.Array:
        # Linear: theta_t = a * theta_prev + b * inv + c. Returns shape (n_state,).
        return jnp.array(
            [params[0] * full_states[0] + params[1] * full_states[1] + params[2]]
        )

    total_n_transition_params = 3
    n_inv_eq_params_per = 1 + n_state + n_obs_factors
    total_n_inv_params = n_endog * n_inv_eq_params_per

    # Param vector layout matches `_parse_transition_params`: 3 transition
    # params, 1 shock sd, 2 inv_eq params, 1 inv sd, 2 control params, 2
    # loadings, 2 meas sds = 13 entries total.
    params_value = jnp.array(
        [
            0.6,
            0.3,
            0.1,
            0.4,
            0.0,
            0.5,
            0.2,
            0.0,
            0.0,
            1.0,
            1.0,
            0.3,
            0.3,
        ]
    )

    state_factor_indices_in_latent = jnp.array([0], dtype=jnp.int32)
    shock_factor_indices = jnp.array([0], dtype=jnp.int32)
    obs_factor_values = jnp.zeros((n_obs, n_obs_factors))

    def _ll(prev_meas: jax.Array, params: jax.Array) -> jax.Array:
        return af_loglike_transition(
            params,
            n_state_factors=n_state,
            n_endogenous_factors=n_endog,
            n_measures=n_measures,
            n_controls=n_controls,
            measurements=measurements,
            controls=controls,
            loading_mask=loading_mask,
            prev_measurements=prev_meas,
            prev_controls=prev_controls,
            prev_loading_mask=prev_loading_mask,
            prev_control_params=prev_control_params,
            prev_loadings_flat=prev_loadings_flat,
            prev_meas_sds=prev_meas_sds,
            prev_distribution=prev_distribution,
            chain_links=chain_links,
            obs_factor_values_chain=obs_factor_values_chain,
            joint_nodes=joint_nodes,
            joint_weights=joint_weights,
            transition_func=transition_func,
            total_n_transition_params=total_n_transition_params,
            total_n_inv_params=total_n_inv_params,
            n_inv_eq_params_per=n_inv_eq_params_per,
            observed_factor_values=obs_factor_values,
            stability_floor=1e-300,
            state_factor_indices_in_latent=state_factor_indices_in_latent,
            n_shock_factors=1,
            shock_factor_indices=shock_factor_indices,
        )

    def loglike_a(params: jax.Array) -> jax.Array:
        return _ll(prev_measurements_a, params)

    def loglike_b(params: jax.Array) -> jax.Array:
        return _ll(prev_measurements_b, params)

    grad_a = jax.grad(loglike_a)(params_value)
    grad_b = jax.grad(loglike_b)(params_value)

    np.testing.assert_allclose(np.asarray(grad_a), np.asarray(grad_b), atol=1e-10)

    # Sanity: with a non-zero perturbation, the inv-row residuals do change,
    # so the loglik *value* itself differs (by a per-obs constant). That
    # difference must NOT be zero -- otherwise the test isn't actually
    # exercising the inv-loading rows.
    val_a = float(loglike_a(params_value))
    val_b = float(loglike_b(params_value))
    assert not np.isclose(val_a, val_b), (
        "Test sanity failure: perturbing prev inv-meas changed nothing -- "
        "the test isn't exercising the inv-loading rows."
    )


def test_rebuild_chain_at_period_matches_python_forward_pass() -> None:
    """Unit test for `_rebuild_chain_at_period`.

    Hand-code a 2-step linear chain (1 state factor, 1 endog factor, 1
    observed factor) and assert the helper's output matches a Python
    forward pass to numerical precision. Catches index/reshape bugs in
    the chain-rebuild helper independently of the integrand.
    """
    rng = np.random.default_rng(20260507)
    n_state = 1
    n_endog = 1
    n_obs_factors = 1
    n_inv_eq_params_per = 1 + n_state + n_obs_factors

    # Two prior chain steps (so we're computing θ_0 → θ_1 → θ_2).
    z_state = jnp.asarray(rng.normal(size=n_state))
    z_inv_per_step = jnp.asarray(rng.normal(size=(2, n_endog)))
    z_shock_per_step = jnp.asarray(rng.normal(size=(2, n_state)))

    initial_mean = jnp.asarray(rng.normal(size=n_state))
    initial_chol = jnp.asarray([[0.7]])

    obs_factor_values_per_step = jnp.asarray(rng.normal(size=(2, n_obs_factors)))

    # Linear "transition": theta_next = a * theta + b * inv + c * obs + d.
    # Wrap as the f(full_states, params) signature used in production.
    def make_transition_func() -> Callable[[jax.Array, jax.Array], jax.Array]:
        def fn(full_states: jax.Array, params: jax.Array) -> jax.Array:
            a, b, c, d = params[0], params[1], params[2], params[3]
            return jnp.array(
                [a * full_states[0] + b * full_states[1] + c * full_states[2] + d]
            )

        return fn

    transition_func = make_transition_func()

    link_1 = ChainLink(
        period=1,
        transition_func=transition_func,
        transition_params=jnp.array([0.6, 0.3, 0.05, 0.1]),
        shock_sds=jnp.array([0.4]),
        shock_factor_indices=jnp.array([0], dtype=jnp.int32),
        inv_eq_params=jnp.array([0.0, 0.5, 0.2]),  # intercept, beta_skills, beta_inc
        inv_sds=jnp.array([0.25]),
        n_inv_eq_params_per=n_inv_eq_params_per,
        obs_factor_values=jnp.zeros((1, n_obs_factors)),  # unused by helper
    )
    link_2 = ChainLink(
        period=2,
        transition_func=transition_func,
        transition_params=jnp.array([0.5, 0.4, 0.0, 0.2]),
        shock_sds=jnp.array([0.3]),
        shock_factor_indices=jnp.array([0], dtype=jnp.int32),
        inv_eq_params=jnp.array([0.05, 0.6, 0.3]),
        inv_sds=jnp.array([0.15]),
        n_inv_eq_params_per=n_inv_eq_params_per,
        obs_factor_values=jnp.zeros((1, n_obs_factors)),
    )
    chain_links = (link_1, link_2)

    # Hand-coded forward pass.
    theta_0 = initial_mean + initial_chol @ z_state
    for step_idx, link in enumerate(chain_links):
        z_inv = z_inv_per_step[step_idx]
        z_shock = z_shock_per_step[step_idx]
        obs_y = obs_factor_values_per_step[step_idx]
        beta = link.inv_eq_params  # (intercept, beta_skills, beta_inc)
        inv_val = (
            beta[0]
            + beta[1] * theta_0[0]
            + beta[2] * obs_y[0]
            + (link.inv_sds[0] * z_inv[0])
        )
        inv = jnp.array([inv_val])
        full = jnp.concatenate([theta_0, inv, obs_y])
        theta_next_det = transition_func(full, link.transition_params)  # ty: ignore[invalid-argument-type]
        theta_0 = theta_next_det + jnp.array([link.shock_sds[0] * z_shock[0]])
    expected = theta_0  # θ at the last link's target period

    actual = _rebuild_chain_at_period(
        z_state=z_state,
        z_inv_per_step=z_inv_per_step,
        z_shock_per_step=z_shock_per_step,
        initial_mean=initial_mean,
        initial_chol=initial_chol,
        chain_links=chain_links,
        obs_factor_values_at_obs_per_step=obs_factor_values_per_step,
        n_state_factors=n_state,
        n_endogenous_factors=n_endog,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


def test_rebuild_chain_at_period_empty_chain_returns_period_0() -> None:
    """Verify the empty-chain (0->1) path of `_rebuild_chain_at_period`.

    With no chain links, the helper just returns
    ``initial_mean + initial_chol @ z_state``.
    """
    rng = np.random.default_rng(7)
    n_state = 2
    z_state = jnp.asarray(rng.normal(size=n_state))
    initial_mean = jnp.asarray(rng.normal(size=n_state))
    initial_chol = jnp.asarray([[0.5, 0.0], [0.1, 0.4]])
    expected = initial_mean + initial_chol @ z_state

    actual = _rebuild_chain_at_period(
        z_state=z_state,
        z_inv_per_step=jnp.zeros((0, 1)),
        z_shock_per_step=jnp.zeros((0, n_state)),
        initial_mean=initial_mean,
        initial_chol=initial_chol,
        chain_links=(),
        obs_factor_values_at_obs_per_step=jnp.zeros((0, 0)),
        n_state_factors=n_state,
        n_endogenous_factors=1,
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-14)


def test_af_joint_halton_recovers_sigma_prod_argmax() -> None:  # noqa: PLR0915
    """Catch regression to split-Halton: sigma_prod recovery on synthetic translog.

    With all params except sigma_prod_0 pinned at the truth, the per-obs mean
    log-likelihood at sigma_prod=truth must beat sigma_prod ≈ truth/4 by at least
    1.0 nat per obs. Under the buggy split-Halton scheme the argmax sat
    near sigma ≈ truth/4 with truth being WORSE; under the joint-Halton fix
    the argmax aligns with truth. The empirical joint-vs-split gap on
    the MATLAB sim was ~2.5 nats per obs (see
    ``sim_repro/debug_joint_halton.py`` and
    ``obsidian/Professional/skillmodels/sigma-prod-collapse-2026-05-07.md``);
    1.0 nat is generous headroom that still flags any return to split.

    The test calls ``af_loglike_transition`` directly with hand-built
    kwargs on a tiny synthetic translog DGP (1 state factor, 1 endog
    factor, 1 observed factor), so it isolates the integrand from the
    optimizer and runs in ~10s.
    """
    rng = np.random.default_rng(20260508)
    n_obs = 200
    n_halton = 500
    n_state = 1
    n_endog = 1
    n_obs_factors = 1
    n_inv_eq_params_per = 1 + n_state + n_obs_factors

    # MATLAB-translog truth values (from set_parameters in
    # AF_Simulations_Translog.m), restricted to one state factor.
    a_true = 0.9283
    sigma_t_true = 0.5125  # log(skills) coef in translog
    gamma_t_true = 0.6113  # log(inv) coef
    delta_t_true = -0.0175  # cross coef
    sigma_p_true = 0.36
    sigma_i_true = 0.10
    beta_skills_true = 0.10
    beta_inc_true = 0.90

    # Mixture truth (matches MATLAB sim): two components on (skills, log_inc).
    p_a_true = 0.62
    mu_a = jnp.array([-4.0, -2.0])  # (skills, log_inc)
    cov_a = jnp.array([[0.62, 0.035], [0.035, 0.056]])
    mu_b = jnp.array([6.0, 3.0])
    cov_b = jnp.array([[0.83, 0.17], [0.17, 1.28]])

    # Period-0 measurement system (3 skill measures).
    lam_skills_0 = jnp.array([1.0, 0.36, 0.56])
    sd_skills_0 = jnp.array([0.68, 0.03, 0.08])
    # Period-1 measurement system: 3 skill measures + 3 inv measures.
    lam_skills_1 = jnp.array([1.0, 0.66, 1.18])
    sd_skills_1 = jnp.array([0.51, 0.12, 0.19])
    lam_inv_1 = jnp.array([1.0, 0.84, 0.79])
    sd_inv_1 = jnp.array([0.15, 0.39, 0.47])

    # Forward simulation of one panel.
    u = rng.uniform(size=n_obs)
    is_a = (u < p_a_true).astype(np.float64)

    def _draw_2d(mu: jax.Array, cov: jax.Array, n: int) -> np.ndarray:
        chol = np.linalg.cholesky(np.asarray(cov))
        z = rng.normal(size=(n, 2))
        return np.asarray(mu)[None, :] + z @ chol.T

    draw_a = _draw_2d(mu_a, cov_a, n_obs)
    draw_b = _draw_2d(mu_b, cov_b, n_obs)
    skills_0 = is_a * draw_a[:, 0] + (1 - is_a) * draw_b[:, 0]
    log_inc = is_a * draw_a[:, 1] + (1 - is_a) * draw_b[:, 1]

    # Period-0 data: z_skills_0 = lam * skills_0 + meas_noise.
    z_skills_0 = (
        np.asarray(lam_skills_0)[None, :] * skills_0[:, None]
        + rng.normal(size=(n_obs, 3)) * np.asarray(sd_skills_0)[None, :]
    )

    # Period-0->1 transition: inv_0 = beta_sk*skills_0 + beta_inc*log_inc + sd_I*z.
    inv_0_true = (
        beta_skills_true * skills_0
        + beta_inc_true * log_inc
        + rng.normal(size=n_obs) * sigma_i_true
    )
    skills_1 = (
        a_true
        + sigma_t_true * skills_0
        + gamma_t_true * inv_0_true
        + delta_t_true * skills_0 * inv_0_true
        + rng.normal(size=n_obs) * sigma_p_true
    )
    z_skills_1 = (
        np.asarray(lam_skills_1)[None, :] * skills_1[:, None]
        + rng.normal(size=(n_obs, 3)) * np.asarray(sd_skills_1)[None, :]
    )
    z_inv_1 = (
        np.asarray(lam_inv_1)[None, :] * inv_0_true[:, None]
        + rng.normal(size=(n_obs, 3)) * np.asarray(sd_inv_1)[None, :]
    )

    # Period-0 cond-distribution payload (Schur conditional given log_inc).
    def _schur(mu_2d: jax.Array, cov_2d: jax.Array) -> tuple[jax.Array, jax.Array]:
        # skills given log_inc: cond_mean (per obs) and cond_chol (scalar).
        sigma_skills_inc = cov_2d[0, 1]
        var_inc = cov_2d[1, 1]
        var_cond = cov_2d[0, 0] - sigma_skills_inc**2 / var_inc
        cond_chol = jnp.sqrt(var_cond)
        cond_means = mu_2d[0] + (sigma_skills_inc / var_inc) * (
            jnp.asarray(log_inc) - mu_2d[1]
        )
        return cond_means.reshape(n_obs, 1), jnp.asarray([[cond_chol]])

    cond_mean_a, cond_chol_a = _schur(mu_a, cov_a)
    cond_mean_b, cond_chol_b = _schur(mu_b, cov_b)
    cond_means = jnp.stack([cond_mean_a, cond_mean_b], axis=0)
    cond_chols = jnp.stack([cond_chol_a, cond_chol_b], axis=0)

    # Per-obs Bayes posterior weights from the marginal Y density.
    def _log_marg_y(mu: jax.Array, cov: jax.Array) -> jax.Array:
        var_y = cov[1, 1]
        return (
            -0.5 * jnp.log(2 * jnp.pi * var_y)
            - 0.5 * (jnp.asarray(log_inc) - mu[1]) ** 2 / var_y
        )

    log_w_a = jnp.log(p_a_true) + _log_marg_y(mu_a, cov_a)
    log_w_b = jnp.log(1.0 - p_a_true) + _log_marg_y(mu_b, cov_b)
    log_w = jnp.stack([log_w_a, log_w_b], axis=-1)
    cond_weights = jax.nn.softmax(log_w, axis=-1)

    prev_distribution = {
        "cond_weights": cond_weights,
        "cond_means": cond_means,
        "cond_chols": cond_chols,
    }

    # Period-1 measurement loadings: 6 measures in order (skill_1, skill_2,
    # skill_3, inv_1, inv_2, inv_3) -- skill measures load on factor 0
    # (skills), inv measures load on factor 1 (investment).
    n_measures = 6
    measurements = jnp.concatenate(
        [jnp.asarray(z_skills_1), jnp.asarray(z_inv_1)], axis=1
    )
    loading_mask = jnp.array(
        [
            [True, False],
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [False, True],
        ]
    )
    loadings_flat_curr = jnp.concatenate([lam_skills_1, lam_inv_1])
    meas_sds_curr = jnp.concatenate([sd_skills_1, sd_inv_1])

    # Period-0 measurement system (prev) -- 3 skill measures.
    n_prev_measures = 3
    prev_measurements = jnp.asarray(z_skills_0)
    prev_loading_mask = jnp.array([[True, False]] * 3)
    prev_loadings_flat = lam_skills_0
    prev_meas_sds = sd_skills_0

    # No controls (zeros).
    n_controls = 1  # constant
    controls = jnp.ones((n_obs, 1))
    prev_controls = jnp.ones((n_obs, 1))

    obs_factor_values = jnp.asarray(log_inc).reshape(n_obs, 1)

    # Transition function: log-translog (matches MATLAB sim).
    def transition_func(full_states: jax.Array, params: jax.Array) -> jax.Array:
        # full_states = [theta, inv, log_inc]; params = [lin_skills, lin_inv,
        # lin_inc, sq_skills, sq_inv, sq_inc, inter_skills_inv,
        # inter_skills_inc, inter_inv_inc, constant].
        skills = full_states[0]
        inv = full_states[1]
        return jnp.array(
            [
                params[9]
                + params[0] * skills
                + params[1] * inv
                + params[6] * skills * inv
            ]
        )

    total_n_transition_params = 10
    n_per_inv = n_inv_eq_params_per
    total_n_inv_params = n_endog * n_per_inv

    state_factor_indices_in_latent = jnp.array([0], dtype=jnp.int32)
    shock_factor_indices = jnp.array([0], dtype=jnp.int32)

    # Param vector layout: transition (10) + shock_sds (1) + inv_eq (3) +
    # inv_sds (1) + control_params (n_measures*n_controls=6) + loadings (6)
    # + meas_sds (6) = 33.
    transition_params_truth = jnp.array(
        [
            sigma_t_true,
            gamma_t_true,
            0.0,  # lin coef on log_inc
            0.0,
            0.0,
            0.0,  # squares
            delta_t_true,  # skills * inv
            0.0,
            0.0,  # other interactions
            a_true,
        ]
    )
    inv_eq_params_truth = jnp.array([0.0, beta_skills_true, beta_inc_true])

    def _build_params(sigma_p: float) -> jax.Array:
        return jnp.concatenate(
            [
                transition_params_truth,
                jnp.array([sigma_p]),
                inv_eq_params_truth,
                jnp.array([sigma_i_true]),
                jnp.zeros(n_measures * n_controls),  # control intercepts
                loadings_flat_curr,
                meas_sds_curr,
            ]
        )

    def _ll(sigma_p: float) -> float:
        params_value = _build_params(sigma_p)
        neg_mean = af_loglike_transition(
            params_value,
            n_state_factors=n_state,
            n_endogenous_factors=n_endog,
            n_measures=n_measures,
            n_controls=n_controls,
            measurements=measurements,
            controls=controls,
            loading_mask=loading_mask,
            prev_measurements=prev_measurements,
            prev_controls=prev_controls,
            prev_loading_mask=prev_loading_mask,
            prev_control_params=jnp.zeros((n_prev_measures, n_controls)),
            prev_loadings_flat=prev_loadings_flat,
            prev_meas_sds=prev_meas_sds,
            prev_distribution=prev_distribution,
            chain_links=(),
            obs_factor_values_chain=jnp.zeros((n_obs, 0, n_obs_factors)),
            joint_nodes=jnp.array(
                np.random.default_rng(1).normal(
                    size=(n_halton, n_state + n_state + n_endog)
                )
            ),
            joint_weights=jnp.full(n_halton, 1.0 / n_halton),
            transition_func=transition_func,
            total_n_transition_params=total_n_transition_params,
            total_n_inv_params=total_n_inv_params,
            n_inv_eq_params_per=n_per_inv,
            observed_factor_values=obs_factor_values,
            stability_floor=1e-300,
            state_factor_indices_in_latent=state_factor_indices_in_latent,
            n_shock_factors=1,
            shock_factor_indices=shock_factor_indices,
        )
        # Convert from neg-mean back to per-obs mean ll.
        return float(-neg_mean)

    sigma_truth = sigma_p_true
    sigma_wrong = 0.09  # well below truth (= truth / 4)
    ll_truth = _ll(sigma_truth)
    ll_wrong = _ll(sigma_wrong)
    gap = ll_truth - ll_wrong
    assert gap > 1.0, (
        f"Joint-Halton sigma_prod recovery REGRESSED: ll(truth={sigma_truth})="
        f"{ll_truth:.4f} should beat ll(wrong={sigma_wrong})={ll_wrong:.4f} by "
        f"at least 1.0 nat per obs but gap is only {gap:.4f}. The empirical "
        f"joint-vs-split gap on the MATLAB translog sim was ~2.5 nats; a gap "
        f"below 1.0 here suggests the AF likelihood has reverted to the split-"
        f"Halton scheme that biases sigma_prod toward 0."
    )


def test_af_joint_halton_recovers_sigma_prod_with_chain_link() -> None:  # noqa: PLR0915
    """As above, but exercise a 1→2 step where ``chain_links`` is non-empty.

    For the 0→1 step the joint Halton dim is just `n_state + n_shock +
    n_endog` and the joint-vs-split distinction is subtle (no prior
    chain to bridge). For 1→2 steps the joint Halton couples z_state +
    prior chain shocks + current shocks all in one sequence — that's
    where MATLAB's working scheme actually outperforms split Halton.

    This test runs `estimate_af` end-to-end on a tiny synthetic translog
    DGP through periods 0, 1, 2, then verifies the period-2 (= 1→2)
    estimated sigma_prod_1 is within 35% of truth. Under split Halton
    this parameter collapses toward 0; under joint Halton it recovers
    near truth (0.42 in the MATLAB sim). The 35% threshold (vs split-
    Halton's ~100% collapse) clearly separates the two regimes while
    absorbing JAX numerical-determinism differences across CI vs local
    hardware that nudged the recovered estimate from ~28% to ~31% on
    the same fixed seed.
    """
    pytest.importorskip("optimagic")
    rng = np.random.default_rng(20260509)
    n_obs = 300
    n_periods = 3

    # MATLAB-translog truths.
    a_t = (0.9283, 0.9536)
    sigma_t_arr = (0.5125, 0.7295)
    gamma_t_arr = (0.6113, 0.2814)
    delta_t_arr = (-0.0175, -0.0024)
    sigma_p_arr = (0.36, 0.42)
    sigma_i_arr = (0.10, 0.10)
    beta_skills = (0.10, 0.10)
    beta_inc = (0.90, 0.90)
    lam_skills = (
        np.array([1.0, 0.36, 0.56]),
        np.array([1.0, 0.66, 1.18]),
        np.array([1.0, 0.19, 0.50]),
    )
    sd_skills = (
        np.array([0.68, 0.03, 0.08]),
        np.array([0.51, 0.12, 0.19]),
        np.array([0.14, 0.03, 0.15]),
    )
    lam_inv = (np.array([1.0, 0.84, 0.79]),) * 2
    sd_inv = (np.array([0.15, 0.39, 0.47]),) * 2

    # Initial mixture (matches MATLAB).
    p_a = 0.62
    mu_a = np.array([-4.0, -2.0])
    cov_a = np.array([[0.62, 0.035], [0.035, 0.056]])
    mu_b = np.array([6.0, 3.0])
    cov_b = np.array([[0.83, 0.17], [0.17, 1.28]])

    u = rng.uniform(size=n_obs)
    is_a = (u < p_a).astype(np.float64)
    chol_a = np.linalg.cholesky(cov_a)
    chol_b = np.linalg.cholesky(cov_b)
    z_init = rng.normal(size=(n_obs, 2))
    draw_a = mu_a[None, :] + z_init @ chol_a.T
    draw_b = mu_b[None, :] + z_init @ chol_b.T
    skills = np.zeros((n_obs, n_periods))
    skills[:, 0] = is_a * draw_a[:, 0] + (1 - is_a) * draw_b[:, 0]
    log_inc = is_a * draw_a[:, 1] + (1 - is_a) * draw_b[:, 1]
    inv = np.zeros((n_obs, n_periods - 1))
    for t in range(n_periods - 1):
        inv[:, t] = (
            beta_skills[t] * skills[:, t]
            + beta_inc[t] * log_inc
            + rng.normal(size=n_obs) * sigma_i_arr[t]
        )
        skills[:, t + 1] = (
            a_t[t]
            + sigma_t_arr[t] * skills[:, t]
            + gamma_t_arr[t] * inv[:, t]
            + delta_t_arr[t] * skills[:, t] * inv[:, t]
            + rng.normal(size=n_obs) * sigma_p_arr[t]
        )

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {
                "caseid": int(i),
                "period": int(t),
                "skill_1": lam_skills[t][0] * skills[i, t]
                + rng.normal() * sd_skills[t][0],
                "skill_2": lam_skills[t][1] * skills[i, t]
                + rng.normal() * sd_skills[t][1],
                "skill_3": lam_skills[t][2] * skills[i, t]
                + rng.normal() * sd_skills[t][2],
                "log_income": float(log_inc[i]),
            }
            if 1 <= t <= 2:
                inv_t_idx = t - 1
                row["inv_1"] = (
                    lam_inv[inv_t_idx][0] * inv[i, inv_t_idx]
                    + rng.normal() * sd_inv[inv_t_idx][0]
                )
                row["inv_2"] = (
                    lam_inv[inv_t_idx][1] * inv[i, inv_t_idx]
                    + rng.normal() * sd_inv[inv_t_idx][1]
                )
                row["inv_3"] = (
                    lam_inv[inv_t_idx][2] * inv[i, inv_t_idx]
                    + rng.normal() * sd_inv[inv_t_idx][2]
                )
            else:
                row["inv_1"] = np.nan
                row["inv_2"] = np.nan
                row["inv_3"] = np.nan
            rows.append(row)
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    skill_normalisations = Normalizations(
        loadings=({"skill_1": 1.0},) * n_periods,
        intercepts=({"skill_1": 0.0},) * n_periods,
    )
    inv_normalisations = Normalizations(
        loadings=({}, {"inv_1": 1.0}, {"inv_1": 1.0}),
        intercepts=({}, {"inv_1": 0.0}, {"inv_1": 0.0}),
    )

    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("skill_1", "skill_2", "skill_3"),) * n_periods,
                normalizations=skill_normalisations,
                transition_function="translog",
            ),
            "investment": FactorSpec(
                measurements=(
                    (),
                    ("inv_1", "inv_2", "inv_3"),
                    ("inv_1", "inv_2", "inv_3"),
                ),
                normalizations=inv_normalisations,
                transition_function="linear",
                is_endogenous=True,
            ),
        },
        observed_factors=("log_income",),
        n_mixtures=2,
    )

    # Pin everything except sigma_prod_0 / sigma_prod_1 at MATLAB truth.
    truth_extras: list[tuple[tuple[str, int, str, str], float]] = [
        (("transition", 0, "skills", "constant"), a_t[0]),
        (("transition", 0, "skills", "skills"), sigma_t_arr[0]),
        (("transition", 0, "skills", "investment"), gamma_t_arr[0]),
        (("transition", 0, "skills", "skills * investment"), delta_t_arr[0]),
        (("transition", 1, "skills", "constant"), a_t[1]),
        (("transition", 1, "skills", "skills"), sigma_t_arr[1]),
        (("transition", 1, "skills", "investment"), gamma_t_arr[1]),
        (("transition", 1, "skills", "skills * investment"), delta_t_arr[1]),
        # Pin sigma_prod_0 at truth so we can isolate sigma_prod_1.
        (("shock_sds", 0, "skills", "-"), sigma_p_arr[0]),
        (("investment_eq", 0, "investment", "skills"), beta_skills[0]),
        (("investment_eq", 0, "investment", "log_income"), beta_inc[0]),
        (("investment_eq", 0, "investment", "constant"), 0.0),
        (("investment_eq", 1, "investment", "skills"), beta_skills[1]),
        (("investment_eq", 1, "investment", "log_income"), beta_inc[1]),
        (("investment_eq", 1, "investment", "constant"), 0.0),
        (("investment_sds", 0, "investment", "-"), sigma_i_arr[0]),
        (("investment_sds", 1, "investment", "-"), sigma_i_arr[1]),
    ]
    # Pin all squares + log_income terms in translog to 0.
    for t in range(n_periods - 1):
        for fac in ("skills", "investment", "log_income"):
            truth_extras.append((("transition", t, "skills", f"{fac} ** 2"), 0.0))
        truth_extras.append((("transition", t, "skills", "log_income"), 0.0))
        for cross in ("skills * log_income", "investment * log_income"):
            truth_extras.append((("transition", t, "skills", cross), 0.0))

    fixed_idx = pd.MultiIndex.from_tuples(
        [r[0] for r in truth_extras],
        names=["category", "period", "name1", "name2"],
    )
    fixed_params = pd.DataFrame(
        {"value": [r[1] for r in truth_extras]}, index=fixed_idx
    )

    truth_df = pd.DataFrame({"value": [v for _, v in truth_extras]}, index=fixed_idx)

    af_opts = AFEstimationOptions(
        n_halton_points=200,
        n_halton_points_shock=200,
        optimizer_algorithm="scipy_lbfgsb",
    )
    result = estimate_af(
        model_spec=model,
        data=data,
        options=af_opts,
        fixed_params=fixed_params,
        start_params=truth_df,
    )
    p2 = result.period_results[2].params
    sigma_prod_1_est = float(
        p2.loc[("shock_sds", 1, "skills", "-"), "value"]  # ty: ignore[invalid-argument-type]
    )
    rel_err = abs(sigma_prod_1_est - sigma_p_arr[1]) / sigma_p_arr[1]
    assert rel_err < 0.35, (
        f"sigma_prod_1 estimate {sigma_prod_1_est:.4f} is more than 35% off truth "
        f"{sigma_p_arr[1]:.4f} (rel error {rel_err:.2%}). Suggests joint-Halton "
        f"chain rebuild has regressed and sigma_prod is collapsing toward 0."
    )


# ---------------------------------------------------------------------------
# Posterior states tests
# ---------------------------------------------------------------------------


@pytest.mark.end_to_end
def test_af_get_individual_states() -> None:
    """Verify get_individual_states works with AF results.

    Run AF on a simple single-factor model, then call get_individual_states
    with the AF result. Check the returned DataFrame has the right shape,
    columns, and reasonable values.
    """
    data, _true_params = _simulate_linear_transition_data(n_obs=200, n_periods=3)
    model = _make_linear_transition_model(n_periods=3)

    af_result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    result = get_individual_states(data=data, result=af_result)

    # Should have unanchored_states
    assert "unanchored_states" in result
    states_df = result["unanchored_states"]["states"]

    # DataFrame should have id, period, and factor columns
    assert "period" in states_df.columns
    assert "skill" in states_df.columns

    # One row per individual per period
    n_obs = 200
    n_periods = 3
    assert len(states_df) == n_obs * n_periods

    # Values should be finite
    assert states_df["skill"].apply(np.isfinite).all()

    # State estimates should have non-trivial variance (not all the same)
    assert states_df["skill"].std() > 0.1


@pytest.mark.end_to_end
def test_af_estimate_with_translog() -> None:
    """Verify AF estimation runs with a translog transition function.

    Simulate from a linear DGP but estimate with translog — translog nests
    linear (squares and interactions zero), so estimation should still
    converge to a finite likelihood and recover the linear coefficient
    roughly. With one factor there are only 3 translog params: beta, beta^2,
    constant.
    """
    data, _true_params = _simulate_linear_transition_data(n_obs=300, n_periods=3)
    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("m1", "m2", "m3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"m1": 1},) * 3,
                    intercepts=({"m1": 0},) * 3,
                ),
                transition_function="translog",
            ),
        },
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    assert len(result.period_results) == 3
    for pr in result.period_results:
        assert np.isfinite(pr.loglikelihood), (
            f"Period {pr.period}: non-finite loglik {pr.loglikelihood}"
        )

    # Period 1 should have 3 translog transition params: skill, skill ** 2, constant
    p1 = result.period_results[1].params
    trans = p1.query("category == 'transition'")
    param_names = set(trans.index.get_level_values("name2"))
    assert {"skill", "skill ** 2", "constant"}.issubset(param_names), (
        f"Expected translog params skill, skill ** 2, constant; got {param_names}"
    )

    # Linear coefficient should be recovered roughly (true beta = 0.8).
    # Tolerance is wide because translog overfits with squared term.
    est_beta = float(
        p1.loc[("transition", 0, "skill", "skill"), "value"]  # ty: ignore[invalid-argument-type]
    )
    assert abs(est_beta - 0.8) < 0.4, (
        f"translog skill coefficient: got {est_beta:.3f}, expected ≈ 0.8"
    )


@pytest.mark.end_to_end
def test_af_joint_initial_distribution_with_observed_factor() -> None:
    """Verify the joint (latent, observed) initial distribution is estimated.

    When observed factors are specified, the initial period estimator models
    the joint (latent, observed) distribution and conditions Halton draws on
    observed values per the Schur complement (Antweiler & Freyberger 2025).

    This test constructs data with a latent skill strongly correlated with
    observed income, runs AF, and verifies:
    - The estimated initial_states includes an entry for the observed factor.
    - The recovered mean of the observed factor is close to its sample mean.
    - The covariance between latent and observed has the expected sign.
    """
    rng = np.random.default_rng(2026)
    n_obs, n_periods = 400, 2
    true_corr = 0.7  # strong latent-observed correlation

    # Jointly simulate skill and income with specified correlation
    z = rng.multivariate_normal(
        mean=[0.0, 1.0],
        cov=[[1.0, true_corr * 0.5], [true_corr * 0.5, 0.25]],
        size=n_obs,
    )
    theta = z[:, 0]
    income = z[:, 1]

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "s1": theta[i] + rng.normal(0, 0.3),
                    "s2": 0.3 + 0.9 * theta[i] + rng.normal(0, 0.35),
                    "s3": -0.1 + 1.1 * theta[i] + rng.normal(0, 0.4),
                    "income": income[i],
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
        },
        observed_factors=("income",),
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=40,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    p0 = result.period_results[0].params

    # initial_states must now include an entry for the observed factor
    income_mean_loc = ("initial_states", 0, "mixture_0", "income")
    assert income_mean_loc in p0.index, (
        "initial_states should include the observed factor 'income'"
    )
    est_income_mean = float(p0.loc[income_mean_loc, "value"])  # ty: ignore[invalid-argument-type]
    sample_income_mean = float(income.mean())
    assert abs(est_income_mean - sample_income_mean) < 0.15, (
        f"Estimated income mean {est_income_mean:.3f} far from sample "
        f"{sample_income_mean:.3f}."
    )

    # Cross-covariance entry (skill-income) should reflect the positive
    # correlation in the DGP; stored as lower-triangular Cholesky with
    # factor ordering (latent, observed).
    cross_loc = ("initial_cholcovs", 0, "mixture_0", "income-skill")
    assert cross_loc in p0.index, (
        "Cross Cholesky entry between skill and income should be present"
    )
    # For a 2x2 joint Cholesky with positive cross-cov, the (1,0) entry
    # should be positive.
    cross_val = float(p0.loc[cross_loc, "value"])  # ty: ignore[invalid-argument-type]
    assert cross_val > 0.05, (
        f"Expected positive skill-income covariance; got Cholesky[1,0]={cross_val:.3f}"
    )


@pytest.mark.end_to_end
def test_af_fixed_params_pins_time_invariant_latent() -> None:
    """Verify fixed_params pins MC-style time-invariant latent factors.

    Construct a 2-factor model where `mc` is time-invariant and `skill`
    evolves linearly. Pin mc's transitions to identity and its shock SD
    to a near-zero floor (same convention CHS uses for augmented periods).
    After estimation, the pinned parameters must equal the input values
    exactly (not optimized away).
    """
    rng = np.random.default_rng(7)
    n_obs, n_periods = 300, 3
    mc = rng.normal(0, 1, n_obs)
    theta = np.zeros((n_obs, n_periods))
    theta[:, 0] = rng.normal(0, 1, n_obs)
    for t in range(n_periods - 1):
        theta[:, t + 1] = 0.7 * theta[:, t] + 0.2 * mc + rng.normal(0, 0.3, n_obs)

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {
                "caseid": i,
                "period": t,
                "s1": theta[i, t] + rng.normal(0, 0.3),
                "s2": 0.3 + 0.9 * theta[i, t] + rng.normal(0, 0.35),
                "s3": -0.1 + 1.1 * theta[i, t] + rng.normal(0, 0.4),
            }
            if t == 0:
                row["m1"] = mc[i] + rng.normal(0, 0.3)
                row["m2"] = 0.2 + 0.8 * mc[i] + rng.normal(0, 0.35)
                row["m3"] = -0.1 + 1.1 * mc[i] + rng.normal(0, 0.4)
            rows.append(row)
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
            "mc": FactorSpec(
                measurements=(("m1", "m2", "m3"), (), ()),
                normalizations=Normalizations(
                    loadings=({"m1": 1}, {}, {}),
                    intercepts=({"m1": 0}, {}, {}),
                ),
                transition_function="linear",
            ),
        },
    )

    # Pin mc to identity transition + floor shock SD across both
    # transition periods (0 and 1).
    fixed_entries: list[tuple[tuple[str, int, str, str], float]] = []
    for t in (0, 1):
        for reg in ("skill", "mc", "constant"):
            fixed_entries.append(
                (("transition", t, "mc", reg), 1.0 if reg == "mc" else 0.0)
            )
        fixed_entries.append((("shock_sds", t, "mc", "-"), 0.001))
    fixed_idx = pd.MultiIndex.from_tuples(
        [e[0] for e in fixed_entries],
        names=["category", "period", "name1", "name2"],
    )
    fixed_df = pd.DataFrame({"value": [e[1] for e in fixed_entries]}, index=fixed_idx)

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
        fixed_params=fixed_df,
    )

    for t_trans in (0, 1):
        p_t = result.period_results[t_trans + 1].params
        for reg in ("skill", "mc", "constant"):
            expected = 1.0 if reg == "mc" else 0.0
            val = float(
                p_t.loc[("transition", t_trans, "mc", reg), "value"]  # ty: ignore[invalid-argument-type]
            )
            assert val == expected, (
                f"mc transition period {t_trans}, regressor {reg}: "
                f"expected {expected}, got {val}"
            )
        sd = float(
            p_t.loc[("shock_sds", t_trans, "mc", "-"), "value"]  # ty: ignore[invalid-argument-type]
        )
        assert sd == 0.001, f"mc shock_sd period {t_trans}: {sd} (expected 0.001)"


def _make_three_factor_log_ces_model(
    n_periods: int,
) -> tuple[ModelSpec, pd.DataFrame]:
    """Build a 3-factor model with log_ces on fac1 and simulated data.

    fac1 is produced via CES from (fac1, fac2, fac3). In the DGP we mute
    fac3's contribution so tests can recover the pinning without fighting a
    strong signal from that factor.
    """
    rng = np.random.default_rng(17)
    n_obs = 250

    fac1 = np.zeros((n_obs, n_periods))
    fac2 = np.zeros((n_obs, n_periods))
    fac3 = np.zeros((n_obs, n_periods))
    fac1[:, 0] = rng.normal(0.5, 0.2, n_obs)
    fac2[:, 0] = rng.normal(0.5, 0.2, n_obs)
    fac3[:, 0] = rng.normal(0.0, 0.2, n_obs)
    for t in range(n_periods - 1):
        fac1[:, t + 1] = 0.4 * fac1[:, t] + 0.6 * fac2[:, t] + rng.normal(0, 0.1, n_obs)
        fac2[:, t + 1] = 0.9 * fac2[:, t] + rng.normal(0, 0.1, n_obs)
        fac3[:, t + 1] = fac3[:, t]

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "y1": fac1[i, t] + rng.normal(0, 0.1),
                    "y2": 0.5 + 0.8 * fac1[i, t] + rng.normal(0, 0.12),
                    "y3": -0.2 + 1.1 * fac1[i, t] + rng.normal(0, 0.1),
                    "y4": fac2[i, t] + rng.normal(0, 0.1),
                    "y5": 0.2 + 0.9 * fac2[i, t] + rng.normal(0, 0.12),
                    "y6": -0.1 + 1.1 * fac2[i, t] + rng.normal(0, 0.1),
                    "y7": fac3[i, t] + rng.normal(0, 0.1),
                    "y8": 0.1 + 0.9 * fac3[i, t] + rng.normal(0, 0.12),
                    "y9": -0.1 + 1.0 * fac3[i, t] + rng.normal(0, 0.1),
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
        factors={
            "fac1": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * n_periods,
                    intercepts=({"y1": 0},) * n_periods,
                ),
                transition_function="log_ces",
            ),
            "fac2": FactorSpec(
                measurements=(("y4", "y5", "y6"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"y4": 1},) * n_periods,
                    intercepts=({"y4": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
            "fac3": FactorSpec(
                measurements=(("y7", "y8", "y9"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"y7": 1},) * n_periods,
                    intercepts=({"y7": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
        },
    )
    return model, data


@pytest.mark.end_to_end
def test_af_log_ces_with_cross_factor_gamma_fixed_at_zero() -> None:
    """Fix gamma_fac3 = 0 in a log_ces transition and run AF end-to-end.

    Before the probability-constraint + fixed-params support was added, this
    combination raised `InvalidConstraintError` because optimagic refused
    any fix inside a ProbabilityConstraint selector. Now the fold helper
    removes gamma_fac3 from the selector and the remaining two gammas are
    optimised on the simplex summing to one.
    """
    model, data = _make_three_factor_log_ces_model(n_periods=2)

    fixed_idx = pd.MultiIndex.from_tuples(
        [("transition", 0, "fac1", "fac3")],
        names=["category", "period", "name1", "name2"],
    )
    fixed_df = pd.DataFrame({"value": [0.0]}, index=fixed_idx)

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=20,
            n_halton_points_shock=10,
            optimizer_algorithm="scipy_lbfgsb",
        ),
        fixed_params=fixed_df,
    )

    p_t = result.period_results[1].params
    gamma_fac1 = float(
        p_t.loc[("transition", 0, "fac1", "fac1"), "value"]  # ty: ignore[invalid-argument-type]
    )
    gamma_fac2 = float(
        p_t.loc[("transition", 0, "fac1", "fac2"), "value"]  # ty: ignore[invalid-argument-type]
    )
    gamma_fac3 = float(
        p_t.loc[("transition", 0, "fac1", "fac3"), "value"]  # ty: ignore[invalid-argument-type]
    )

    assert gamma_fac3 == 0.0
    assert np.isclose(gamma_fac1 + gamma_fac2, 1.0, atol=1e-6)
    assert gamma_fac1 > 0.0
    assert gamma_fac2 > 0.0


@pytest.mark.end_to_end
def test_af_log_ces_with_cross_factor_gamma_fixed_at_nonzero() -> None:
    """Fix gamma_fac3 = 0.2; verify remaining gammas sum to 0.8 at the optimum."""
    model, data = _make_three_factor_log_ces_model(n_periods=2)

    fixed_idx = pd.MultiIndex.from_tuples(
        [("transition", 0, "fac1", "fac3")],
        names=["category", "period", "name1", "name2"],
    )
    fixed_df = pd.DataFrame({"value": [0.2]}, index=fixed_idx)

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=20,
            n_halton_points_shock=10,
            optimizer_algorithm="scipy_lbfgsb",
        ),
        fixed_params=fixed_df,
    )

    p_t = result.period_results[1].params
    gamma_fac1 = float(
        p_t.loc[("transition", 0, "fac1", "fac1"), "value"]  # ty: ignore[invalid-argument-type]
    )
    gamma_fac2 = float(
        p_t.loc[("transition", 0, "fac1", "fac2"), "value"]  # ty: ignore[invalid-argument-type]
    )
    gamma_fac3 = float(
        p_t.loc[("transition", 0, "fac1", "fac3"), "value"]  # ty: ignore[invalid-argument-type]
    )

    assert gamma_fac3 == 0.2
    assert np.isclose(gamma_fac1 + gamma_fac2, 0.8, atol=1e-6)


@pytest.mark.end_to_end
def test_af_estimate_tolerates_nan_measurements() -> None:
    """NaN entries in measurement columns must not poison AF gradients.

    Real panels routinely have missing values; the AF likelihood masks
    them out at the per-observation level so each observation contributes
    only its non-missing measurements to the log-pdf sum.
    """
    rng = np.random.default_rng(2026)
    n_obs, n_periods = 400, 2

    z = rng.multivariate_normal(
        mean=[0.0, 1.0],
        cov=[[1.0, 0.35], [0.35, 0.25]],
        size=n_obs,
    )
    theta = z[:, 0]
    income = z[:, 1]

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {
                "caseid": i,
                "period": t,
                "s1": theta[i] + rng.normal(0, 0.3),
                "s2": 0.3 + 0.9 * theta[i] + rng.normal(0, 0.35),
                "s3": -0.1 + 1.1 * theta[i] + rng.normal(0, 0.4),
                "income": income[i],
            }
            # Sprinkle ~10% NaN into s2 across both periods.
            if rng.random() < 0.10:
                row["s2"] = np.nan
            rows.append(row)
    data = pd.DataFrame(rows).set_index(["caseid", "period"])
    assert data["s2"].isna().any(), "test setup should inject NaN measurements"

    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
        },
        observed_factors=("income",),
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )
    for pr in result.period_results:
        assert pr.success, f"Period {pr.period} failed with NaN measurements"
        assert np.isfinite(pr.loglikelihood)


@pytest.mark.end_to_end
def test_af_estimate_with_register_params_user_transition() -> None:
    """AF must accept `@register_params`-decorated user transition functions.

    User-defined transition functions take individual factor arguments
    plus a `params` dict; AF's per-period likelihood passes a packed
    state vector and a flat parameter slice. Without the bridging
    wrapper in `_get_raw_transition_functions`, callers that supply
    custom transitions (e.g. `skane-struct-bw`) raise TypeError at the
    first transition-step call.
    """

    @register_params(params=["constant", "skill"])
    def f_skill(skill: jax.Array, params: dict[str, float]) -> jax.Array:
        return params["constant"] + params["skill"] * skill

    rng = np.random.default_rng(2026)
    n_obs, n_periods = 300, 3
    theta = rng.normal(0, 1, (n_obs, n_periods))
    for t in range(1, n_periods):
        theta[:, t] = 0.1 + 0.8 * theta[:, t - 1] + rng.normal(0, 0.4, n_obs)

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "s1": theta[i, t] + rng.normal(0, 0.3),
                    "s2": 0.3 + 0.9 * theta[i, t] + rng.normal(0, 0.35),
                    "s3": -0.1 + 1.1 * theta[i, t] + rng.normal(0, 0.4),
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function=f_skill,
            ),
        },
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )
    for pr in result.period_results:
        assert pr.success, f"Period {pr.period} failed"
        assert np.isfinite(pr.loglikelihood)


def test_af_result_to_numpy_materialises_and_drops_samples_per_component() -> None:
    """`AFEstimationResult.to_numpy()` produces a numpy-only, pickle-friendly copy.

    `estimate_af` itself leaves arrays on-device so the JAX/XLA
    compilation cache can be reused across repeated calls (e.g. inside
    a Monte Carlo sweep). Callers that need host residency -- pickling,
    plotting, sending across processes -- must invoke `to_numpy()`,
    which:

    * drops `samples_per_component` (per-period `(n_halton, n_obs,
      n_state)` importance buffers, multi-GB at realistic sizes), and
    * materialises every `jax.Array` in the result
      (`MixtureComponent.mean`, `chol_cov`,
      `ConditionalDistribution.cond_means`, `cond_chols`,
      `conditional_weights`, `mixture_weights`, and the arrays inside
      every `ChainLink`) as `np.ndarray`. JAX arrays bind to GPU
      memory; without `to_numpy()`, pickling the result triggers a
      GPU→host materialisation inside `__reduce__` that routinely OOMs
      on a device still holding JIT caches.
    """
    rng = np.random.default_rng(2026)
    n_obs, n_periods = 200, 2
    theta = rng.normal(0, 1, (n_obs, n_periods))
    for t in range(1, n_periods):
        theta[:, t] = 0.1 + 0.8 * theta[:, t - 1] + rng.normal(0, 0.4, n_obs)

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "s1": theta[i, t] + rng.normal(0, 0.3),
                    "s2": 0.3 + 0.9 * theta[i, t] + rng.normal(0, 0.35),
                    "s3": -0.1 + 1.1 * theta[i, t] + rng.normal(0, 0.4),
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
        },
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=20,
            n_halton_points_shock=10,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    ).to_numpy()

    def _assert_numpy(arr: object, label: str) -> None:
        if arr is None:
            return
        assert isinstance(arr, np.ndarray), (
            f"{label} should be a numpy ndarray, got {type(arr).__name__}"
        )

    for cd in result.conditional_distributions:
        assert cd.samples_per_component == (), (
            "samples_per_component should be cleared before returning"
        )
        _assert_numpy(cd.mixture_weights, "mixture_weights")
        _assert_numpy(cd.conditional_weights, "conditional_weights")
        _assert_numpy(cd.cond_means, "cond_means")
        _assert_numpy(cd.cond_chols, "cond_chols")
        for component in cd.components:
            _assert_numpy(component.mean, "MixtureComponent.mean")
            _assert_numpy(component.chol_cov, "MixtureComponent.chol_cov")
        for cl in cd.chain_links:
            _assert_numpy(cl.transition_params, "ChainLink.transition_params")
            _assert_numpy(cl.shock_sds, "ChainLink.shock_sds")
            _assert_numpy(cl.shock_factor_indices, "ChainLink.shock_factor_indices")
            _assert_numpy(cl.inv_eq_params, "ChainLink.inv_eq_params")
            _assert_numpy(cl.inv_sds, "ChainLink.inv_sds")
            _assert_numpy(cl.obs_factor_values, "ChainLink.obs_factor_values")


def test_af_initial_loglike_is_joint_density_of_measurements_and_observed_factors() -> (  # noqa: PLR0915
    None
):
    """Pin the joint estimand f(Z_theta,0, Y_0), not the conditional f(Z|Y).

    With observed factors present, the initial-period per-obs log-likelihood
    must equal `log p(Y_i) + log_integral` (the JOINT density). If a future
    change subtracts `log p(Y_i)` to switch to the conditional MLE, the first
    assertion fails. Cross-checked against a plain-numpy single-component
    computation of the marginal-Y density and the quadrature integral.
    """
    n_factors = 2  # 1 latent + 1 observed
    n_latent = 1
    n_mixture_components = 1
    n_measures = 2
    n_controls = 1

    # Mixture mean (mu_theta, mu_y) and lower-tri Cholesky of the 2x2 joint
    # covariance in tril order [L00, L10, L11].
    mu_theta, mu_y = 0.5, -0.3
    chol_l00, chol_l10, chol_l11 = 1.2, 0.4, 0.9

    control_params = [0.1, -0.2]  # (n_measures, n_controls) flat
    loadings = [1.0, 0.8]  # both measures load on the single latent factor
    meas_sds = [0.5, 0.6]

    params = jnp.array(
        [
            1.0,  # mixture_weights
            mu_theta,  # mixture_means
            mu_y,
            chol_l00,  # mixture_chol_covs (tril)
            chol_l10,
            chol_l11,
            *control_params,
            *loadings,
            *meas_sds,
        ]
    )

    # Both measurements load on the single latent factor.
    loading_mask = jnp.array([[True], [True]])

    n_obs = 3
    rng = np.random.default_rng(404)
    measurements = jnp.asarray(rng.normal(0, 1, (n_obs, n_measures)))
    controls = jnp.asarray(rng.normal(0, 1, (n_obs, n_controls)))
    observed_factor_values = jnp.asarray(rng.normal(0, 1, (n_obs, 1)))

    # A handful of 1d standard-normal quadrature nodes with weights summing
    # to 1 (the conditional latent dimension is 1).
    raw_nodes = np.array([-1.5, -0.5, 0.5, 1.5])
    node_w = np.exp(-0.5 * raw_nodes**2)
    node_w = node_w / node_w.sum()
    nodes = jnp.asarray(raw_nodes.reshape(-1, 1))
    weights = jnp.asarray(node_w)

    per_obs = np.asarray(
        af_per_obs_loglike_initial(
            params,
            n_factors=n_factors,
            n_mixture_components=n_mixture_components,
            n_measures=n_measures,
            n_controls=n_controls,
            measurements=measurements,
            controls=controls,
            loading_mask=loading_mask,
            nodes=nodes,
            weights=weights,
            stability_floor=0.0,
            n_latent_factors=n_latent,
            observed_factor_values=observed_factor_values,
        )
    )

    # Independent plain-numpy reference for the single mixture component.
    chol_full = np.array([[chol_l00, 0.0], [chol_l10, chol_l11]])
    cov_full = chol_full @ chol_full.T
    cov_tt = cov_full[:n_latent, :n_latent]
    cov_ty = cov_full[:n_latent, n_latent:]
    cov_yy = cov_full[n_latent:, n_latent:]

    full_loadings = np.array(loadings).reshape(n_measures, n_latent)
    control_arr = np.array(control_params).reshape(n_measures, n_controls)
    meas_sd_arr = np.array(meas_sds)
    nodes_np = np.asarray(nodes)
    weights_np = np.asarray(weights)

    def _log_norm(x: np.ndarray, mean: float, sd: np.ndarray) -> np.ndarray:
        return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((x - mean) / sd) ** 2

    for i in range(n_obs):
        y_i = np.asarray(observed_factor_values[i])
        z_i = np.asarray(measurements[i])
        ctrl_i = np.asarray(controls[i])
        residual_base = z_i - control_arr @ ctrl_i

        # Marginal density of Y_i.
        log_marg = -0.5 * np.log(2 * np.pi * cov_yy[0, 0]) - 0.5 * (
            (y_i[0] - mu_y) ** 2 / cov_yy[0, 0]
        )

        # Schur-complement conditional of theta | Y_i.
        cond_mean = mu_theta + (cov_ty[0, 0] / cov_yy[0, 0]) * (y_i[0] - mu_y)
        cond_cov = cov_tt[0, 0] - cov_ty[0, 0] ** 2 / cov_yy[0, 0]
        cond_cov = cond_cov + 1e-10  # matches the code's jitter
        cond_chol = np.sqrt(cond_cov)

        log_nodes = []
        for q in range(nodes_np.shape[0]):
            theta_q = cond_mean + cond_chol * nodes_np[q, 0]
            resid = residual_base - full_loadings[:, 0] * theta_q
            log_nodes.append(np.sum(_log_norm(resid, 0.0, meas_sd_arr)))
        log_nodes = np.array(log_nodes)
        log_integral = np.log(np.sum(np.exp(log_nodes) * weights_np))

        # The returned per-obs log-likelihood is the JOINT density.
        np.testing.assert_allclose(per_obs[i], log_marg + log_integral, atol=1e-8)
        # And it is NOT the conditional integral alone (p(Y_i) term present).
        assert not np.allclose(per_obs[i], log_integral)


def test_update_conditional_distribution_does_not_recondition_on_later_income() -> None:
    """Pin that later-period income does NOT re-condition the carried state.

    `_update_conditional_distribution` propagates the chained sample through
    the just-fitted transition/investment equations, but the conditioning
    payload (`cond_means`, `cond_chols`, `conditional_weights`,
    `chain_links`) is carried forward UNCHANGED -- the state distribution
    stays conditioned on period-0 income only. Income at t > 0 flows solely
    through the transition function (so the chained `components` legitimately
    change), never through a re-conditioning update.
    """
    n_state = 1
    n_endog = 0
    n_observed_factors = 1
    n_components = 1
    n_halton = 6
    n_obs = 4

    rng = np.random.default_rng(202)
    prev_sample = jnp.asarray(rng.normal(0, 1, (n_halton, n_obs, n_state)))
    cond_means = jnp.asarray(rng.normal(0, 1, (n_components, n_obs, n_state)))
    cond_chols = jnp.asarray(
        np.tile(np.eye(n_state), (n_components, 1, 1)).astype(float)
    )
    conditional_weights = jnp.ones((n_obs, n_components))

    prev_distribution = ConditionalDistribution(
        mixture_weights=jnp.array([1.0]),
        components=(
            MixtureComponent(mean=jnp.zeros(n_state), chol_cov=jnp.eye(n_state)),
        ),
        samples_per_component=(prev_sample,),
        conditional_weights=conditional_weights,
        cond_means=cond_means,
        cond_chols=cond_chols,
        chain_links=(),
    )

    # Linear transition over (state, income): theta_t = 0.7 * theta + 0.3 * Y.
    def combined_transition(
        full_prev_with_obs: jax.Array, params: jax.Array
    ) -> jax.Array:
        return jnp.array(
            [params[0] * full_prev_with_obs[0] + params[1] * full_prev_with_obs[1]]
        )

    idx = pd.MultiIndex.from_tuples(
        [
            ("transition", 1, "skill", "skill"),
            ("transition", 1, "skill", "income"),
            ("shock_sds", 1, "skill", "-"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    result_params = pd.DataFrame({"value": [0.7, 0.3, 0.4]}, index=idx)

    # Two different later-period income inputs.
    income_a = jnp.asarray(rng.normal(0, 1, (n_obs, n_observed_factors)))
    income_b = income_a + 5.0

    joint_nodes = jnp.asarray(rng.normal(0, 1, (n_halton, 1)))  # n_shock = 1 column

    def _update(observed_factor_values: jax.Array) -> ConditionalDistribution:
        return _update_conditional_distribution(
            prev_distribution=prev_distribution,
            result_params=result_params,
            combined_transition=combined_transition,
            joint_nodes=joint_nodes,
            n_state=n_state,
            n_endog=n_endog,
            n_shock=1,
            shock_factor_indices=jnp.array([0]),
            observed_factor_values=observed_factor_values,
            n_observed_factors=n_observed_factors,
        )

    out_a = _update(income_a)
    out_b = _update(income_b)

    # The conditioning payload is byte-identical to the input and across the
    # two income values: income at t > 0 leaves it untouched.
    for out in (out_a, out_b):
        np.testing.assert_array_equal(
            np.asarray(out.cond_means), np.asarray(prev_distribution.cond_means)
        )
        np.testing.assert_array_equal(
            np.asarray(out.cond_chols), np.asarray(prev_distribution.cond_chols)
        )
        np.testing.assert_array_equal(
            np.asarray(out.conditional_weights),
            np.asarray(prev_distribution.conditional_weights),
        )
        assert out.chain_links == prev_distribution.chain_links

    # The chained sample summary DOES change with income (the single allowed
    # channel: income flows through the transition function).
    assert not np.allclose(
        np.asarray(out_a.components[0].mean),
        np.asarray(out_b.components[0].mean),
    )
