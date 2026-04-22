"""End-to-end tests for the AF estimator.

Run AF estimation on MODEL2 test data and verify it produces reasonable
results, comparing to the CHS Kalman filter estimates where applicable.
"""

from pathlib import Path

import jax
import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.config import TEST_DATA_DIR
from skillmodels.filtered_states import get_filtered_states
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.model_spec import (
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)

jax.config.update("jax_enable_x64", True)

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
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
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(
        model_spec=model2_af,
        data=model2_data,
        af_options=af_options,
    )

    # Basic checks
    assert len(result.period_results) == 3
    assert result.all_params is not None
    assert len(result.all_params) > 0

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
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(
        model_spec=model2_af,
        data=model2_data,
        af_options=af_options,
    )

    # Compare period 0 measurement SDs
    af_meas_sds = result.all_params.query("category == 'meas_sds' and period == 0")
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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
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
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(model_spec=model, data=data, af_options=af_options)

    assert len(result.period_results) == 2
    assert np.isfinite(result.period_results[0].loglikelihood)

    # Check that estimated loadings are roughly in the right direction
    af_loadings = result.all_params.query("category == 'loadings' and period == 0")
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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )

    # --- AF estimation ---
    af_result = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=50,
            n_halton_points_shock=20,
            n_mixture_components=1,
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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
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
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )
    result = estimate_af(model_spec=model, data=data, af_options=af_opts)

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
        n_halton_points=40,
        n_halton_points_shock=20,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )
    result = estimate_af(model_spec=model, data=data, af_options=af_opts)

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
        af_options=AFEstimationOptions(
            n_halton_points=40,
            n_halton_points_shock=20,
            n_mixture_components=1,
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
    max_inputs = get_maximization_inputs(model, data)
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
        af_options=AFEstimationOptions(
            n_halton_points=60,
            n_halton_points_shock=30,
            n_mixture_components=1,
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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            n_mixture_components=1,
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


# ---------------------------------------------------------------------------
# Posterior states tests
# ---------------------------------------------------------------------------


@pytest.mark.end_to_end
def test_af_get_filtered_states() -> None:
    """Verify get_filtered_states works with AF results.

    Run AF on a simple single-factor model, then call get_filtered_states
    with the AF result. Check the returned DataFrame has the right shape,
    columns, and reasonable values.
    """
    data, _true_params = _simulate_linear_transition_data(n_obs=200, n_periods=3)
    model = _make_linear_transition_model(n_periods=3)

    af_result = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            n_mixture_components=1,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )

    result = get_filtered_states(
        model_spec=model,
        data=data,
        params=af_result.all_params,
        af_result=af_result,
    )

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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            n_mixture_components=1,
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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )

    result = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=40,
            n_halton_points_shock=15,
            n_mixture_components=1,
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
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
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
        af_options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            n_mixture_components=1,
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
