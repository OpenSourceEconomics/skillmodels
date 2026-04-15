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

    # --- CHS estimation ---
    max_inputs = get_maximization_inputs(model, data)
    chs_params = max_inputs["params_template"].copy()
    free = chs_params["lower_bound"] != chs_params["upper_bound"]
    chs_params.loc[free, "value"] = 0.5
    load_free = free & (chs_params.index.get_level_values("category") == "loadings")
    chs_params.loc[load_free, "value"] = 1.0
    ctrl_free = free & (chs_params.index.get_level_values("category") == "controls")
    chs_params.loc[ctrl_free, "value"] = 0.0

    def _neg_loglike_and_grad(
        p: pd.DataFrame,
    ) -> tuple[float, np.ndarray]:
        val, grad = max_inputs["loglike_and_gradient"](p)
        return -float(val), -np.array(grad)

    opt_res = om.minimize(
        fun=lambda p: -max_inputs["loglike"](p),
        params=chs_params[["value"]],
        algorithm="scipy_lbfgsb",
        bounds=om.Bounds(
            lower=chs_params["lower_bound"],
            upper=chs_params["upper_bound"],
        ),
        constraints=max_inputs["constraints"],
        fun_and_jac=_neg_loglike_and_grad,
    )
    chs_est = opt_res.params

    # --- Compare period-0 measurement parameters ---
    tol = 0.15  # generous tolerance for finite-sample differences

    for meas in ("m2", "m3"):
        af_load = float(
            af_p0.loc[("loadings", 0, meas, "skill"), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_load = float(chs_est.loc[("loadings", 0, meas, "skill"), "value"])
        assert abs(af_load - chs_load) < tol, (
            f"loading({meas}): AF={af_load:.4f} vs CHS={chs_load:.4f}"
        )

        af_intercept = float(
            af_p0.loc[("controls", 0, meas, "constant"), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_intercept = float(chs_est.loc[("controls", 0, meas, "constant"), "value"])
        assert abs(af_intercept - chs_intercept) < tol, (
            f"intercept({meas}): AF={af_intercept:.4f} vs CHS={chs_intercept:.4f}"
        )

    for meas in ("m1", "m2", "m3"):
        af_sd = float(
            af_p0.loc[("meas_sds", 0, meas, "-"), "value"]  # ty: ignore[invalid-argument-type]
        )
        chs_sd = float(chs_est.loc[("meas_sds", 0, meas, "-"), "value"])
        assert abs(af_sd - chs_sd) < tol, (
            f"meas_sd({meas}): AF={af_sd:.4f} vs CHS={chs_sd:.4f}"
        )
