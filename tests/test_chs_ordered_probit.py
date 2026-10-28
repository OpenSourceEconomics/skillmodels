"""CHS estimation with ordered-probit measurements.

The reference likelihood is a scalar assumed-density filter written directly from the
update equations for a one-factor linear model, as for probit and Tobit: an ordered
observation `y = k` contributes `log(Phi(beta) - Phi(alpha))` with
`alpha = (c_k - mu) / s`, `beta = (c_{k+1} - mu) / s` (`c_0 = -inf`, `c_K = +inf`),
and moves the state to the exact posterior mean and variance under the Gaussian
predicted state, from the closed-form truncated-normal moments. It shares no code
with the production scan.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
from test_chs_limited_measurements import _N_PERIODS, _TRUE, _shifted

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.individual_states import get_individual_states_from_params
from skillmodels.common.measurement_models import (
    OrderedProbitMeasurement,
    ProbitMeasurement,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations

_MEASURES = ("g1", "o1", "g2")
_CUTS = (0.0, 0.6, 1.5)


def _model(measurement_models) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(_MEASURES,) * _N_PERIODS,
                normalizations=Normalizations(
                    loadings=({"g1": 1},) * _N_PERIODS,
                    intercepts=({"g1": 0},) * _N_PERIODS,
                ),
                transition_function="linear",
            ),
        },
        measurement_models=measurement_models,
    )


_ORDERED = {"o1": OrderedProbitMeasurement(n_categories=len(_CUTS) + 1)}


def _data(cuts=_CUTS, n_obs=40, seed=1) -> pd.DataFrame:
    """Simulate the model; persons 0-4 have no measurement in period 2."""
    rng = np.random.default_rng(seed)
    theta = _TRUE["initial_mean"] + _TRUE["initial_sd"] * rng.normal(size=n_obs)
    rows = []
    for t in range(_N_PERIODS):
        if t > 0:
            theta = (
                _TRUE["rho"][t - 1] * theta
                + _TRUE["trans_const"][t - 1]
                + _TRUE["shock_sd"][t - 1] * rng.normal(size=n_obs)
            )
        g1 = theta + _TRUE["g1_sd"] * rng.normal(size=n_obs)
        index = _TRUE["p1_const"][t] + _TRUE["p1_loading"][t] * theta
        o1 = np.searchsorted(np.array(cuts), index + rng.normal(size=n_obs))
        g2 = (
            _TRUE["g2_const"]
            + _TRUE["g2_loading"] * theta
            + _TRUE["g2_sd"] * rng.normal(size=n_obs)
        )
        for i in range(n_obs):
            row = {"caseid": i, "period": t, "g1": g1[i], "o1": float(o1[i])}
            row["g2"] = g2[i]
            if i < 5 and t == 2:
                row.update(g1=np.nan, o1=np.nan, g2=np.nan)
            if i == 7:
                row["o1"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def _params(template: pd.DataFrame, cuts=_CUTS) -> pd.DataFrame:
    params = template.copy()

    def put(loc, value):
        params.loc[loc, "value"] = value

    for t in range(_N_PERIODS):
        put(("controls", t, "g1", "constant"), 0.0)
        put(("controls", t, "o1", "constant"), _TRUE["p1_const"][t])
        put(("controls", t, "g2", "constant"), _TRUE["g2_const"])
        put(("loadings", t, "g1", "skills"), 1.0)
        put(("loadings", t, "o1", "skills"), _TRUE["p1_loading"][t])
        put(("loadings", t, "g2", "skills"), _TRUE["g2_loading"])
        put(("meas_sds", t, "g1", "-"), _TRUE["g1_sd"])
        put(("meas_sds", t, "o1", "-"), 1.0)
        put(("meas_sds", t, "g2", "-"), _TRUE["g2_sd"])
        for k, cut in enumerate(cuts, start=1):
            if ("cutpoints", t, "o1", f"cutpoint_{k}") in params.index:
                put(("cutpoints", t, "o1", f"cutpoint_{k}"), cut)
    for t in range(_N_PERIODS - 1):
        put(("shock_sds", t, "skills", "-"), _TRUE["shock_sd"][t])
        put(("transition", t, "skills", "skills"), _TRUE["rho"][t])
        put(("transition", t, "skills", "constant"), _TRUE["trans_const"][t])
    put(("initial_states", 0, "mixture_0", "skills"), _TRUE["initial_mean"])
    put(("initial_cholcovs", 0, "mixture_0", "skills-skills"), _TRUE["initial_sd"])
    put(("mixture_weights", 0, "mixture_0", "-"), 1.0)
    return params


def _truncated_normal(alpha: float, beta: float) -> tuple[float, float, float]:
    """Return `log Z`, `E[z]` and `Var[z]` of `N(0, 1)` truncated to `(alpha, beta]`."""
    z = ndtr(beta) - ndtr(alpha)

    def phi(x):
        return 0.0 if math.isinf(x) else math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def x_phi(x):
        return 0.0 if math.isinf(x) else x * phi(x)

    mean = (phi(alpha) - phi(beta)) / z
    var = 1 + (x_phi(alpha) - x_phi(beta)) / z - mean**2
    return math.log(z), mean, var


def _scalar_adf_reference(data: pd.DataFrame, cuts=_CUTS):
    """Return per-person log likelihoods and filtered means after each period."""
    table = (-math.inf, *cuts, math.inf)
    loglikes, filtered = [], []
    for _, person in data.groupby(level="caseid"):
        mean, var = _TRUE["initial_mean"], _TRUE["initial_sd"] ** 2
        total, means = 0.0, []
        for t in range(_N_PERIODS):
            if t > 0:
                mean = _TRUE["rho"][t - 1] * mean + _TRUE["trans_const"][t - 1]
                var = _TRUE["rho"][t - 1] ** 2 * var + _TRUE["shock_sd"][t - 1] ** 2
            row = person.iloc[t]
            specs = {
                "g1": (0.0, 1.0, _TRUE["g1_sd"]),
                "o1": (_TRUE["p1_const"][t], _TRUE["p1_loading"][t], 1.0),
                "g2": (_TRUE["g2_const"], _TRUE["g2_loading"], _TRUE["g2_sd"]),
            }
            for name in _MEASURES:
                y = row[name]
                if np.isnan(y):
                    continue
                const, loading, sd = specs[name]
                mu = const + loading * mean
                s2 = sd**2 + loading**2 * var
                cross = var * loading
                if name == "o1":
                    k = int(y)
                    s = math.sqrt(s2)
                    log_z, z_mean, z_var = _truncated_normal(
                        (table[k] - mu) / s, (table[k + 1] - mu) / s
                    )
                    total += log_z
                    mean += cross * z_mean / s
                    var -= cross**2 * (1 - z_var) / s2
                else:
                    resid = y - mu
                    total += -0.5 * (math.log(2 * math.pi * s2) + resid**2 / s2)
                    mean += cross * resid / s2
                    var -= cross**2 / s2
            means.append(mean)
        loglikes.append(total)
        filtered.append(means)
    return np.array(loglikes), np.array(filtered)


@pytest.fixture(scope="module")
def ordered_inputs():
    data = _data()
    inputs = get_maximization_inputs(
        model_spec=_model(_ORDERED),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    return data, inputs, _params(inputs["params_template"])


def test_ordered_loglikeobs_equals_scalar_assumed_density_reference(
    ordered_inputs,
) -> None:
    data, inputs, params = ordered_inputs
    expected, _ = _scalar_adf_reference(data)
    np.testing.assert_allclose(inputs["loglikeobs"](params), expected, atol=1e-10)


def test_filtered_states_use_the_interval_update(ordered_inputs) -> None:
    data, _, params = ordered_inputs
    _, expected_means = _scalar_adf_reference(data)
    states = get_individual_states_from_params(
        model_spec=_model(_ORDERED),
        data=data,
        params=params,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["unanchored_states"]["states"]
    got = states.pivot_table(index="id", columns="period", values="skills")
    observed = ~np.isnan(got.to_numpy())
    np.testing.assert_allclose(
        got.to_numpy()[observed], expected_means[observed], atol=1e-10
    )


def test_debug_contributions_equal_the_jitted_objective(ordered_inputs) -> None:
    _, inputs, params = ordered_inputs
    debug = inputs["debug_loglike"](params)
    np.testing.assert_allclose(
        debug["contributions"], inputs["loglikeobs"](params), rtol=1e-13
    )


def test_gradient_matches_step_swept_finite_differences(ordered_inputs) -> None:
    _, inputs, params = ordered_inputs
    _, grad = inputs["loglike_and_gradient"](params)
    locs = [
        ("cutpoints", 0, "o1", "cutpoint_2"),
        ("cutpoints", 1, "o1", "cutpoint_3"),
        ("loadings", 1, "o1", "skills"),
        ("controls", 2, "o1", "constant"),
        ("transition", 0, "skills", "skills"),
        ("initial_cholcovs", 0, "mixture_0", "skills-skills"),
    ]
    for loc in locs:
        pos = params.index.get_loc(loc)
        errors = []
        for step in (1e-3, 1e-4, 1e-5, 1e-6):
            up, down = _shifted(params, loc, step), _shifted(params, loc, -step)
            fd = (inputs["loglike"](up) - inputs["loglike"](down)) / (2 * step)
            errors.append(abs(fd - grad[pos]) / max(1.0, abs(grad[pos])))
        assert min(errors) < 1e-5, loc
    unit_sd = params.index.get_loc(("meas_sds", 1, "o1", "-"))
    assert grad[unit_sd] == 0.0


def test_two_categories_reproduce_the_probit_model_exactly() -> None:
    data = _data(cuts=(0.0,))
    options = CHSEstimationOptions(start_params_strategy="none")
    ordered = get_maximization_inputs(
        model_spec=_model({"o1": OrderedProbitMeasurement(n_categories=2)}),
        data=data,
        chs_options=options,
    )
    probit = get_maximization_inputs(
        model_spec=_model({"o1": ProbitMeasurement()}),
        data=data,
        chs_options=options,
    )
    ordered_params = _params(ordered["params_template"], cuts=(0.0,))
    probit_params = _params(probit["params_template"], cuts=(0.0,))
    np.testing.assert_array_equal(
        ordered["loglikeobs"](ordered_params), probit["loglikeobs"](probit_params)
    )
    ordered_grad = pd.Series(
        ordered["loglike_and_gradient"](ordered_params)[1], index=ordered_params.index
    )
    probit_grad = pd.Series(
        probit["loglike_and_gradient"](probit_params)[1], index=probit_params.index
    )
    np.testing.assert_array_equal(
        ordered_grad.loc[probit_grad.index].to_numpy(), probit_grad.to_numpy()
    )


def test_spearman_start_values_give_a_finite_likelihood() -> None:
    inputs = get_maximization_inputs(
        model_spec=_model(_ORDERED),
        data=_data(),
        chs_options=CHSEstimationOptions(start_params_strategy="spearman"),
    )
    value, grad = inputs["loglike_and_gradient"](inputs["params_template"])
    assert np.isfinite(value)
    assert np.isfinite(grad).all()
