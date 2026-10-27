"""CHS estimation with probit and Tobit measurements.

The reference likelihood is a scalar assumed-density filter written directly from
the update equations for a one-factor linear model: Gaussian measurements and Tobit
values inside their bounds use the Kalman update; a probit observation or a Tobit
value at a bound contributes `log Phi(t)` and moves the state to the exact posterior
mean and variance under the Gaussian predicted state. It shares no code with the
production scan.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr

from skillmodels.chs.estimate import estimate_chs
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.individual_states import get_individual_states_from_params
from skillmodels.common.measurement_models import (
    ProbitMeasurement,
    TobitMeasurement,
)
from skillmodels.common.model_spec import (
    AnchoringSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)

_MEASURES = ("g1", "p1", "g2")
_N_PERIODS = 3
_TRUE: dict[str, Any] = {
    "initial_mean": 0.2,
    "initial_sd": 0.8,
    "rho": (0.8, 0.7),
    "trans_const": (0.1, -0.1),
    "shock_sd": (0.4, 0.5),
    "g1_sd": 0.7,
    "p1_loading": (0.9, 1.1, 0.7),
    "p1_const": (0.2, -0.1, 0.3),
    "g2_loading": 0.6,
    "g2_const": 0.4,
    "g2_sd": 0.5,
}


def _model(measurement_models=None, anchoring=None) -> ModelSpec:
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
        anchoring=anchoring,
    )


_PROBIT = {"p1": ProbitMeasurement()}


def _data(n_obs=40, seed=0) -> pd.DataFrame:
    """Simulate the model; persons 0-4 die (p1 = 0, then missing) in period 1."""
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
        p1 = (index + rng.normal(size=n_obs) > 0).astype(float)
        g2 = (
            _TRUE["g2_const"]
            + _TRUE["g2_loading"] * theta
            + _TRUE["g2_sd"] * rng.normal(size=n_obs)
        )
        for i in range(n_obs):
            row = {"caseid": i, "period": t, "g1": g1[i], "p1": p1[i], "g2": g2[i]}
            if i < 5 and t == 1:
                row["p1"] = 0.0
            if i < 5 and t == 2:
                row.update(g1=np.nan, p1=np.nan, g2=np.nan)
            if i == 7:
                row["g2"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def _params(template: pd.DataFrame, *, probit_sd=1.0) -> pd.DataFrame:
    params = template.copy()

    def put(loc, value):
        params.loc[loc, "value"] = value

    for t in range(_N_PERIODS):
        put(("controls", t, "g1", "constant"), 0.0)
        put(("controls", t, "p1", "constant"), _TRUE["p1_const"][t])
        put(("controls", t, "g2", "constant"), _TRUE["g2_const"])
        put(("loadings", t, "g1", "skills"), 1.0)
        put(("loadings", t, "p1", "skills"), _TRUE["p1_loading"][t])
        put(("loadings", t, "g2", "skills"), _TRUE["g2_loading"])
        put(("meas_sds", t, "g1", "-"), _TRUE["g1_sd"])
        put(("meas_sds", t, "p1", "-"), probit_sd)
        put(("meas_sds", t, "g2", "-"), _TRUE["g2_sd"])
    for t in range(_N_PERIODS - 1):
        put(("shock_sds", t, "skills", "-"), _TRUE["shock_sd"][t])
        put(("transition", t, "skills", "skills"), _TRUE["rho"][t])
        put(("transition", t, "skills", "constant"), _TRUE["trans_const"][t])
    put(("initial_states", 0, "mixture_0", "skills"), _TRUE["initial_mean"])
    put(("initial_cholcovs", 0, "mixture_0", "skills-skills"), _TRUE["initial_sd"])
    put(("mixture_weights", 0, "mixture_0", "-"), 1.0)
    return params


def _shifted(params: pd.DataFrame, loc: tuple, step: float) -> pd.DataFrame:
    values = params["value"].to_numpy(dtype=float, copy=True)
    values[params.index.get_loc(loc)] += step
    return params.assign(value=values)


def _scalar_adf_reference(data: pd.DataFrame, families: dict[str, tuple]):
    """Return per-person log likelihoods and filtered means after each period.

    `families` maps a measurement to `("probit",)` or `("tobit", lower, upper)`;
    others are Gaussian.
    """
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
                "p1": (_TRUE["p1_const"][t], _TRUE["p1_loading"][t], 1.0),
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
                family = families.get(name, ("gaussian",))
                if family[0] == "probit":
                    direction, threshold = 2.0 * y - 1.0, 0.0
                elif family[0] == "tobit" and y <= family[1]:
                    direction, threshold = -1.0, family[1]
                elif family[0] == "tobit" and y >= family[2]:
                    direction, threshold = 1.0, family[2]
                else:
                    direction = None
                if direction is not None:
                    tt = direction * (mu - threshold) / math.sqrt(s2)
                    log_p = float(log_ndtr(tt))
                    r = math.exp(-0.5 * tt * tt - 0.5 * math.log(2 * math.pi) - log_p)
                    total += log_p
                    mean += direction * cross * r / math.sqrt(s2)
                    var -= cross**2 * r * (r + tt) / s2
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
def probit_inputs():
    data = _data()
    inputs = get_maximization_inputs(
        model_spec=_model(_PROBIT),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    return data, inputs, _params(inputs["params_template"])


def test_probit_loglikeobs_equals_scalar_assumed_density_reference(
    probit_inputs,
) -> None:
    data, inputs, params = probit_inputs
    expected, _ = _scalar_adf_reference(data, {"p1": ("probit",)})
    np.testing.assert_allclose(inputs["loglikeobs"](params), expected, atol=1e-10)


def test_debug_contributions_equal_the_jitted_objective(probit_inputs) -> None:
    _, inputs, params = probit_inputs
    debug = inputs["debug_loglike"](params)
    np.testing.assert_allclose(
        debug["contributions"], inputs["loglikeobs"](params), rtol=1e-13
    )
    np.testing.assert_allclose(debug["value"], inputs["loglike"](params), rtol=1e-13)


def test_debug_residuals_of_probit_updates_are_missing(probit_inputs) -> None:
    _, inputs, params = probit_inputs
    residuals = inputs["debug_loglike"](params)["residuals"]
    assert residuals.query("measurement == 'p1'")["residual"].isna().all()
    assert residuals.query("measurement == 'g2'")["residual"].notna().any()


def test_probit_meas_sd_is_fixed_to_one_and_does_not_enter_the_likelihood(
    probit_inputs,
) -> None:
    _, inputs, params = probit_inputs
    template = inputs["params_template"]
    probit_rows = template.query("category == 'meas_sds' and name1 == 'p1'")
    assert (probit_rows["value"] == 1.0).all()
    assert np.isneginf(probit_rows["lower_bound"]).all()
    fixed = {
        c.loc
        for c in inputs["constraints"]
        if getattr(c, "value", None) == 1.0 and c.loc[0] == "meas_sds"
    }
    assert set(probit_rows.index) <= fixed
    shifted = _params(template, probit_sd=3.0)
    assert inputs["loglike"](shifted) == inputs["loglike"](params)


def test_gradient_matches_step_swept_finite_differences(probit_inputs) -> None:
    _, inputs, params = probit_inputs
    _, grad = inputs["loglike_and_gradient"](params)
    locs = [
        ("loadings", 1, "p1", "skills"),
        ("controls", 2, "p1", "constant"),
        ("meas_sds", 0, "g2", "-"),
        ("transition", 0, "skills", "skills"),
        ("initial_cholcovs", 0, "mixture_0", "skills-skills"),
    ]
    positions = [params.index.get_loc(loc) for loc in locs]
    for loc, pos in zip(locs, positions, strict=True):
        errors = []
        for step in (1e-3, 1e-4, 1e-5, 1e-6):
            up, down = _shifted(params, loc, step), _shifted(params, loc, -step)
            fd = (inputs["loglike"](up) - inputs["loglike"](down)) / (2 * step)
            errors.append(abs(fd - grad[pos]) / max(1.0, abs(grad[pos])))
        assert min(errors) < 1e-5, loc
    probit_sd = params.index.get_loc(("meas_sds", 1, "p1", "-"))
    assert grad[probit_sd] == 0.0


def test_split_dataset_gradient_equals_unsplit_gradient(probit_inputs) -> None:
    data, inputs, params = probit_inputs
    split = get_maximization_inputs(
        model_spec=_model(_PROBIT),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none", split_dataset=3),
    )
    np.testing.assert_allclose(
        split["loglike_and_gradient"](params)[1],
        inputs["loglike_and_gradient"](params)[1],
        rtol=1e-11,
        atol=1e-11,
    )


def test_filtered_states_use_the_probit_update(probit_inputs) -> None:
    data, _, params = probit_inputs
    _, expected_means = _scalar_adf_reference(data, {"p1": ("probit",)})
    states = get_individual_states_from_params(
        model_spec=_model(_PROBIT),
        data=data,
        params=params,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["unanchored_states"]["states"]
    got = states.pivot_table(index="id", columns="period", values="skills")
    observed = ~np.isnan(got.to_numpy())
    np.testing.assert_allclose(
        got.to_numpy()[observed], expected_means[observed], atol=1e-10
    )


def test_constant_probit_sample_is_a_valid_likelihood_input() -> None:
    data = _data()
    data.loc[pd.IndexSlice[:, 0], "p1"] = 1.0
    inputs = get_maximization_inputs(
        model_spec=_model(_PROBIT),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    expected, _ = _scalar_adf_reference(data, {"p1": ("probit",)})
    params = _params(inputs["params_template"])
    np.testing.assert_allclose(inputs["loglikeobs"](params), expected, atol=1e-10)


@pytest.mark.parametrize("strategy", ["spearman", "amn"])
def test_gaussian_seeding_keeps_the_probit_model(strategy) -> None:
    """Seeds come from a Gaussian working model; the likelihood stays probit."""
    data = _data(n_obs=200)
    inputs = get_maximization_inputs(
        model_spec=_model(_PROBIT),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy=strategy),
    )
    template = inputs["params_template"]
    assert np.isfinite(template["value"]).all()
    assert (
        template.query("category == 'meas_sds' and name1 == 'p1'")["value"] == 1
    ).all()
    expected, _ = _scalar_adf_reference(data, {"p1": ("probit",)})
    np.testing.assert_allclose(
        inputs["loglikeobs"](_params(template)), expected, atol=1e-10
    )


def test_fixed_params_contradicting_the_probit_scale_raise() -> None:
    fixed = pd.DataFrame(
        {"value": [2.0]},
        index=pd.MultiIndex.from_tuples(
            [("meas_sds", 0, "p1", "-")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )
    with pytest.raises(ValueError, match="probit measurement is fixed to 1"):
        get_maximization_inputs(
            model_spec=_model(_PROBIT),
            data=_data(),
            chs_options=CHSEstimationOptions(start_params_strategy="none"),
            fixed_params=fixed,
        )


def test_limited_anchoring_outcome_is_rejected() -> None:
    model = _model(
        _PROBIT, anchoring=AnchoringSpec(outcomes={"skills": "p1"}, free_loadings=True)
    )
    with pytest.raises(NotImplementedError, match="anchoring requires Gaussian"):
        get_maximization_inputs(
            model_spec=model,
            data=_data(),
            chs_options=CHSEstimationOptions(start_params_strategy="none"),
        )


_LOWER, _UPPER = 0.1, 0.7


def _tobit_data(lower, upper) -> pd.DataFrame:
    """Censor g2 at the bounds and place float64 boundary witnesses.

    Persons 10-13 observe g2 exactly at a bound or at the adjacent float inside it in
    period 0; float32 rounding would swap the mass and density events of these values.
    """
    data = _data()
    data["g2"] = data["g2"].clip(lower, upper)
    witnesses = {
        10: lower,
        11: math.nextafter(lower, math.inf),
        12: upper,
        13: math.nextafter(upper, -math.inf),
    }
    for person, value in witnesses.items():
        if math.isfinite(value):
            data.loc[(person, 0), "g2"] = value
    return data


@pytest.mark.parametrize(
    ("lower", "upper"), [(_LOWER, _UPPER), (_LOWER, math.inf), (-math.inf, _UPPER)]
)
def test_tobit_loglikeobs_equals_scalar_assumed_density_reference(lower, upper) -> None:
    data = _tobit_data(lower, upper)
    tobit = TobitMeasurement(
        lower=lower if math.isfinite(lower) else None,
        upper=upper if math.isfinite(upper) else None,
    )
    inputs = get_maximization_inputs(
        model_spec=_model({**_PROBIT, "g2": tobit}),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    expected, _ = _scalar_adf_reference(
        data, {"p1": ("probit",), "g2": ("tobit", lower, upper)}
    )
    params = _params(inputs["params_template"])
    np.testing.assert_allclose(inputs["loglikeobs"](params), expected, atol=1e-10)


def test_tobit_gradient_matches_finite_differences() -> None:
    data = _tobit_data(_LOWER, _UPPER)
    inputs = get_maximization_inputs(
        model_spec=_model({**_PROBIT, "g2": TobitMeasurement(_LOWER, _UPPER)}),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    params = _params(inputs["params_template"])
    _, grad = inputs["loglike_and_gradient"](params)
    locs = [
        ("meas_sds", 0, "g2", "-"),
        ("loadings", 1, "g2", "skills"),
        ("controls", 2, "g2", "constant"),
    ]
    for loc in locs:
        pos = params.index.get_loc(loc)
        errors = []
        for step in (1e-3, 1e-4, 1e-5, 1e-6):
            up, down = _shifted(params, loc, step), _shifted(params, loc, -step)
            fd = (inputs["loglike"](up) - inputs["loglike"](down)) / (2 * step)
            errors.append(abs(fd - grad[pos]) / max(1.0, abs(grad[pos])))
        assert grad[pos] != 0.0
        assert min(errors) < 1e-5, loc


def test_tobit_without_censored_values_equals_the_gaussian_model() -> None:
    """Interior Tobit values are Gaussian observations, through either scan."""
    data = _data().astype(np.float32).astype(np.float64)
    gaussian = get_maximization_inputs(
        model_spec=_model(),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    tobit = get_maximization_inputs(
        model_spec=_model({"g2": TobitMeasurement(lower=-100.0, upper=100.0)}),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    params = _params(gaussian["params_template"], probit_sd=0.9)
    np.testing.assert_allclose(
        tobit["loglikeobs"](params), gaussian["loglikeobs"](params), rtol=1e-12
    )
    np.testing.assert_allclose(
        tobit["loglike_and_gradient"](params)[1],
        gaussian["loglike_and_gradient"](params)[1],
        rtol=1e-10,
        atol=1e-12,
    )


def test_user_bounds_excluding_the_probit_scale_raise() -> None:
    bounds = pd.Series(
        [2.0],
        index=pd.MultiIndex.from_tuples(
            [("meas_sds", 1, "p1", "-")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )
    with pytest.raises(ValueError, match="probit"):
        estimate_chs(
            model_spec=_model(_PROBIT),
            data=_data(),
            options=CHSEstimationOptions(start_params_strategy="none"),
            lower_bounds=bounds,
        )


@pytest.mark.parametrize("transition", ["translog", "asymmetric_translog"])
def test_probit_and_tobit_with_nonlinear_transitions(transition) -> None:
    """The limited update consumes the unscented prediction like a Gaussian one."""
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(_MEASURES,) * _N_PERIODS,
                normalizations=Normalizations(
                    loadings=({"g1": 1},) * _N_PERIODS,
                    intercepts=({"g1": 0},) * _N_PERIODS,
                ),
                transition_function=transition,
            ),
        },
        measurement_models={**_PROBIT, "g2": TobitMeasurement(_LOWER, _UPPER)},
    )
    data = _tobit_data(_LOWER, _UPPER)
    inputs = get_maximization_inputs(
        model_spec=model,
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    params = inputs["params_template"].copy()
    linear = _params(
        get_maximization_inputs(
            model_spec=_model({**_PROBIT, "g2": TobitMeasurement(_LOWER, _UPPER)}),
            data=data,
            chs_options=CHSEstimationOptions(start_params_strategy="none"),
        )["params_template"]
    )
    shared = params.index.intersection(linear.index)
    params.loc[shared, "value"] = linear.loc[shared, "value"]
    params["value"] = params["value"].fillna(0.05)
    value, grad = inputs["loglike_and_gradient"](params)
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(
        inputs["debug_loglike"](params)["value"], value, rtol=1e-12
    )
