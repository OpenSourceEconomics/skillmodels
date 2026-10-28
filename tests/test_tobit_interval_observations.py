"""Tobit measurements with observation-specific intervals.

A `TobitMeasurement` with `interval_columns` observes each person either exactly (a
value strictly inside the fixed bounds, or a value at a fixed censoring bound) or only
through an interval `lo < y* <= hi` of the latent response, read from two data
columns. Both kinds of observation share the intercept, the loadings and the error SD
`sigma`. Given the Gaussian predicted state, an interval contributes
`log(Phi(beta) - Phi(alpha))` with `alpha = (lo - mu) / s`, `beta = (hi - mu) / s`,
`s^2 = sigma^2 + lambda'P lambda`, and moves the state to the exact posterior mean and
variance from the truncated-normal moments of `(y* - mu) / s`. An infinite end makes
the interval a half-line.

The likelihood reference is a scalar assumed-density filter written directly from
these equations for a one-factor linear model; it shares no code with the production
scan.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import log_ndtr
from test_chs_halfline import _assert_update_close, _prior, _quadrature_update
from test_chs_limited_measurements import _N_PERIODS, _TRUE, _shifted
from test_chs_ordered_probit import _truncated_normal

from skillmodels.af.estimate import estimate_af
from skillmodels.chs.halfline import kalman_update_limited
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.individual_states import get_individual_states_from_params
from skillmodels.common.measurement_models import (
    MeasurementFamily,
    ObservationEvent,
    TobitMeasurement,
    classify_observations,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.process_data import process_data
from skillmodels.common.process_model import process_model
from skillmodels.common.simulate_data import simulate_dataset

jax.config.update("jax_enable_x64", True)

E = ObservationEvent
_INF = math.inf
_MEASURES = ("g1", "c1", "g2")
_LOWER, _UPPER = -1.0, 1.6
_COLUMNS = ("c1_lo", "c1_hi")
_C1 = TobitMeasurement(lower=_LOWER, upper=_UPPER, interval_columns=_COLUMNS)
_C1_TRUE: dict[str, Any] = {
    "const": (0.3, 0.1, -0.2),
    "loading": (0.9, 1.2, 0.8),
    "sd": 0.6,
}
# Class intervals of the persons observed only through an interval.
_CLASSES = ((-_INF, -0.3), (-0.3, 0.6), (0.6, _INF))


def _model(c1_model=_C1) -> ModelSpec:
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
        measurement_models={"c1": c1_model},
    )


def _data(n_obs=40, seed=3) -> pd.DataFrame:
    """Simulate the model; odd persons report `c1` only by its class interval.

    - person 0 has a narrow interval in period 0;
    - person 1 has only its interval observation in the last period;
    - person 2 has no `c1` observation in period 1;
    - person 4 has a value at the lower and person 6 at the upper bound.
    """
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
        latent = (
            _C1_TRUE["const"][t]
            + _C1_TRUE["loading"][t] * theta
            + _C1_TRUE["sd"] * rng.normal(size=n_obs)
        )
        c1 = np.clip(latent, _LOWER, _UPPER)
        g2 = (
            _TRUE["g2_const"]
            + _TRUE["g2_loading"] * theta
            + _TRUE["g2_sd"] * rng.normal(size=n_obs)
        )
        for i in range(n_obs):
            row = {"caseid": i, "period": t, "g1": g1[i], "g2": g2[i]}
            row |= {"c1": c1[i], "c1_lo": np.nan, "c1_hi": np.nan}
            if i % 2:
                lo, hi = next(c for c in _CLASSES if c[0] < latent[i] <= c[1])
                row |= {"c1": np.nan, "c1_lo": lo, "c1_hi": hi}
            if i == 0 and t == 0:
                row |= {"c1": np.nan, "c1_lo": 0.2, "c1_hi": 0.2005}
            if i == 1 and t == _N_PERIODS - 1:
                row |= {"g1": np.nan, "g2": np.nan}
            if i == 2 and t == 1:
                row |= {"c1": np.nan}
            if i == 4 and t == 1:
                row |= {"c1": _LOWER}
            if i == 6 and t == 2:
                row |= {"c1": _UPPER}
            rows.append(row)
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def _params(template: pd.DataFrame) -> pd.DataFrame:
    params = template.copy()

    def put(loc, value):
        params.loc[loc, "value"] = value

    for t in range(_N_PERIODS):
        put(("controls", t, "g1", "constant"), 0.0)
        put(("controls", t, "c1", "constant"), _C1_TRUE["const"][t])
        put(("controls", t, "g2", "constant"), _TRUE["g2_const"])
        put(("loadings", t, "g1", "skills"), 1.0)
        put(("loadings", t, "c1", "skills"), _C1_TRUE["loading"][t])
        put(("loadings", t, "g2", "skills"), _TRUE["g2_loading"])
        put(("meas_sds", t, "g1", "-"), _TRUE["g1_sd"])
        put(("meas_sds", t, "c1", "-"), _C1_TRUE["sd"])
        put(("meas_sds", t, "g2", "-"), _TRUE["g2_sd"])
    for t in range(_N_PERIODS - 1):
        put(("shock_sds", t, "skills", "-"), _TRUE["shock_sd"][t])
        put(("transition", t, "skills", "skills"), _TRUE["rho"][t])
        put(("transition", t, "skills", "constant"), _TRUE["trans_const"][t])
    put(("initial_states", 0, "mixture_0", "skills"), _TRUE["initial_mean"])
    put(("initial_cholcovs", 0, "mixture_0", "skills-skills"), _TRUE["initial_sd"])
    put(("mixture_weights", 0, "mixture_0", "-"), 1.0)
    return params


def _scalar_adf_reference(data: pd.DataFrame):
    """Return per-person log likelihoods and filtered means after each period."""
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
                "c1": (_C1_TRUE["const"][t], _C1_TRUE["loading"][t], _C1_TRUE["sd"]),
                "g2": (_TRUE["g2_const"], _TRUE["g2_loading"], _TRUE["g2_sd"]),
            }
            for name in _MEASURES:
                y = row[name]
                const, loading, sd = specs[name]
                mu = const + loading * mean
                s2 = sd**2 + loading**2 * var
                s = math.sqrt(s2)
                cross = var * loading
                if name == "c1" and np.isnan(y) and not np.isnan(row["c1_lo"]):
                    lo, hi = row["c1_lo"], row["c1_hi"]
                elif name == "c1" and y <= _LOWER:
                    lo, hi = -_INF, _LOWER
                elif name == "c1" and y >= _UPPER:
                    lo, hi = _UPPER, _INF
                elif np.isnan(y):
                    continue
                else:
                    resid = y - mu
                    total += -0.5 * (math.log(2 * math.pi * s2) + resid**2 / s2)
                    mean += cross * resid / s2
                    var -= cross**2 / s2
                    continue
                if math.isinf(lo) or math.isinf(hi):
                    direction = 1.0 if math.isinf(hi) else -1.0
                    threshold = lo if math.isinf(hi) else hi
                    tt = direction * (mu - threshold) / s
                    log_z = float(log_ndtr(tt))
                    r = math.exp(-0.5 * tt * tt - 0.5 * math.log(2 * math.pi) - log_z)
                    z_mean, z_var = direction * r, 1.0 - r * (r + tt)
                else:
                    log_z, z_mean, z_var = _truncated_normal(
                        (lo - mu) / s, (hi - mu) / s
                    )
                total += log_z
                mean += cross * z_mean / s
                var -= cross**2 * (1 - z_var) / s2
            means.append(mean)
        loglikes.append(total)
        filtered.append(means)
    return np.array(loglikes), np.array(filtered)


@pytest.fixture(scope="module")
def interval_inputs():
    data = _data()
    inputs = get_maximization_inputs(
        model_spec=_model(),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    return data, inputs, _params(inputs["params_template"])


def test_tobit_with_interval_columns_needs_no_fixed_bound() -> None:
    model = TobitMeasurement(lower=None, upper=None, interval_columns=("lo", "hi"))
    assert model.interval_columns == ("lo", "hi")


def test_interval_columns_must_name_two_distinct_columns() -> None:
    with pytest.raises(ValueError, match="two distinct"):
        TobitMeasurement(interval_columns=("lo", "lo"))


def test_from_dict_parses_interval_columns() -> None:
    spec = {
        "factors": {
            "skills": {
                "measurements": [["g1", "c1"], ["g1", "c1"]],
                "normalizations": {
                    "loadings": [{"g1": 1}, {"g1": 1}],
                    "intercepts": [{"g1": 0}, {"g1": 0}],
                },
                "transition_function": "linear",
            },
        },
        "measurement_models": {
            "c1": {
                "family": "tobit",
                "lower": None,
                "upper": 27.0,
                "interval_columns": ["c1_lo", "c1_hi"],
            },
        },
    }
    model = ModelSpec.from_dict(spec)
    assert model.measurement_model("c1") == TobitMeasurement(
        lower=None, upper=27.0, interval_columns=("c1_lo", "c1_hi")
    )


def _classify(values, lows, highs, lower=_LOWER, upper=_UPPER):
    return classify_observations(
        np.asarray(values, dtype=np.float64),
        MeasurementFamily.TOBIT,
        lower,
        upper,
        interval_bounds=np.column_stack([lows, highs]).astype(np.float64),
    ).tolist()


def test_interval_observations_are_half_lines_or_intervals() -> None:
    nan = np.nan
    events = _classify(
        values=[nan, nan, nan, nan, 0.3, _LOWER, _UPPER],
        lows=[nan, -_INF, 0.5, -0.2, nan, nan, nan],
        highs=[nan, 0.5, _INF, 0.4, nan, nan, nan],
    )
    assert events == [
        E.MISSING,
        E.LOWER_TAIL,
        E.UPPER_TAIL,
        E.INTERVAL,
        E.DENSITY,
        E.LOWER_TAIL,
        E.UPPER_TAIL,
    ]


@pytest.mark.parametrize(
    ("value", "low", "high", "match"),
    [
        (0.3, -0.2, 0.4, "both a value and an interval"),
        (np.nan, -0.2, np.nan, "both ends"),
        (np.nan, 0.4, 0.4, "below its upper end"),
        (np.nan, 0.5, 0.4, "below its upper end"),
        (np.nan, -_INF, _INF, "at least one finite end"),
    ],
)
def test_invalid_interval_observations_raise(value, low, high, match) -> None:
    with pytest.raises(ValueError, match=match):
        _classify([value], [low], [high])


def _process(model, data):
    pm = process_model(model)
    return process_data(
        df=data,
        has_endogenous_factors=False,
        labels=pm.labels,
        update_info=pm.update_info,
        anchoring_info=pm.anchoring,
        measurement_models=model.measurement_models,
    )


def test_process_data_returns_the_interval_bounds_of_each_update() -> None:
    data = _data(n_obs=6)
    out = _process(_model(), data)
    bounds = np.asarray(out["interval_bounds"])
    assert bounds.shape == (3 * _N_PERIODS, 6, 2)
    # Update rows per period: g1, c1, g2.
    np.testing.assert_array_equal(
        bounds[1], data.xs(0, level="period")[list(_COLUMNS)].to_numpy()
    )
    assert np.isnan(bounds[[0, 2]]).all()
    events = np.asarray(out["events"])
    assert events[1].tolist() == [E.INTERVAL, *events[1, 1:].tolist()]


def test_process_data_reports_a_missing_interval_column() -> None:
    data = _data(n_obs=6).drop(columns="c1_hi")
    with pytest.raises(ValueError, match="c1_hi"):
        _process(_model(), data)


def test_interval_observations_with_missing_controls_are_dropped() -> None:
    model = _model().with_controls(("x",))
    data = _data(n_obs=6).assign(x=1.0)
    data.loc[(1, 0), "x"] = np.nan
    with pytest.warns(UserWarning, match="NaNs in the controls"):
        out = _process(model, data)
    assert np.asarray(out["events"])[1, 1] == E.MISSING


def test_loglikeobs_equals_scalar_assumed_density_reference(interval_inputs) -> None:
    data, inputs, params = interval_inputs
    expected, _ = _scalar_adf_reference(data)
    np.testing.assert_allclose(inputs["loglikeobs"](params), expected, atol=1e-10)


def test_filtered_states_use_the_interval_update(interval_inputs) -> None:
    data, _, params = interval_inputs
    _, expected_means = _scalar_adf_reference(data)
    states = get_individual_states_from_params(
        model_spec=_model(),
        data=data,
        params=params,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["unanchored_states"]["states"]
    got = states.pivot_table(index="id", columns="period", values="skills")
    assert not np.isnan(got.to_numpy()).any()
    np.testing.assert_allclose(got.to_numpy(), expected_means, atol=1e-10)


def test_debug_contributions_equal_the_jitted_objective(interval_inputs) -> None:
    _, inputs, params = interval_inputs
    debug = inputs["debug_loglike"](params)
    np.testing.assert_allclose(
        debug["contributions"], inputs["loglikeobs"](params), rtol=1e-13
    )


def test_gradient_matches_step_swept_finite_differences(interval_inputs) -> None:
    _, inputs, params = interval_inputs
    _, grad = inputs["loglike_and_gradient"](params)
    locs = [
        ("meas_sds", 0, "c1", "-"),
        ("meas_sds", 2, "c1", "-"),
        ("loadings", 1, "c1", "skills"),
        ("controls", 2, "c1", "constant"),
        ("initial_states", 0, "mixture_0", "skills"),
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


def test_split_dataset_gradient_equals_unsplit_gradient(interval_inputs) -> None:
    data, inputs, params = interval_inputs
    split = get_maximization_inputs(
        model_spec=_model(),
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none", split_dataset=3),
    )
    np.testing.assert_allclose(
        split["loglike_and_gradient"](params)[1],
        inputs["loglike_and_gradient"](params)[1],
        rtol=1e-11,
        atol=1e-11,
    )


def test_without_interval_observations_the_model_equals_the_plain_tobit() -> None:
    data = _data().assign(c1_lo=np.nan, c1_hi=np.nan)
    options = CHSEstimationOptions(start_params_strategy="none")
    with_columns = get_maximization_inputs(
        model_spec=_model(), data=data, chs_options=options
    )
    plain = get_maximization_inputs(
        model_spec=_model(TobitMeasurement(lower=_LOWER, upper=_UPPER)),
        data=data,
        chs_options=options,
    )
    params = _params(plain["params_template"])
    np.testing.assert_array_equal(
        with_columns["loglikeobs"](params), plain["loglikeobs"](params)
    )


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("lo", "hi", "sigma", "offset"),
    [(0.0, 0.8, 0.6, 0.1), (-0.5, 0.4, 1.7, 2.5), (0.3, 0.301, 0.4, -0.2)],
)
def test_interval_update_with_free_error_sd_matches_prior_quadrature(
    dim, lo, hi, sigma, offset
) -> None:
    mean, upper, loading = _prior(dim)
    cov = upper.T @ upper
    states, chols, _, loglikes, _ = kalman_update_limited(
        states=jnp.asarray(mean)[None, None, :],
        upper_chols=jnp.asarray(upper)[None, None, :, :],
        loadings=jnp.asarray(loading),
        control_params=jnp.array([offset]),
        meas_sd=jnp.array(sigma),
        measurements=jnp.array([jnp.nan]),
        events=jnp.array([E.INTERVAL], dtype=jnp.int8),
        thresholds=jnp.array([[hi, lo]]),
        controls=jnp.ones((1, 1)),
        log_mixture_weights=jnp.zeros((1, 1)),
        has_intervals=True,
    )
    chol = np.asarray(chols[0, 0])
    actual = {
        "loglike": float(loglikes[0]),
        "mean": np.asarray(states[0, 0]),
        "cov": chol.T @ chol,
    }
    expected = _quadrature_update(
        mean=mean,
        covariance=cov,
        loading=loading,
        offset=offset,
        error_sd=sigma,
        kind="interval",
        threshold=lo,
        upper_threshold=hi,
    )
    _assert_update_close(actual, expected, cov)


def test_narrow_interval_gradient_matches_the_density_approximation() -> None:
    """For width `w = 1e-9`, `log Z = log(phi(beta) w / s)` up to `O(w^2)`."""
    mean, upper, loading = _prior(2)
    lo, hi = 0.3, 0.3 + 1e-9

    def loglike(args):
        sigma, offset, load = args
        return kalman_update_limited(
            states=jnp.asarray(mean)[None, None, :],
            upper_chols=jnp.asarray(upper)[None, None, :, :],
            loadings=load,
            control_params=jnp.array([offset]),
            meas_sd=sigma,
            measurements=jnp.array([jnp.nan]),
            events=jnp.array([E.INTERVAL], dtype=jnp.int8),
            thresholds=jnp.array([[hi, lo]]),
            controls=jnp.ones((1, 1)),
            log_mixture_weights=jnp.zeros((1, 1)),
            has_intervals=True,
        )[3][0]

    args = (jnp.array(0.6), jnp.array(-0.2), jnp.asarray(loading))
    grad = jax.jit(jax.grad(loglike))(args)
    s = math.sqrt(0.6**2 + loading @ upper.T @ upper @ loading)
    mu = -0.2 + loading @ mean
    beta = (hi - mu) / s
    # d/d offset of log(phi(beta) w / s) is beta / s.
    np.testing.assert_allclose(float(grad[1]), beta / s, rtol=1e-6)
    # d/d sigma: (beta^2 - 1) / s * ds/dsigma with ds/dsigma = sigma / s.
    np.testing.assert_allclose(float(grad[0]), (beta**2 - 1) / s * 0.6 / s, rtol=1e-6)
    assert np.isfinite(np.asarray(grad[2])).all()


def test_af_estimation_rejects_interval_columns() -> None:
    with pytest.raises(NotImplementedError, match="interval"):
        estimate_af(model_spec=_model(), data=_data(n_obs=6))


def test_simulation_draws_the_censored_value_of_every_observation() -> None:
    template = get_maximization_inputs(
        model_spec=_model(),
        data=_data(n_obs=6),
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["params_template"]
    observed = simulate_dataset(
        model_spec=_model(), params=_params(template), n_obs=200, seed=0
    )["aug_observed_data"]
    c1 = observed["c1"].to_numpy()
    assert np.isfinite(c1).all()
    assert c1.min() == _LOWER
    assert c1.max() == _UPPER


def test_endogenous_factor_periods_keep_the_interval_columns() -> None:
    base = _model()
    investment = FactorSpec(
        measurements=(("i1",),) * _N_PERIODS,
        normalizations=Normalizations(
            loadings=({"i1": 1},) * _N_PERIODS,
            intercepts=({"i1": 0},) * _N_PERIODS,
        ),
        transition_function="linear",
        is_endogenous=True,
    )
    model = base._replace(factors={**base.factors, "investment": investment})
    data = _data(n_obs=6).assign(i1=np.linspace(0.0, 1.0, 6 * _N_PERIODS))
    pm = process_model(model)
    out = process_data(
        df=data,
        has_endogenous_factors=True,
        labels=pm.labels,
        update_info=pm.update_info,
        anchoring_info=pm.anchoring,
        measurement_models=model.measurement_models,
    )
    row = list(pm.update_info.index).index((2, "c1"))
    np.testing.assert_array_equal(
        np.asarray(out["interval_bounds"])[row],
        data.xs(1, level="period")[list(_COLUMNS)].to_numpy(),
    )
