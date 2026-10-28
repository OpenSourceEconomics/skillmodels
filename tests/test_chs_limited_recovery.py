"""Recovery of models with limited measurements from simulated data.

`simulate_dataset` draws each measurement by its family (probit: Bernoulli of the
latent index, Tobit: the latent response clipped to its bounds, ordered probit: the
number of cutpoints below the latent index). CHS estimation
from the true parameters' neighbourhood must recover them up to sampling noise and
the small assumed-density approximation.

In the interval test, persons with a low anchor measure `g1` are more likely to
report the Tobit measure `t1` only by the class interval of its latent value and to
miss `g2`. Shared measurement parameters put both kinds of report on one scale.
Selection depends on observed data only, so the mean filtered skills of each group
equal its mean true skills up to sampling noise and the assumed-density approximation.
"""

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.individual_states import get_individual_states_from_params
from skillmodels.common.measurement_models import (
    OrderedProbitMeasurement,
    ProbitMeasurement,
    TobitMeasurement,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.simulate_data import simulate_dataset

_MODEL = ModelSpec(
    factors={
        "skills": FactorSpec(
            measurements=(("g1", "g2", "p1", "t1"),) * 2,
            normalizations=Normalizations(
                loadings=({"g1": 1},) * 2, intercepts=({"g1": 0},) * 2
            ),
            transition_function="linear",
        ),
    },
    measurement_models={
        "p1": ProbitMeasurement(),
        "t1": TobitMeasurement(lower=0.0),
    },
)

_TRUE = {
    ("loadings", "g2"): 0.8,
    ("loadings", "p1"): 1.2,
    ("loadings", "t1"): 0.7,
    ("controls", "g2"): 0.3,
    ("controls", "p1"): -0.2,
    ("controls", "t1"): 0.1,
    ("meas_sds", "g1"): 0.5,
    ("meas_sds", "g2"): 0.6,
    ("meas_sds", "p1"): 1.0,
    ("meas_sds", "t1"): 0.4,
}


_ORDERED_MODEL = ModelSpec(
    factors={
        "skills": FactorSpec(
            measurements=(("g1", "g2", "o1"),) * 2,
            normalizations=Normalizations(
                loadings=({"g1": 1},) * 2, intercepts=({"g1": 0},) * 2
            ),
            transition_function="linear",
        ),
    },
    measurement_models={"o1": OrderedProbitMeasurement(n_categories=4)},
)

_ORDERED_TRUE = {
    ("loadings", "g2"): 0.8,
    ("loadings", "o1"): 1.1,
    ("controls", "g2"): 0.3,
    ("controls", "o1"): 0.4,
    ("meas_sds", "g1"): 0.5,
    ("meas_sds", "g2"): 0.6,
    ("meas_sds", "o1"): 1.0,
}
_CUTPOINTS = {"cutpoint_1": 0.0, "cutpoint_2": 0.7, "cutpoint_3": 1.6}


def _true_params(template: pd.DataFrame, true=_TRUE) -> pd.DataFrame:
    params = template.copy()
    for category, aug_period, name1, name2 in params.index:
        loc = (category, aug_period, name1, name2)
        if (category, name1) in true:
            params.loc[loc, "value"] = true[(category, name1)]
        if category == "cutpoints":
            params.loc[loc, "value"] = _CUTPOINTS[name2]
    params.loc[("controls", slice(None), "g1", "constant"), "value"] = 0.0
    params.loc[("loadings", slice(None), "g1", "skills"), "value"] = 1.0
    params.loc[("shock_sds", 0, "skills", "-"), "value"] = 0.5
    params.loc[("transition", 0, "skills", "skills"), "value"] = 0.8
    params.loc[("transition", 0, "skills", "constant"), "value"] = 0.2
    params.loc[("initial_states", 0, "mixture_0", "skills"), "value"] = 0.0
    params.loc[("initial_cholcovs", 0, "mixture_0", "skills-skills"), "value"] = 1.0
    params.loc[("mixture_weights", 0, "mixture_0", "-"), "value"] = 1.0
    return params


@pytest.mark.long_running
def test_simulated_probit_and_tobit_parameters_are_recovered() -> None:
    estimate, truth, inputs = _estimate_from_simulated_data(_MODEL, _TRUE)
    assert inputs["loglike"](estimate) >= inputs["loglike"](truth)
    for (category, name1), value in _TRUE.items():
        got = estimate.query(f"category == '{category}' and name1 == '{name1}'")
        np.testing.assert_allclose(got["value"], value, atol=0.12)


@pytest.mark.long_running
def test_simulated_ordered_probit_parameters_are_recovered() -> None:
    estimate, truth, inputs = _estimate_from_simulated_data(
        _ORDERED_MODEL, _ORDERED_TRUE
    )
    assert inputs["loglike"](estimate) >= inputs["loglike"](truth)
    for (category, name1), value in _ORDERED_TRUE.items():
        got = estimate.query(f"category == '{category}' and name1 == '{name1}'")
        np.testing.assert_allclose(got["value"], value, atol=0.12)
    for name2, value in _CUTPOINTS.items():
        got = estimate.query(f"category == 'cutpoints' and name2 == '{name2}'")
        np.testing.assert_allclose(got["value"], value, atol=0.12)


_INTERVAL_MODEL = ModelSpec(
    factors={
        "skills": FactorSpec(
            measurements=(("g1", "g2", "t1"),) * 2,
            normalizations=Normalizations(
                loadings=({"g1": 1},) * 2, intercepts=({"g1": 0},) * 2
            ),
            transition_function="linear",
        ),
    },
    measurement_models={
        "t1": TobitMeasurement(
            lower=-1.0, upper=None, interval_columns=("t1_lo", "t1_hi")
        ),
    },
)
_INTERVAL_TRUE = {
    ("loadings", "g2"): 0.8,
    ("loadings", "t1"): 0.9,
    ("controls", "g2"): 0.3,
    ("controls", "t1"): 0.1,
    ("meas_sds", "g1"): 0.5,
    ("meas_sds", "g2"): 0.6,
    ("meas_sds", "t1"): 0.5,
}
_INTERVAL_CLASSES = ((-np.inf, -0.3), (-0.3, 0.6), (0.6, np.inf))


def _report_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Replace `t1` and `g2` by the class interval of `t1`, mostly for low `g1`."""
    rng = np.random.default_rng(5)
    proxy = rng.random(len(data)) < np.where(data["g1"].to_numpy() < 0.0, 0.7, 0.1)
    t1 = data["t1"].to_numpy()
    lows = np.select([t1 <= c[1] for c in _INTERVAL_CLASSES], [-np.inf, -0.3, 0.6])
    highs = np.select([t1 <= c[1] for c in _INTERVAL_CLASSES], [-0.3, 0.6, np.inf])
    return data.assign(
        t1=np.where(proxy, np.nan, t1),
        t1_lo=np.where(proxy, lows, np.nan),
        t1_hi=np.where(proxy, highs, np.nan),
        g2=np.where(proxy, np.nan, data["g2"]),
        proxy=proxy,
    )


@pytest.mark.long_running
def test_interval_reports_recover_parameters_and_the_skill_level() -> None:
    estimate, truth, inputs, data, skills = _estimate_from_simulated_data(
        _INTERVAL_MODEL, _INTERVAL_TRUE, interval_reports=True
    )
    assert inputs["loglike"](estimate) >= inputs["loglike"](truth)
    for (category, name1), value in _INTERVAL_TRUE.items():
        got = estimate.query(f"category == '{category}' and name1 == '{name1}'")
        np.testing.assert_allclose(got["value"], value, atol=0.12)
    filtered = get_individual_states_from_params(
        model_spec=_INTERVAL_MODEL,
        data=data.drop(columns="proxy"),
        params=estimate,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["unanchored_states"]["states"]
    proxy = data["proxy"].to_numpy()
    filtered_skills = filtered.sort_values(["id", "period"])["skills"].to_numpy()
    true_gap = skills[proxy].mean() - skills[~proxy].mean()
    filtered_gap = filtered_skills[proxy].mean() - filtered_skills[~proxy].mean()
    assert true_gap < -0.5
    np.testing.assert_allclose(filtered_gap, true_gap, atol=0.05)


def _estimate_from_simulated_data(model, true, *, interval_reports=False):
    template = get_maximization_inputs(
        model_spec=model,
        data=_simulated_data(model, n_obs=20, params=None),
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["params_template"]
    truth = _true_params(template, true)
    data = _simulated_data(model, n_obs=3000, params=truth)
    if interval_reports:
        skills = (
            simulate_dataset(model_spec=model, params=truth, n_obs=3000, seed=42)[
                "unanchored_states"
            ]["states"]
            .sort_values(["id", "period"])["skills"]
            .to_numpy()
        )
        data = _report_intervals(data.sort_index())
    inputs = get_maximization_inputs(
        model_spec=model,
        data=data.drop(columns="proxy", errors="ignore"),
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    rng = np.random.default_rng(1)
    fixed = {
        c.loc
        for c in inputs["constraints"]
        if isinstance(getattr(c, "loc", None), tuple)
    }
    # The weight of the single mixture component is 1 by its probability
    # constraint; a perturbed start would scale the reported filtered states.
    free = np.array(
        [loc not in fixed and loc[0] != "mixture_weights" for loc in truth.index]
    )
    start = truth.copy()
    start.loc[free, "value"] += rng.uniform(-0.15, 0.15, size=int(free.sum()))
    start["value"] = np.maximum(start["value"], start["lower_bound"] + 0.05)
    res = om.maximize(
        fun=inputs["loglike"],
        fun_and_jac=inputs["loglike_and_gradient"],
        params=start,
        algorithm="scipy_lbfgsb",
        constraints=inputs["constraints"],
        bounds=om.Bounds(lower=start["lower_bound"], upper=start["upper_bound"]),
    )
    if interval_reports:
        return res.params, truth, inputs, data, skills
    return res.params, truth, inputs


def _simulated_data(
    model: ModelSpec, n_obs: int, params: pd.DataFrame | None
) -> pd.DataFrame:
    """Return simulated measurements, or a placeholder panel without `params`."""
    if params is None:
        rng = np.random.default_rng(0)
        placeholders = {
            "g1": rng.normal,
            "g2": rng.normal,
            "p1": lambda: float(rng.random() < 0.5),
            "t1": lambda: max(rng.normal(), 0.0),
            "t1_lo": lambda: np.nan,
            "t1_hi": lambda: np.nan,
            "o1": lambda: float(rng.integers(4)),
        }
        names = (*model.factors["skills"].measurements[0], "t1_lo", "t1_hi")
        rows = [
            {"caseid": i, "period": t} | {n: placeholders[n]() for n in names}
            for i in range(n_obs)
            for t in (0, 1)
        ]
        return pd.DataFrame(rows).set_index(["caseid", "period"])
    observed = simulate_dataset(model_spec=model, params=params, n_obs=n_obs, seed=42)[
        "aug_observed_data"
    ]
    return observed.rename(columns={"id": "caseid", "aug_period": "period"}).set_index(
        ["caseid", "period"]
    )
