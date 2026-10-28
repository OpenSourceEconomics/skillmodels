"""Recovery of models with limited measurements from simulated data.

`simulate_dataset` draws each measurement by its family (probit: Bernoulli of the
latent index, Tobit: the latent response clipped to its bounds, ordered probit: the
number of cutpoints below the latent index). CHS estimation
from the true parameters' neighbourhood must recover them up to sampling noise and
the small assumed-density approximation.
"""

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
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


def _estimate_from_simulated_data(model, true):
    template = get_maximization_inputs(
        model_spec=model,
        data=_simulated_data(model, n_obs=20, params=None),
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["params_template"]
    truth = _true_params(template, true)
    data = _simulated_data(model, n_obs=3000, params=truth)
    inputs = get_maximization_inputs(
        model_spec=model,
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    rng = np.random.default_rng(1)
    fixed = {
        c.loc
        for c in inputs["constraints"]
        if isinstance(getattr(c, "loc", None), tuple)
    }
    free = np.array([loc not in fixed for loc in truth.index])
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
            "o1": lambda: float(rng.integers(4)),
        }
        names = model.factors["skills"].measurements[0]
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
