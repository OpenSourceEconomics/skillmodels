import json
from itertools import product
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import model_spec_from_yaml_dict
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.config import TEST_DATA_DIR
from skillmodels.decorators import register_params
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.model_spec import ModelSpec
from skillmodels.utilities import reduce_n_periods

jax.config.update("jax_enable_x64", True)

MODEL_NAMES = [
    "no_stages_anchoring",
    "one_stage",
    "one_stage_anchoring",
    "two_stages_anchoring",
    "one_stage_anchoring_custom_functions",
]

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2():
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        return yaml.load(y, Loader=yaml.SafeLoader)


@pytest.fixture
def model2_data():
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


def _convert_model(base_model, model_name):
    model = base_model.copy()
    if model_name == "no_stages_anchoring":
        model.pop("stagemap")
    elif model_name == "one_stage":
        model.pop("anchoring")
    elif model_name == "one_stage_anchoring":
        pass
    elif model_name == "two_stages_anchoring":
        model["stagemap"] = [0, 0, 0, 0, 1, 1, 1]
    elif model_name == "one_stage_anchoring_custom_functions":

        @register_params(params=[])
        def constant(fac3, params):
            return fac3

        @register_params(params=["fac1", "fac2", "fac3", "constant"])
        def linear(fac1, fac2, fac3, params):
            p = params
            return (
                p["constant"] + fac1 * p["fac1"] + fac2 * p["fac2"] + fac3 * p["fac3"]
            )

        model["factors"]["fac2"]["transition_function"] = linear
        model["factors"]["fac3"]["transition_function"] = constant
    else:
        raise ValueError("Invalid model name.")
    return model_spec_from_yaml_dict(model)


@pytest.mark.parametrize(
    ("model_name", "fun_key"), product(MODEL_NAMES, ["loglike", "debug_loglike"])
)
def test_likelihood_values_have_not_changed(
    model2, model2_data, model_name, fun_key
) -> None:
    regvault = REGRESSION_VAULT
    model = _convert_model(model2, model_name)
    params = pd.read_csv(regvault / f"{model_name}.csv").set_index(
        ["category", "period", "name1", "name2"],
    )

    inputs = get_maximization_inputs(model, model2_data)

    params = params.loc[inputs["params_template"].index]

    fun = inputs[fun_key]
    new_loglike = fun(params)["value"] if "debug" in fun_key else fun(params)

    with (regvault / f"{model_name}_result.json").open() as j:
        old_loglike = np.array(json.load(j)).sum()
    aaae(new_loglike, old_loglike)


def test_splitting_does_not_change_gradient(model2, model2_data) -> None:
    model = model_spec_from_yaml_dict(model2)
    inputs = get_maximization_inputs(model, model2_data)
    inputs_split = get_maximization_inputs(model, model2_data, 13)

    params = inputs["params_template"]
    params["value"] = 0.1

    _, gradient = inputs["loglike_and_gradient"](params)
    _, gradient_split = inputs_split["loglike_and_gradient"](params)

    aaae(gradient, gradient_split)


@pytest.mark.parametrize(
    ("model_name", "fun_key"), product(MODEL_NAMES, ["loglikeobs"])
)
def test_likelihood_contributions_have_not_changed(
    model2, model2_data, model_name, fun_key
) -> None:
    regvault = REGRESSION_VAULT
    model = _convert_model(model2, model_name)
    params = pd.read_csv(regvault / f"{model_name}.csv").set_index(
        ["category", "period", "name1", "name2"],
    )

    inputs = get_maximization_inputs(model, model2_data)

    params = params.loc[inputs["params_template"].index]

    fun = inputs[fun_key]
    new_loglikes = fun(params)["contributions"] if "debug" in fun_key else fun(params)

    with (regvault / f"{model_name}_result.json").open() as j:
        old_loglikes = np.array(json.load(j))
    aaae(new_loglikes, old_loglikes)


@pytest.mark.parametrize(
    ("model_type", "fun_key"),
    product(["no_stages_anchoring", "with_missings"], ["loglike_and_gradient"]),
)
def test_likelihood_contributions_large_nobs(
    model2, model2_data, model_type, fun_key
) -> None:
    rng = np.random.default_rng(42)
    regvault = REGRESSION_VAULT
    model = _convert_model(model2, "no_stages_anchoring")
    params = pd.read_csv(regvault / "no_stages_anchoring.csv").set_index(
        ["category", "period", "name1", "name2"],
    )

    to_concat = [model2_data]
    idx_names = model2_data.index.names
    n_repetitions = 5
    n_ids = model2_data.index.get_level_values("caseid").max()
    for i in range(1, 1 + n_repetitions):
        increment = i * n_ids
        this_round = model2_data.copy().reset_index()
        for col in ("caseid", "id"):
            this_round[col] += increment
        this_round = this_round.set_index(idx_names)
        cols = [
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
            "y6",
            "y7",
            "y8",
            "y9",
            "Q1",
            "dy7",
            "dy8",
            "dy9",
            "x1",
        ]
        if model_type == "no_stages_anchoring":
            for col in cols:
                this_round[col] += rng.normal(0, 0.1, (len(model2_data),))
        elif model_type == "with_missings":
            fraction_to_set_missing = 0.9
            n_rows = len(this_round)
            n_missing = int(n_rows * fraction_to_set_missing)
            rows_to_set_missing = this_round.sample(n=n_missing, random_state=rng).index
            this_round.loc[rows_to_set_missing, cols] = np.nan
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        to_concat.append(this_round)

    stacked_data = pd.concat(to_concat)

    inputs = get_maximization_inputs(model, stacked_data)

    params = params.loc[inputs["params_template"].index]

    loglike = inputs[fun_key](params)

    assert np.isfinite(loglike[0])
    assert np.isfinite(loglike[1]).all()


def test_likelihood_runs_with_empty_periods(model2, model2_data) -> None:
    del model2["anchoring"]
    for factor in ["fac1", "fac2"]:
        model2["factors"][factor]["measurements"][-1] = []
        model2["factors"][factor]["normalizations"]["loadings"][-1] = {}

    func_dict = get_maximization_inputs(model_spec_from_yaml_dict(model2), model2_data)

    params = func_dict["params_template"]
    params["value"] = 0.1

    debug_loglike = func_dict["debug_loglike"]
    debug_loglike(params)


def test_likelihood_runs_with_too_long_data(model2, model2_data) -> None:
    full_model = model_spec_from_yaml_dict(model2)
    reduced = reduce_n_periods(full_model, 2)
    assert isinstance(reduced, ModelSpec)
    model = reduced
    func_dict = get_maximization_inputs(model, model2_data)

    params = func_dict["params_template"]
    params["value"] = 0.1

    debug_loglike = func_dict["debug_loglike"]
    debug_loglike(params)


def test_likelihood_runs_with_observed_factors(model2, model2_data) -> None:
    model2["observed_factors"] = ["ob1", "ob2"]
    model2_data["ob1"] = np.arange(len(model2_data))
    model2_data["ob2"] = np.ones(len(model2_data))
    func_dict = get_maximization_inputs(model_spec_from_yaml_dict(model2), model2_data)

    params = func_dict["params_template"]
    params["value"] = 0.1

    debug_loglike = func_dict["debug_loglike"]
    debug_loglike(params)
