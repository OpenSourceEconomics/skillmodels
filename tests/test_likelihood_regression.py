import json
from itertools import product
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.decorators import register_params
from skillmodels.likelihood_function import get_maximization_inputs
from skillmodels.utilities import reduce_n_periods

jax.config.update("jax_enable_x64", True)

MODEL_NAMES = [
    "no_stages_anchoring",
    "one_stage",
    "one_stage_anchoring",
    "two_stages_anchoring",
    "one_stage_anchoring_custom_functions",
]
# keys in dict returned by get_maximization_inputs
LIKELIHOODS_VALUES = ["loglike", "debug_loglike"]
LIKELIHOODS_CONTRIBUTIONS = ["loglikeobs"]

# importing the TEST_DIR from config does not work for test run in conda build
TEST_DIR = Path(__file__).parent.resolve()


@pytest.fixture
def model2():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    return model_dict


@pytest.fixture
def model2_data():
    data = pd.read_stata(TEST_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])
    return data


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
            out = p["constant"] + fac1 * p["fac1"] + fac2 * p["fac2"] + fac3 * p["fac3"]
            return out

        model["factors"]["fac2"]["transition_function"] = linear
        model["factors"]["fac3"]["transition_function"] = constant
    else:
        raise ValueError("Invalid model name.")
    return model


@pytest.mark.parametrize(
    ("model_name", "fun_key"), product(MODEL_NAMES, LIKELIHOODS_CONTRIBUTIONS)
)
def test_likelihood_contributions_have_not_changed(
    model2, model2_data, model_name, fun_key
):
    regvault = TEST_DIR / "regression_vault"
    model = _convert_model(model2, model_name)
    params = pd.read_csv(regvault / f"{model_name}.csv").set_index(
        ["category", "period", "name1", "name2"],
    )

    inputs = get_maximization_inputs(model, model2_data)

    params = params.loc[inputs["params_template"].index]

    fun = inputs[fun_key]
    new_loglikes = fun(params)["contributions"] if "debug" in fun_key else fun(params)

    with open(regvault / f"{model_name}_result.json") as j:
        old_loglikes = np.array(json.load(j))
    aaae(new_loglikes, old_loglikes)


def test_likelihood_runs_with_empty_periods(model2, model2_data):
    del model2["anchoring"]
    for factor in ["fac1", "fac2"]:
        model2["factors"][factor]["measurements"][-1] = []
        model2["factors"][factor]["normalizations"]["loadings"][-1] = {}

    func_dict = get_maximization_inputs(model2, model2_data)

    params = func_dict["params_template"]
    params["value"] = 0.1

    debug_loglike = func_dict["debug_loglike"]
    debug_loglike(params)


def test_likelihood_runs_with_too_long_data(model2, model2_data):
    model = reduce_n_periods(model2, 2)
    func_dict = get_maximization_inputs(model, model2_data)

    params = func_dict["params_template"]
    params["value"] = 0.1

    debug_loglike = func_dict["debug_loglike"]
    debug_loglike(params)


def test_likelihood_runs_with_observed_factors(model2, model2_data):
    model2["observed_factors"] = ["ob1", "ob2"]
    model2_data["ob1"] = np.arange(len(model2_data))
    model2_data["ob2"] = np.ones(len(model2_data))
    func_dict = get_maximization_inputs(model2, model2_data)

    params = func_dict["params_template"]
    params["value"] = 0.1

    debug_loglike = func_dict["debug_loglike"]
    debug_loglike(params)
