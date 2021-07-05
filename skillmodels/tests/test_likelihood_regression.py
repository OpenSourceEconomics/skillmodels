import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from jax import config
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.likelihood_function import get_maximization_inputs

config.update("jax_enable_x64", True)

model_names = [
    "no_stages_anchoring",
    "one_stage",
    "one_stage_anchoring",
    "two_stages_anchoring",
]

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
    else:
        raise ValueError("Invalid model name.")
    return model


@pytest.mark.parametrize("model_name", model_names)
def test_likelihood_contributions_have_not_changed(model2, model2_data, model_name):
    regvault = TEST_DIR / "regression_vault"
    model = _convert_model(model2, model_name)
    params = pd.read_csv(regvault / f"{model_name}.csv")

    func_dict = get_maximization_inputs(model, model2_data)
    debug_loglike = func_dict["debug_loglike"]

    new_loglikes = debug_loglike(params)["contributions"]

    with open(regvault / f"{model_name}_result.json") as j:
        old_loglikes = np.array(json.load(j))

    aaae(new_loglikes, old_loglikes)
