import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from jax import config
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.likelihood_function_jax import get_log_likelihood_contributions_func

config.update("jax_enable_x64", True)


model_names = [
    "no_stages_anchoring",
    "one_stage",
    "one_stage_anchoring",
    "two_stages_anchoring",
]


@pytest.fixture
def model2():
    test_dir = Path(__file__).parent.resolve()
    with open(test_dir / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    return model_dict


@pytest.fixture
def model2_data():
    test_dir = Path(__file__).parent.resolve()
    data = pd.read_stata(test_dir / "model2_simulated_data.dta")
    data["period"] = data["period"].astype(int)
    data["id"] = data["id"].astype(int)
    data.loc[data["period"] != 7, "Q1"] = np.nan
    data.set_index(["id", "period"], inplace=True)
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
    regvault = Path(__file__).parent.resolve() / "regression_vault"
    model = _convert_model(model2, model_name)
    params = pd.read_csv(regvault / f"{model_name}.csv")
    contribs_func = get_log_likelihood_contributions_func(model, model2_data)

    params_vec = jnp.array(params["value"].to_numpy())
    contribs = contribs_func(params_vec)
    new_loglikes = np.array(contribs.sum(axis=0))

    with open(regvault / f"{model_name}_result.json") as j:
        old_loglikes = np.array(json.load(j))

    aaae(new_loglikes, old_loglikes)
