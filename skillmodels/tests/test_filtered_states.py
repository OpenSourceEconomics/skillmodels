from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from skillmodels.filtered_states import get_filtered_states
from skillmodels.likelihood_function import get_maximization_inputs


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


def test_get_filtered_states(model2, model2_data):
    params = pd.read_csv(TEST_DIR / "regression_vault" / f"one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    max_inputs = get_maximization_inputs(model2, model2_data)
    params = params.loc[max_inputs["params_template"].index]

    calculated = get_filtered_states(model_dict=model2, data=model2_data, params=params)

    factors = ["fac1", "fac2", "fac3"]
    expected_ratios = [1.187757, 1, 1]
    for factor, expected_ratio in zip(factors, expected_ratios):
        anch_ranges = calculated["anchored_states"]["state_ranges"][factor]
        unanch_ranges = calculated["unanchored_states"]["state_ranges"][factor]
        ratio = (anch_ranges / unanch_ranges).to_numpy()
        assert np.allclose(ratio, expected_ratio)
