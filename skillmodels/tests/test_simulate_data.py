"""Tests for functions in simulate_data module."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.simulate_data import measurements_from_states
from skillmodels.simulate_data import simulate_dataset


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


def test_simulate_dataset(model2, model2_data):
    model_dict = model2
    params = pd.read_csv(TEST_DIR / "regression_vault" / f"one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    calculated = simulate_dataset(
        model_dict=model_dict, params=params, data=model2_data
    )

    factors = ["fac1", "fac2", "fac3"]
    expected_ratios = [1.187757, 1, 1]
    for factor, expected_ratio in zip(factors, expected_ratios):
        anch_ranges = calculated["anchored_states"]["state_ranges"][factor]
        unanch_ranges = calculated["unanchored_states"]["state_ranges"][factor]
        ratio = (anch_ranges / unanch_ranges).to_numpy()
        assert np.allclose(ratio, expected_ratio)


def test_measurements_from_factors():
    inputs = {
        "states": np.array([[0, 0, 0], [1, 1, 1]]),
        "controls": np.array([[1, 1], [1, 1]]),
        "loadings": np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]),
        "control_params": np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
        "sds": np.zeros(3),
    }
    expected = np.array([[1, 1, 1], [1.9, 1.9, 1.9]])
    aaae(measurements_from_states(**inputs), expected)
