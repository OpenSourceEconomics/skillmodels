"""Tests for functions in simulate_data module
"""
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal as adfeq
import pytest
from unittest.mock import patch
import sys

sys.path.append("../../")

import simulation.simulate_data as sd

# test measuerments_from_factors
@pytest.fixture
def set_up_meas():
    out = {}
    out["factors"] = np.array([[0, 0, 0], [1, 1, 1]])
    out["controls"] = np.array([[1, 1], [1, 1]])
    out["loadings"] = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    out["deltas"] = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    out["variances"] = np.zeros(3)
    out["nmeas"] = 3
    return out


@pytest.fixture
def expected_meas():
    out = np.array([[1, 1, 1], [1.9, 1.9, 1.9]])
    return out


def test_measurements_from_factors(set_up_meas, expected_meas):
    aaae(sd.measurements_from_factors(**set_up_meas), expected_meas)


# Test next_period_factors


@pytest.fixture
def set_up_npfac():
    out = {}
    out["factors"] = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1, 1]])
    out["transition_names"] = [
        "translog",
        "linear",
        "log_ces",
        "ar1",
        "constant",
        "linear_with_constant",
    ]
    out["transition_argument_dicts"] = [
        {"coeffs": np.array([0.02] * 28), "included_positions": [0, 1, 2, 3, 5]},
        {
            "coeffs": np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.2]),
            "included_positions": [0, 1, 2, 3, 4, 5],
        },
        {
            "coeffs": np.array([0.1, 0.1, 0.4, 0.4, 0.5, 0.5, 0.6]),
            "included_positions": [0, 1, 2, 3, 4, 5],
        },
        {"coeffs": np.array([2]), "included_positions": [1]},
        {"coeffs": np.array([1]), "included_positions": [1]},
        {
            "coeffs": np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.05]),
            "included_positions": [0, 1, 2, 3, 4, 5],
        },
    ]
    out["shock_variances"] = np.array([0, 0, 0, 0, 0, 0])
    return out


@pytest.fixture
def expected_npfac():
    # The values have been computed by using the functions from the module transition_functions.py
    # since the main calculations are made through those functions, what the test actualy tests
    # is whether the loops of getatr in sumlated_next_period_factors work correctly
    d = {}
    d["tl"] = np.array([[0.145, 0.42]])
    d["lin"] = np.array([[0.7, 1.4]])
    d["lces"] = np.array([[1.6552453, 2.1552453]])
    d["ar1"] = np.array([[1, 2]])
    d["constant"] = np.array([[0.5, 1]])
    d["linwc"] = np.array([[0.75, 1.45]])
    npfac = np.concatenate(list(d.values())).T
    return npfac


def test_next_period_factors(set_up_npfac, expected_npfac):
    aaae(sd.next_period_factors(**set_up_npfac), expected_npfac)


@pytest.fixture
def set_up_generate_datasets():
    out = {}
    out["factor_names"] = ["f1", "f2"]
    out["control_names"] = ["c1", "c2"]
    out["meas_names"] = ["m1", "m2"]
    out["nobs"] = 1
    out["nper"] = 2
    out["means"] = np.array([0, 0, 0.5, 0.5])
    out["covs"] = np.zeros((4, 4))
    out["weights"] = 1
    out["transition_names"] = ["linear", "linear_with_constant"]
    out["transition_argument_dicts"] = [
        {"coeffs": np.array([0.2, 0.2]), "included_positions": [0, 1]},
        {"coeffs": np.array([0.2, 0.2, 0.3]), "included_positions": [0, 1]},
    ]
    out["shock_variances"] = np.zeros(2)
    out["loadings"] = np.array([[0.5, 0.5], [0.5, 0.5]])
    out["deltas"] = np.array([[0.5, 0.5], [0.5, 0.5]])
    out["meas_variances"] = np.zeros(2)
    out["controls_mock"] = np.array([[0.5, 0.5]])
    out["start_states_mock"] = np.array([[0, 0]])
    return out


@pytest.fixture
def expected_dataset():
    out = {}
    id_obs = np.array([0, 0])
    controls = pd.DataFrame(
        data=np.array([[0.5, 0.5], [0.5, 0.5]]), columns=["c1", "c2"], index=id_obs
    )  # constant over time
    states_p0 = np.array([[0, 0]])
    states_p1 = np.array([[0, 0.3]])
    meas_p0 = np.array([[0.5, 0.5]])
    meas_p1 = np.array([[0.65, 0.65]])
    periods = np.array([[0, 1]]).T
    meas = pd.DataFrame(
        data=np.concatenate((periods, np.concatenate((meas_p0, meas_p1))), axis=1),
        columns=["time_period", "m1", "m2"],
        index=id_obs,
    )

    factors = pd.DataFrame(
        data=np.concatenate((periods, np.concatenate((states_p0, states_p1))), axis=1),
        columns=["time_period", "f1", "f2"],
        index=id_obs,
    )
    out["observed_data"] = pd.concat([meas, controls], axis=1)
    out["latent_data"] = factors
    return out


def test_simulate_latent_data(set_up_generate_datasets, expected_dataset):
    d = set_up_generate_datasets
    mock_npfac = patch("simulate_data.generate_start_factors_and_control_variables")
    mock_npfac.return_value = (d["start_states_mock"], d["controls_mock"])
    results = sd.simulate_datasets(
        d["factor_names"],
        d["control_names"],
        d["meas_names"],
        d["nobs"],
        d["nper"],
        d["means"],
        d["covs"],
        d["weights"],
        d["transition_names"],
        d["transition_argument_dicts"],
        d["shock_variances"],
        d["loadings"],
        d["deltas"],
        d["meas_variances"],
    )
    adfeq(results[1], expected_dataset["latent_data"], check_dtype=False)


# @patch("simulate_data.generate_start_factors_and_control_variables")
def test_simulate_observed_data(set_up_generate_datasets, expected_dataset):
    d = set_up_generate_datasets
    # mock_npfac = patch("simulate_data.generate_start_factors_and_control_variables")
    # mock_npfac.return_value = (d["start_states_mock"], d["controls_mock"])
    results = sd.simulate_datasets(
        d["factor_names"],
        d["control_names"],
        d["meas_names"],
        d["nobs"],
        d["nper"],
        d["means"],
        d["covs"],
        d["weights"],
        d["transition_names"],
        d["transition_argument_dicts"],
        d["shock_variances"],
        d["loadings"],
        d["deltas"],
        d["meas_variances"],
    )
    adfeq(results[0], expected_dataset["observed_data"], check_dtype=False)


# def test_simulate_datasets_single_obs(set_up_generate_datasets, expected_dataset):
