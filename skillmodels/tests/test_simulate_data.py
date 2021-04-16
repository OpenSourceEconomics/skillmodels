"""Tests for functions in simulate_data module."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from skillmodels.simulate_data import _simulate_datasets
from skillmodels.simulate_data import measurements_from_states
from skillmodels.simulate_data import next_period_states

# ===============================
# test measuerments_from_factors
# ===============================
@pytest.fixture
def set_up_meas():
    out = {}
    out["states"] = np.array([[0, 0, 0], [1, 1, 1]])
    out["controls"] = np.array([[1, 1], [1, 1]])
    out["loadings"] = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    out["control_params"] = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    out["sds"] = np.zeros(3)
    return out


@pytest.fixture
def expected_meas():
    out = np.array([[1, 1, 1], [1.9, 1.9, 1.9]])
    return out


def test_measurements_from_factors(set_up_meas, expected_meas):
    aaae(measurements_from_states(**set_up_meas), expected_meas)


# =========================
# Test next_period_factors
# =========================


@pytest.fixture
def set_up_npfac():
    out = {}
    out["states"] = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1, 1]])
    out["transition_names"] = [
        "translog",
        "linear",
        "log_ces",
        "linear",
        "log_ces",
        "linear",
    ]
    out["transition_params"] = [
        np.array([0.02] * 28),
        np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.0]),
        np.array([0.1, 0.1, 0.4, 0.4, 0.5, 0.5, 0.6]),
        np.array([0.0, 2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.01, 1, 0.01, 0.01, 0.01, 0.01, 1]),
        np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.05]),
    ]
    out["shock_sds"] = np.array([0, 0, 0, 0, 0, 0])
    return out


@pytest.fixture
def expected_npfac():
    # The values have been computed by using the functions from the module transition_
    # functions.py since the main calculations are made through those functions,
    # what the test actually tests is whether the loops of getatr in #
    # simlated_next_period_factors work correctly
    d = {}
    d["tl"] = np.array([[0.18499996, 0.56000006]])
    d["lin"] = np.array([[0.7, 1.4]])
    d["lces"] = np.array([[1.6552453, 2.1552453]])
    d["ar1"] = np.array([[1, 2]])
    d["lces_"] = np.array([[0.54879016, 1.04879016]])
    d["linwc"] = np.array([[0.75, 1.45]])
    npfac = np.concatenate(list(d.values())).T
    return npfac


def test_next_period_factors(set_up_npfac, expected_npfac):
    aaae(next_period_states(**set_up_npfac), expected_npfac)


# ===============================
# test generate datasets, nmixtures=1
# ===============================


@pytest.fixture
def set_up_generate_datasets():
    out = {}
    out["states"] = np.array([0, 0] * 5).reshape(5, 1, 2)
    out["covs"] = np.zeros((1, 1, 2, 2))
    out["log_weights"] = 0
    pardict = {}
    pardict["loadings"] = np.array([[0.5, 0.4], [0.2, 0.7]] * 3)
    pardict["controls"] = np.array([[0, 0.5, 0.3], [0, 0.5, 0.6]] * 3)
    pardict["meas_sds"] = np.zeros(2 * 3)
    pardict["transition"] = [[np.array([0.2, 0.2, 0.0]), np.array([0.2, 0.2, 0.3])]] * 2
    pardict["shock_sds"] = [np.zeros(2)] * 2
    out["pardict"] = pardict
    labels = {}
    labels["factors"] = ["f1", "f2"]
    labels["controls"] = ["constant", "c1", "c2"]
    labels["transition_names"] = ["linear", "linear"]
    out["labels"] = labels
    out["dimensions"] = {
        "n_states": 2,
        "n_controls": 3,
        "n_periods": 3,
        "n_mixtures": 1,
    }
    out["n_obs"] = 5
    update_info = pd.DataFrame(np.empty(2 * 3))
    update_info.index = pd.MultiIndex.from_tuples(
        [(0, "m1"), (0, "m2"), (1, "m1"), (1, "m2"), (2, "m1"), (2, "m2")]
    )
    out["update_info"] = update_info
    out["control_means"] = np.array([0.5, 0.5, 0.5])
    out["control_sds"] = np.zeros(3)
    out["policies"] = [
        {"period": 0, "factor": "f1", "effect_size": 0.2, "standard_deviation": 0.0},
        {"period": 1, "factor": "f2", "effect_size": 0.1, "standard_deviation": 0.0},
    ]
    return out


@pytest.fixture
def expected_dataset():

    out = {}
    id_obs = np.array([0, 1, 2, 3, 4] * 3)
    controls = pd.DataFrame(
        data=np.array([[0.5, 0.5]] * 15), columns=["c1", "c2"], index=id_obs
    )  # constant over time
    controls["constant"] = 1
    states_p0 = np.array([[0.2, 0]] * 5)  # setup[means][0:2]
    states_p1 = np.array(
        [[0.04, 0.44]] * 5
    )  # transition_name(states_p0), called manually
    states_p2 = np.array(
        [[0.096, 0.396]] * 5
    )  # transition_name(states_p1), called manually
    meas_p0 = np.array(
        [[0.5, 0.59]] * 5
    )  # meas_from_factor(factors_p0,controls), called manually
    meas_p1 = np.array(
        [[0.596, 0.866]] * 5
    )  # meas_from_factor(factors_p1,controls), called manually
    meas_p2 = np.array(
        [[0.6064, 0.8464]] * 5
    )  # meas_from_factor(factors_p2,controls), called manually
    periods = np.repeat(np.arange(3), 5).reshape(15, 1)
    meas = pd.DataFrame(
        data=np.concatenate(
            (periods, np.concatenate((meas_p0, meas_p1, meas_p2))), axis=1
        ),
        columns=["period", "m1", "m2"],
        index=id_obs,
    )

    latent_data = pd.DataFrame(
        data=np.concatenate(
            (periods, np.concatenate((states_p0, states_p1, states_p2))), axis=1
        ),
        columns=["period", "f1", "f2"],
        index=id_obs,
    )

    observed_data = pd.concat([controls, meas], axis=1)

    for df in [observed_data, latent_data]:
        df["period"] = df["period"].astype(int)
        df["id"] = df.index
        df.sort_values(["id", "period"], inplace=True)
        df.set_index(["id", "period"], inplace=True)
    out["observed_data"] = observed_data
    out["latent_data"] = latent_data
    return out


def test_simulate_latent_data(set_up_generate_datasets, expected_dataset):
    latent_data = _simulate_datasets(**set_up_generate_datasets)[1]
    assert_frame_equal(latent_data, expected_dataset["latent_data"], check_dtype=False)


def test_simulate_observed_data(set_up_generate_datasets, expected_dataset):
    obs_data = _simulate_datasets(**set_up_generate_datasets)[0]
    assert_frame_equal(obs_data, expected_dataset["observed_data"], check_dtype=False)


# =================
# test with nmixtures=2
# =================


@pytest.fixture
def set_up_generate_datasets_2_mix():
    out = {}
    out["states"] = np.array([0, 0] * 10).reshape(5, 2, 2)
    out["covs"] = np.zeros((1, 2, 2, 2))
    out["log_weights"] = np.log(np.array([0.5, 0.5]))
    pardict = {}
    pardict["loadings"] = np.array([[0.5, 0.5], [0.6, 0.6]] * 3)
    pardict["controls"] = np.array([[0, 0.5, 0.5], [0, 0.6, 0.6]] * 3)
    pardict["meas_sds"] = np.zeros(2 * 3)
    pardict["transition"] = [[np.array([0.2, 0.2, 0.0]), np.array([0.2, 0.2, 0.3])]] * 2
    pardict["shock_sds"] = [np.zeros(2)] * 2
    out["pardict"] = pardict
    labels = {}
    labels["factors"] = ["f1", "f2"]
    labels["controls"] = ["constant", "c1", "c2"]
    labels["transition_names"] = ["linear", "linear"]
    out["labels"] = labels
    out["dimensions"] = {
        "n_states": 2,
        "n_controls": 3,
        "n_periods": 3,
        "n_mixtures": 2,
    }
    out["n_obs"] = 5
    update_info = pd.DataFrame(np.empty(2 * 3))
    update_info.index = pd.MultiIndex.from_tuples(
        [(0, "m1"), (0, "m2"), (1, "m1"), (1, "m2"), (2, "m1"), (2, "m2")]
    )
    out["update_info"] = update_info
    out["control_means"] = np.array([0.5, 0.5, 0.5])
    out["control_sds"] = np.zeros(3)
    out["policies"] = None

    return out


@pytest.fixture
def expected_dataset_2_mix():
    out = {}
    id_obs = np.array([0, 1, 2, 3, 4] * 3)
    controls = pd.DataFrame(
        data=np.array([[0.5, 0.5]] * 15), columns=["c1", "c2"], index=id_obs
    )  # constant over time
    controls["constant"] = 1
    states_p0 = np.array([[0, 0]] * 5)
    states_p1 = np.array([[0, 0.3]] * 5)
    states_p2 = np.array([[0.06, 0.36]] * 5)
    meas_p0 = np.array([[0.5, 0.6]] * 5)
    meas_p1 = np.array([[0.65, 0.78]] * 5)
    meas_p2 = np.array([[0.71, 0.852]] * 5)
    periods = np.repeat(np.arange(3), 5).reshape(15, 1)
    meas = pd.DataFrame(
        data=np.concatenate(
            (periods, np.concatenate((meas_p0, meas_p1, meas_p2))), axis=1
        ),
        columns=["period", "m1", "m2"],
        index=id_obs,
    )

    latent_data = pd.DataFrame(
        data=np.concatenate(
            (periods, np.concatenate((states_p0, states_p1, states_p2))), axis=1
        ),
        columns=["period", "f1", "f2"],
        index=id_obs,
    )
    observed_data = pd.concat([controls, meas], axis=1)
    for df in [observed_data, latent_data]:
        df["period"] = df["period"].astype(int)
        df["id"] = df.index
        df.sort_values(["id", "period"], inplace=True)
        df.set_index(["id", "period"], inplace=True)
    out["observed_data"] = observed_data
    out["latent_data"] = latent_data
    return out


def test_simulate_latent_data_2_mix(
    set_up_generate_datasets_2_mix, expected_dataset_2_mix
):
    latent_data = _simulate_datasets(**set_up_generate_datasets_2_mix)[1]
    assert_frame_equal(
        latent_data, expected_dataset_2_mix["latent_data"], check_dtype=False
    )


def test_simulate_observed_data_2_mix(
    set_up_generate_datasets_2_mix, expected_dataset_2_mix
):
    obs_data = _simulate_datasets(**set_up_generate_datasets_2_mix)[0]
    assert_frame_equal(
        obs_data, expected_dataset_2_mix["observed_data"], check_dtype=False
    )
