"""Tests for the linear Kalman predict step."""
import json

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import skillmodels.fast_routines.kalman_filters as kf

# ======================================================================================
# manual tests
# ======================================================================================


def make_unique(qr_result_arr):
    long_side, m, n = qr_result_arr.shape
    for u in range(long_side):
        for j in range(n):
            if qr_result_arr[u, j, j] < 0:
                for k in range(n):
                    qr_result_arr[u, j, k] *= -1


@pytest.fixture
def setup_linear_predict():
    out = {}

    out["state"] = np.array([[7, 9, 3], [8, 3, 5]])

    out["cov"] = np.array(
        [
            [[0.3, 0, 0], [0, 0.5, 0], [0, 0, 0.9]],
            [[0.3, -0.2, -0.1], [-0.2, 0.3, -0.1], [-0.1, -0.1, 0.5]],
        ]
    )

    out["root_cov"] = np.array(
        [
            [[0.3, 0, 0], [0, 0.5, 0], [0, 0, 0.9]],
            [[0.3, 0.2, 0.1], [0, 0.3, 0.1], [0, 0, 0.5]],
        ]
    )

    out["shocks_sds"] = np.array([1.2, 0.3, 0.2])

    out["transition_matrix"] = np.array([[0.2, 1, 2], [0.5, 0.9, 0.7], [0.2, 0.4, 0.4]])

    return out


@pytest.fixture
def expected_linear_predict():
    out = {}

    out["predicted_states"] = np.array([[16.4, 13.7, 6.2], [14.6, 10.2, 4.8]])

    out["predicted_covs"] = np.array(
        [
            [[5.552, 1.74, 0.932], [1.74, 1.011, 0.462], [0.932, 0.462, 0.276]],
            [[3.192, 0.5, 0.308], [0.5, 0.277, 0.104], [0.308, 0.104, 0.1]],
        ]
    )

    out["predicted_sqrt_covs"] = np.array(
        [
            [
                [5.427423, -0.424064, 0.0809716],
                [-0.424064, 0.374016, -0.0487112],
                [0.0809716, -0.0487112, 0.0572607],
            ],
            [
                [3.11777, 0.273778, 0.0502302],
                [0.273778, 0.354309, 0.0453752],
                [0.0502302, 0.0453752, 0.05562],
            ],
        ]
    )

    return out


def test_normal_linear_predict_states(setup_linear_predict, expected_linear_predict):
    d = setup_linear_predict
    calc_pred_state, calc_pred_cov = kf.normal_linear_predict(
        d["state"], d["cov"], d["shocks_sds"], d["transition_matrix"]
    )
    aaae(calc_pred_state, expected_linear_predict["predicted_states"])


def test_normal_linear_predict_covs(setup_linear_predict, expected_linear_predict):
    d = setup_linear_predict
    calc_pred_state, calc_pred_cov = kf.normal_linear_predict(
        d["state"], d["cov"], d["shocks_sds"], d["transition_matrix"]
    )
    aaae(calc_pred_cov, expected_linear_predict["predicted_covs"])


def test_sqrt_predict_states(setup_linear_predict, expected_linear_predict):
    d = setup_linear_predict
    calc_pred_state, calc_pred_cov = kf.sqrt_linear_predict(
        d["state"], d["root_cov"], d["shocks_sds"], d["transition_matrix"]
    )
    aaae(calc_pred_state, expected_linear_predict["predicted_states"])


def test_sqrt_predict_root_covs(setup_linear_predict, expected_linear_predict):
    d = setup_linear_predict
    calc_pred_state, calc_pred_root_cov = kf.sqrt_linear_predict(
        d["state"], d["root_cov"], d["shocks_sds"], d["transition_matrix"]
    )
    calc_cov = np.matmul(
        calc_pred_root_cov, np.transpose(calc_pred_root_cov, axes=(0, 2, 1))
    )
    aaae(calc_cov, expected_linear_predict["predicted_sqrt_covs"])


@pytest.fixture
def setup_unscented_predict():
    out = {}

    nmixtures, nind, nsigma, nfac = 2, 3, 7, 3

    out["stage"] = 1

    first = np.array([1.1, 1.2, 1.3])
    second = np.array([1.4, 1.5, 1.6])
    third = np.array([2.1, 2.2, 2.3])
    fourth = np.array([2.4, 2.5, 2.6])

    # these are sigma_points for the test with focus on columns
    sps1 = np.zeros((nmixtures, nind, nsigma, nfac))
    sps1[0, 0, :, :] = np.tile(first, nsigma).reshape(nsigma, nfac)
    sps1[0, 1, :, :] = np.tile(second, nsigma).reshape(nsigma, nfac)
    sps1[1, 0, :, :] = np.tile(third, nsigma).reshape(nsigma, nfac)
    sps1[1, 1, :, :] = np.tile(fourth, nsigma).reshape(nsigma, nfac)
    out["sps1"] = sps1.reshape(nmixtures * nind, nsigma, nfac)
    out["flat_sps1"] = sps1.reshape(nmixtures * nind * nsigma, nfac)

    expected_states1 = np.zeros((nmixtures, nind, nfac))
    expected_states1[0, 0, :] = first
    expected_states1[0, 1, :] = second
    expected_states1[1, 0, :] = third
    expected_states1[1, 1, :] = fourth
    out["expected_states1"] = expected_states1.reshape(nmixtures * nind, nfac)

    # these are sigma_points for the test with focus on weighting
    sps2 = np.zeros((nmixtures, nind, nsigma, nfac))
    sps2[:, :, :, :] = np.arange(nsigma).repeat(nfac).reshape(nsigma, nfac)
    out["sps2"] = sps2.reshape(nmixtures * nind, nsigma, nfac)
    out["flat_sps2"] = sps2.reshape(nmixtures * nind * nsigma, nfac)
    out["expected_states2"] = np.ones((nmixtures * nind, nfac)) * 3

    # these are sigma_points for the test with focus on the covariances
    sps3 = np.zeros((nmixtures, nind, nsigma, nfac))
    sps3[:, :, 1, :] += 1
    sps3[:, :, 2, :] += 2
    sps3[:, :, 3, :] += 3
    sps3[:, :, 4, :] -= 1
    sps3[:, :, 5, :] -= 2
    sps3[:, :, 6, :] -= 3
    out["sps3"] = sps3.reshape(nmixtures * nind, nsigma, nfac)
    out["flat_sps3"] = sps3.reshape(nmixtures * nind * nsigma, nfac)

    sws_m = np.ones(nsigma) / nsigma
    out["sws_m"] = sws_m
    out["sws_c"] = sws_m

    qq = np.eye(nfac)
    q = np.zeros((2, nfac, nfac))
    q[:] = qq
    out["q"] = q

    out["transform_sps_args"] = {}

    exp_covs = np.zeros((nmixtures * nind, nfac, nfac))
    exp_covs[:] = np.array([[4.75, 4.5, 4.5], [4.5, 4.75, 4.5], [4.5, 4.5, 4.75]])
    out["exp_covs"] = exp_covs

    exp_cholcovs = np.zeros_like(exp_covs)
    exp_cholcovs[:] = np.array(
        [
            [2.23606798, 0.00000000, 0.00000000],
            [1.78885438, 1.34164079, 0.00000000],
            [1.78885438, 0.596284794, 1.20185043],
        ]
    ).T
    out["exp_cholcovs"] = exp_cholcovs

    out["out_states"] = np.zeros((nmixtures * nind, nfac))
    out_sqrt_covs = np.zeros((nmixtures * nind, nfac + 1, nfac + 1))
    out["out_sqrt_covs"] = out_sqrt_covs
    out["out_covs"] = out_sqrt_covs[:, 1:, 1:]

    return out


def test_normal_unscented_predict_focus_on_colums(setup_unscented_predict, mocker):
    d = setup_unscented_predict
    mock_transform = mocker.patch(
        "skillmodels.fast_routines.kalman_filters.transform_sigma_points"
    )
    mock_transform.return_value = d["sps1"]
    kf.normal_unscented_predict(
        d["stage"],
        d["sps1"],
        d["flat_sps1"],
        d["sws_m"],
        d["sws_c"],
        d["q"],
        d["transform_sps_args"],
        d["out_states"],
        d["out_covs"],
    )

    aaae(d["out_states"], d["expected_states1"])


def test_normal_unscented_predict_focus_on_weighting(setup_unscented_predict, mocker):
    d = setup_unscented_predict
    mock_transform = mocker.patch(
        "skillmodels.fast_routines.kalman_filters.transform_sigma_points"
    )
    mock_transform.return_value = d["sps2"]
    kf.normal_unscented_predict(
        d["stage"],
        d["sps2"],
        d["flat_sps2"],
        d["sws_m"],
        d["sws_c"],
        d["q"],
        d["transform_sps_args"],
        d["out_states"],
        d["out_covs"],
    )

    aaae(d["out_states"], d["expected_states2"])


def test_normal_unscented_predict_focus_on_covs(setup_unscented_predict, mocker):
    d = setup_unscented_predict
    mock_transform = mocker.patch(
        "skillmodels.fast_routines.kalman_filters.transform_sigma_points"
    )
    mock_transform.return_value = d["sps3"]

    d["q"][:] = np.eye(3) * 0.25 + np.ones((3, 3)) * 0.5
    kf.normal_unscented_predict(
        d["stage"],
        d["sps3"],
        d["flat_sps3"],
        d["sws_m"],
        d["sws_c"],
        d["q"],
        d["transform_sps_args"],
        d["out_states"],
        d["out_covs"],
    )

    aaae(d["out_covs"], d["exp_covs"])


def test_sqrt_unscented_predict_focus_on_colums(setup_unscented_predict, mocker):
    d = setup_unscented_predict
    mock_transform = mocker.patch(
        "skillmodels.fast_routines.kalman_filters.transform_sigma_points"
    )
    mock_transform.return_value = d["sps1"]
    kf.sqrt_unscented_predict(
        d["stage"],
        d["sps1"],
        d["flat_sps1"],
        d["sws_m"],
        d["sws_c"],
        d["q"],
        d["transform_sps_args"],
        d["out_states"],
        d["out_sqrt_covs"],
    )

    aaae(d["out_states"], d["expected_states1"])


def test_sqrt_unscented_predict_focus_on_weighting(setup_unscented_predict, mocker):
    d = setup_unscented_predict
    mock_transform = mocker.patch(
        "skillmodels.fast_routines.kalman_filters.transform_sigma_points"
    )
    mock_transform.return_value = d["sps2"]

    kf.sqrt_unscented_predict(
        d["stage"],
        d["sps2"],
        d["flat_sps2"],
        d["sws_m"],
        d["sws_c"],
        d["q"],
        d["transform_sps_args"],
        d["out_states"],
        d["out_sqrt_covs"],
    )

    aaae(d["out_states"], d["expected_states2"])


def test_sqrt_unscented_predict_focus_on_covs(setup_unscented_predict, mocker):
    d = setup_unscented_predict
    mock_transform = mocker.patch(
        "skillmodels.fast_routines.kalman_filters.transform_sigma_points"
    )
    mock_transform.return_value = d["sps3"]
    kf.sqrt_unscented_predict(
        d["stage"],
        d["sps3"],
        d["flat_sps3"],
        d["sws_m"],
        d["sws_c"],
        d["q"],
        d["transform_sps_args"],
        d["out_states"],
        d["out_sqrt_covs"],
    )
    make_unique(d["out_covs"])
    aaae(d["out_covs"], d["exp_cholcovs"])


shocks_sd = np.array(
    [
        [[1.2, 0.3, 0.2]],
        [[0.1, 0.9, 0.2]],
        [[0.3, 0.01, 0.3]],
        [[0.2, 0.5, 0.6]],
        [[0.7, 0.1, 0.4]],
        [[0.2, 0.1, 0.1]],
    ]
)

# ======================================================================================
# tests from filterpy
# ======================================================================================


def unpack_predict_fixture(fixture):
    nfac = len(fixture["state"])

    args = (
        np.array(fixture["state"]).reshape(1, nfac),
        np.array(fixture["state_cov"]).reshape(1, nfac, nfac),
        np.array(fixture["shock_sds"]),
        np.array(fixture["transition_matrix"]),
    )
    exp_state = np.array(fixture["expected_post_means"])
    exp_cov = np.array(fixture["expected_post_state_cov"])
    return args, exp_state, exp_cov


def convert_normal_to_sqrt_args(args):
    args_list = list(args)
    covs = args[1]
    all_diagonal = True
    for i in range(len(covs)):
        if (np.diag(np.diagonal(covs[i])) != covs[i]).any():
            all_diagonal = False

    if all_diagonal is True:
        args_list[1] = np.sqrt(covs)
    else:
        args_list[1] = np.transpose(np.linalg.cholesky(covs), axes=(0, 2, 1))
    return tuple(args_list)


# for the normal linear predict
# ------------------------------

fix_path = "skillmodels/tests/fast_routines/generated_fixtures_predict.json"
with open(fix_path, "r") as f:
    id_to_fix = json.load(f)
ids, fixtures = zip(*id_to_fix.items())


@pytest.mark.parametrize("fixture", fixtures, ids=ids)
def test_normal_linear_predicted_state_against_filterpy(fixture):
    args, exp_state, exp_cov = unpack_predict_fixture(fixture)
    after_state, after_covs = kf.normal_linear_predict(*args)
    aaae(after_state.flatten(), exp_state)


@pytest.mark.parametrize("fixture", fixtures, ids=ids)
def test_normal_linear_predicted_cov_against_filterpy(fixture):
    args, exp_state, exp_cov = unpack_predict_fixture(fixture)
    after_state, after_covs = kf.normal_linear_predict(*args)
    aaae(after_covs.reshape(exp_cov.shape), exp_cov)


# for the square root linear predict
# -----------------------------------


@pytest.mark.parametrize("fixture", fixtures, ids=ids)
def test_sqrt_linear_predicted_state_against_filterpy(fixture):
    args, exp_state, exp_cov = unpack_predict_fixture(fixture)
    args = convert_normal_to_sqrt_args(args)
    after_state, after_covs = kf.sqrt_linear_predict(*args)
    aaae(after_state.flatten(), exp_state)


np.set_printoptions(formatter={"float": "{: 0.3f}".format})


@pytest.mark.parametrize("fixture", fixtures, ids=ids)
def test_sqrt_linear_predicted_cov_against_filterpy(fixture):
    # this gives the covariance matrix, not a square root of it!
    args, exp_state, exp_cov = unpack_predict_fixture(fixture)
    args = convert_normal_to_sqrt_args(args)
    after_state, after_cov_sqrt = kf.sqrt_linear_predict(*args)
    after_cov_sqrt = after_cov_sqrt[0]

    implied_cov = after_cov_sqrt.T.dot(after_cov_sqrt)

    aaae(implied_cov, exp_cov)
