import skillmodels.fast_routines.kalman_filters as kf
from numpy.testing import assert_array_almost_equal as aaae
import numpy as np
import pytest
import json


# ======================================================================================
# helper functions
# ======================================================================================


def make_unique(qr_result_arr):
    long_side, m, n = qr_result_arr.shape
    for u in range(long_side):
        for j in range(n):
            if qr_result_arr[u, j, j] < 0:
                for k in range(n):
                    qr_result_arr[u, j, k] *= -1


def skillmodels_kwargs_from_fixture(
        factors, state, measurement, state_cov, loadings, meas_var,
        expected_post_means=None, expected_post_state_cov=None):
    kwargs = {
        "state": np.array([state]).astype(float),
        "covs": np.array([state_cov]).astype(float),
        "like_vec": np.array([1]).astype(float),
        "y": np.array([measurement]).astype(float),
        "c": np.array(range(0)),
        "delta": np.array(range(0)),
        "h": np.array(loadings).astype(float),
        "r": np.array([meas_var]).astype(float),
        "positions": np.array([i for i, x in enumerate(loadings) if x != 0]).astype(int),
        "weights": np.array([[1]]).astype(float),
        "kf": np.zeros((1, len(state))).astype(float)}
    sk_type_exp_states = np.array([expected_post_means])
    sk_type_exp_cov = np.array([expected_post_state_cov])
    return kwargs, sk_type_exp_states, sk_type_exp_cov

# =============================================================================
# TESTS FOR THE UPDATE FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# MANUAL TESTS OF THE UPDATE FUNCTIONS
# -----------------------------------------------------------------------------


@pytest.fixture
def setup_linear_update():
    out = {}

    nemf, nind, nfac = 2, 6, 3
    states = np.ones((nind, nemf, nfac))
    states[:, 1, 2] *= 2
    out["state"] = states

    covs = np.zeros((nind, nemf, nfac, nfac))
    covs[:] = np.ones((nfac, nfac)) * 0.1 + np.eye(nfac) * 0.6
    out["covs"] = covs

    mcovs = np.zeros((nind, nemf, nfac + 1, nfac + 1))
    out["mcovs"] = mcovs

    mcovs[:, :, 1:, 1:] = np.transpose(np.linalg.cholesky(covs), axes=(0, 1, 3, 2))
    cholcovs = mcovs[:, :, 1:, 1:]
    out["cholcovs"] = cholcovs

    out["like_vector"] = np.ones(nind)

    out["y"] = np.array([3.5, 2.3, np.nan, 3.1, 4, np.nan])

    out["c"] = np.ones((nind, 2))

    out["delta"] = np.ones(2) / 2

    out["h"] = np.array([1, 1, 0.5])

    out["sqrt_r"] = np.sqrt(np.array([0.3]))

    out["positions"] = np.array([0, 1, 2])

    out["r"] = np.array([0.3])

    out["kf"] = np.zeros((nind, nfac))

    weights = np.ones((nind, nemf))
    weights[:, 0] *= 0.4
    weights[:, 1] *= 0.6
    out["weights"] = weights

    return out


@pytest.fixture
def expected_linear_update():

    out = {}

    out["expected_states"] = np.array(
        [
            [
                [1.00000000, 1.00000000, 1.00000000],
                [0.81318681, 0.81318681, 1.87912088],
            ],
            [
                [0.55164835, 0.55164835, 0.70989011],
                [0.36483516, 0.36483516, 1.58901099],
            ],
            [
                [1.00000000, 1.00000000, 1.00000000],
                [1.00000000, 1.00000000, 2.00000000],
            ],
            [
                [0.85054945, 0.85054945, 0.90329670],
                [0.66373626, 0.66373626, 1.78241758],
            ],
            [
                [1.18681319, 1.18681319, 1.12087912],
                [1.00000000, 1.00000000, 2.00000000],
            ],
            [
                [1.00000000, 1.00000000, 1.00000000],
                [1.00000000, 1.00000000, 2.00000000],
            ],
        ]
    )

    out["expected_weights"] = np.array(
        [
            [0.41325632, 0.58674368],
            [0.47831766, 0.52168234],
            [0.40000000, 0.60000000],
            [0.43472272, 0.56527728],
            [0.38688853, 0.61311147],
            [0.40000000, 0.60000000],
        ]
    )

    out["expected_like_vector"] = np.array(
        [0.25601173, 0.16118176, 1.0, 0.23496064, 0.25883987, 1.0]
    )

    exp_covs = np.array(
        [
            [
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
            ],
            [
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
            ],
            [
                [[0.7, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.7]],
                [[0.7, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.7]],
            ],
            [
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
            ],
            [
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
                [
                    [0.38241758, -0.21758242, -0.10549451],
                    [-0.21758242, 0.38241758, -0.10549451],
                    [-0.10549451, -0.10549451, 0.56703297],
                ],
            ],
            [
                [[0.7, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.7]],
                [[0.7, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.7]],
            ],
        ]
    )
    out["expected_covs"] = exp_covs

    exp_cholcovs = np.transpose(np.linalg.cholesky(exp_covs), axes=(0, 1, 3, 2))
    out["exp_cholcovs"] = exp_cholcovs

    return out


def test_sqrt_state_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.sqrt_linear_update(
        d["state"],
        d["mcovs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["sqrt_r"],
        d["positions"],
        d["weights"],
    )
    modified_states = d["state"]
    aaae(modified_states, expected_linear_update["expected_states"])


def test_sqrt_cov_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.sqrt_linear_update(
        d["state"],
        d["mcovs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["sqrt_r"],
        d["positions"],
        d["weights"],
    )
    modified_mcovs = d["cholcovs"]
    make_unique(modified_mcovs.reshape(12, 3, 3))
    aaae(modified_mcovs, expected_linear_update["exp_cholcovs"])


def test_sqrt_like_vec_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.sqrt_linear_update(
        d["state"],
        d["mcovs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["sqrt_r"],
        d["positions"],
        d["weights"],
    )
    aaae(d["like_vector"], expected_linear_update["expected_like_vector"])


def test_sqrt_weight_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.sqrt_linear_update(
        d["state"],
        d["mcovs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["sqrt_r"],
        d["positions"],
        d["weights"],
    )
    aaae(d["weights"], expected_linear_update["expected_weights"])


def test_normal_state_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.normal_linear_update(
        d["state"],
        d["covs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["r"],
        d["positions"],
        d["weights"],
        d["kf"],
    )
    aaae(d["state"], expected_linear_update["expected_states"])


def test_normal_cov_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.normal_linear_update(
        d["state"],
        d["covs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["r"],
        d["positions"],
        d["weights"],
        d["kf"],
    )
    aaae(d["covs"], expected_linear_update["expected_covs"])


def test_normal_like_vec_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.normal_linear_update(
        d["state"],
        d["covs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["r"],
        d["positions"],
        d["weights"],
        d["kf"],
    )
    aaae(d["like_vector"], expected_linear_update["expected_like_vector"])


def test_normal_weight_update_with_nans(setup_linear_update, expected_linear_update):
    d = setup_linear_update
    kf.normal_linear_update(
        d["state"],
        d["covs"],
        d["like_vector"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["r"],
        d["positions"],
        d["weights"],
        d["kf"],
    )
    aaae(d["weights"], expected_linear_update["expected_weights"])


@pytest.fixture
def setup_linear_update_2():
    # to conform with the jsons that contain setup and result of filterpy
    # this fixture contains the setup and expected result
    factors = ['c', 'n', 'i', 'cp', 'np']
    state = np.array([11, 11.5, 12, 12.5, 10])
    root_cov = np.linalg.cholesky(
        [[1.00, 0.10, 0.15, 0.20, 0.05],
         [0.10, 1.40, 0.20, 0.30, 0.25],
         [0.15, 0.20, 1.10, 0.20, 0.10],
         [0.20, 0.30, 0.20, 1.50, 0.15],
         [0.05, 0.25, 0.10, 0.15, 0.90]])
    measurement = 29
    loadings = np.array([1.5, 0, 1.2, 0, 0])
    meas_var = 2.25

    expected_updated_means = np.array(
        [10.51811594, 11.38813406, 11.55683877, 12.3451087, 9.94406703])
    expected_updated_cov = np.array([
        [5.73913043e-01, 1.08695652e-03, -2.41847826e-01,
         6.30434783e-02, 5.43478261e-04],
        [1.08695652e-03, 1.37703804e+00, 1.09035326e-01,
         2.68206522e-01, 2.38519022e-01],
        [-2.41847826e-01, 1.09035326e-01, 7.39639946e-01,
         7.40489130e-02, 5.45176630e-02],
        [6.30434783e-02, 2.68206522e-01, 7.40489130e-02,
         1.45597826e+00, 1.34103261e-01],
        [5.43478261e-04, 2.38519022e-01, 5.45176630e-02,
         1.34103261e-01, 8.94259511e-01]])

    return {
        'factors': factors,
        'state': state,
        'measurement': measurement,
        'state_cov': root_cov.dot(root_cov.T),
        'loadings': loadings,
        'meas_var': meas_var,
        'expected_post_means': expected_updated_means,
        'expected_post_state_cov': expected_updated_cov}


def test_normal_state_and_cov_update_without_nan(setup_linear_update_2):
    d, exp_states, exp_cov = \
        skillmodels_kwargs_from_fixture(**setup_linear_update_2)
    kf.normal_linear_update(
        d["state"],
        d["covs"],
        d["like_vec"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["r"],
        d["positions"],
        d["weights"],
        d["kf"],
    )
    aaae(d["state"], exp_states)
    aaae(d["covs"], exp_cov)

# -----------------------------------------------------------------------------
# TESTS OF THE UPDATE FUNCTION FROM FILTERPY
# -----------------------------------------------------------------------------

with open('skillmodels/tests/fast_routines/generated_fixtures.json', 'r') as f:
    id_to_fix = json.load(f)

ids, fixtures = zip(*id_to_fix.items())


@pytest.mark.parametrize("fixture", fixtures, ids=ids)
def test_normal_state_and_cov_against_filterpy(fixture):
    d, exp_states, exp_cov = skillmodels_kwargs_from_fixture(**fixture)
    kf.normal_linear_update(
        d["state"],
        d["covs"],
        d["like_vec"],
        d["y"],
        d["c"],
        d["delta"],
        d["h"],
        d["r"],
        d["positions"],
        d["weights"],
        d["kf"],
    )
    aaae(d["state"], exp_states)
    aaae(d["covs"], exp_cov)

# =============================================================================
# TESTS FOR THE PREDICT FUNCTIONS
# =============================================================================


# tests for normal linear predict and sqrt linear functions
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

    nemf, nind, nsigma, nfac = 2, 3, 7, 3

    out["stage"] = 1

    first = np.array([1.1, 1.2, 1.3])
    second = np.array([1.4, 1.5, 1.6])
    third = np.array([2.1, 2.2, 2.3])
    fourth = np.array([2.4, 2.5, 2.6])

    # these are sigma_points for the test with focus on columns
    sps1 = np.zeros((nemf, nind, nsigma, nfac))
    sps1[0, 0, :, :] = np.tile(first, nsigma).reshape(nsigma, nfac)
    sps1[0, 1, :, :] = np.tile(second, nsigma).reshape(nsigma, nfac)
    sps1[1, 0, :, :] = np.tile(third, nsigma).reshape(nsigma, nfac)
    sps1[1, 1, :, :] = np.tile(fourth, nsigma).reshape(nsigma, nfac)
    out["sps1"] = sps1.reshape(nemf * nind, nsigma, nfac)
    out["flat_sps1"] = sps1.reshape(nemf * nind * nsigma, nfac)

    expected_states1 = np.zeros((nemf, nind, nfac))
    expected_states1[0, 0, :] = first
    expected_states1[0, 1, :] = second
    expected_states1[1, 0, :] = third
    expected_states1[1, 1, :] = fourth
    out["expected_states1"] = expected_states1.reshape(nemf * nind, nfac)

    # these are sigma_points for the test with focus on weighting
    sps2 = np.zeros((nemf, nind, nsigma, nfac))
    sps2[:, :, :, :] = np.arange(nsigma).repeat(nfac).reshape(nsigma, nfac)
    out["sps2"] = sps2.reshape(nemf * nind, nsigma, nfac)
    out["flat_sps2"] = sps2.reshape(nemf * nind * nsigma, nfac)
    out["expected_states2"] = np.ones((nemf * nind, nfac)) * 3

    # these are sigma_points for the test with focus on the covariances
    sps3 = np.zeros((nemf, nind, nsigma, nfac))
    sps3[:, :, 1, :] += 1
    sps3[:, :, 2, :] += 2
    sps3[:, :, 3, :] += 3
    sps3[:, :, 4, :] -= 1
    sps3[:, :, 5, :] -= 2
    sps3[:, :, 6, :] -= 3
    out["sps3"] = sps3.reshape(nemf * nind, nsigma, nfac)
    out["flat_sps3"] = sps3.reshape(nemf * nind * nsigma, nfac)

    sws_m = np.ones(nsigma) / nsigma
    out["sws_m"] = sws_m
    out["sws_c"] = sws_m

    q = np.eye(nfac)
    Q = np.zeros((2, nfac, nfac))
    Q[:] = q
    out["Q"] = Q

    out["transform_sps_args"] = {}

    exp_covs = np.zeros((nemf * nind, nfac, nfac))
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

    out["out_states"] = np.zeros((nemf * nind, nfac))
    out_sqrt_covs = np.zeros((nemf * nind, nfac + 1, nfac + 1))
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
        d["Q"],
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
        d["Q"],
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

    d["Q"][:] = np.eye(3) * 0.25 + np.ones((3, 3)) * 0.5
    kf.normal_unscented_predict(
        d["stage"],
        d["sps3"],
        d["flat_sps3"],
        d["sws_m"],
        d["sws_c"],
        d["Q"],
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
        d["Q"],
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
        d["Q"],
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
        d["Q"],
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
