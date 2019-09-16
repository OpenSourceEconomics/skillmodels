"""Tests for the linear Kalman update step."""
import json

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import skillmodels.fast_routines.kalman_filters as kf

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


def unpack_update_fixture(
    factors,
    state,
    measurement,
    state_cov,
    loadings,
    meas_var,
    expected_post_means=None,
    expected_post_state_cov=None,
):
    kwargs = {
        "state": np.array([state]).astype(float),
        "covs": np.array([state_cov]).astype(float),
        "like_vec": np.array([0]).astype(float),
        "y": np.array([measurement]).astype(float),
        "c": np.array(range(0)),
        "delta": np.array(range(0)),
        "h": np.array(loadings).astype(float),
        "r": np.array([meas_var]).astype(float),
        "positions": np.array([i for i, x in enumerate(loadings) if x != 0]).astype(
            int
        ),
        "weights": np.array([[1]]).astype(float),
        "kf": np.zeros((1, len(state))).astype(float),
    }
    sk_type_exp_states = np.array([expected_post_means])
    sk_type_exp_cov = np.array([expected_post_state_cov])
    return kwargs, sk_type_exp_states, sk_type_exp_cov


# ======================================================================================
# manual tests
# ======================================================================================


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

    out["like_vector"] = np.zeros(nind)

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

    out["expected_like_vector"] = np.log(
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
    factors = ["c", "n", "i", "cp", "np"]
    state = np.array([11, 11.5, 12, 12.5, 10])
    root_cov = np.linalg.cholesky(
        [
            [1.00, 0.10, 0.15, 0.20, 0.05],
            [0.10, 1.40, 0.20, 0.30, 0.25],
            [0.15, 0.20, 1.10, 0.20, 0.10],
            [0.20, 0.30, 0.20, 1.50, 0.15],
            [0.05, 0.25, 0.10, 0.15, 0.90],
        ]
    )
    measurement = 29
    loadings = np.array([1.5, 0, 1.2, 0, 0])
    meas_var = 2.25

    expected_updated_means = np.array(
        [10.51811594, 11.38813406, 11.55683877, 12.3451087, 9.94406703]
    )
    expected_updated_cov = np.array(
        [
            [
                5.73913043e-01,
                1.08695652e-03,
                -2.41847826e-01,
                6.30434783e-02,
                5.43478261e-04,
            ],
            [
                1.08695652e-03,
                1.37703804e00,
                1.09035326e-01,
                2.68206522e-01,
                2.38519022e-01,
            ],
            [
                -2.41847826e-01,
                1.09035326e-01,
                7.39639946e-01,
                7.40489130e-02,
                5.45176630e-02,
            ],
            [
                6.30434783e-02,
                2.68206522e-01,
                7.40489130e-02,
                1.45597826e00,
                1.34103261e-01,
            ],
            [
                5.43478261e-04,
                2.38519022e-01,
                5.45176630e-02,
                1.34103261e-01,
                8.94259511e-01,
            ],
        ]
    )

    return {
        "factors": factors,
        "state": state,
        "measurement": measurement,
        "state_cov": root_cov.dot(root_cov.T),
        "loadings": loadings,
        "meas_var": meas_var,
        "expected_post_means": expected_updated_means,
        "expected_post_state_cov": expected_updated_cov,
    }


def test_normal_state_and_cov_update_without_nan(setup_linear_update_2):
    d, exp_states, exp_cov = unpack_update_fixture(**setup_linear_update_2)
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
# tests from filterpy
# =============================================================================

fix_path = "skillmodels/tests/fast_routines/generated_fixtures_update.json"
with open(fix_path, "r") as f:
    id_to_fix = json.load(f)
ids, fixtures = zip(*id_to_fix.items())


@pytest.mark.parametrize("fixture", fixtures, ids=ids)
def test_normal_state_and_cov_against_filterpy(fixture):
    d, exp_states, exp_cov = unpack_update_fixture(**fixture)
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
