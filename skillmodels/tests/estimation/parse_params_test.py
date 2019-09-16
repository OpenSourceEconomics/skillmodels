import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae

from skillmodels.estimation.parse_params import _map_params_to_delta
from skillmodels.estimation.parse_params import _map_params_to_h
from skillmodels.estimation.parse_params import _map_params_to_p
from skillmodels.estimation.parse_params import _map_params_to_q
from skillmodels.estimation.parse_params import _map_params_to_r
from skillmodels.estimation.parse_params import _map_params_to_trans_coeffs
from skillmodels.estimation.parse_params import _map_params_to_w
from skillmodels.estimation.parse_params import _map_params_to_x


def test_map_params_to_delta():
    ind_tups = [
        ("delta", 0, "m1", "constant"),
        ("delta", 0, "m1", "c1"),
        ("delta", 0, "m1", "c2"),
        ("delta", 0, "m2", "constant"),
        ("delta", 0, "m2", "c1"),
        ("delta", 0, "m2", "c2"),
        ("delta", 0, "m3", "constant"),
        ("delta", 0, "m3", "c1"),
        ("delta", 0, "m3", "c2"),
        ("delta", 0, "m4", "constant"),
        ("delta", 0, "m4", "c1"),
        ("delta", 0, "m4", "c2"),
        ("delta", 1, "m3", "constant"),
        ("delta", 1, "m3", "c1"),
        ("delta", 1, "m4", "constant"),
        ("delta", 1, "m4", "c1"),
        ("delta", 1, "bla", "constant"),
        ("delta", 1, "bla", "c1"),
        ("delta", 1, "m1", "constant"),
        ("delta", 1, "m1", "c1"),
        ("delta", 1, "m5", "constant"),
        ("delta", 1, "m5", "c1"),
        ("delta", 1, "m6", "constant"),
        ("delta", 1, "m6", "c1"),
    ]
    params_vec = [
        10,
        11,
        12,
        13,
        14,
        15,
        0,
        16,
        17,
        18,
        19,
        20,
        0,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
    ]

    params = pd.Series(index=pd.MultiIndex.from_tuples(ind_tups), data=params_vec)
    initial = np.zeros((4, 3)), np.zeros((6, 2))

    _map_params_to_delta(params, initial)

    expected0 = np.array([[10.0, 11, 12], [13, 14, 15], [0, 16, 17], [18, 19, 20]])
    expected1 = np.array([[0.0, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]])

    aae(initial[0], expected0)
    aae(initial[1], expected1)


def test_map_params_to_h():

    ind_tups = [
        ("h", 0, "m1", "f1"),
        ("h", 0, "m1", "f2"),
        ("h", 0, "m1", "f3"),
        ("h", 0, "m2", "f1"),
        ("h", 0, "m2", "f2"),
        ("h", 0, "m2", "f3"),
        ("h", 0, "bla", "f1"),
        ("h", 0, "bla", "f2"),
        ("h", 0, "bla", "f3"),
        ("h", 1, "bla", "f1"),
        ("h", 1, "bla", "f2"),
        ("h", 1, "bla", "f3"),
        ("h", 1, "m1", "f1"),
        ("h", 1, "m1", "f2"),
        ("h", 1, "m1", "f3"),
        ("h", 2, "m1", "f1"),
        ("h", 2, "m1", "f2"),
        ("h", 2, "m1", "f3"),
    ]
    params_vec = [10, 0, 0, 0, 11, 0, 0, 3, 12, 13, 0, 14, 0, 15, 16, 0, 0, 0]
    expected = np.array(
        [[10.0, 0, 0], [0, 11, 0], [0, 3, 12], [13, 0, 14], [0, 15, 16], [0, 0, 0]]
    )
    initial = np.zeros((6, 3))
    params = pd.Series(index=pd.MultiIndex.from_tuples(ind_tups), data=params_vec)

    _map_params_to_h(params, initial)
    aae(initial, expected)


def test_map_params_to_r_square_root_filters():
    params = pd.Series(index=["r"] * 5 + ["bla", "blubb"], data=np.arange(7))
    initial = np.zeros(5)
    _map_params_to_r(params, initial, True)
    expected = np.sqrt(np.arange(5))
    aae(initial, expected)


def test_map_params_to_r_normal_filters():
    params = pd.Series(index=["r"] * 5 + ["bla", "blubb"], data=np.arange(7))
    initial = np.zeros(5)
    _map_params_to_r(params, initial, False)
    expected = np.arange(5)
    aae(initial, expected)


def test_map_params_to_q():
    initial = np.zeros((2, 3, 3))
    params_vec = [2.0, 4, 6, 8, 10, 12]
    ind_tups = [
        ("q", 0, "fac1"),
        ("q", 0, "fac2"),
        ("q", 0, "fac3"),
        ("q", 1, "fac1"),
        ("q", 1, "fac2"),
        ("q", 1, "fac3"),
    ]
    params = pd.Series(index=pd.MultiIndex.from_tuples(ind_tups), data=params_vec)

    expected = np.vstack([np.diag([2.0, 4, 6]), np.diag([8.0, 10, 12])]).reshape(
        (2, 3, 3)
    )

    _map_params_to_q(params, initial)
    aae(initial, expected)


def test_map_params_to_x():
    ind_tups = [
        ("x", 0, 0, "fac1"),
        ("x", 0, 0, "fac2"),
        ("x", 0, 1, "fac1"),
        ("x", 0, 1, "fac2"),
        ("x", 0, 2, "fac1"),
        ("x", 0, 2, "fac2"),
    ]
    params = pd.Series(index=pd.MultiIndex.from_tuples(ind_tups), data=np.arange(6))
    initial = np.zeros((8, 3, 2))

    expected = np.zeros_like(initial)
    for obs in range(8):
        expected[obs] = np.arange(6).reshape(3, 2)

    _map_params_to_x(params, initial)
    aae(initial, expected)


@pytest.fixture
def p_params():
    ind_tups = [
        ("p", 0, 0, "fac1-fac1"),
        ("p", 0, 0, "fac2-fac1"),
        ("p", 0, 0, "fac2-fac2"),
        ("p", 0, 0, "fac3-fac1"),
        ("p", 0, 0, "fac3-fac2"),
        ("p", 0, 0, "fac3-fac3"),
        ("p", 0, 1, "fac1-fac1"),
        ("p", 0, 1, "fac2-fac1"),
        ("p", 0, 1, "fac2-fac2"),
        ("p", 0, 1, "fac3-fac1"),
        ("p", 0, 1, "fac3-fac2"),
        ("p", 0, 1, "fac3-fac3"),
    ]
    params_vec = [1, 0.1, 2, 0.2, 0.3, 3, 4, 0.5, 5, 0.6, 0.7, 6]
    params = pd.Series(index=pd.MultiIndex.from_tuples(ind_tups), data=params_vec)
    return params


def test_map_params_to_p_square_root_filters(p_params):
    initial = np.zeros((5, 2, 4, 4))
    expected0 = np.linalg.cholesky(
        np.array([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]])
    ).T
    expected1 = np.linalg.cholesky(
        np.array([[4, 0.5, 0.6], [0.5, 5, 0.7], [0.6, 0.7, 6]])
    ).T

    expected = np.zeros_like(initial)
    expected[:, 0, 1:, 1:] = expected0
    expected[:, 1, 1:, 1:] = expected1

    _map_params_to_p(p_params, initial, True)
    aae(initial, expected)


def test_maps_params_to_p_normal_filters(p_params):
    initial = np.zeros((5, 2, 3, 3))
    expected0 = np.array([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]])
    expected1 = np.array([[4, 0.5, 0.6], [0.5, 5, 0.7], [0.6, 0.7, 6]])

    expected = np.zeros_like(initial)
    expected[:, 0] = expected0
    expected[:, 1] = expected1

    _map_params_to_p(p_params, initial, False)
    aae(initial, expected)


def test_map_params_to_w():
    params = pd.Series(index=["w"] * 5 + ["bla", "blubb"], data=np.arange(7))
    initial = np.zeros(5)
    _map_params_to_w(params, initial)
    expected = np.arange(5)
    aae(initial, expected)


def test_map_params_to_trans_coeffs():
    factors = ["fac1", "fac2", "fac3"]

    ind_tups = [
        ("trans", 0, "fac1", "lincoeff-fac1"),
        ("trans", 0, "fac1", "lincoeff-fac2"),
        ("trans", 0, "fac1", "lincoeff-constant"),
        ("trans", 0, "fac2", "ar1coeff"),
        ("trans", 0, "fac3", "gamma-fac2"),
        ("trans", 0, "fac3", "gamma-fac3"),
        ("trans", 0, "fac3", "phi"),
        ("trans", 1, "fac1", "lincoeff-fac1"),
        ("trans", 1, "fac1", "lincoeff-fac2"),
        ("trans", 1, "fac1", "lincoeff-constant"),
        ("trans", 1, "fac2", "ar1coeff"),
        ("trans", 1, "fac3", "gamma-fac2"),
        ("trans", 1, "fac3", "gamma-fac3"),
        ("trans", 1, "fac3", "phi"),
    ]

    params = pd.Series(index=pd.MultiIndex.from_tuples(ind_tups), data=np.arange(14))

    initial = [np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((2, 3))]

    expected = [
        np.array([[0, 1, 2], [7, 8, 9.0]]),
        np.array([[3.0], [10]]),
        np.array([[4.0, 5, 6], [11, 12, 13]]),
    ]

    _map_params_to_trans_coeffs(params, initial, factors)

    for arr_calc, arr_exp in zip(initial, expected):
        aae(arr_calc, arr_exp)
