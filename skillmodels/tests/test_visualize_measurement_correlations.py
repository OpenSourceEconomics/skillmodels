import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal as afe

from skillmodels.visualize_measurement_correlations import _get_factors
from skillmodels.visualize_measurement_correlations import _get_mask
from skillmodels.visualize_measurement_correlations import (
    _process_data_for_plotting_with_multiple_periods,
)
from skillmodels.visualize_measurement_correlations import (
    _process_data_for_plotting_with_single_period,
)


def test_process_data_with_single_period():
    period = 1
    factors = ["f3", "f1"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, True, False, False, False, False] * 2,
                [False, False, False, False, True, True] * 2,
                [False, False, True, True, False, False] * 2,
            ]
        ).T,
        columns=["f1", "f2", "f3"],
        index=pd.MultiIndex.from_product(
            [[0, 1], [f"y{i}" for i in range(6)]], names=["stage", "variable"]
        ),
    )
    update_info["purpose"] = "measurement"

    data = pd.DataFrame(
        np.array(
            [
                [1, 1, 0, 0],
                [2, 2, 0, 0],
                [3, 3, 0, 0],
                [4, 4, 0, 0],
                [5, 5, 0, 0],
                [6, 6, 0, 0],
                [7, 7, 0, 0],
            ]
        ).T,
        columns=["stage"] + [f"y{i}" for i in range(6)],
        index=[0, 1] * 2,
    )
    exp = pd.DataFrame(
        np.array([[4, 4], [5, 5], [2, 2], [3, 3]]).T,
        columns=["y2", "y3", "y0", "y1"],
    )
    res = _process_data_for_plotting_with_single_period(
        data, update_info, period, factors
    )
    afe(res, exp)


def test_process_data_with_multiple_periods():
    period = [1, 2]
    factors = ["f3", "f1"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, True, False, False, False, False] * 3,
                [False, False, False, False, True, True] * 3,
                [False, False, True, True, False, False] * 3,
            ]
        ).T,
        columns=["f1", "f2", "f3"],
        index=pd.MultiIndex.from_product(
            [[0, 1, 2], [f"y{i}" for i in range(6)]], names=["stage", "variable"]
        ),
    )
    update_info["purpose"] = "measurement"

    data = pd.DataFrame(
        np.array(
            [
                [1, 1, 0, 0, 2, 2],
                [2, 2, 0, 0, -2, -2],
                [3, 3, 0, 0, -3, -3],
                [4, 4, 0, 0, -4, -4],
                [5, 5, 0, 0, -5, -5],
                [6, 6, 0, 0, -6, -6],
                [7, 7, 0, 0, -7, -7],
            ]
        ).T,
        columns=["stage"] + [f"y{i}" for i in range(6)],
        index=[0, 1] * 3,
    )
    exp = pd.DataFrame(
        np.array(
            [[4, 4], [5, 5], [2, 2], [3, 3], [-4, -4], [-5, -5], [-2, -2], [-3, -3]]
        ).T,
        columns=["y2_1", "y3_1", "y0_1", "y1_1", "y2_2", "y3_2", "y0_2", "y1_2"],
    )
    res = _process_data_for_plotting_with_multiple_periods(
        data, update_info, period, factors
    )
    afe(res, exp)


def test_get_factors():
    model = {"labels": {"all_factors": list("abcd")}}
    factor = "c"
    factors = ["b", "d"]
    all_factors = None
    assert list("abcd") == _get_factors(model, all_factors)
    assert [factor] == _get_factors(model, factor)
    assert factors == _get_factors(model, factors)


def test_get_mask_lower_triangle_only():
    corr = np.ones((4, 4))
    show_upper = False
    show_diag = False
    exp = np.array(
        [
            [False] * 4,
            [True] + [False] * 3,
            [True, True] + [False, False],
            [True] * 3 + [False],
        ]
    )
    res = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(res, exp)


def test_get_mask_lower_triangle_and_diag():
    corr = np.ones((4, 4))
    show_upper = False
    show_diag = True
    exp = np.array(
        [
            [True] + [False] * 3,
            [True] * 2 + [False] * 2,
            [True] * 3 + [False],
            [True] * 4,
        ]
    )
    res = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(res, exp)


def test_get_mask_lower_and_upper_triangle_no_diag():
    corr = np.ones((4, 4))
    show_upper = True
    show_diag = False
    exp = np.array(
        [
            [False] + [True] * 3,
            [True] + [False] + [True] * 2,
            [True] * 2 + [False] + [True],
            [True] * 3 + [False],
        ]
    )
    res = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(res, exp)


def test_get_mask_full_square_matrix():
    corr = np.ones((4, 4))
    show_upper = True
    show_diag = True
    exp = corr.astype(bool)
    res = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(res, exp)
