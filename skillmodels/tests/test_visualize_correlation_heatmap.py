import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal as afe

from skillmodels.visualize_correlation_heatmap import _get_factors
from skillmodels.visualize_correlation_heatmap import _get_mask
from skillmodels.visualize_correlation_heatmap import (
    _get_measurement_data_for_multiple_periods,
)
from skillmodels.visualize_correlation_heatmap import (
    _get_measurement_data_for_single_period,
)
from skillmodels.visualize_correlation_heatmap import (
    _get_quasi_factor_scores_data_for_multiple_periods,
)
from skillmodels.visualize_correlation_heatmap import (
    _get_quasi_factor_scores_data_for_single_period,
)


def test_get_measurement_data_with_single_period():
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
    expected = pd.DataFrame(
        np.array([[4, 4], [5, 5], [2, 2], [3, 3]]).T,
        columns=["y2", "y3", "y0", "y1"],
    )
    result = _get_measurement_data_for_single_period(data, update_info, period, factors)
    afe(result, expected)


def test_get_factor_scores_data_with_single_period():
    period = 1
    factors = ["f1", "f2"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, False, True, False] * 2,
                [False, True, False, True] * 2,
            ]
        ).T,
        columns=factors,
        index=pd.MultiIndex.from_product(
            [[0, 1], [f"y{i}" for i in range(4)]], names=["period", "variable"]
        ),
    )
    update_info["purpose"] = "measurement"
    data = pd.DataFrame(
        np.array(
            [
                [1, 1, 0, 0],
                [2, 0, 2, 0],
                [3, 0, 3, 0],
                [4, 0, 4, 0],
                [5, 0, 5, 0],
            ]
        ).T,
        columns=["period"] + [f"y{i}" for i in range(4)],
        index=[0, 1] * 2,
    )
    data_std = data.iloc[:2][[f"y{i}" for i in range(4)]].copy(deep=True)
    for m in data_std.columns:
        data_std[m] = (data_std[m] - np.mean(data_std[m])) / np.std(data_std[m], ddof=1)
    expected = (
        pd.concat(
            [data_std["y0"] + data_std["y2"], data_std["y1"] + data_std["y3"]], axis=1
        )
        / 2
    )
    expected.columns = ["f1", "f2"]
    result = _get_quasi_factor_scores_data_for_single_period(
        data, update_info, period, factors
    )
    afe(expected, result)


def test_get_measurement_data_with_multiple_periods():
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
    expected = pd.DataFrame(
        np.array(
            [[4, 4], [5, 5], [2, 2], [3, 3], [-4, -4], [-5, -5], [-2, -2], [-3, -3]]
        ).T,
        columns=["y2_1", "y3_1", "y0_1", "y1_1", "y2_2", "y3_2", "y0_2", "y1_2"],
    )
    result = _get_measurement_data_for_multiple_periods(
        data, update_info, period, factors
    )
    afe(result, expected)


def test_get_factor_scores_data_with_multiple_period():
    periods = [0, 1]
    factors = ["f1", "f2"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, False, True, False] * 2,
                [False, True, False, True] * 2,
            ]
        ).T,
        columns=factors,
        index=pd.MultiIndex.from_product(
            [[0, 1], [f"y{i}" for i in range(4)]], names=["period", "variable"]
        ),
    )
    update_info["purpose"] = "measurement"
    data = pd.DataFrame(
        np.array(
            [
                [1, 1, 0, 0],
                [2, 0, 2, 0],
                [3, 0, 3, 0],
                [4, 0, 4, 0],
                [5, 0, 5, 0],
            ]
        ).T,
        columns=["period"] + [f"y{i}" for i in range(4)],
        index=[0, 1] * 2,
    )
    to_concat = []

    data_std = data.iloc[:2][[f"y{i}" for i in range(4)]].copy(deep=True)
    for m in data_std.columns:
        data_std[m] = (data_std[m] - np.mean(data_std[m])) / np.std(data_std[m], ddof=1)
    temp = (
        pd.concat(
            [data_std["y0"] + data_std["y2"], data_std["y1"] + data_std["y3"]], axis=1
        )
        / 2
    )
    temp.columns = ["f1_0", "f2_0"]
    to_concat.append(temp.reset_index(drop=True))

    data_std = data.iloc[2:][[f"y{i}" for i in range(4)]].copy(deep=True)
    for m in data_std.columns:
        data_std[m] = (data_std[m] - np.mean(data_std[m])) / np.std(data_std[m], ddof=1)
    temp = (
        pd.concat(
            [data_std["y0"] + data_std["y2"], data_std["y1"] + data_std["y3"]], axis=1
        )
        / 2
    )
    temp.columns = ["f1_1", "f2_1"]
    to_concat.append(temp.reset_index(drop=True))

    expected = pd.concat(to_concat, axis=1)
    result = _get_quasi_factor_scores_data_for_multiple_periods(
        data, update_info, periods, factors
    )
    afe(expected, result)


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
    expected = np.array(
        [
            [False] * 4,
            [True] + [False] * 3,
            [True, True] + [False, False],
            [True] * 3 + [False],
        ]
    )
    result = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_lower_triangle_and_diag():
    corr = np.ones((4, 4))
    show_upper = False
    show_diag = True
    expected = np.array(
        [
            [True] + [False] * 3,
            [True] * 2 + [False] * 2,
            [True] * 3 + [False],
            [True] * 4,
        ]
    )
    result = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_lower_and_upper_triangle_no_diag():
    corr = np.ones((4, 4))
    show_upper = True
    show_diag = False
    expected = np.array(
        [
            [False] + [True] * 3,
            [True] + [False] + [True] * 2,
            [True] * 2 + [False] + [True],
            [True] * 3 + [False],
        ]
    )
    result = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_full_square_matrix():
    corr = np.ones((4, 4))
    show_upper = True
    show_diag = True
    expected = corr.astype(bool)
    result = _get_mask(corr, show_upper, show_diag)
    np.testing.assert_array_equal(result, expected)
