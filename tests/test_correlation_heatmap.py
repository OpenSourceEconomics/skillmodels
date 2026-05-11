"""Tests for correlation heatmap."""

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pandas.testing import assert_frame_equal as afe

from skillmodels.common.correlation_heatmap import (
    _get_mask,
    _get_measurement_data_for_multiple_periods,
    _get_measurement_data_for_single_period,
    _get_quasi_factor_scores_data_for_multiple_periods,
    _get_quasi_factor_scores_data_for_single_period,
    _process_factors,
    get_measurements_corr,
    get_quasi_scores_corr,
    get_scores_corr,
    plot_correlation_heatmap,
)
from skillmodels.common.types import Labels

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def test_get_measurement_data_with_single_period() -> None:
    period = 1
    factors = ["f3", "f1"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, True, False, False, False, False] * 2,
                [False, False, False, False, True, True] * 2,
                [False, False, True, True, False, False] * 2,
            ],
        ).T,
        columns=["f1", "f2", "f3"],
        index=pd.MultiIndex.from_product(
            [[0, 1], [f"y{i}" for i in range(6)]],
            names=["stage", "variable"],
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
            ],
        ).T,
        columns=["stage"] + [f"y{i}" for i in range(6)],
        index=[0, 1] * 2,
    )
    expected = pd.DataFrame(
        np.array([[4, 4], [5, 5], [2, 2], [3, 3], [7, 7]]).T,
        columns=["y2", "y3", "y0", "y1", "y5"],
    )
    result = _get_measurement_data_for_single_period(
        data,
        update_info,
        period,
        latent_factors=factors,
        observed_factors=["y5"],
    )
    afe(result, expected)


def test_get_factor_scores_data_with_single_period() -> None:
    period = 1
    factors = ["f1", "f2"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, False, True, False] * 2,
                [False, True, False, True] * 2,
            ],
        ).T,
        columns=factors,
        index=pd.MultiIndex.from_product(
            [[0, 1], [f"y{i}" for i in range(4)]],
            names=["period", "variable"],
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
                [6, 0, 6, 0],
            ],
        ).T,
        columns=["period"] + [f"y{i}" for i in range(5)],
        index=[0, 1] * 2,
    )
    data_std = data.iloc[:2][[f"y{i}" for i in range(4)]].copy(deep=True)
    for m in data_std.columns:
        data_std[m] = (data_std[m] - np.mean(data_std[m])) / np.std(data_std[m], ddof=1)
    expected = pd.concat(
        [
            data_std["y0"] / 2 + data_std["y2"] / 2,
            data_std["y1"] / 2 + data_std["y3"] / 2,
            data.iloc[:2]["y4"],
        ],
        axis=1,
    )
    expected.columns = ["f1", "f2", "y4"]
    result = _get_quasi_factor_scores_data_for_single_period(
        data,
        update_info,
        period,
        latent_factors=factors,
        observed_factors=["y4"],
    )
    afe(expected, result, check_dtype=False)


def test_get_measurement_data_with_multiple_periods() -> None:
    period = [1, 2]
    factors = ["f3", "f1"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, True, False, False, False, False] * 3,
                [False, False, False, False, True, True] * 3,
                [False, False, True, True, False, False] * 3,
            ],
        ).T,
        columns=["f1", "f2", "f3"],
        index=pd.MultiIndex.from_product(
            [[0, 1, 2], [f"y{i}" for i in range(6)]],
            names=["stage", "variable"],
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
            ],
        ).T,
        columns=["stage"] + [f"y{i}" for i in range(6)],
        index=[0, 1] * 3,
    )
    expected = pd.DataFrame(
        np.array(
            [[4, 4], [5, 5], [2, 2], [3, 3], [-4, -4], [-5, -5], [-2, -2], [-3, -3]],
        ).T,
        columns=[
            "y2, 1",
            "y3, 1",
            "y0, 1",
            "y1, 1",
            "y2, 2",
            "y3, 2",
            "y0, 2",
            "y1, 2",
        ],
    )
    result = _get_measurement_data_for_multiple_periods(
        data,
        update_info,
        period,
        latent_factors=factors,
        observed_factors=[],
    )
    afe(result, expected)


def test_get_factor_scores_data_with_multiple_period() -> None:
    periods = [0, 1]
    factors = ["f1", "f2"]
    update_info = pd.DataFrame(
        np.array(
            [
                [True, False, True, False] * 2,
                [False, True, False, True] * 2,
            ],
        ).T,
        columns=factors,
        index=pd.MultiIndex.from_product(
            [[0, 1], [f"y{i}" for i in range(4)]],
            names=["period", "variable"],
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
            ],
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
            [data_std["y0"] + data_std["y2"], data_std["y1"] + data_std["y3"]],
            axis=1,
        )
        / 2
    )
    temp.columns = ["f1, 0", "f2, 0"]
    to_concat.append(temp.reset_index(drop=True))

    data_std = data.iloc[2:][[f"y{i}" for i in range(4)]].copy(deep=True)
    for m in data_std.columns:
        data_std[m] = (data_std[m] - np.mean(data_std[m])) / np.std(data_std[m], ddof=1)
    temp = (
        pd.concat(
            [data_std["y0"] + data_std["y2"], data_std["y1"] + data_std["y3"]],
            axis=1,
        )
        / 2
    )
    temp.columns = ["f1, 1", "f2, 1"]
    to_concat.append(temp.reset_index(drop=True))

    expected = pd.concat(to_concat, axis=1)
    result = _get_quasi_factor_scores_data_for_multiple_periods(
        data,
        update_info,
        periods,
        latent_factors=factors,
        observed_factors=[],
    )
    afe(expected, result)


def test_process_factors() -> None:
    model = SimpleNamespace(
        labels=Labels(
            latent_factors=tuple("abcd"),
            observed_factors=tuple("efg"),
            controls=("constant",),
            periods=(0,),
            stagemap=(0,),
            stages=(0,),
            aug_periods=(0,),
            aug_periods_to_periods=MappingProxyType({0: 0}),
            aug_stagemap=(0,),
            aug_stages=(0,),
            aug_stages_to_stages=MappingProxyType({0: 0}),
        ),
    )
    latent_factor = "c"
    observed_factor = "g"
    factors = ["b", "d", "g"]
    all_factors = None
    assert tuple("abcd") == _process_factors(model, all_factors)[0]  # ty: ignore[invalid-argument-type]
    assert tuple("efg") == _process_factors(model, all_factors)[1]  # ty: ignore[invalid-argument-type]
    assert (latent_factor,) == _process_factors(model, latent_factor)[0]  # ty: ignore[invalid-argument-type]
    assert (observed_factor,) == _process_factors(model, observed_factor)[1]  # ty: ignore[invalid-argument-type]
    assert tuple(factors[:-1]) == _process_factors(model, factors)[0]  # ty: ignore[invalid-argument-type]
    assert (factors[-1],) == _process_factors(model, factors)[1]  # ty: ignore[invalid-argument-type]


def test_get_mask_lower_triangle_only() -> None:
    corr = pd.DataFrame(np.ones((4, 4)))
    show_upper = False
    show_diag = False
    expected = np.array(
        [
            [False] * 4,
            [True] + [False] * 3,
            [True, True, False, False],
            [True] * 3 + [False],
        ],
    )
    result = _get_mask(corr, show_upper_triangle=show_upper, show_diagonal=show_diag)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_lower_triangle_and_diag() -> None:
    corr = pd.DataFrame(np.ones((4, 4)))
    show_upper = False
    show_diag = True
    expected = np.array(
        [
            [True] + [False] * 3,
            [True] * 2 + [False] * 2,
            [True] * 3 + [False],
            [True] * 4,
        ],
    )
    result = _get_mask(corr, show_upper_triangle=show_upper, show_diagonal=show_diag)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_lower_and_upper_triangle_no_diag() -> None:
    corr = pd.DataFrame(np.ones((4, 4)))
    show_upper = True
    show_diag = False
    expected = np.array(
        [
            [False] + [True] * 3,
            [True] + [False] + [True] * 2,
            [True] * 2 + [False] + [True],
            [True] * 3 + [False],
        ],
    )
    result = _get_mask(corr, show_upper_triangle=show_upper, show_diagonal=show_diag)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_full_square_matrix() -> None:
    corr = pd.DataFrame(np.ones((4, 4)))
    show_upper = True
    show_diag = True
    expected = corr.to_numpy().astype(bool)
    result = _get_mask(corr, show_upper_triangle=show_upper, show_diagonal=show_diag)
    np.testing.assert_array_equal(result, expected)


def _synthetic_corr():
    """Return a synthetic 3x3 correlation DataFrame."""
    data = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]])
    return pd.DataFrame(data, columns=["a", "b", "c"], index=["a", "b", "c"])


def test_plot_correlation_heatmap_basic() -> None:
    corr = _synthetic_corr()
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_correlation_heatmap_no_diagonal() -> None:
    corr = _synthetic_corr()
    fig = plot_correlation_heatmap(corr, show_diagonal=False)
    assert isinstance(fig, go.Figure)


def test_plot_correlation_heatmap_no_upper_triangle() -> None:
    corr = _synthetic_corr()
    fig = plot_correlation_heatmap(corr, show_upper_triangle=False)
    assert isinstance(fig, go.Figure)


def test_plot_correlation_heatmap_annotations() -> None:
    corr = _synthetic_corr()
    fig = plot_correlation_heatmap(corr, annotate=True)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations is not None
    assert len(fig.layout.annotations) > 0


def test_plot_correlation_heatmap_custom_kwargs() -> None:
    corr = _synthetic_corr()
    fig = plot_correlation_heatmap(
        corr,
        heatmap_kwargs={"colorscale": "Viridis"},
        layout_kwargs={"title": "My Heatmap"},
    )
    assert fig.layout.title.text == "My Heatmap"


def test_plot_correlation_heatmap_trim() -> None:
    corr = _synthetic_corr()
    fig = plot_correlation_heatmap(
        corr, trim_heatmap=True, show_upper_triangle=False, show_diagonal=False
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_get_measurements_corr(model2, model2_data) -> None:
    result = get_measurements_corr(
        data=model2_data, model_spec=model2, factors=None, periods=None
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # square


@pytest.mark.integration
def test_get_measurements_corr_single_period(model2, model2_data) -> None:
    result = get_measurements_corr(
        data=model2_data, model_spec=model2, factors=None, periods=0
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]


@pytest.mark.integration
def test_get_quasi_scores_corr(model2, model2_data) -> None:
    result = get_quasi_scores_corr(
        data=model2_data, model_spec=model2, factors=None, periods=None
    )
    assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
def test_get_quasi_scores_corr_single_period(model2, model2_data) -> None:
    result = get_quasi_scores_corr(
        data=model2_data, model_spec=model2, factors=None, periods=0
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]


@pytest.fixture
def vault_params_for_scores():
    """Load vault params with aug_period index level for get_scores_corr."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.rename(columns={"period": "aug_period"})
    return params.set_index(["category", "aug_period", "name1", "name2"])


@pytest.mark.integration
def test_get_scores_corr(model2, model2_data, vault_params_for_scores) -> None:
    result = get_scores_corr(
        data=model2_data,
        params=vault_params_for_scores,
        model_spec=model2,
        factors=None,
        periods=None,
    )
    assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
def test_get_scores_corr_single_period(
    model2, model2_data, vault_params_for_scores
) -> None:
    result = get_scores_corr(
        data=model2_data,
        params=vault_params_for_scores,
        model_spec=model2,
        factors=None,
        periods=0,
    )
    assert isinstance(result, pd.DataFrame)
