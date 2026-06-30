"""Tests for utils_plotting module."""

import numpy as np

from skillmodels.common.utils_plotting import get_layout_kwargs, get_make_subplot_kwargs


def test_get_layout_kwargs_defaults() -> None:
    result = get_layout_kwargs()
    assert result["template"] == "simple_white"
    assert result["showlegend"] is False


def test_get_layout_kwargs_override() -> None:
    result = get_layout_kwargs(layout_kwargs={"title": "X"})
    assert result["title"] == "X"


def test_get_layout_kwargs_with_legend_kwargs() -> None:
    result = get_layout_kwargs(legend_kwargs={"x": 0.5})
    assert np.isclose(result["legend"]["x"], 0.5)


def test_get_layout_kwargs_with_title_kwargs() -> None:
    result = get_layout_kwargs(title_kwargs={"text": "Hello"})
    assert result["title"] == {"text": "Hello"}


def test_get_layout_kwargs_with_rows_and_cols() -> None:
    result = get_layout_kwargs(rows=["a", "b"], columns=["c"])
    assert result["height"] == 600
    assert result["width"] == 300


def test_get_make_subplot_kwargs_defaults() -> None:
    result = get_make_subplot_kwargs(
        sharex=False,
        sharey=False,
        column_order=["a", "b"],
        row_order=["r1"],
        make_subplot_kwargs=None,
    )
    assert result["rows"] == 1
    assert result["cols"] == 2


def test_get_make_subplot_kwargs_override() -> None:
    result = get_make_subplot_kwargs(
        sharex=True,
        sharey=True,
        column_order=["a"],
        row_order=["r1"],
        make_subplot_kwargs={"print_grid": True},
    )
    assert result["print_grid"] is True
