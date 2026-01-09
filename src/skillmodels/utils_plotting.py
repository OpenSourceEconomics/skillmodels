"""Utility functions for configuring plot layouts and subplots."""

from typing import Any

import numpy as np


def get_layout_kwargs(
    layout_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    *,
    showlegend: bool = False,
    columns: list[str] | None = None,
    rows: list[str] | None = None,
) -> dict[str, Any]:
    """Define and update default kwargs for update_layout.

    Defines some default keyword arguments to update figure layout, such as
    title and legend.

    """
    default_kwargs = {
        "template": "simple_white",
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "legend": {},
        "title": {},
        "showlegend": showlegend,
    }
    if rows is not None:
        default_kwargs["height"] = 300 * len(rows)
    if columns is not None:
        default_kwargs["width"] = 300 * len(columns)
    if title_kwargs:
        default_kwargs["title"] = title_kwargs
    if legend_kwargs:
        default_kwargs["legend"].update(legend_kwargs)  # ty: ignore[possibly-missing-attribute]
    if layout_kwargs:
        default_kwargs.update(layout_kwargs)
    return default_kwargs


def get_make_subplot_kwargs(
    *,
    sharex: bool,
    sharey: bool,
    column_order: list[str],
    row_order: list[str],
    make_subplot_kwargs: dict[str, Any] | None,
    add_scenes: bool = False,
) -> dict[str, Any]:
    """Define and update keywargs for instantiating figure with subplots."""
    nrows = len(row_order)
    ncols = len(column_order)
    default_kwargs = {
        "rows": nrows,
        "cols": ncols,
        "start_cell": "top-left",
        "print_grid": False,
        "shared_yaxes": sharey,
        "shared_xaxes": sharex,
        "horizontal_spacing": 1 / (ncols * 6),
    }
    if nrows > 1:
        default_kwargs["vertical_spacing"] = (1 / (nrows - 1)) / 4
    if not sharey:
        default_kwargs["horizontal_spacing"] = 2 * default_kwargs["horizontal_spacing"]
    if add_scenes:
        specs = np.array([[{}] * ncols] * nrows)
        for i in range(nrows):
            for j in range(ncols):
                if j > i:
                    specs[i, j] = {"type": "scene"}
        default_kwargs["specs"] = specs.tolist()
    if make_subplot_kwargs is not None:
        default_kwargs.update(make_subplot_kwargs)
    return default_kwargs
