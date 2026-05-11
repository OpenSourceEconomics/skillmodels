"""Functions to create diagnostic plots for model evaluation."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model


def plot_residual_boxplots(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params: pd.DataFrame,
    period: int | None = None,
    *,
    show_reference_line: bool = True,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure | dict[int, go.Figure]:
    """Create boxplots of measurement residuals by measurement variable.

    Residuals are computed as the difference between observed measurements and
    their predicted values based on filtered states.

    Args:
        model_spec: The model specification.
        data: Empirical dataset used to estimate the model.
        params: Estimated model parameters.
        period: If provided, create a single figure for that period. If None,
            returns a dictionary mapping periods to figures.
        show_reference_line: Whether to show a horizontal reference line at zero.
        layout_kwargs: Dictionary of keyword arguments for Plotly layout.

    Returns:
        If period is specified, returns a single go.Figure. Otherwise, returns
        a dictionary mapping period numbers to figures.

    """
    max_inputs = get_maximization_inputs(model_spec=model_spec, data=data)
    # debug_loglike already returns processed debug data
    processed_debug = max_inputs["debug_loglike"](params)

    processed_model = process_model(model_spec)

    residuals_df = processed_debug["residuals"]
    update_info = processed_model.update_info

    # Get period column name
    period_col = "aug_period"

    # Map measurement names
    residuals_df = residuals_df.merge(
        update_info.reset_index()[["aug_period", "variable"]].rename(
            columns={"variable": "measurement"}
        ),
        left_on=[period_col, "measurement"],
        right_on=["aug_period", "measurement"],
        how="left",
    )

    # Map aug_period → period for the public API
    ap_to_p = processed_model.labels.aug_periods_to_periods

    available_periods = sorted(residuals_df[period_col].unique())

    if period is not None:
        # Find aug_period(s) matching the requested period
        aug_periods_for_period = [ap for ap, p in ap_to_p.items() if p == period]
        aug_period = aug_periods_for_period[0] if aug_periods_for_period else period
        return _create_residual_boxplot_for_period(
            residuals_df=residuals_df,
            period=aug_period,
            period_col=period_col,
            show_reference_line=show_reference_line,
            layout_kwargs=layout_kwargs,
        )

    return {
        ap_to_p.get(p, p): _create_residual_boxplot_for_period(
            residuals_df=residuals_df,
            period=p,
            period_col=period_col,
            show_reference_line=show_reference_line,
            layout_kwargs=layout_kwargs,
        )
        for p in available_periods
    }


def _create_residual_boxplot_for_period(
    residuals_df: pd.DataFrame,
    period: int,
    period_col: str,
    *,
    show_reference_line: bool,
    layout_kwargs: dict[str, Any] | None,
) -> go.Figure:
    """Create a single residual boxplot figure for one period."""
    period_data = residuals_df[residuals_df[period_col] == period]

    measurements = period_data["measurement"].unique()

    fig = go.Figure()

    for measurement in measurements:
        meas_residuals = period_data[period_data["measurement"] == measurement][
            "residual"
        ]
        fig.add_trace(
            go.Box(
                y=meas_residuals,
                name=str(measurement),
                boxpoints=False,
            )
        )

    if show_reference_line:
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)

    default_layout = {
        "title": f"Residuals by Measurement (Period {period})",
        "xaxis_title": "Measurement",
        "yaxis_title": "Residual",
        "showlegend": False,
    }

    if layout_kwargs:
        default_layout.update(layout_kwargs)

    fig.update_layout(**default_layout)

    return fig


def plot_likelihood_contributions(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params: pd.DataFrame,
    period: int | None = None,
    *,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure | dict[int, go.Figure]:
    """Create boxplots of log-likelihood contributions by measurement.

    Args:
        model_spec: The model specification.
        data: Empirical dataset used to estimate the model.
        params: Estimated model parameters.
        period: If provided, create a single figure for that period. If None,
            returns a dictionary mapping periods to figures.
        layout_kwargs: Dictionary of keyword arguments for Plotly layout.

    Returns:
        If period is specified, returns a single go.Figure. Otherwise, returns
        a dictionary mapping period numbers to figures.

    """
    max_inputs = get_maximization_inputs(model_spec=model_spec, data=data)
    # debug_loglike already returns processed debug data
    processed_debug = max_inputs["debug_loglike"](params)

    processed_model = process_model(model_spec)

    contributions_df = processed_debug["all_contributions"]
    update_info = processed_model.update_info

    period_col = "aug_period"

    # Map measurement names
    contributions_df = contributions_df.merge(
        update_info.reset_index()[["aug_period", "variable"]].rename(
            columns={"variable": "measurement"}
        ),
        left_on=[period_col, "measurement"],
        right_on=["aug_period", "measurement"],
        how="left",
    )

    # Map aug_period → period for the public API
    ap_to_p = processed_model.labels.aug_periods_to_periods

    available_periods = sorted(contributions_df[period_col].unique())

    if period is not None:
        # Find aug_period(s) matching the requested period
        aug_periods_for_period = [ap for ap, p in ap_to_p.items() if p == period]
        aug_period = aug_periods_for_period[0] if aug_periods_for_period else period
        return _create_likelihood_boxplot_for_period(
            contributions_df=contributions_df,
            period=aug_period,
            period_col=period_col,
            layout_kwargs=layout_kwargs,
        )

    return {
        ap_to_p.get(p, p): _create_likelihood_boxplot_for_period(
            contributions_df=contributions_df,
            period=p,
            period_col=period_col,
            layout_kwargs=layout_kwargs,
        )
        for p in available_periods
    }


def _create_likelihood_boxplot_for_period(
    contributions_df: pd.DataFrame,
    period: int,
    period_col: str,
    layout_kwargs: dict[str, Any] | None,
) -> go.Figure:
    """Create a single likelihood contribution boxplot figure for one period."""
    period_data = contributions_df[contributions_df[period_col] == period]

    measurements = period_data["measurement"].unique()

    fig = go.Figure()

    for measurement in measurements:
        meas_contribs = period_data[period_data["measurement"] == measurement][
            "contribution"
        ]
        # Filter out -inf values for visualization
        meas_contribs = meas_contribs.replace([np.inf, -np.inf], np.nan).dropna()
        fig.add_trace(
            go.Box(
                y=meas_contribs,
                name=str(measurement),
                boxpoints=False,
            )
        )

    default_layout = {
        "title": f"Log-Likelihood Contributions (Period {period})",
        "xaxis_title": "Measurement",
        "yaxis_title": "Log-Likelihood Contribution",
        "showlegend": False,
    }

    if layout_kwargs:
        default_layout.update(layout_kwargs)

    fig.update_layout(**default_layout)

    return fig
