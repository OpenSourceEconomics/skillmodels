"""Functions to visualize transition equations and production functions."""

import itertools
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from skillmodels.filtered_states import get_filtered_states
from skillmodels.model_spec import ModelSpec
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info, parse_params
from skillmodels.process_data import process_data
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model
from skillmodels.types import ParsedParams, ProcessedModel
from skillmodels.utils_plotting import get_layout_kwargs, get_make_subplot_kwargs


def combine_transition_plots(
    plots_dict: dict[tuple[str, str], go.Figure],
    column_order: list[str] | tuple[str, ...] | str | None = None,
    row_order: list[str] | tuple[str, ...] | str | None = None,
    factor_mapping: dict[str, str] | None = None,
    make_subplot_kwargs: dict[str, Any] | None = None,
    *,
    sharex: bool = False,
    sharey: bool = True,
    showlegend: bool = True,
    layout_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Combine individual plots into figure with subplots.

    Use dictionary with plotly images as values to build plotly figure with subplots.

    Args:
        plots_dict: Dictionary with plots of transition functions for each
            factor.
        column_order: List of (output) factor names according
            to which transition plots should be ordered horizontally. If None, infer
            from the keys of of plots_dict
        row_order: List of (input) factor names according
            to which transition plots should be ordered vertically. If None, infer
            from the keys of of plots_dict
        factor_mapping: A dictionary with custom factor names to
            display as axes labels.
        make_subplot_kwargs: Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots. If None,
            default arguments defined in the function are used.
        sharex: Whether to share the properties of x-axis across subplots.
            Default False.
        sharey: Whether to share the properties ofy-axis across subplots.
            Default True.
        showlegend: Display legend if True.
        layout_kwargs: Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs: Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs: Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title. If None, infers title based on the value of
            `quntiles_of_other_factors`.

    Returns:
        fig: Plotly figure with subplots that combines individual
            transition functions.

    """
    plots_dict = deepcopy(plots_dict)

    column_order, row_order = _process_orders(
        columns=column_order, rows=row_order, plots_dict=plots_dict
    )
    make_subplot_kwargs = get_make_subplot_kwargs(
        sharex=sharex,
        sharey=sharey,
        column_order=column_order,
        row_order=row_order,
        make_subplot_kwargs=make_subplot_kwargs,
    )
    factor_mapping = _process_factor_mapping_trans(
        factor_mapper=factor_mapping,
        output_factors=row_order,
        input_factors=column_order,
    )
    fig = make_subplots(**make_subplot_kwargs)
    for (output_factor, input_factor), (row, col) in zip(
        itertools.product(row_order, column_order),
        itertools.product(np.arange(len(row_order)), np.arange(len(column_order))),
        strict=False,
    ):
        try:
            subfig = plots_dict[(input_factor, output_factor)]
        except KeyError:
            subfig = go.Figure()
        if not (row == 0 and col == 0):
            for d in subfig.data:
                d.update({"showlegend": False})
                fig.add_trace(d, col=col + 1, row=row + 1)
        else:
            for d in subfig.data:
                fig.add_trace(
                    d,
                    col=col + 1,
                    row=row + 1,
                )
        fig.update_xaxes(
            title_text=f"{factor_mapping[input_factor]}",
            row=row + 1,
            col=col + 1,
        )
        if col == 0:
            fig.update_yaxes(
                title_text=f"{factor_mapping[output_factor]}",
                row=row + 1,
                col=col + 1,
            )

    layout_kwargs = get_layout_kwargs(
        layout_kwargs=layout_kwargs,
        legend_kwargs=legend_kwargs,
        title_kwargs=title_kwargs,
        showlegend=showlegend,
        columns=column_order,
        rows=row_order,
    )
    fig.update_layout(**layout_kwargs)
    return fig


def get_transition_plots(  # noqa: C901, PLR0912
    model_spec: ModelSpec,
    params: pd.DataFrame,
    data: pd.DataFrame | None = None,
    period: int | None = None,
    periods: Sequence[int] | None = None,
    state_ranges: dict[str, pd.DataFrame] | None = None,
    quantiles_of_other_factors: tuple[float, ...] | list[float] | float | None = (
        0.25,
        0.5,
        0.75,
    ),
    aggregation_method: Literal["quantiles", "median", "average"] = "quantiles",
    n_points: int = 50,
    n_draws: int = 50,
    colorscale: str | list[str] = "Magenta_r",
    state_range_quantile_cutoff: float | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    *,
    states: pd.DataFrame | None = None,
    include_correction_factors: bool = False,
) -> dict[tuple[str, str], go.Figure]:
    """Get dictionary with individual plots of transition equations for each factor.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
        params: Model parameters.
        data: Empirical dataset used to estimate the model. Required when `states`
            is not provided or when the model has observed factors.
        period: The start period of the transition equations that are plotted.
            Deprecated in favor of `periods`. If both are provided, `periods` is used.
        periods: List of periods to overlay on each plot. Each period gets a different
            color. If None and period is None, uses the first valid period.
        state_ranges: The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        quantiles_of_other_factors: Quantiles at which the factors
            that are not varied in a given plot are fixed. Only used when
            aggregation_method is "quantiles". If None, those factors are
            not fixed but integrated out (equivalent to aggregation_method="average").
        aggregation_method: How to handle factors not being varied in a plot.
            - "quantiles": Fix other factors at specified quantiles (default).
            - "median": Fix other factors at their median (0.5 quantile).
            - "average": Average over random draws of other factors.
        n_points: Number of grid points per input. Default 50.
        n_draws: Number of randomly drawn values of the factors that are averaged
            out. Only relevant if aggregation_method is "average". Default 50.
        colorscale: The color scale to use for line legends. Can be a string
            (plotly.express.colors.sequential attribute) or a list of colors.
            Default 'Magenta_r'.
        state_range_quantile_cutoff: If provided, compute state ranges by cutting
            this quantile from both tails (e.g., 0.01 cuts 1st and 99th percentiles).
            Only used if state_ranges is None.
        layout_kwargs: Dictionary of key word arguments used to
            update layout of plotly image object. If None, the default kwargs
            defined in the function will be used.
        states: Pre-computed filtered states DataFrame (with a `period`
            column). If provided, skip the internal `get_filtered_states` call.
        include_correction_factors: Whether to include correction factors in the
            plots. Default False.

    Returns:
        plots_dict: Dictionary with individual plots of transition equations
            for each combination of input and output factors.

    """
    # Handle period/periods arguments
    if periods is not None:
        periods_list = list(periods)
    elif period is not None:
        periods_list = [period]
    else:
        periods_list = None  # Will be set after processing model

    # Convert aggregation_method to quantiles_of_other_factors format
    if aggregation_method == "median":
        quantiles_of_other_factors = [0.5]
    elif aggregation_method == "average":
        quantiles_of_other_factors = None
    else:  # "quantiles"
        quantiles_of_other_factors = _process_quantiles_of_other_factors(
            quantiles_of_other_factors,
        )

    processed_model = process_model(model_spec)

    # Set default periods if not provided
    if periods_list is None:
        periods_list = [0]

    # Validate periods
    max_period = processed_model.labels.periods[-1]
    for p in periods_list:
        if p >= max_period:
            raise ValueError(
                f"Period {p} is invalid. Must be less than {max_period} "
                "(the last period has no transition).",
            )

    if (
        include_correction_factors
        or not processed_model.endogenous_factors_info.has_endogenous_factors
    ):
        latent_factors = processed_model.labels.latent_factors
    else:
        latent_factors = [
            lf
            for lf in processed_model.labels.latent_factors
            if not processed_model.endogenous_factors_info.factor_info[lf].is_correction
        ]
    all_factors = processed_model.labels.all_factors
    if states is None:
        if data is None:
            msg = "Either 'data' or 'states' must be provided."
            raise TypeError(msg)
        states = get_filtered_states(model_spec=model_spec, data=data, params=params)[
            "anchored_states"
        ]["states"]

    states = _normalize_states_columns(
        states,
        aug_periods_to_periods=processed_model.labels.aug_periods_to_periods,
    )

    return _get_dictionary_with_plots(
        model=processed_model,
        data=data,
        params=params,
        states=states,
        state_ranges=state_ranges,
        latent_factors=latent_factors,  # ty: ignore[invalid-argument-type]
        all_factors=all_factors,
        quantiles_of_other_factors=quantiles_of_other_factors,
        periods=periods_list,
        n_points=n_points,
        n_draws=n_draws,
        colorscale=colorscale,
        state_range_quantile_cutoff=state_range_quantile_cutoff,
        layout_kwargs=layout_kwargs,
    )


def _get_dictionary_with_plots(
    model: ProcessedModel,
    data: pd.DataFrame | None,
    params: pd.DataFrame,
    states: pd.DataFrame,
    state_ranges: dict[str, pd.DataFrame] | None,
    latent_factors: tuple[str, ...],
    all_factors: tuple[str, ...],
    quantiles_of_other_factors: list[float] | None,
    periods: list[int],
    n_points: int,
    n_draws: int,
    colorscale: str | list[str],
    state_range_quantile_cutoff: float | None,
    layout_kwargs: dict[str, Any] | None,
    *,
    showlegend: bool = True,
) -> dict[tuple[str, str], go.Figure]:
    """Get plots of transition functions for each input and output combination.

    Return a dictionary with individual plots of transition functions for each input
    and output factors.

    Args:
        model: The model specification. See: :ref:`model_specs`
        data: Panel dataset in long format for getting observed factors.
        params: DataFrame with model parameters.
        states: Tidy DataFrame with filtered or simulated states.
            They are used to estimate the state ranges in each period (if state_ranges
            are not given explicitly) and to estimate the distribution of the factors
            that are not visualized.
        state_ranges: The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        latent_factors: Latent factors of the model that are outputs of
            transition factors.
        all_factors: All factors of the model that are the inputs of transition
            functions.
        quantiles_of_other_factors: Quantiles at which the factors
            that are not varied in a given plot are fixed. If None, those factors are
            not fixed but integrated out.
        periods: List of periods to plot. Each period gets a different line/color.
        n_points: Number of grid points per input. Default 50.
        n_draws: Number of randomly drawn values of the factors that are averaged
            out. Only relevant if quantiles_of_other_factors is *None*. Default 50.
        colorscale: The color scale to use for line legends. Can be a string
            (plotly.express.colors.sequential attribute) or a list of colors.
        state_range_quantile_cutoff: If provided, compute state ranges by cutting
            this quantile from both tails.
        layout_kwargs: Dictionary of key word arguments used to
            update layout of plotly image object. If None, the default kwargs defined
            in the function will be used.
        showlegend: Display legend if True. Default True.

    Returns:
        plots_dict: Dictionary with individual plots of transition functions
            for each input and output factors.

    """
    observed_factors = model.labels.observed_factors

    # Collect states data for all periods
    all_states_data = []
    for period in periods:
        period_states = _get_states_data(
            model=model,
            period=period,
            data=data,
            states=states,
            observed_factors=observed_factors,
        )
        all_states_data.append(period_states)
    states_data = pd.concat(all_states_data, ignore_index=True)

    params = _set_index_params(model=model, params=params)
    parsed_params = _get_parsed_params(model=model, params=params)
    state_ranges = _get_state_ranges(
        state_ranges=state_ranges,
        states_data=states_data,
        all_factors=all_factors,
        quantile_cutoff=state_range_quantile_cutoff,
    )
    layout_kwargs = get_layout_kwargs(
        layout_kwargs=layout_kwargs,
        legend_kwargs=None,
        title_kwargs=None,
        showlegend=showlegend,
    )

    # Get color sequence
    if isinstance(colorscale, str):
        colors = getattr(px.colors.sequential, colorscale)
    else:
        colors = colorscale

    plots_dict = {}
    for output_factor, input_factor in itertools.product(latent_factors, all_factors):
        combined_data = _prepare_plot_data_for_factor_pair(
            model=model,
            states_data=states_data,
            state_ranges=state_ranges,
            parsed_params=parsed_params,
            periods=periods,
            input_factor=input_factor,
            output_factor=output_factor,
            all_factors=all_factors,
            quantiles_of_other_factors=quantiles_of_other_factors,
            n_points=n_points,
            n_draws=n_draws,
        )

        color = _determine_color_column(len(periods), quantiles_of_other_factors)

        subfig = px.line(
            combined_data,
            y=f"output_{output_factor}",
            x=f"input_{input_factor}",
            color=color,
            color_discrete_sequence=colors,
        )
        subfig.update_xaxes(title={"text": input_factor})
        subfig.update_yaxes(title={"text": output_factor})
        subfig.update_traces(line={"width": 2})
        subfig.update_layout(**layout_kwargs)
        plots_dict[(input_factor, output_factor)] = deepcopy(subfig)

    return plots_dict


def _prepare_plot_data_for_factor_pair(
    model: ProcessedModel,
    states_data: pd.DataFrame,
    state_ranges: dict[str, pd.DataFrame],
    parsed_params: ParsedParams,
    periods: list[int],
    input_factor: str,
    output_factor: str,
    all_factors: tuple[str, ...],
    quantiles_of_other_factors: list[float] | None,
    n_points: int,
    n_draws: int,
) -> pd.DataFrame:
    """Prepare plot data for one input/output factor pair across all periods."""
    transition_function = model.transition_info.individual_functions[output_factor]
    has_endogenous_factors = model.endogenous_factors_info.has_endogenous_factors

    period_data_frames = []
    for period in periods:
        aug_period = _get_aug_period_for_output(
            model=model,
            period=period,
            output_factor=output_factor,
            has_endogenous_factors=has_endogenous_factors,
        )

        transition_params = {
            output_factor: parsed_params.transition[output_factor][aug_period]
        }
        period_states = states_data[states_data["aug_period"] == aug_period]

        plot_data = _prepare_single_period_plot_data(
            states_data=period_states,
            state_ranges=state_ranges,
            aug_period=aug_period,
            input_factor=input_factor,
            output_factor=output_factor,
            all_factors=all_factors,
            quantiles_of_other_factors=quantiles_of_other_factors,
            n_points=n_points,
            n_draws=n_draws,
            transition_function=transition_function,
            transition_params=transition_params,
        )
        plot_data["period"] = period
        period_data_frames.append(plot_data)

    return pd.concat(period_data_frames, ignore_index=True)


def _get_aug_period_for_output(
    model: ProcessedModel,
    period: int,
    output_factor: str,
    *,
    has_endogenous_factors: bool,
) -> int:
    """Determine the augmented period for an output factor."""
    if not has_endogenous_factors:
        return period

    _aug_periods = model.endogenous_factors_info.aug_periods_from_period(period)
    if model.endogenous_factors_info.factor_info[output_factor].is_endogenous:
        return min(_aug_periods)
    return max(_aug_periods)


def _prepare_single_period_plot_data(
    states_data: pd.DataFrame,
    state_ranges: dict[str, pd.DataFrame],
    aug_period: int,
    input_factor: str,
    output_factor: str,
    all_factors: tuple[str, ...],
    quantiles_of_other_factors: list[float] | None,
    n_points: int,
    n_draws: int,
    transition_function: Callable[..., Array],
    transition_params: dict[str, Any],
) -> pd.DataFrame:
    """Prepare plot data for a single period."""
    if quantiles_of_other_factors is not None:
        return _prepare_data_for_one_plot_fixed_quantile_2d(
            states_data=states_data,
            state_ranges=state_ranges,
            aug_period=aug_period,
            input_factor=input_factor,
            output_factor=output_factor,
            n_points=n_points,
            quantiles_of_other_factors=quantiles_of_other_factors,
            transition_function=transition_function,
            transition_params=transition_params,
            all_factors=all_factors,
        )
    return _prepare_data_for_one_plot_average_2d(
        states_data=states_data,
        state_ranges=state_ranges,
        aug_period=aug_period,
        input_factor=input_factor,
        output_factor=output_factor,
        n_points=n_points,
        n_draws=n_draws,
        transition_function=transition_function,
        transition_params=transition_params,
        all_factors=all_factors,
    )


def _determine_color_column(
    n_periods: int,
    quantiles_of_other_factors: list[float] | None,
) -> str | None:
    """Determine which column to use for color encoding."""
    if n_periods > 1:
        return "period"
    if (
        isinstance(quantiles_of_other_factors, list)
        and len(quantiles_of_other_factors) > 1
    ):
        return "quantile"
    return None


def _get_state_ranges(
    state_ranges: dict[str, pd.DataFrame] | None,
    states_data: pd.DataFrame,
    all_factors: tuple[str, ...],
    quantile_cutoff: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Create state ranges if none is given."""
    if state_ranges is None:
        state_ranges = create_state_ranges(
            filtered_states=states_data,
            factors=list(all_factors),
            quantile_cutoff=quantile_cutoff,
        )
    return state_ranges


def _get_parsed_params(
    model: ProcessedModel,
    params: pd.DataFrame,
) -> ParsedParams:
    """Get parsed params dataclass."""
    parsing_info = create_parsing_info(
        params_index=params.index,  # ty: ignore[invalid-argument-type]
        update_info=model.update_info,
        labels=model.labels,
        anchoring=model.anchoring,
        has_endogenous_factors=model.endogenous_factors_info.has_endogenous_factors,
    )

    _, _, _, parsed_params = parse_params(
        params=jnp.array(params["value"].to_numpy()),
        parsing_info=parsing_info,
        dimensions=model.dimensions,
        labels=model.labels,
        n_obs=1,
    )
    return parsed_params


def _set_index_params(
    model: ProcessedModel,
    params: pd.DataFrame,
) -> pd.DataFrame:
    """Reset index of params data frame to model implied values."""
    params_index = get_params_index(
        update_info=model.update_info,
        labels=model.labels,
        dimensions=model.dimensions,
        transition_info=model.transition_info,
        endogenous_factors_info=model.endogenous_factors_info,
    )

    return params.reindex(params_index)


def _get_states_data(
    model: ProcessedModel,
    period: int,
    data: pd.DataFrame | None,
    states: pd.DataFrame,
    observed_factors: tuple[str, ...],
) -> pd.DataFrame:
    if observed_factors:
        if data is None:
            msg = (
                "The model has observed factors. You must pass the empirical data to "
                "'get_transition_plots' via the keyword 'data'."
            )
            raise TypeError(msg)
        _observed_arr = process_data(
            df=data,
            has_endogenous_factors=model.endogenous_factors_info.has_endogenous_factors,
            labels=model.labels,
            update_info=model.update_info,
            anchoring_info=model.anchoring,
        )["observed_factors"]
        # convert from jax to numpy
        _observed_arr = np.array(_observed_arr)
        if model.endogenous_factors_info.has_endogenous_factors:
            both_aug_periods = [
                aug_p
                for aug_p, p in model.labels.aug_periods_to_periods.items()
                if p == period
            ]
            to_concat = []
            for aug_p in both_aug_periods:
                df = pd.DataFrame(
                    data=_observed_arr[aug_p],
                    columns=observed_factors,
                )
                df["id"] = df.index
                df["aug_period"] = aug_p
                to_concat.append(df)
            observed_data = pd.concat(to_concat)
        else:
            observed_data = pd.DataFrame(
                data=_observed_arr[period],
                columns=observed_factors,
            )
            observed_data["id"] = observed_data.index
            observed_data["aug_period"] = period
        # Do a left merge because we need all periods for the ranges
        states_data = states.merge(
            observed_data,
            left_on=["id", "aug_period"],
            right_on=["id", "aug_period"],
            how="left",
        )
    else:
        states_data = states.copy(deep=True)
    return states_data


def _normalize_states_columns(
    states: pd.DataFrame,
    aug_periods_to_periods: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Ensure `aug_period` and `id` are columns, not index levels.

    Pre-computed states DataFrames may carry period information as `period`
    (in the index or a column) instead of `aug_period`.  Downstream code
    uniformly expects `aug_period` as a column, so this helper promotes
    index levels to columns and, when a mapping is provided, expands each
    period row into one row per corresponding aug_period.

    Args:
        states: DataFrame with latent factor columns and either `period` or
            `aug_period` identifying the time dimension.
        aug_periods_to_periods: Mapping from aug_period to period.  When
            provided and the DataFrame has `period` but not `aug_period`,
            rows are expanded so that each period produces one row per
            aug_period that maps to it.
    """
    # Promote relevant index levels to columns.
    names_to_reset = [
        n for n in states.index.names if n in ("period", "aug_period", "id")
    ]
    if names_to_reset:
        states = states.reset_index(level=names_to_reset)

    if "aug_period" in states.columns:
        return states

    if "period" not in states.columns:
        return states

    # Expand period rows into aug_period rows using the mapping.
    if aug_periods_to_periods is not None:
        mapping_df = pd.DataFrame(
            list(aug_periods_to_periods.items()),
            columns=["aug_period", "period"],
        )
        states = states.merge(mapping_df, on="period", how="left")
        states = states.drop(columns=["period"])
    else:
        states = states.rename(columns={"period": "aug_period"})

    return states


def _prepare_data_for_one_plot_fixed_quantile_2d(
    states_data: pd.DataFrame,
    state_ranges: dict[str, pd.DataFrame],
    aug_period: int,
    input_factor: str,
    output_factor: str,
    n_points: int,
    quantiles_of_other_factors: list[float],
    transition_function: Callable[..., Array],
    transition_params: dict[str, Any],
    all_factors: tuple[str, ...],
) -> pd.DataFrame:
    period_data = states_data.query(f"aug_period == {aug_period}")[list(all_factors)]
    input_min = state_ranges[input_factor].loc[aug_period]["minimum"]
    input_max = state_ranges[input_factor].loc[aug_period]["maximum"]
    to_concat = []
    for quantile in quantiles_of_other_factors:
        input_data = pd.DataFrame()
        input_data[input_factor] = np.linspace(input_min, input_max, n_points)
        fixed_quantiles = period_data.drop(columns=input_factor).quantile(quantile)
        for col, val in fixed_quantiles.items():
            input_data[col] = val
        input_arr = jnp.array(input_data[list(all_factors)].to_numpy())
        # convert from jax to numpy array
        output_arr = np.array(transition_function(transition_params, input_arr))
        quantile_data = pd.DataFrame()
        quantile_data[f"input_{input_factor}"] = input_data[input_factor]
        quantile_data[f"output_{output_factor}"] = np.array(output_arr)
        quantile_data["quantile"] = quantile
        to_concat.append(quantile_data)

    return pd.concat(to_concat).reset_index()


def _process_quantiles_of_other_factors(
    quantiles_of_other_factors: tuple[float, ...] | list[float] | float | None,
) -> list[float] | None:
    """Process quantiles of other factors to always have list as type."""
    if isinstance(quantiles_of_other_factors, float | int):
        quantiles_of_other_factors = [quantiles_of_other_factors]
    elif isinstance(quantiles_of_other_factors, tuple | list):
        quantiles_of_other_factors = list(quantiles_of_other_factors)
    return quantiles_of_other_factors


def _prepare_data_for_one_plot_average_2d(
    states_data: pd.DataFrame,
    state_ranges: dict[str, pd.DataFrame],
    aug_period: int,
    input_factor: str,
    output_factor: str,
    n_points: int,
    n_draws: int,
    transition_function: Callable[..., Array],
    transition_params: dict[str, Any],
    all_factors: tuple[str, ...],
) -> pd.DataFrame:
    period_data = states_data.query(f"aug_period == {aug_period}")

    sampled_factors = [factor for factor in all_factors if factor != input_factor]
    draws = period_data[sampled_factors].sample(n=n_draws)
    input_min = state_ranges[input_factor].loc[aug_period]["minimum"]
    input_max = state_ranges[input_factor].loc[aug_period]["maximum"]

    input_grid = np.linspace(input_min, input_max, n_points)
    draws_arr = draws.to_numpy()  # (n_draws, n_sampled_factors)

    # Build (n_draws * n_points, n_factors) array with broadcasting
    tiled_input = np.tile(input_grid, n_draws)
    repeated_draws = np.repeat(draws_arr, n_points, axis=0)

    full_arr = np.empty((n_draws * n_points, len(all_factors)))
    for i, factor in enumerate(all_factors):
        if factor == input_factor:
            full_arr[:, i] = tiled_input
        else:
            col_idx = sampled_factors.index(factor)
            full_arr[:, i] = repeated_draws[:, col_idx]

    output_arr = np.array(
        transition_function(transition_params, jnp.array(full_arr)),
    )
    output_mean = output_arr.reshape(n_draws, n_points).mean(axis=0)

    return pd.DataFrame(
        {
            f"input_{input_factor}": input_grid,
            f"output_{output_factor}": output_mean,
        }
    )


def _process_factor_mapping_trans(
    factor_mapper: dict[str, str] | None,
    output_factors: tuple[str, ...],
    input_factors: tuple[str, ...],
) -> dict[str, str]:
    """Process mapper to return dictionary with old and new factor names."""
    all_factors = input_factors + output_factors
    if factor_mapper is None:
        factor_mapper = {fac: fac for fac in all_factors}
    else:
        for fac in all_factors:
            if fac not in factor_mapper:
                factor_mapper[fac] = fac
    return factor_mapper


def _process_orders(
    columns: list[str] | tuple[str, ...] | str | None,
    rows: list[str] | tuple[str, ...] | str | None,
    plots_dict: dict[tuple[str, str], go.Figure],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Process axes orders to return tuples of strings."""
    out_columns: tuple[str, ...]
    out_rows: tuple[str, ...]
    if columns is None:
        seen: list[str] = []
        for f in plots_dict:
            if f[0] not in seen:
                seen.append(f[0])
        out_columns = tuple(seen)
    elif isinstance(columns, str):
        out_columns = (columns,)
    else:
        out_columns = tuple(columns)
    if rows is None:
        seen = []
        for f in plots_dict:
            if f[1] not in seen:
                seen.append(f[1])
        out_rows = tuple(seen)
    elif isinstance(rows, str):
        out_rows = (rows,)
    else:
        out_rows = tuple(rows)
    return out_columns, out_rows
