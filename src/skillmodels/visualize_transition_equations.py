import itertools
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from skillmodels.filtered_states import get_filtered_states
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info, parse_params
from skillmodels.process_data import process_data
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model
from skillmodels.utils_plotting import get_layout_kwargs, get_make_subplot_kwargs

if TYPE_CHECKING:
    from skillmodels.types import ProcessedModel


def combine_transition_plots(
    plots_dict: dict[tuple[str, str], go.Figure],
    column_order: list[str] | str | None = None,
    row_order: list[str] | str | None = None,
    factor_mapping: dict[str, str] | None = None,
    make_subplot_kwargs: dict[str, Any] | None = None,
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
        plots_dict (dict): Dictionary with plots of transition functions for each
            factor.
        column_order (list, str or NoneType): List of (output) factor names according
            to which transition plots should be ordered horizontally. If None, infer
            from the keys of of plots_dict
        row_order (list, str or NoneType): List of (input) factor names according
            to which transition plots should be ordered vertically. If None, infer
            from the keys of of plots_dict
        factor_mapping (dict or NoneType): A dictionary with custom factor names to
            display as axes labels.
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots. If None,
            default arguments defined in the function are used.
        sharex (bool): Whether to share the properties of x-axis across subplots.
            Default False.
        sharey (bool): Whether to share the properties ofy-axis across subplots.
            Default True.
        showlegend (bool): Display legend if True.
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title. If None, infers title based on the value of
            `quntiles_of_other_factors`.

    Returns:
        fig (plotly.Figure): Plotly figure with subplots that combines individual
            transition functions.

    """
    plots_dict = deepcopy(plots_dict)

    column_order, row_order = _process_orders(column_order, row_order, plots_dict)
    make_subplot_kwargs = get_make_subplot_kwargs(
        sharex,
        sharey,
        column_order,
        row_order,
        make_subplot_kwargs,
    )
    factor_mapping = _process_factor_mapping_trans(
        factor_mapping,
        row_order,
        column_order,
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
        layout_kwargs,
        legend_kwargs,
        title_kwargs,
        showlegend,
        column_order,
        row_order,
    )
    fig.update_layout(**layout_kwargs)
    return fig


def get_transition_plots(
    model_dict: dict[str, Any],
    params: pd.DataFrame,
    data: pd.DataFrame,
    period: int,
    state_ranges: dict[str, pd.DataFrame] | None = None,
    quantiles_of_other_factors: tuple[float, ...] | list[float] | float | None = (
        0.25,
        0.5,
        0.75,
    ),
    n_points: int = 50,
    n_draws: int = 50,
    colorscale: str = "Magenta_r",
    layout_kwargs: dict[str, Any] | None = None,
    include_correction_factors: bool = False,
) -> dict[tuple[str, str], go.Figure]:
    """Get dictionary with individual plots of transition equations for each factor.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        params (pandas.DataFrame): DataFrame with model parameters.
        data (pd.DataFrame): Empirical dataset that is used to estimate the model.
        period (int): The start period of the transition equations that are plotted.
        state_ranges (dict or NoneType): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        quantiles_of_other_factors (float, list or None): Quantiles at which the factors
            that are not varied in a given plot are fixed. If None, those factors are
            not fixed but integrated out.
        n_points (int): Number of grid points per input. Default 50.
        n_draws (int): Number of randomly drawn values of the factors that are averaged
            out. Only relevant if quantiles_of_other_factors is *None*. Default 50.
        colorscale (str): The color scale to use for line legends. Must be a valid
            plotly.express.colors.sequential attribute. Default 'Magenta_r'.
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly image object. If None, the default kwargs
            defined in the function will be used.
        include_correction_factors (bool): Whether to include correction factors in the
            plots. Default False.

    Returns:
        plots_dict (dict): Dictionary with individual plots of transition equations
            for each combination of input and output factors.

    """
    quantiles_of_other_factors = _process_quantiles_of_other_factors(
        quantiles_of_other_factors,
    )

    model = process_model(model_dict)

    if period >= model.labels.periods[-1]:
        raise ValueError(
            "*period* must be the penultimate period of the model or earlier.",
        )

    if (
        include_correction_factors
        or not model.endogenous_factors_info.has_endogenous_factors
    ):
        latent_factors = model.labels.latent_factors
    else:
        latent_factors = [
            lf
            for lf in model.labels.latent_factors
            if not model.endogenous_factors_info.factor_info[lf].is_correction  # ty: ignore[invalid-argument-type]
        ]
    all_factors = model.labels.all_factors
    states = get_filtered_states(model_dict=model_dict, data=data, params=params)[
        "anchored_states"
    ]["states"]
    plots_dict = _get_dictionary_with_plots(
        model=model,
        data=data,
        params=params,
        states=states,
        state_ranges=state_ranges,
        latent_factors=latent_factors,  # ty: ignore[invalid-argument-type]
        all_factors=all_factors,
        quantiles_of_other_factors=quantiles_of_other_factors,
        period=period,
        n_points=n_points,
        n_draws=n_draws,
        colorscale=colorscale,
        layout_kwargs=layout_kwargs,
    )
    return plots_dict


def _get_dictionary_with_plots(
    model: "ProcessedModel",
    data: pd.DataFrame,
    params: pd.DataFrame,
    states: pd.DataFrame,
    state_ranges: dict[str, pd.DataFrame] | None,
    latent_factors: list[str],
    all_factors: tuple[str, ...],
    quantiles_of_other_factors: list[float] | None,
    period: int,
    n_points: int,
    n_draws: int,
    colorscale: str,
    layout_kwargs: dict[str, Any] | None,
    showlegend: bool = True,
) -> dict[tuple[str, str], go.Figure]:
    """Get plots of transition functions for each input and output combination.

    Return a dictionary with individual plots of transition functions for each input
    and output factors.

    Args:
        model (dict): The model specification. See: :ref:`model_specs`
        params (pandas.DataFrame): DataFrame with model parameters.
        states (pandas.DataFrame): Tidy DataFrame with filtered or simulated states.
            They are used to estimate the state ranges in each period (if state_ranges
            are not given explicitly) and to estimate the distribution of the factors
            that are not visualized.
        state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.

        latent_factors (list): Latent factors of the model that are outputs of
            transition factors.
        all_factors (list): All factors of the model that are the inputs of transition
            functions.
        quantiles_of_other_factors (float, list or None): Quantiles at which the factors
            that are not varied in a given plot are fixed. If None, those factors are
            not fixed but integrated out.
        period (int): The start period of the transition equations that are plotted.
        n_points (int): Number of grid points per input. Default 50.
        n_draws (int): Number of randomly drawn values of the factors that are averaged
            out. Only relevant if quantiles_of_other_factors is *None*. Default 50.
        colorscale (str): The color scale to use for line legends. Must be a valid
            plotly.express.colors.sequential attribute. Default 'Magenta_r'.
        subfig_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly image object. If None, the default kwargs defined
            in the function will be used.

    Returns:
        plots_dict (dict): Dictionary with individual plots of transition functions
            for each input and output factors.

    """
    observed_factors = model.labels.observed_factors
    states_data = _get_states_data(model, period, data, states, observed_factors)
    params = _set_index_params(model, params)
    pardict = _get_pardict(model, params)
    state_ranges = _get_state_ranges(state_ranges, states_data, all_factors)
    layout_kwargs = get_layout_kwargs(
        layout_kwargs=layout_kwargs,
        legend_kwargs=None,
        title_kwargs=None,
        showlegend=showlegend,
    )
    has_endogenous_factors = model.endogenous_factors_info.has_endogenous_factors
    if has_endogenous_factors:
        _aug_periods = model.endogenous_factors_info.aug_periods_from_period(period)
    else:
        _aug_periods = [period]
    plots_dict = {}
    for output_factor, input_factor in itertools.product(latent_factors, all_factors):
        transition_function = model.transition_info.individual_functions[output_factor]  # ty: ignore[invalid-argument-type]
        if (
            has_endogenous_factors
            and model.endogenous_factors_info.factor_info[output_factor].is_endogenous  # ty: ignore[invalid-argument-type]
        ):
            aug_period = min(_aug_periods)
        else:
            aug_period = max(_aug_periods)
        transition_params = {
            output_factor: pardict["transition"][output_factor][aug_period]
        }

        if quantiles_of_other_factors is not None:
            plot_data = _prepare_data_for_one_plot_fixed_quantile_2d(
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

        else:
            plot_data = _prepare_data_for_one_plot_average_2d(
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

        if (
            isinstance(quantiles_of_other_factors, list)
            and len(quantiles_of_other_factors) > 1
        ):
            color = "quantile"
        else:
            color = None
        subfig = px.line(
            plot_data,
            y=f"output_{output_factor}",
            x=f"input_{input_factor}",
            color=color,
            color_discrete_sequence=getattr(px.colors.sequential, colorscale),
        )
        subfig.update_xaxes(title={"text": input_factor})
        subfig.update_yaxes(title={"text": output_factor})
        subfig.update_layout(**layout_kwargs)
        plots_dict[(input_factor, output_factor)] = deepcopy(subfig)

    return plots_dict


def _get_state_ranges(
    state_ranges: dict[str, pd.DataFrame] | None,
    states_data: pd.DataFrame,
    all_factors: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Create state ranges if none is given."""
    if state_ranges is None:
        state_ranges = create_state_ranges(states_data, list(all_factors))
    return state_ranges


def _get_pardict(
    model: "ProcessedModel",
    params: pd.DataFrame,
) -> dict[str, Any]:
    """Get parsed params dictionary."""
    parsing_info = create_parsing_info(
        params_index=params.index,  # ty: ignore[invalid-argument-type]
        update_info=model.update_info,
        labels=model.labels,
        anchoring=model.anchoring,
        has_endogenous_factors=model.endogenous_factors_info.has_endogenous_factors,
    )

    _, _, _, pardict = parse_params(
        params=jnp.array(params["value"].to_numpy()),
        parsing_info=parsing_info,
        dimensions=model.dimensions,
        labels=model.labels,
        n_obs=1,
    )
    return pardict


def _set_index_params(
    model: "ProcessedModel",
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

    params = params.reindex(params_index)
    return params


def _get_states_data(
    model: "ProcessedModel",
    period: int,
    data: pd.DataFrame,
    states: pd.DataFrame,
    observed_factors: tuple[str, ...],
) -> pd.DataFrame:
    if observed_factors and data is None:
        raise ValueError(
            "The model has observed factors. You must pass the empirical data to "
            "'visualize_transition_equations' via the keyword *data*.",
        )

    if observed_factors:
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
        states_data = pd.merge(
            left=states,
            right=observed_data,
            left_on=["id", "aug_period"],
            right_on=["id", "aug_period"],
            how="left",
        )
    else:
        states_data = states.copy(deep=True)
    return states_data


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

    out = pd.concat(to_concat).reset_index()
    return out


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

    to_concat = []
    for _, draw in draws.iterrows():
        input_data = pd.DataFrame()
        input_data[input_factor] = np.linspace(input_min, input_max, n_points)
        for col, val in draw.items():
            input_data[col] = val
        input_arr = jnp.array(input_data[list(all_factors)].to_numpy())
        # convert from jax to numpy array
        output_arr = np.array(transition_function(transition_params, input_arr))
        draw_data = pd.DataFrame()
        draw_data[f"input_{input_factor}"] = input_data[input_factor]
        draw_data[f"output_{output_factor}"] = np.array(output_arr)
        to_concat.append(draw_data)

    out = pd.concat(to_concat).groupby(f"input_{input_factor}").mean().reset_index()
    return out


def _process_factor_mapping_trans(
    factor_mapper: dict[str, str] | None,
    output_factors: list[str],
    input_factors: list[str],
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
    columns: list[str] | str | None,
    rows: list[str] | str | None,
    plots_dict: dict[tuple[str, str], go.Figure],
) -> tuple[list[str], list[str]]:
    """Process axes orders to return list of strings."""
    out_columns: list[str]
    out_rows: list[str]
    if columns is None:
        out_columns = []
        for f in plots_dict:
            if f[0] not in out_columns:
                out_columns.append(f[0])
    elif isinstance(columns, str):
        out_columns = [columns]
    else:
        out_columns = columns
    if rows is None:
        out_rows = []
        for f in plots_dict:
            if f[1] not in out_rows:
                out_rows.append(f[1])
    elif isinstance(rows, str):
        out_rows = [rows]
    else:
        out_rows = rows
    return out_columns, out_rows
