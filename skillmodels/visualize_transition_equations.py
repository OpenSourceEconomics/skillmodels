import itertools
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from skillmodels.filtered_states import get_filtered_states
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.parse_params import parse_params
from skillmodels.process_data import process_data
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model
from skillmodels.utils_plotting import get_layout_kwargs
from skillmodels.utils_plotting import get_make_subplot_kwargs


def combine_transition_plots(
    plots_dict,
    column_order=None,
    row_order=None,
    factor_mapping=None,
    make_subplot_kwargs=None,
    sharex=False,
    sharey=True,
    showlegend=True,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
):
    """Combine individual plots into figure with subplots.
    Uses dictionary with plotly images as values to build plotly Figure with subplots.

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
        sharex, sharey, column_order, row_order, make_subplot_kwargs
    )
    factor_mapping = _process_factor_mapping_trans(
        factor_mapping, row_order, column_order
    )
    fig = make_subplots(**make_subplot_kwargs)
    for (output_factor, input_factor), (row, col) in zip(
        itertools.product(row_order, column_order),
        itertools.product(np.arange(len(row_order)), np.arange(len(column_order))),
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
            title_text=f"{factor_mapping[input_factor]}", row=row + 1, col=col + 1
        )
        if col == 0:
            fig.update_yaxes(
                title_text=f"{factor_mapping[output_factor]}",
                row=row + 1,
                col=col + 1,
            )

    layout_kwargs = get_layout_kwargs(
        layout_kwargs, legend_kwargs, title_kwargs, showlegend, column_order, row_order
    )
    fig.update_layout(**layout_kwargs)
    return fig


def get_transition_plots(
    model_dict,
    params,
    data,
    period,
    state_ranges=None,
    quantiles_of_other_factors=(0.25, 0.5, 0.75),
    n_points=50,
    n_draws=50,
    colorscale="Magenta_r",
    layout_kwargs=None,
):
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

    Returns:
        plots_dict (dict): Dictionary with individual plots of transition equations
            for each combination of input and output factors.

    """
    quantiles_of_other_factors = _process_quantiles_of_other_factors(
        quantiles_of_other_factors
    )

    model = process_model(model_dict)

    if period >= model["labels"]["periods"][-1]:
        raise ValueError(
            "*period* must be the penultimate period of the model or earlier."
        )

    latent_factors = model["labels"]["latent_factors"]
    all_factors = model["labels"]["all_factors"]
    states = get_filtered_states(model_dict=model_dict, data=data, params=params)[
        "anchored_states"
    ]["states"]
    plots_dict = _get_dictionary_with_plots(
        model,
        data,
        params,
        states,
        state_ranges,
        latent_factors,
        all_factors,
        quantiles_of_other_factors,
        period,
        n_points,
        n_draws,
        colorscale,
        layout_kwargs,
    )
    return plots_dict


def _get_dictionary_with_plots(
    model,
    data,
    params,
    states,
    state_ranges,
    latent_factors,
    all_factors,
    quantiles_of_other_factors,
    period,
    n_points,
    n_draws,
    colorscale,
    layout_kwargs,
    showlegend=True,
):
    """Get plots of transition functions for each input and output combination.
    Returns a dictionary with individual plots of transition fanctions for each input
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
        all_factors (list): All factors of the model that are the inuts of transition
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
    observed_factors = model["labels"]["observed_factors"]
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
    plots_dict = {}
    for output_factor, input_factor in itertools.product(latent_factors, all_factors):
        transition_function = model["transition_info"]["individual_functions"][
            output_factor
        ]
        transition_params = {
            output_factor: pardict["transition"][output_factor][period]
        }

        if quantiles_of_other_factors is not None:
            plot_data = _prepare_data_for_one_plot_fixed_quantile_2d(
                states_data=states_data,
                state_ranges=state_ranges,
                period=period,
                input_factor=input_factor,
                output_factor=output_factor,
                quantiles_of_other_factors=quantiles_of_other_factors,
                n_points=n_points,
                transition_function=transition_function,
                transition_params=transition_params,
                all_factors=all_factors,
            )

        else:
            plot_data = _prepare_data_for_one_plot_average_2d(
                states_data=states_data,
                state_ranges=state_ranges,
                period=period,
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


def _get_state_ranges(state_ranges, states_data, all_factors):
    """Create state ranges if none is given"""
    if state_ranges is None:
        state_ranges = create_state_ranges(states_data, all_factors)
    return state_ranges


def _get_pardict(model, params):
    """Get parsed params dictionary."""
    parsing_info = create_parsing_info(
        params_index=params.index,
        update_info=model["update_info"],
        labels=model["labels"],
        anchoring=model["anchoring"],
    )

    _, _, _, pardict = parse_params(
        params=jnp.array(params["value"].to_numpy()),
        parsing_info=parsing_info,
        dimensions=model["dimensions"],
        labels=model["labels"],
        n_obs=1,
    )
    return pardict


def _set_index_params(model, params):
    """Reset index of params data frame to model implied values."""
    params_index = get_params_index(
        update_info=model["update_info"],
        labels=model["labels"],
        dimensions=model["dimensions"],
        transition_info=model["transition_info"],
    )

    params = params.reindex(params_index)
    return params


def _get_states_data(model, period, data, states, observed_factors):

    if observed_factors and data is None:
        raise ValueError(
            """The model has observed factors. You must pass the empirical data to
        'visualize_transition_equations' via the keyword *data*."""
        )

    if observed_factors:
        _, _, _observed_arr = process_data(
            df=data,
            labels=model["labels"],
            update_info=model["update_info"],
            anchoring_info=model["anchoring"],
        )
        # convert from jax to numpy
        _observed_arr = np.array(_observed_arr)
        observed_data = pd.DataFrame(
            data=_observed_arr[period], columns=observed_factors
        )
        observed_data["id"] = observed_data.index
        observed_data["period"] = period
        states_data = pd.merge(
            left=states,
            right=observed_data,
            left_on=["id", "period"],
            right_on=["id", "period"],
            how="left",
        )
    else:
        states_data = states.copy(deep=True)
    return states_data


def _prepare_data_for_one_plot_fixed_quantile_2d(
    states_data,
    state_ranges,
    period,
    input_factor,
    output_factor,
    quantiles_of_other_factors,
    n_points,
    transition_function,
    transition_params,
    all_factors,
):

    period_data = states_data.query(f"period == {period}")[all_factors]
    input_min = state_ranges[input_factor].loc[period]["minimum"]
    input_max = state_ranges[input_factor].loc[period]["maximum"]
    to_concat = []
    for quantile in quantiles_of_other_factors:
        input_data = pd.DataFrame()
        input_data[input_factor] = np.linspace(input_min, input_max, n_points)
        fixed_quantiles = period_data.drop(columns=input_factor).quantile(quantile)
        input_data[fixed_quantiles.index] = fixed_quantiles
        input_arr = jnp.array(input_data[all_factors].to_numpy())
        # convert from jax to numpy array
        output_arr = np.array(transition_function(transition_params, input_arr))
        quantile_data = pd.DataFrame()
        quantile_data[f"input_{input_factor}"] = input_data[input_factor]
        quantile_data[f"output_{output_factor}"] = np.array(output_arr)
        quantile_data["quantile"] = quantile
        to_concat.append(quantile_data)

    out = pd.concat(to_concat).reset_index()
    return out


def _process_quantiles_of_other_factors(quantiles_of_other_factors):
    """Process quantiles of other factors to always have list as type."""
    if isinstance(quantiles_of_other_factors, (float, int)):
        quantiles_of_other_factors = [quantiles_of_other_factors]
    elif isinstance(quantiles_of_other_factors, (tuple, list)):
        quantiles_of_other_factors = list(quantiles_of_other_factors)
    return quantiles_of_other_factors


def _prepare_data_for_one_plot_average_2d(
    states_data,
    state_ranges,
    period,
    input_factor,
    output_factor,
    n_points,
    n_draws,
    transition_function,
    transition_params,
    all_factors,
):

    period_data = states_data.query(f"period == {period}")[all_factors].reset_index()

    sampled_factors = [factor for factor in all_factors if factor != input_factor]
    draws = period_data[sampled_factors].sample(n=n_draws)
    input_min = state_ranges[input_factor].loc[period]["minimum"]
    input_max = state_ranges[input_factor].loc[period]["maximum"]

    to_concat = []
    for _, draw in draws.iterrows():
        input_data = pd.DataFrame()
        input_data[input_factor] = np.linspace(input_min, input_max, n_points)
        input_data[draw.index] = draw
        input_arr = jnp.array(input_data[all_factors].to_numpy())
        # convert from jax to numpy array
        output_arr = np.array(transition_function(transition_params, input_arr))
        draw_data = pd.DataFrame()
        draw_data[f"input_{input_factor}"] = input_data[input_factor]
        draw_data[f"output_{output_factor}"] = np.array(output_arr)
        to_concat.append(draw_data)

    out = pd.concat(to_concat).groupby(f"input_{input_factor}").mean().reset_index()
    return out


def _process_factor_mapping_trans(factor_mapper, output_factors, input_factors):
    """Process mapper to return dictionary with old and new factor names"""
    all_factors = input_factors + output_factors
    if factor_mapper is None:
        factor_mapper = {fac: fac for fac in all_factors}
    else:
        for fac in all_factors:
            if fac not in factor_mapper:
                factor_mapper[fac] = fac
    return factor_mapper


def _process_orders(columns, rows, plots_dict):
    """Process axes orders to return list of strings."""
    if columns is None:
        columns = []
        for f in plots_dict.keys():
            if f[0] not in columns:
                columns.append(f[0])
    elif isinstance(columns, str):
        columns = [columns]
    if rows is None:
        rows = []
        for f in plots_dict.keys():
            if f[1] not in rows:
                rows.append(f[1])
    elif isinstance(rows, str):
        rows = [rows]
    return columns, rows
