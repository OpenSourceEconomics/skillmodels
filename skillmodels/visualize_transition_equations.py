import itertools

import jax.numpy as jnp
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.parse_params import parse_params
from skillmodels.process_data import process_data
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model


def visualize_transition_equations(
    model_dict,
    params,
    states,
    period,
    state_ranges=None,
    quantiles_of_other_factors=(0.25, 0.5, 0.75),
    plot_marginal_effects=False,
    n_points=50,
    n_draws=50,
    sharex=False,
    sharey="row",
    data=None,
):
    """Visualize transition equations.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        params (pandas.DataFrame): DataFrame with model parameters.
        states (pandas.DataFrame): Tidy DataFrame with filtered or simulated states.
            They are used to estimate the state ranges in each period (if state_ranges
            are not given explicitly) and to estimate the distribution of the factors
            that are not visualized.
        period (int): The start period of the transition equations that are plotted.
        combine_plots_in_grid (boolen): Return a figure containing subplots for each
            pair of factors or a dictionary of individual plots. Default True.
        state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        quantiles_of_other_factors (float, list or None): Quantiles at which the factors
            that are not varied in a given plot are fixed. If None, those factors are
            not fixed but integrated out.
        n_points (int): Number of grid points per input. Default 50.
        n_draws (int): Number of randomly drawn values of the factors that are averaged
            out. Only relevant if quantiles_of_other_factors is *None*. Default 50.
        sharex (bool or {'none', 'all', 'col'}): Whether to share the properties of
            x-axis across subplots. See API docs of matplotlib.pyplot.subplots.
            Default False.
        sharey (bool or {'none', 'all', 'row'}): : Whether to share the properties of
            y-axis across subplots. See API docs of matplotlib.pyplot.subplots.
            Default 'row'.
        data (pd.DataFrame): Empirical dataset that is used to estimate the model. Only
            needed if the model has observed factors. Those factors are directly taken
            from the data to calculate their quantiles or averages.

    Returns:
        matplotlib.Figure: The plot
        pandas.DataFrame: The data from which the plot was generated.
    """

    if plot_marginal_effects:
        raise NotImplementedError()

    quantiles_of_other_factors = _process_quantiles_of_other_factors(
        quantiles_of_other_factors
    )
    model = process_model(model_dict)

    if period >= model["labels"]["periods"][-1]:
        raise ValueError(
            "*period* must be the penultimate period of the model or earlier."
        )

    latent_factors = model["labels"]["latent_factors"]
    observed_factors = model["labels"]["observed_factors"]
    all_factors = model["labels"]["all_factors"]
    states_data = _get_states_data(model, period, data, states, observed_factors)
    params = _set_index_params(model, params)
    pardict = _get_pardict(model, params)
    state_ranges = _get_state_ranges(state_ranges, states_data, all_factors)

    nlatent = len(latent_factors)
    nall = len(all_factors)
    fig = make_subplots(
        rows=nlatent,
        cols=nall,
        start_cell="top-left",
        print_grid=False,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.2,
    )
    for (output_factor, input_factor), (row, col) in zip(
        itertools.product(latent_factors, all_factors),
        itertools.product(np.arange(nlatent), np.arange(nall)),
    ):
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
            for q, df in plot_data.groupby("quantile"):
                if row == 0 and col == 0:
                    fig.add_trace(
                        go.Scatter(
                            y=df[f"{output_factor} in period {period + 1}"],
                            x=df[f"{input_factor} in period {period}"],
                            line_shape="linear",
                            legendgroup=f"{q}",
                            name=q,
                            line={"color": "Viridis"},
                        ),
                        col=col + 1,
                        row=row + 1,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            y=df[f"{output_factor} in period {period + 1}"],
                            x=df[f"{input_factor} in period {period}"],
                            line_shape="linear",
                            showlegend=False,
                            legendgroup=f"{q}",
                            name=q,
                            line={"color": "Viridis"},
                        ),
                        col=col + 1,
                        row=row + 1,
                    )
        fig.update_xaxes(
            title_text=f"{input_factor} in period {period}", row=row + 1, col=col + 1
        )
        if col == 0:
            fig.update_yaxes(
                title_text=f"{output_factor} in period {period+1}",
                row=row + 1,
                col=col + 1,
            )

    fig.update_layout(template="simple_white")
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    return fig


def _get_state_ranges(state_ranges, states_data, all_factors):
    if state_ranges is None:
        state_ranges = create_state_ranges(states_data, all_factors)
    return state_ranges


def _get_pardict(model, params):
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
        quantile_data[f"{input_factor} in period {period}"] = input_data[input_factor]
        quantile_data[f"{output_factor} in period {period + 1}"] = np.array(output_arr)
        quantile_data["quantile"] = quantile
        to_concat.append(quantile_data)

    out = pd.concat(to_concat).reset_index()
    return out


def _process_quantiles_of_other_factors(quantiles_of_other_factors):
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
        draw_data[f"{input_factor} in period {period}"] = input_data[input_factor]
        draw_data[f"{output_factor} in period {period + 1}"] = np.array(output_arr)
        to_concat.append(draw_data)

    out = (
        pd.concat(to_concat)
        .groupby(f"{input_factor} in period {period}")
        .mean()
        .reset_index()
    )
    return out
