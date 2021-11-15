import itertools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        quantiles_of_other_factors (float, list or None): Quantiles at which the factors
            that are not varied in a given plot are fixed. If None, those factors are
            not fixed but integrated out.
        n_points (int): Number of grid points per plot. For 3d plots this is per
            dimension.
        n_draws (int): Number of randomly drawn values of the non visualized factors
            if those factors are not fixed at a quantile but averaged out.
        data (pd.DataFrame): Empirical dataset that is used to estimate the model. This
            is only needed if the model has observed factors. Those factors are then
            extracted from the data and treated like other factors that are kept fixed
            in a given plot.

    Returns:
        matplotlib.Figure: The plot
        pandas.DataFrame: The data from which the plot was generated.
    """
    if isinstance(quantiles_of_other_factors, float):
        quantiles_of_other_factors = [quantiles_of_other_factors]
    elif isinstance(quantiles_of_other_factors, tuple):
        quantiles_of_other_factors = list(quantiles_of_other_factors)

    if plot_marginal_effects:
        raise NotImplementedError()

    model = process_model(model_dict)

    has_observed_factors = bool(model["labels"]["observed_factors"])

    if has_observed_factors and data is None:
        raise ValueError(
            "The model has observed factors. You must pass the empirical data to "
            "'visualize_transition_equations' via the keyword *data*."
        )

    if has_observed_factors:
        _, _, observed_arr = process_data(
            df=data,
            labels=model["labels"],
            update_info=model["update_info"],
            anchoring_info=model["anchoring"],
        )
        observed_factor_data = pd.DataFrame(
            data=observed_arr[period], columns=model["labels"]["observed_factors"]
        )
    else:
        observed_factor_data = None

    if period >= model["labels"]["periods"][-1]:
        raise ValueError(
            "*period* must be the penultimate period of the model or earlier."
        )

    params_index = get_params_index(
        update_info=model["update_info"],
        labels=model["labels"],
        dimensions=model["dimensions"],
    )

    params = params.reindex(params_index)

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

    factors = model["labels"]["latent_factors"]

    if state_ranges is None:
        state_ranges = create_state_ranges(states, model["labels"]["latent_factors"])

    figsize = (2.5 * len(factors), 2 * len(factors))
    fig, axes = plt.subplots(
        nrows=len(factors), ncols=len(factors), figsize=figsize, sharey="row"
    )

    for (output_factor, input_factor), ax in zip(
        itertools.product(factors, repeat=2), axes.flatten()
    ):
        output_factor_position = factors.index(output_factor)
        transition_function = model["transition_functions"][output_factor_position]
        transition_params = pardict["transition"][period][output_factor_position]

        if quantiles_of_other_factors is not None:
            plot_data = _prepare_data_for_one_plot_fixed_quantile_2d(
                states=states,
                state_ranges=state_ranges,
                period=period,
                input_factor=input_factor,
                output_factor=output_factor,
                quantiles_of_other_factors=quantiles_of_other_factors,
                n_points=n_points,
                transition_function=transition_function,
                transition_params=transition_params,
                latent_factors=factors,
                all_factors=model["labels"]["all_factors"],
                observed_factor_data=observed_factor_data,
            )
        else:
            plot_data = _prepare_data_for_one_plot_average_2d(
                states=states,
                state_ranges=state_ranges,
                period=period,
                input_factor=input_factor,
                output_factor=output_factor,
                n_points=n_points,
                n_draws=n_draws,
                transition_function=transition_function,
                transition_params=transition_params,
                latent_factors=factors,
                all_factors=model["labels"]["all_factors"],
                observed_factor_data=observed_factor_data,
            )

        if (
            isinstance(quantiles_of_other_factors, list)
            and len(quantiles_of_other_factors) > 1
        ):
            hue = "quantile"
        else:
            hue = None

        sns.lineplot(
            data=plot_data,
            x=f"{input_factor} in period {period}",
            y=f"{output_factor} in period {period + 1}",
            hue=hue,
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if hue is not None:
        fig.legend(
            handles, labels, loc="upper center", ncol=len(quantiles_of_other_factors)
        )
    fig.tight_layout()
    sns.despine()
    return fig


def _prepare_data_for_one_plot_fixed_quantile_2d(
    states,
    state_ranges,
    period,
    input_factor,
    output_factor,
    quantiles_of_other_factors,
    n_points,
    transition_function,
    transition_params,
    latent_factors,
    all_factors,
    observed_factor_data,
):

    period_data = states.query(f"period == {period}")[latent_factors]
    transition_name, transition_function = transition_function
    input_min = state_ranges[input_factor].loc[period]["minimum"]
    input_max = state_ranges[input_factor].loc[period]["maximum"]
    to_concat = []
    for quantile in quantiles_of_other_factors:
        input_data = pd.DataFrame()
        input_data[input_factor] = np.linspace(input_min, input_max, n_points)
        fixed_quantiles = period_data.drop(columns=input_factor).quantile(quantile)
        input_data[fixed_quantiles.index] = fixed_quantiles
        if observed_factor_data is not None:
            observed_quantiles = observed_factor_data.quantile(quantile)
            input_data[observed_quantiles.index] = observed_quantiles
        input_arr = jnp.array(input_data[all_factors].to_numpy())
        if transition_name != "constant":
            output_arr = transition_function(input_arr, transition_params)
        else:
            output_arr = input_data[output_factor].to_numpy()

        quantile_data = pd.DataFrame()
        quantile_data[f"{input_factor} in period {period}"] = input_data[input_factor]
        quantile_data[f"{output_factor} in period {period + 1}"] = np.array(output_arr)
        quantile_data["quantile"] = quantile
        to_concat.append(quantile_data)

    out = pd.concat(to_concat).reset_index()
    return out


def _prepare_data_for_one_plot_average_2d(
    states,
    state_ranges,
    period,
    input_factor,
    output_factor,
    n_points,
    n_draws,
    transition_function,
    transition_params,
    latent_factors,
    all_factors,
    observed_factor_data,
):

    transition_name, transition_function = transition_function
    period_data = states.query(f"period == {period}")[latent_factors].reset_index()
    if observed_factor_data is not None:
        period_data = pd.concat([period_data, observed_factor_data], axis=1)

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
        if transition_name != "constant":
            output_arr = transition_function(input_arr, transition_params)
        else:
            output_arr = input_data[output_factor].to_numpy()

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
