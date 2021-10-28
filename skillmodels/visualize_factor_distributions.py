import itertools
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model


def plot_pairwise_factor_distributions(
    states,
    model_dict,
    period,
    combine_plots_in_grid=True,
    add_3d_plots=False,
    state_ranges=None,
    n_points=50,
):
    """Visualize pairwise_factor_distributions in certain period.

    Args:
        states (list, pandas.DataFrame): list of tidy DataFrames with filtered
            or simulated states or only one DataFrame with filtered or
            simulated states.They are used to estimate the state ranges in
            each period (if state_ranges are not given explicitly) and to
            estimate the distribution of the latent factors.
        model_dict (dict): The model specification. See: :ref:'model_specs'
        period (int): The selected period of the filtered states that are plotted.
        combine_plots_in_grid (boolen): decide whether to retrun a grid of plots
            or return a dict of individual plots. Default True.
        add_3d_plots (boolen):decide whether to add 3D plots in grid of plots
            or in the dict of individual plots. Default False.
        state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        n_points (int): Number of grid points per plot. For 3d plots this is per
            dimension. Default 50.

    Returns:
        matplotlib.Figure: The grid plot or dict of individual plots

    """
    model = process_model(model_dict)
    if state_ranges is None:
        if isinstance(states, list):
            state_ranges = []
            for state in states:
                state_range = create_state_ranges(state[1], model["labels"]["factors"])
                state_ranges.append(state_range)
        elif isinstance(states, tuple):
            state_range = create_state_ranges(states[1], model["labels"]["factors"])
            state_ranges = [state_range]
        else:
            state_range = create_state_ranges(states, model["labels"]["factors"])
            state_ranges = [state_range]
    if isinstance(states, list):
        pass
    else:
        states = [states]
    plot_dict = {}
    if combine_plots_in_grid:
        if add_3d_plots:
            plot_dict = _plot_grid_2d_3d(states, state_ranges, period, n_points)
        else:
            plot_dict = _plot_changes_grid_2d(states, state_ranges, period, n_points)
    else:
        if add_3d_plots:
            dict2 = _plot_changes_seperately_2d(states, state_ranges, period, n_points)
            dict1 = _plot_3d_seperately(states, state_ranges, period)
            plot_dict = {**dict1, **dict2}
        else:
            plot_dict = _plot_changes_seperately_2d(
                states, state_ranges, period, n_points
            )
    return plot_dict


def _prepare_for_filtered_data(states, period):
    data_period = states.query(f"period == {period}")
    data_cleaned = data_period.drop(columns=["mixture", "period", "id"])
    names = data_cleaned.keys()
    factors = list(itertools.product(names, repeat=2))
    return factors, data_cleaned


def _prepare_for_simulated_data(state, period):
    data = state[1]
    data = data.reset_index()
    data_period = data.query(f"period == {period}")
    data_cleaned = data_period.drop(columns=["period", "id"])
    names = data_cleaned.keys()
    factors = list(itertools.product(names, repeat=2))
    return factors, data_cleaned


def prepare_axes_for_2d_changes(ax, state_ranges, a, b, period):
    lower_bounds = []
    upper_bounds = []
    for state_range in state_ranges:
        stat_min = min(
            state_range[a].loc[period]["minimum"], state_range[b].loc[period]["minimum"]
        )
        stat_max = max(
            state_range[a].loc[period]["maximum"], state_range[b].loc[period]["maximum"]
        )
        lower_bounds.append(stat_min)
        upper_bounds.append(stat_max)
    lower_bound = math.floor(min(lower_bounds)) - 1
    upper_bound = math.ceil(max(upper_bounds)) + 1
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound))
    return ax


def _plot_changes_grid_2d(states, state_ranges, period, n_points):
    factors = []
    datas = []
    for state in states:
        if isinstance(state, tuple):
            factor, data = _prepare_for_simulated_data(state, period)
        else:
            factor, data = _prepare_for_filtered_data(state, period)
        factors.append(factor)
        datas.append(data)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f"Grid of plots for Period {period}")
    size = int(np.sqrt(len(factors[0])))
    gs = fig.add_gridspec(size, size)
    for a, b in factors[0]:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row > col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            for data in datas:
                sns.kdeplot(
                    data=data,
                    x=data[a],
                    y=data[b],
                    gridsize=n_points,
                    ax=ax,
                    fill=True,
                    alpha=0.7,
                )

        elif row == col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            for data in datas:
                sns.kdeplot(data=data, x=data[a], gridsize=n_points, ax=ax)
            ax.set(ylim=(0, None))
    return fig


def _plot_changes_seperately_2d(states, state_ranges, period, n_points):
    figure_dict = {}
    factors = []
    datas = []
    for state in states:
        if isinstance(state, tuple):
            factor, data = _prepare_for_simulated_data(state, period)
        else:
            factor, data = _prepare_for_filtered_data(state, period)
        factors.append(factor)
        datas.append(data)
    for a, b in factors[0]:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row > col:
            fig, ax = plt.subplots()
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            for data in datas:
                sns.kdeplot(
                    data=data,
                    x=data[a],
                    y=data[b],
                    gridsize=n_points,
                    ax=ax,
                    fill=True,
                    alpha=0.7,
                )
            fig.suptitle(f"{a}_{b}_2D_Period {period}")
            figure_dict[f"{a}_{b}_2D_Period {period}"] = fig

        elif row == col:
            fig, ax = plt.subplots()
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            for data in datas:
                sns.kdeplot(data=data, x=data[a], gridsize=n_points, ax=ax)
                fig.suptitle(f"{a}_2D_Period {period}")
                figure_dict[f"{a}_2D_Period {period}"] = fig
            ax.set(ylim=(0, None))
    return figure_dict


def _calculate_kde_for_3d(data_cleaned, a, b, lower_bound, upper_bound):

    x = data_cleaned[a]
    y = data_cleaned[b]
    xx, yy = np.mgrid[lower_bound:upper_bound:50j, lower_bound:upper_bound:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f


def _prepare_axes_for_3d(ax, state_ranges, a, b, period):
    lower_bounds = []
    upper_bounds = []
    for state_range in state_ranges:
        stat_min = min(
            state_range[a].loc[period]["minimum"], state_range[b].loc[period]["minimum"]
        )
        stat_max = max(
            state_range[a].loc[period]["maximum"], state_range[b].loc[period]["maximum"]
        )
        lower_bounds.append(stat_min)
        upper_bounds.append(stat_max)
    lower_bound = math.floor(min(lower_bounds))
    upper_bound = math.ceil(max(upper_bounds))

    ax.set(
        xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound), zlim=(0, None)
    )
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlabel("KDE")
    ax.grid(False)
    ax.view_init(30, 35)
    return ax, lower_bound, upper_bound


def _plot_grid_2d_3d(states, state_ranges, period, n_points):
    if isinstance(states[0], tuple):
        factors, data_cleaned = _prepare_for_simulated_data(states[0], period)
    else:
        factors, data_cleaned = _prepare_for_filtered_data(states[0], period)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f"Grid of plots for Period {period}")
    size = int(np.sqrt(len(factors)))
    gs = fig.add_gridspec(size, size)
    for a, b in factors:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row > col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            sns.kdeplot(
                data=data_cleaned,
                x=data_cleaned[a],
                y=data_cleaned[b],
                gridsize=n_points,
                ax=ax,
                fill=True,
                alpha=0.7,
            )

        elif row == col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = prepare_axes_for_2d_changes(ax, state_ranges, a, b, period)
            sns.kdeplot(data=data_cleaned, x=data_cleaned[a], gridsize=n_points, ax=ax)
            ax.set(ylim=(0, None))

        else:
            ax = fig.add_subplot(gs[row - 1, col - 1], projection="3d")
            ax, lower_bound, upper_bound = _prepare_axes_for_3d(
                ax, state_ranges, a, b, period
            )
            xx, yy, f = _calculate_kde_for_3d(
                data_cleaned, a, b, lower_bound, upper_bound
            )
            surf = ax.plot_surface(
                xx,
                yy,
                f,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                cmap="coolwarm",
                edgecolor="none",
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig


def _plot_3d_seperately(states, state_ranges, period):
    if isinstance(states[0], tuple):
        factors, data_cleaned = _prepare_for_simulated_data(states[0], period)
    else:
        factors, data_cleaned = _prepare_for_filtered_data(states[0], period)
    fig = plt.figure(figsize=(15, 15))
    figure_dict = {}
    for a, b in factors:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row > col:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            fig.suptitle(f"{a}_{b}_3D_Period {period}")
            ax, lower_bound, upper_bound = _prepare_axes_for_3d(
                ax, state_ranges, a, b, period
            )
            xx, yy, f = _calculate_kde_for_3d(
                data_cleaned, a, b, lower_bound, upper_bound
            )
            surf = ax.plot_surface(
                xx,
                yy,
                f,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                cmap="coolwarm",
                edgecolor="none",
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
            figure_dict[f"{a}_{b}_3D_Period {period}"] = fig
    return figure_dict
