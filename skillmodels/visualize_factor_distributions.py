import itertools
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
    combined_plots,
    add_3d_plots,
    state_ranges=None,
    n_points=50,
):
    """Visualize pairwise_factor_distributions.
    Args:
        states (pandas.DataFrame): Tidy DataFrame with filtered or simulated states.
            They are used to estimate the state ranges in each period (if state_ranges
            are not given explicitly) and to estimate the distribution of the factors
            that are not visualized.
        model_dict (dict): The model specification. See: :ref:`model_specs`
        params (pandas.DataFrame): DataFrame with model parameters.
        period (int): The selected period of the filtered states that are plotted.
        combined_plots (boolen): decide whether to retrun a grid of plots
            or return a dict of individual plots
        add_3d_plots (boolen):decide whether to adda 3D plots in grid of plots
            or in the dict of individual plots
        state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        n_points (int): Number of grid points per plot. For 3d plots this is per
            dimension.
    Returns:
        matplotlib.Figure: The grid plot or dict of individual plots
    """
    model = process_model(model_dict)
    if state_ranges is None:
        state_ranges = create_state_ranges(states, model["labels"]["factors"])
    plot_dict = {}

    if combined_plots is not None:
        if add_3d_plots is not None:
            plot_dict = _plot_grid_2d_3d(states, state_ranges, period, n_points)
        else:
            plot_dict = _plot_grid_2d(states, state_ranges, period, n_points)
    else:
        if add_3d_plots is not None:
            dict2 = _plot_2d_seperately(states, state_ranges, period, n_points)
            dict1 = _plot_3d_seperately(states, state_ranges, period, n_points)
            plot_dict = {**dict1, **dict2}
        else:
            plot_dict = _plot_2d_seperately(states, state_ranges, period, n_points)
    return plot_dict


def _prepare_data(states, period):
    data_period = states.query(f"period == {period}")
    data_cleaned = data_period.drop(columns=["mixture", "period", "id"])
    names = data_cleaned.keys()
    factors = list(itertools.product(names, repeat=2))
    return factors, data_cleaned


def _prepare_axes_for_2d(ax, state_ranges, a, b, period):
    lower_bound = min(
        state_ranges[a].loc[period]["minimum"], state_ranges[b].loc[period]["minimum"]
    )
    upper_bound = max(
        state_ranges[a].loc[period]["maximum"], state_ranges[b].loc[period]["maximum"]
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound))
    return ax


def _prepare_axes_for_3d(ax, state_ranges, a, b, period):
    lower_bound = min(
        state_ranges[a].loc[period]["minimum"], state_ranges[b].loc[period]["minimum"]
    )
    upper_bound = max(
        state_ranges[a].loc[period]["maximum"], state_ranges[b].loc[period]["maximum"]
    )
    ax.set(
        xlim=(lower_bound, upper_bound), ylim=(lower_bound, upper_bound), zlim=(0, 1)
    )
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlabel("KDE")
    ax.grid(False)
    ax.view_init(30, 35)

    return ax, lower_bound, upper_bound


def _calculate_kde_for_3d(data_cleaned, a, b, lower_bound, upper_bound):

    x = data_cleaned[a]
    y = data_cleaned[b]
    xx, yy = np.mgrid[lower_bound:upper_bound:50j, lower_bound:upper_bound:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f


def _plot_grid_2d_3d(states, state_ranges, period, n_points):

    factors, data_cleaned = _prepare_data(states, period)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f"Grid of plots for Period {period}")
    size = int(np.sqrt(len(factors)))
    gs = fig.add_gridspec(size, size)
    for a, b in factors:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row > col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = _prepare_axes_for_2d(ax, state_ranges, a, b, period)
            sns.kdeplot(
                data=data_cleaned,
                x=data_cleaned[a],
                y=data_cleaned[b],
                gridsize=n_points,
                ax=ax,
            )

        elif row == col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = _prepare_axes_for_2d(ax, state_ranges, a, b, period)
            sns.kdeplot(data=data_cleaned, x=data_cleaned[a], gridsize=n_points, ax=ax)
            ax.set(ylim=(0, 2))

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


def _plot_grid_2d(states, state_ranges, period, n_points):

    factors, data_cleaned = _prepare_data(states, period)
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(f"Grid of plots for Period {period}")
    size = int(np.sqrt(len(factors)))
    gs = fig.add_gridspec(size, size)
    for a, b in factors:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row > col:
            ax = fig.add_subplot(gs[row - 1, col - 1])
            ax = _prepare_axes_for_2d(ax, state_ranges, a, b, period)
            sns.kdeplot(
                data=data_cleaned,
                x=data_cleaned[a],
                y=data_cleaned[b],
                gridsize=n_points,
                ax=ax,
            )

        elif row == col:
            ax = fig.add_subplot(gs[row - 1, col - 1])

            ax = _prepare_axes_for_2d(ax, state_ranges, a, b, period)
            sns.kdeplot(data=data_cleaned, x=data_cleaned[a], gridsize=n_points, ax=ax)
            ax.set(ylim=(0, 2))
    return fig


def _plot_2d_seperately(states, state_ranges, period, n_points):
    figure_dict = {}
    factors, data_cleaned = _prepare_data(states, period)
    for a, b in factors:
        row = int(re.sub("[^0-9]", "", a))
        col = int(re.sub("[^0-9]", "", b))
        if row == col:
            fig, ax = plt.subplots()
            sns.kdeplot(data=data_cleaned, x=data_cleaned[a], gridsize=n_points, ax=ax)
            ax = _prepare_axes_for_2d(ax, state_ranges, a, b, period)
            ax.set(ylim=(0, 2))
            fig.suptitle(f"{a}_2D_Period {period}")
            figure_dict[f"{a}_2D_Period {period}"] = fig
        elif row < col:
            fig, ax = plt.subplots()
            sns.kdeplot(
                data=data_cleaned,
                x=data_cleaned[a],
                y=data_cleaned[b],
                gridsize=n_points,
                ax=ax,
            )
            ax = _prepare_axes_for_2d(ax, state_ranges, a, b, period)
            fig.suptitle(f"{a}_{b}_2D_Period {period}")
            figure_dict[f"{a}_{b}_2D_Period {period}"] = fig
    return figure_dict


def _plot_3d_seperately(states, state_ranges, period, n_points):
    factors, data_cleaned = _prepare_data(states, period)
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
