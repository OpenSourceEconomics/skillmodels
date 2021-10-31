import sys
import warnings
from traceback import format_exception

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from skillmodels.process_model import process_model


def plot_factor_distributions(
    states,
    model_dict,
    period,
    combine_plots_in_grid=True,
    add_3d_plots=False,
    n_points=50,
    lower_kde_kws=None,
    diag_kde_kws=None,
    surface_kws=None,
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
        combine_plots_in_grid (boolen): decide whether to retrun a one figure
            containing subplots for each factor pair or a dictionary of
            individual plots. Default True.
        add_3d_plots (boolen):decide whether to add 3D plots in grid of plots
            or in the dict of individual plots. Default False.
        state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            The state_ranges are used to define the axis limits of the plots.
        n_points (int): Number of grid points per plot. For 3d plots this is per
            dimension. Default 50.
        lower_kde_kws (dict): Keyword arguments for seaborn.kdeplot, used to generate
            the plots in the lower triangle of the grid, i.e. the two dimensional
            kdeplot for each factor pair.
        diag_kde_kws (dict): Keyword arguments for seaborn.kdeplot, used to generate
            the plots on the diagonal of the grid, i.e. the one dimensional
            kdeplot for each factor.
        surface_kws (dict): Keyword arguments for Axes.plot_surface, used to generate
            the plots in the upper triangle of the grid, i.e. the surface plot of the
            kernel density estimates for each factor pair.

    Returns:
        matplotlib.Figure: The grid plot or dict of individual plots

    """
    if add_3d_plots and not isinstance(states, pd.DataFrame):
        raise ValueError("3d plots are only supported if states is a DataFrame")

    lower_kde_kws = {} if lower_kde_kws is None else lower_kde_kws
    diag_kde_kws = {} if diag_kde_kws is None else diag_kde_kws
    surface_kws = {} if surface_kws is None else surface_kws

    model = process_model(model_dict)
    factors = model["labels"]["factors"]

    data, hue = _process_data(states, period, factors)

    grid = _get_axes_grid(
        factors=factors,
        combine_into_grid=combine_plots_in_grid,
        add_3d_plots=add_3d_plots,
    )
    for row, fac1 in enumerate(factors):
        for col, fac2 in enumerate(factors):
            ax = grid[row][col]

            if col < row:
                kwargs = {
                    "gridsize": n_points,
                    **lower_kde_kws,
                    "y": fac1,
                    "x": fac2,
                    "data": data,
                    "hue": hue,
                    "ax": ax,
                }
                try:
                    _ = sns.kdeplot(**kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    msg = _get_error_message(data, [fac1, fac2], "bivariate kdeplot")
                    warnings.warn(msg)

            elif col == row:
                kwargs = {
                    "gridsize": n_points,
                    **diag_kde_kws,
                    "y": None,
                    "x": fac1,
                    "data": data,
                    "hue": hue,
                    "ax": ax,
                }
                try:
                    _ = sns.kdeplot(**kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    msg = _get_error_message(data, fac1, "univariate kdeplot")
                    warnings.warn(msg)

            elif add_3d_plots:
                try:
                    _ = _3d_kdeplot(
                        x=fac1,
                        y=fac2,
                        data=data,
                        n_points=n_points,
                        ax=ax,
                        surface_kws=surface_kws,
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    msg = _get_error_message(data, [fac1, fac2], "surface plot")
                    warnings.warn(msg)

    sns.despine()

    if combine_plots_in_grid:
        out = grid[0][0].get_figure()
    else:
        out = {}
        for row, fac1 in enumerate(factors):
            for col, fac2 in enumerate(factors):
                _, visible = _get_ax_properties(row, col, add_3d_plots)
                if visible:
                    out[(fac1, fac2)] = grid[row][col].get_figure()

    return out


def _get_error_message(data, factors, plot_type):
    summary = data[factors].describe().round(3).to_string()
    tb = get_traceback()
    msg = (
        f"\n\n\nAn error occured while trying to generate a {plot_type} for the\n"
        f"factors\n\n\n    {factors}\n\n\nHere is some information on the factors:\n\n\n{summary}\n\n\n"
        f"The error was:\n{tb}"
    )
    return msg


def _3d_kdeplot(x, y, data, n_points, ax, surface_kws):
    xx, yy, f = _calculate_kde_for_3d(data_cleaned=data, a=x, b=y, n_points=n_points)
    kwargs = {
        "rstride": 1,
        "cstride": 1,
        "linewidth": 0,
        "cmap": "coolwarm",
        "edgecolor": "none",
        **surface_kws,
    }
    _ = ax.plot_surface(xx, yy, f, **kwargs)

    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False


def _process_data(states, period, factors):
    if isinstance(states, pd.DataFrame):
        data = states.query(f"period == {period}")[factors]
        hue = None
    else:
        if not isinstance(states, dict):
            states = {i: df for i, df in enumerate(states)}
        to_concat = []
        for name, df in states.items():
            df = df.query(f"period == {period}")[factors]
            df["scenario"] = name
            to_concat.append(df)
        data = pd.concat(to_concat)
        hue = "scenario"
    return data, hue


def _get_axes_grid(factors, combine_into_grid, add_3d_plots):
    dim = len(factors)
    axes = []
    if combine_into_grid:
        fig = plt.figure(figsize=(dim * 5, dim * 5))
        gs = fig.add_gridspec(dim, dim)
        for row in range(len(factors)):
            grid_row = []
            for col in range(len(factors)):
                proj, visible = _get_ax_properties(row, col, add_3d_plots)
                ax = fig.add_subplot(gs[row, col], projection=proj)
                ax.set_visible(visible)
                grid_row.append(ax)
            axes.append(grid_row)
    else:
        for row in range(len(factors)):
            grid_row = []
            for col in range(len(factors)):
                proj, visible = _get_ax_properties(row, col, add_3d_plots)
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": proj})
                grid_row.append(ax)
                ax.set_visible(visible)
            axes.append(grid_row)

    return axes


def _get_ax_properties(row, col, add_3d):
    projection = "3d" if add_3d and col > row else None
    visible = col <= row or add_3d
    return projection, visible


def _calculate_kde_for_3d(data_cleaned, a, b, n_points):
    x = data_cleaned[a]
    y = data_cleaned[b]
    variables = [a, b]
    lb1, lb2 = data_cleaned[variables].min()
    ub1, ub2 = data_cleaned[variables].max()

    cp = complex(n_points)
    xx, yy = np.mgrid[lb1:ub1:cp, lb2:ub2:cp]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f


def get_traceback():
    tb = format_exception(*sys.exc_info())
    if isinstance(tb, list):
        tb = "".join(tb)
    return tb
