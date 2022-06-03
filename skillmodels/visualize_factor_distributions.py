from copy import deepcopy
from logging import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from skillmodels.filtered_states import get_filtered_states
from skillmodels.process_model import process_model


def combine_distribution_plots(
    kde_plots,
    contour_plots,
    surface_plots=None,
    make_subplot_kwargs=None,
    sharex=False,
    sharey=False,
    vertical_spacing=0.1,
    horizontal_spacing=0.1,
    line_width=1.5,
    showlegend=False,
    eye_x=2.2,
    eye_y=2.2,
    eye_z=1,
):
    """Combine individual plots into figure with subplots.

    Uses dictionary with plotly images as values to build plotly Figure with subplots.

    Args:
        kde_plots (dict): Dictionary with plots of indivudal factor kde plots.
        contour_plots (dict): Dictionary with plots of pairwise factor density
            contours.
        surface_plots (dict): Dictionary with plots of pairwise factor density
            3d plots.
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots. If None,
            default arguments defined in the function are used.
        sharex (bool): Whether to share the properties of x-axis across subplots.
            Default False.
        sharey (bool): Whether to share the properties ofy-axis across subplots.
            Default True.
        vertical_spacing (float): Vertical spacing between subplots.
        horizaontal_spacing (float): Horizontal spacing between subplots.
        line_width (float): A float used to set same line width across subplots.
        showlegend (bool): A boolean for displaying plots' legend.
        eye_x, eye_y and eye_z (float): Control camera (view point) of the 3d plots.
            Together they form the a norm, and the larger the norm, the more zoomed out
            is the view. Setting eye_z to a lower value lowers the view point.

    Returns:
        fig (plotly.Figure): Plotly figure with subplots that combines pairwise
            distrubtion plots.

    """
    kde_plots = deepcopy(kde_plots)
    contour_plots = deepcopy(contour_plots)
    surface_plots = deepcopy(surface_plots)
    factors = list(kde_plots.keys())
    make_subplot_kwargs = _get_make_subplot_kwargs_with_scenes(
        sharex,
        sharey,
        factors,
        vertical_spacing,
        horizontal_spacing,
        make_subplot_kwargs,
    )

    fig = make_subplots(**make_subplot_kwargs)
    fig.update_layout(
        height=len(factors) * 300,
        width=len(factors) * 300,
        template="simple_white",
        showlegend=showlegend,
    )
    for col, fac1 in enumerate(factors):
        for row, fac2 in enumerate(factors):

            if row > col:
                for d in contour_plots[(fac1, fac2)].data:

                    d.update({"showlegend": False})
                    fig.add_trace(d, col=col + 1, row=row + 1)
                    fig.update_xaxes(title=fac1, col=col + 1, row=row + 1)
                    fig.update_yaxes(title=fac2, col=col + 1, row=row + 1)
                    fig.update_traces(line_width=line_width, row=row + 1, col=col + 1)
            elif row == col:
                for d in kde_plots[fac1].data:

                    if not row == 0:
                        d.update({"showlegend": False})
                    fig.add_trace(d, col=col + 1, row=row + 1)
                    fig.update_xaxes(title=fac1, col=col + 1, row=row + 1)
                    fig.update_yaxes(title="Density", col=col + 1, row=row + 1)
                    fig.update_traces(line_width=line_width, row=row + 1, col=col + 1)

            else:
                if surface_plots is not None:
                    camera = {"eye": {"x": eye_x, "y": eye_y, "z": eye_z}}
                    fig.add_trace(
                        surface_plots[(fac2, fac1)].data[0], col=col + 1, row=row + 1
                    )
                    fig.update_scenes(camera=camera, row=row + 1, col=col + 1)
                    fig.update_scenes(
                        xaxis={"title": "", "showgrid": True}, row=row + 1, col=col + 1
                    )
                    fig.update_scenes(
                        yaxis={"title": "", "showgrid": True}, row=row + 1, col=col + 1
                    )
                    fig.update_scenes(
                        zaxis={"title": "", "showgrid": True}, row=row + 1, col=col + 1
                    )
    return fig


def univariate_densities(
    data,
    model_dict,
    params,
    period,
    factors=None,
    states=None,
    show_curve=True,
    show_hist=False,
    show_rug=False,
    curve_type="kde",
    colorscale="D3",
    bin_size=1,
    distplot_kwargs=None,
    layout_kwargs=None,
):
    """Get dictionary with kernel density estimate plots for each factor.

    Plots kernel densities for latent factors and collects them in a dictionary
    with factor names as keys.

    Args:
        data (DataFrame): Model estimation input data.
        model_dict (dict): Dictionary with model specifications.
        params (DataFrame): DataFrame with estimated parameter values.
        period (int or float): Model period for which to plot the distributions for.
        factors (list or NoneType): List of factors for which to plot the densities.
            If None, plot pairwise distributions for all latent factors.
        states (dict, list, pd.DataFrame or NoneType): List or dictionary with tidy
            DataFrames with filtered or simulated states or only one DataFrame with
            filtered or simulated states. If None, retrieve data frame with filtered
            states using model_dict and data. States are used to estimate the state
            ranges in each period (if state_ranges are not given explicitly) and to
            estimate the distribution of the latent factors.
        show_hist (bool): Add histogram to the distplot.
        show_curve (bool): Add density curve to the displot.
        show_rug (bool): Add rug to the distplot.
        curve_type (str): Curve type, 'normal' or 'kde', to add to the distplot.
        colorscale (str): The color palette used when plotting multiple data. Must be
            a valid attribute of px.colors.qualitative.
        bin_size (float): Size of the histogram bins.
        distplot_kwargs (NoneType or dict): Dictionary with additional keyword
            arguments passed to ff.create_distplot() to initiate
            the distplot.
        layout_kwargs (NoneType or dict): Dictionary of keyword arguments to update
            layout of the plot figures. Some essential layout kwargs are:
            - xaxis_title (str): label label
            - yaxis_title (str): label of y axis
            - xaxis_showgrid (bool): display axis grid
            - yaxis_showgrid (bool): display axis grid
            - template (str): figure background theme
            - showlegend (bool): add legend
    Returns:
        plots_dict (dict): Dictionary with density plots.

    """
    if states is None:
        states = get_filtered_states(model_dict=model_dict, data=data, params=params,)[
            "anchored_states"
        ]["states"]
    model = process_model(model_dict)
    if factors is None:
        factors = model["labels"]["latent_factors"]
    df = _process_data(states, period, factors)
    scenarios = df["scenario"].unique()
    plots_dict = {}
    distplot_kwargs = _process_distplot_kwargs(
        show_curve,
        show_hist,
        show_rug,
        curve_type,
        bin_size,
        scenarios,
        colorscale,
        distplot_kwargs,
    )
    plots_dict = {}
    layout_kwargs = _process_layout_kwargs(layout_kwargs)
    for fac in factors:
        hist_data = [df[fac][df["scenario"] == s] for s in scenarios]
        try:
            fig = ff.create_distplot(hist_data, **distplot_kwargs)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title=fac)
            fig.update_layout(yaxis_title="Density")
            fig.update_layout(**layout_kwargs)
            plots_dict[fac] = fig
        except Exception as e:
            warnings.warn(
                f"Plotting univariate density failed for {fac} in period {period} with error:\n\n{e}"
            )

    return plots_dict


def bivariate_density_contours(
    data,
    model_dict,
    params,
    period,
    factors=None,
    states=None,
    n_points=50,
    contour_kwargs=None,
    layout_kwargs=None,
    contours_showlabels=False,
    contours_coloring="none",
    contours_colorscale="RdBu_r",
    lines_colorscale="D3",
    showcolorbar=False,
):
    """Get dictionary with pariwise density contour plots.

    Plots pairwise bivariate density contours for latent factors
    and collects them in a dictionary with factor combinations as keys.

    Args:
        data (DataFrame): Model estimation input data.
        model_dict (dict): Dictionary with model specifications.
        params (DataFrame): DataFrame with estimated parameter values.
        period (int or float): Model period for which to plot the distributions for.
        factors (list or NoneType): List of factors for which to plot the densities.
            If None, plot pairwise distributions for all latent factors.
        states (dict, list, pd.DataFrame or NoneType): List or dictionary with tidy
            DataFrames with filtered or simulated states or only one DataFrame with
            filtered or simulated states. If None, retrieve data frame with filtered
            states using model_dict and data. States are used to estimate the state
            ranges in each period (if state_ranges are not given explicitly) and to
            estimate the distribution of the latent factors.
        n_points (int): Number of grid points used to create the mesh for calculation
            of kernel densities.
        contour_kwargs (dict or NoneType): Dictionary with keyword arguments to set
            contour line properties (such as annotation, colorscale).
        layout_kwargs (dict or NoneType): Dictionary with keyword arguments to set
            figure layout properties.

        The following are various essential keyword arguments defining various features
        of plots. All features can also be changed ex-post via 'update_layout' or
        'update_traces'. Some default figure layout properties (such as background
        theme) are defined if layout_kwargs is None.

        contours_showlabels (bool): If True, annotate density contours.
        contours_coloring (str): Defines how to apply color scale to density contours.
            Possible values are in ['lines', 'fill', 'heatmap', 'none']. Default is
            'none' which implies no colorscale.
        contours_colorscale (str): The color scale to use for line legends. Must be
            a valid plotly.express.colors.sequential attribute. Default 'RdBu_r'.
        showcolorbar (bool): A boolean variable for displaying color bar.

    Returns:
        plots_dict (dict): Dictionary with factor combinations as keys and respective
            pariwise plots of density contours as values.

    """

    if states is None:
        states = get_filtered_states(model_dict=model_dict, data=data, params=params,)[
            "anchored_states"
        ]["states"]
    model = process_model(model_dict)
    if factors is None:
        factors = model["labels"]["latent_factors"]
    df = _process_data(states, period, factors)
    plots_dict = {}
    contour_kwargs = _process_contour_kwargs(
        contour_kwargs,
        contours_showlabels,
        contours_coloring,
        contours_colorscale,
        showcolorbar,
    )
    layout_kwargs = _process_layout_kwargs(layout_kwargs)
    pairs = []
    for fac1 in factors:
        for fac2 in factors:
            if fac1 != fac2:
                pairs.append((fac1, fac2))
    pairs = list(set(pairs))
    for pair in pairs:
        try:
            fig = go.Figure()
            for i, scenario in enumerate(df["scenario"].unique()):
                x, y, z = _calculate_kde_for_3d(
                    df[df["scenario"] == scenario], pair, n_points
                )
                contour = go.Contour(
                    x=x[:, 0],
                    y=y[0, :],
                    z=z,
                    line={"color": getattr(px.colors.qualitative, lines_colorscale)[i]},
                )
                fig.add_trace(contour)
                fig.update_traces(**contour_kwargs)
            fig.update_xaxes(title={"text": pair[0]})
            fig.update_yaxes(title={"text": pair[1]})
            fig.update_layout(**layout_kwargs)
            plots_dict[pair] = fig
        except Exception as e:
            warnings.warn(
                f"Contour plot failed for {pair} in period {period} with error:\n\n{e}"
            )

    return plots_dict


def bivariate_density_surfaces(
    data,
    model_dict,
    params,
    period,
    factors=None,
    states=None,
    n_points=50,
    layout_kwargs=None,
    colorscale="RdBu_r",
    opacity=0.9,
    showcolorbar=False,
    showgrids=True,
    showaxlines=True,
    showlabels=True,
):
    """Get dictionary with pariwise 3d density surface plots.

    Plots pairwise 3d density surfaces for latent factors
    and collects them in a dictionary with factor name combinations keys.

    Args:
        data (DataFrame): Model estimation input data.
        model_dict (dict): Dictionary with model specifications.
        params (DataFrame): DataFrame with estimated parameter values.
        period (int or float): Model period for which to plot the distributions for.
        factors (list or NoneType): List of factors for which to plot the densities.
            If None, plot pairwise distributions for all latent factors.
        states (dict, list, pd.DataFrame or NoneType): List or dictionary with tidy
            DataFrames with filtered or simulated states or only one DataFrame with
            filtered or simulated states. If None, retrieve data frame with filtered
            states using model_dict and data. States are used to estimate the state
            ranges in each period (if state_ranges are not given explicitly) and to
            estimate the distribution of the latent factors.
        n_points (int): Number of grid points used to create the mesh for calculation
            of kernel densities.
        The following are various essential keyword arguments defining various features
        of plots. All features can also be changed ex-post via 'update_layout' or
        'update_traces'. Some default figure layout properties (such as background
        theme) are defined if layout_kwargs is None.

        layout_kwargs (dict or NoneType): Dictionary with keyword arguments to set
            figure layout properties.
        colorscale (str): The color scale to use for line legends. Must be a valid
            plotly.express.colors.sequential attribute. Default 'RdBu_r'.
        showcolorbar (bool): A boolean variable for displaying the colorbar associated
            with the surface color scale.
        showgrids (bool): A boolean variable for showing axes grids.
        showaxlines (bool): A boolean variable for showing axes lines.
        showlabels (bool): A boolean variable for displaying axes labels.

    Returns:
        plots_dict (dict): Dictionary with factor combinations as keys and respective
            pariwise plots of 3d density plots as values.

    """
    if states is None:
        states = get_filtered_states(model_dict=model_dict, data=data, params=params,)[
            "anchored_states"
        ]["states"]
    elif not isinstance(states, pd.DataFrame):
        raise ValueError("3d plots are only supported if states is a DataFrame")
    model = process_model(model_dict)
    if factors is None:
        factors = model["labels"]["latent_factors"]
    df = _process_data(states, period, factors)
    plots_dict = {}
    layout_kwargs = _process_layout_kwargs_3d(
        layout_kwargs, showgrids, showaxlines, showlabels
    )
    pairs = []
    for fac1 in factors:
        for fac2 in factors:
            if fac1 != fac2:
                pairs.append((fac1, fac2))
    pairs = list(set(pairs))
    for pair in pairs:
        x, y, z = _calculate_kde_for_3d(df, pair, n_points)
        fig = go.Figure(
            go.Surface(
                x=x,
                y=y,
                z=z,
                showscale=showcolorbar,
                colorscale=colorscale,
                opacity=opacity,
            )
        )
        fig.update_layout(
            scene={
                "xaxis": {"title": pair[0]},
                "yaxis": {"title": pair[1]},
                "zaxis": {"title": ""},
            }
        )
        fig.update_layout(**layout_kwargs)
        plots_dict[pair] = fig
    return plots_dict


def _process_data(states, period, factors):
    if isinstance(states, pd.DataFrame):
        data = states.query(f"period == {period}")[factors]
        data["scenario"] = "none"
    else:
        if not isinstance(states, dict):
            states = {i: df for i, df in enumerate(states)}
        to_concat = []
        for name, df in states.items():
            df = df.query(f"period == {period}")[factors]
            df["scenario"] = name
            to_concat.append(df)
        data = pd.concat(to_concat)
    data = data.reset_index()
    return data


def _process_distplot_kwargs(
    show_curve,
    show_hist,
    show_rug,
    curve_type,
    bin_size,
    scenarios,
    colorscale,
    distplot_kwargs,
):
    """Define and update default distplot kwargs."""
    default_kwargs = {
        "show_hist": show_hist,
        "show_rug": show_rug,
        "show_curve": show_curve,
        "curve_type": curve_type,
        "bin_size": bin_size,
        "group_labels": scenarios,
        "colors": getattr(px.colors.qualitative, colorscale),
    }
    if distplot_kwargs:
        default_kwargs.update(distplot_kwargs)
    return default_kwargs


def _calculate_kde_for_3d(data, factors, n_points):
    """Create grid mesh and calculate Gaussian kernel over the grid."""
    x = data[factors[0]]
    y = data[factors[1]]
    lbx = x.min()
    lby = y.min()
    ubx = x.max()
    uby = y.max()
    xx, yy = np.mgrid[lbx : ubx : complex(n_points), lby : uby : complex(n_points)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(np.vstack([x, y]))
    zz = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, zz


def _process_contour_kwargs(
    contour_kwargs,
    contours_showlabels,
    contours_coloring,
    contours_colorscale,
    contours_showscale,
):
    """Define and update default density contour kwargs."""
    if contours_coloring is None:
        contours_coloring = "none"
    default_kwargs = {
        "contours_coloring": contours_coloring,
        "contours_showlabels": contours_showlabels,
        "colorscale": contours_colorscale,
        "showscale": contours_showscale,
    }

    if contour_kwargs:
        default_kwargs.update(contour_kwargs)
    return default_kwargs


def _process_layout_kwargs(layout_kwargs):
    """Define and update default figure layout kwargs."""
    default_kwargs = {
        "template": "simple_white",
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
    }
    if layout_kwargs:
        default_kwargs.update(layout_kwargs)
    return default_kwargs


def _process_layout_kwargs_3d(layout_kwargs, showgrids, showaxlines, showlabels):
    """Define and update default figure layout kwargs for 3d plots."""
    default_kwargs = {
        "template": "none",
    }
    scene = {}
    for ax in list("xyz"):
        scene[f"{ax}axis"] = {
            "showgrid": showgrids,
            "showline": showaxlines,
        }
        if showlabels is False:
            scene[f"{ax}axis"]["title"] = ""
    default_kwargs["scene"] = scene
    if layout_kwargs:
        default_kwargs.update(layout_kwargs)
    return default_kwargs


def _get_make_subplot_kwargs_with_scenes(
    sharex, sharey, factors, vertical_spacing, horizontal_spacing, make_subplot_kwargs
):
    """Define and update keywargs for instantiating figure with subplots."""
    default_kwargs = {
        "rows": len(factors),
        "cols": len(factors),
        "start_cell": "top-left",
        "print_grid": False,
        "shared_yaxes": sharey,
        "shared_xaxes": sharex,
        "vertical_spacing": vertical_spacing,
        "horizontal_spacing": horizontal_spacing,
    }

    specs = np.array([[{}] * len(factors)] * len(factors))
    for i in range(len(factors)):
        for j in range(len(factors)):
            if i < j:
                specs[i, j] = {"type": "scene"}
    default_kwargs["specs"] = specs.tolist()
    default_kwargs["vertical_spacing"] = vertical_spacing
    if make_subplot_kwargs is not None:
        default_kwargs.update(make_subplot_kwargs)
    return default_kwargs
