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
from skillmodels.utils_plotting import get_layout_kwargs, get_make_subplot_kwargs


def combine_distribution_plots(
    kde_plots,
    contour_plots,
    surface_plots=None,
    factor_order=None,
    factor_mapping=None,
    make_subplot_kwargs=None,
    sharex=False,
    sharey=False,
    line_width=1.5,
    showlegend=False,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
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
        factor_mapping (dct): Dictionary to change displayed factor names.
        sharex (bool): Whether to share the properties of x-axis across subplots.
            Default False.
        sharey (bool): Whether to share the properties ofy-axis across subplots.
            Default True.
        line_width (float): A float used to set same line width across subplots.
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
    factor_names = _process_factor_mapping_dist(factor_mapping, factors)
    ordered_factors = _get_ordered_factors(factor_order, factors)
    make_subplot_kwargs = get_make_subplot_kwargs(
        sharex=sharex,
        sharey=sharey,
        column_order=ordered_factors,
        row_order=ordered_factors,
        make_subplot_kwargs=make_subplot_kwargs,
        add_scenes=True,
    )

    fig = make_subplots(**make_subplot_kwargs)
    layout_kwargs = get_layout_kwargs(
        layout_kwargs=layout_kwargs,
        legend_kwargs=legend_kwargs,
        title_kwargs=title_kwargs,
        showlegend=showlegend,
        columns=ordered_factors,
        rows=ordered_factors,
    )
    fig.update_layout(**layout_kwargs)

    for col, fac1 in enumerate(ordered_factors):
        for row, fac2 in enumerate(ordered_factors):
            if row > col:
                plot = contour_plots[(fac1, fac2)]
                for d in plot.data:
                    d.update({"showlegend": False})
                    fig.add_trace(d, col=col + 1, row=row + 1)
                    fig.update_xaxes(
                        title=factor_names[plot.layout.xaxis.title.text],
                        col=col + 1,
                        row=row + 1,
                    )
                    fig.update_yaxes(
                        title=factor_names[plot.layout.yaxis.title.text],
                        col=col + 1,
                        row=row + 1,
                    )
                    fig.update_traces(line_width=line_width, row=row + 1, col=col + 1)
            elif row == col:
                for d in kde_plots[fac1].data:
                    if row != 0:
                        d.update({"showlegend": False})
                    fig.add_trace(d, col=col + 1, row=row + 1)
                    fig.update_xaxes(title=factor_names[fac1], col=col + 1, row=row + 1)
                    fig.update_yaxes(title="Density", col=col + 1, row=row + 1)
                    fig.update_traces(line_width=line_width, row=row + 1, col=col + 1)

            elif surface_plots is not None:
                plot = surface_plots[(fac1, fac2)]
                camera = {"eye": {"x": eye_x, "y": eye_y, "z": eye_z}}
                fig.add_trace(plot.data[0], col=col + 1, row=row + 1)
                fig.update_scenes(camera=camera, row=row + 1, col=col + 1)
                fig.update_scenes(
                    xaxis={"title": "", "showgrid": True},
                    row=row + 1,
                    col=col + 1,
                )
                fig.update_scenes(
                    yaxis={"title": "", "showgrid": True},
                    row=row + 1,
                    col=col + 1,
                )
                fig.update_scenes(
                    zaxis={"title": "", "showgrid": True},
                    row=row + 1,
                    col=col + 1,
                )
    return fig


def univariate_densities(
    data,
    model_dict,
    params,
    period,
    factors=None,
    observed_factors=False,
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
        observed_factors (bool): If True, plot densities of observed factors too.
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
        states = get_filtered_states(model_dict=model_dict, data=data, params=params)[
            "anchored_states"
        ]["states"]
    model = process_model(model_dict)
    factors = _get_factors(
        model=model,
        factors=factors,
        observed_factors=observed_factors,
    )
    observed_states = _get_data_observed_factors(data, factors)
    df = _process_data(states, period, factors, observed_states)
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
    layout_kwargs = get_layout_kwargs(layout_kwargs)
    for fac in factors:
        hist_data = [df[fac][df["scenario"] == s] for s in scenarios]
        try:
            fig = ff.create_distplot(hist_data, **distplot_kwargs)
        except Exception as e:
            warnings.warn(
                f"""Plotting univariate density failed for {fac} in
                period {period} with error:\n\n{e}""",
            )
            fig = go.Figure()
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title=fac)
        fig.update_layout(yaxis_title="Density")
        fig.update_layout(**layout_kwargs)
        plots_dict[fac] = fig
    return plots_dict


def bivariate_density_contours(
    data,
    model_dict,
    params,
    period,
    factors=None,
    observed_factors=False,
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
        observed_factors (bool): If True, plot densities of observed factors too.
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
        states = get_filtered_states(model_dict=model_dict, data=data, params=params)[
            "anchored_states"
        ]["states"]
    model = process_model(model_dict)
    factors = _get_factors(
        model=model,
        factors=factors,
        observed_factors=observed_factors,
    )
    observed_states = _get_data_observed_factors(data=data, factors=factors)
    df = _process_data(states, period, factors, observed_states)
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
        fig = go.Figure()
        for i, scenario in enumerate(df["scenario"].unique()):
            try:
                x, y, z = _calculate_kde_for_3d(
                    df[df["scenario"] == scenario],
                    pair,
                    n_points,
                )
                contour = go.Contour(
                    x=x[:, 0],
                    y=y[0, :],
                    z=z,
                    line={"color": getattr(px.colors.qualitative, lines_colorscale)[i]},
                )
                fig.add_trace(contour)
                fig.update_traces(**contour_kwargs)
            except Exception as e:
                warnings.warn(
                    f"""
                    Contour plot failed for {pair} in period {period}
                    with error:\n\n{e}
                    """,
                )
        fig.update_xaxes(title={"text": pair[0]})
        fig.update_yaxes(title={"text": pair[1]})
        fig.update_layout(**layout_kwargs)
        plots_dict[pair] = fig

    return plots_dict


def bivariate_density_surfaces(
    data,
    model_dict,
    params,
    period,
    factors=None,
    observed_factors=False,
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
        observed_factors (bool): If True, plot densities of observed factors too.
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
        states = get_filtered_states(model_dict=model_dict, data=data, params=params)[
            "anchored_states"
        ]["states"]
    elif not isinstance(states, pd.DataFrame):
        raise ValueError("3d plots are only supported if states is a DataFrame")
    model = process_model(model_dict)
    factors = _get_factors(
        model=model,
        factors=factors,
        observed_factors=observed_factors,
    )
    observed_states = _get_data_observed_factors(data, factors)
    df = _process_data(states, period, factors, observed_states)
    plots_dict = {}
    layout_kwargs = _process_layout_kwargs_3d(
        layout_kwargs,
        showgrids,
        showaxlines,
        showlabels,
    )
    pairs = []
    for fac1 in factors:
        for fac2 in factors:
            if fac1 != fac2:
                pairs.append((fac1, fac2))
    pairs = list(set(pairs))
    for pair in pairs:
        try:
            x, y, z = _calculate_kde_for_3d(df, pair, n_points)
            fig = go.Figure(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    showscale=showcolorbar,
                    colorscale=colorscale,
                    opacity=opacity,
                ),
            )
        except Exception as e:
            warnings.warn(
                f"""Plotting bivariate density surfaces for {pair} in
                period {period} with error:\n\n{e}""",
            )
            fig = go.Figure()
        fig.update_layout(
            scene={
                "xaxis": {"title": pair[0]},
                "yaxis": {"title": pair[1]},
                "zaxis": {"title": ""},
            },
        )
        fig.update_layout(**layout_kwargs)
        plots_dict[pair] = fig
    return plots_dict


def _process_data(states, period, factors, observed_states=None):
    if isinstance(states, pd.DataFrame):
        to_concat = []
        for fac in factors:
            if fac in states:
                to_concat.append(states.query(f"period == {period}")[fac])
        data = pd.concat(to_concat, axis=1)
        data["scenario"] = "none"
    else:
        if not isinstance(states, dict):
            states = dict(enumerate(states))
        to_concat = []
        for name, df in states.items():
            to_keep = df.query(f"period == {period}")[factors].copy()
            to_keep["scenario"] = name
            to_concat.append(to_keep)
        data = pd.concat(to_concat)
    data = data.reset_index()
    if observed_states is not None:
        data = pd.concat(
            [data, observed_states.query(f"period == {period}").reset_index()],
            axis=1,
        )
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


def _process_factor_mapping_dist(mapper, factors):
    """Process mapper to return dictionary with old and new factor names."""
    if mapper is None:
        mapper = {fac: fac for fac in factors}
    else:
        for fac in factors:
            if fac not in mapper:
                mapper[fac] = fac
    return mapper


def _get_ordered_factors(factor_order, factors):
    """Process factor orders to return list of strings."""
    if factor_order is None:
        ordered_factors = factors
    elif isinstance(factor_order, str):
        ordered_factors = [factor_order]
    else:
        ordered_factors = factor_order
    return ordered_factors


def _get_factors(factors, observed_factors, model):
    """Proccess factor names to return list of strings."""
    if factors is None:
        if observed_factors:
            factors = model["labels"]["all_factors"]
        else:
            factors = model["labels"]["latent_factors"]
    return factors


def _get_data_observed_factors(data, factors):
    """Get data with observed factors if any."""
    to_concat = []
    for fac in factors:
        if fac in data:
            to_concat.append(data[fac])
    if len(to_concat) >= 1:
        observed_states = pd.DataFrame(pd.concat(to_concat))
    else:
        observed_states = None
    return observed_states
