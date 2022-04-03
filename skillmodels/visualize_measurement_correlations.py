"""Visualize the correlations of measurement variables in a given period."""
import numpy as np
import pandas as pd
from plotly import graph_objects as go

from skillmodels.process_data import _pre_process_data
from skillmodels.process_model import process_model


def visualize_measurement_correlations(
    periods,
    model_dict,
    data,
    *,
    factors=None,
    heatmap_kwargs=None,
    layout_kwargs=None,
    rounding=2,
    zmin=-1,
    zmax=1,
    zmid=0,
    colorscale="RdBu_r",
    show_diagonal=False,
    show_upper_triangle=False,
    show_title=True,
    annotate=False,
    annotation_fontsize=13,
    annotation_text_color="black",
    annotation_text_angle=0,
    axes_tick_fontsize=(12, 12),
    axes_tick_label_angle=(0, 0),
    axes_tick_label_color=("black", "black"),
):
    """Plot correlation heatmaps for factor measurements.
    Args:
        periods (int,float or list): If int, the period within which to calculate
            measurement correlations. If a list, calculate correlations over periods.
        model_dict (dct): Dictionary of model attributes to be passed to process_model
            and extract measurements for each period.
        data (pd.DataFrame): DataFrame with observed measurements.
        factors (list): List of factors, whose measurement correlation to calculate. If
            the default value of None is passed, then calculate and plot correlations
            of all measurements.
        heatmap_kwargs (dct): Dictionary of key word arguments to pass to go.Heatmap ().
            If None, the default kwargs defined in the function will be used.
        layout_kwargs (dct): Dictionary of key word arguments used to update layout of
            go.Figure object. If None, the default kwargs defined in the function will
            be used.
        rounding (int): Number of digits after the decimal point to round the
            correlation values to. Default 2.
        zmin (float): Lower bound to set on correlation color map. Default -1.
        zmax (float): Upper bound to set on correlation color map. Default 1.
        zmid (float): Midpoint to set on correlation color map. Default 0.
        colorscale (str): Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        show_diagonal (bool): A boolean for displaying the correlations on the diagonal.
            Default False.
        show_upper_triangle (bool): A boolean for displaying upper triangular part
            of the correlation heatmap. Default False.
        show_title (bool): if True, show figure title.
        annotate (bool): If True, annotate the heatmap figure with correlation values.
            Default False.
        annotation_font_size (int): Font size of the annotation text. Default 13.
        annotation_font_color (str): Collor of the annotation text. Default 'black'.
        annotation_text_angle (float): The angle at which to rotate annotation text.
            Default 0.
        axes_tick_fontsize (list, tuple, other iterable or dict): Fontsize of axes
            ticks. Default (12,12)
        axes_tick_label_angle (list, tuple, other iterable or dict): Rotation angles of
            axes tick labels. Default (0,0).
        axes_tick_label_color (list, tuple, other iterable or dict): Colors of the axes
            tick labels. Default ('black', 'black').
    Returns:
        fig (plotly graph object): The figure with correlaiton heatmap.


    """
    data = data.copy(deep=True)
    data = _pre_process_data(data)
    model = process_model(model_dict)
    factors = _get_factors(model, model)
    update_info = model["update_info"]
    corr = _get_correlation_matrix(
        data,
        update_info,
        periods,
        factors,
        rounding,
        show_diagonal,
        show_upper_triangle,
    )
    heatmap_kwargs = _get_heatmap_kwargs(heatmap_kwargs, colorscale, zmin, zmax, zmid)
    layout_kwargs = _get_layout_kwargs(
        corr,
        periods,
        layout_kwargs=layout_kwargs,
        show_title=show_title,
        annotate=annotate,
        annotation_fontsize=annotation_fontsize,
        annotation_text_color=annotation_text_color,
        annotation_text_angle=annotation_text_angle,
        axes_tick_fontsize=axes_tick_fontsize,
        axes_tick_label_angle=axes_tick_label_angle,
        axes_tick_label_color=axes_tick_label_color,
    )
    goh = go.Heatmap(
        z=corr,
        x=corr.index.values,
        y=corr.columns.values,
        **heatmap_kwargs,
    )
    fig = go.Figure(goh)
    fig.layout.update(**layout_kwargs)
    return fig


def _get_correlation_matrix(
    data,
    update_info,
    periods,
    factors,
    rounding,
    show_diagonal,
    show_upper_triangle,
):
    """Get correlation data frame to plot heatmap for.
    Process data, calculate correlations and process correlation DataFrame.
    Args:
        data (pd.DataFrame): DataFrame with observed measurements.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (int,float or list): If int, the period within which to calculate
            measurement correlations. If a list, calculate correlations over periods.
        factors (list): List of factors, whose measurement correlation to calculate. If
            the default value of None is passed, then calculate and plot correlations
            of all measurements.
        rounding (int): Number of digits after the decimal point to round the.
        show_diagonal (bool): A boolean for displaying the correlations on the diagonal.
        show_upper_triangle (bool): A boolean for displaying upper triangular part
            of the correlation heatmap.
    Returns:
        corr (pd.DataFrame): Processed correlation dataframe.

    """
    data = _process_data_for_plotting(data, update_info, periods, factors)
    corr = data.corr().round(rounding)
    mask = _get_mask(corr, show_upper_triangle, show_diagonal)
    corr = corr.where(mask)
    return corr


def _get_mask(corr, show_upper_triangle, show_diagonal):
    """Get array to mask the correlation DataFrame."""
    mask = np.zeros_like(corr, dtype=bool)
    dim = mask.shape[0]
    mask[np.tril_indices(dim)] = True
    if show_upper_triangle:
        mask[np.triu_indices(dim)] = True
    if not show_diagonal:
        np.fill_diagonal(mask, False)
    return mask


def _process_data_for_plotting(data, update_info, periods, factors):
    """Process data for passing to heatmap plot.
    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (int or list): The period or list of periods that correlations are
            calculated for.
        factors (list or tuple): List of factors the measurements of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): Processed DataFrame to calculate correlations over.

    """
    if isinstance(periods, list) and len(periods) == 1:
        periods = periods[0]
    if isinstance(periods, (int, float)):
        df = _process_data_for_plotting_with_single_period(
            data, update_info, periods, factors
        )
    elif isinstance(periods, list):
        df = _process_data_for_plotting_with_multiple_periods(
            data, update_info, periods, factors
        )
    return df


def _process_data_for_plotting_with_single_period(data, update_info, period, factors):
    """Extract measurements of factors for the given period.
    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (int or float): The period to extract measurements for.
        factors (list or tuple): List factors the measurements of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): DataFrame with measurements of factors for period 'period'.

    """

    period_info = update_info.loc[period].reset_index()
    measurements = []

    for fac in factors:
        measurements += period_info.query(
            f"{fac} == True and purpose == 'measurement'"
        )["variable"].to_list()

    df = data.query(f"{update_info.index.names[0]}=={period}")[measurements]
    return df


def _process_data_for_plotting_with_multiple_periods(
    data, update_info, periods, factors
):
    """Extract measurements for factors for given periods.
    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (list): The periods to extract measurements for.
        factors (list or str): List of or a single factor the measurements of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): DataFrame with measurements of factors in each period as
            columns.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _process_data_for_plotting_with_single_period(
                data, update_info, period, factors
            )
            .add_suffix(f"_{period}")
            .reset_index(drop=True)
        )
    df = pd.concat(to_concat, axis=1)
    return df


def _get_factors(model, factors):
    "Get list of factors."
    if not factors:
        factors = model["labels"]["all_factors"]
    elif isinstance(factors, str):
        factors = [factors]
    return factors


def _get_layout_kwargs(
    corr,
    periods,
    layout_kwargs,
    show_title,
    annotate,
    annotation_fontsize,
    annotation_text_color,
    annotation_text_angle,
    axes_tick_fontsize,
    axes_tick_label_angle,
    axes_tick_label_color,
):
    """Get kwargs to update figure layout.
    Args:
        periods (list): The periods to extract measurements for.
        layout_kwargs (dct): Dictionary of keyword arguments used to update layout of
            go.Figure object.
        show_title (bool): Show figure titel if True.
        annotate (bool): Add annotations to the figure if True.
        annotation_font_size (int): Fontsize of the annotation text.
        annotation_font_color (str): Color of the annotation text.
        annotation_text_angle (float): The angle at which to rotate annotation text.
        axes_tick_fontsize(tuple,list or dict): Fontsizes of axes tick labels.
            Default (11,11).
        axes_tick_label_angle(tuple,list or dict): The angle at which to rotate axes
            tick labels. Default (0,0).
        axes_tick_label_color(tuple,list or dict): Collor of axes labels.
            Default ('black', 'black').
    Returns:
        default_layout_kwargs (dict): Dictionary to update figure layout.

    """
    default_layout_kwargs = {
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "yaxis_autorange": "reversed",
        "template": "plotly_white",
    }
    default_layout_kwargs.update(
        _get_annotations(
            corr,
            annotate,
            annotation_fontsize,
            annotation_text_color,
            annotation_text_angle,
        )
    )
    default_layout_kwargs.update(
        _get_axes_ticks_kwargs(
            axes_tick_fontsize, axes_tick_label_angle, axes_tick_label_color
        )
    )
    if show_title:
        title = _get_fig_title(periods)
        default_layout_kwargs["title"] = title
    if layout_kwargs:
        default_layout_kwargs.update(layout_kwargs)
    return default_layout_kwargs


def _get_axes_ticks_kwargs(
    axes_tick_fontsize, axes_tick_label_angle, axes_tick_label_color
):
    """Get kwargs for axes ticks label formating."""
    axes_tick_fontsize = _process_axes_tick_args(axes_tick_fontsize)
    axes_tick_label_angle = _process_axes_tick_args(axes_tick_label_angle)
    axes_tick_label_color = _process_axes_tick_args(axes_tick_label_color)
    out = {}
    for ax in ["x", "y"]:
        out[f"{ax}axis"] = {
            "tickangle": axes_tick_label_angle[ax],
            "tickfont": {
                "color": axes_tick_label_color[ax],
                "size": axes_tick_fontsize[ax],
            },
        }
    return out


def _get_annotations(
    df, annotate, annotation_fontsize, annotation_text_color, annotation_text_angle
):
    """Get annotations and formatting kwargs."""
    annotation_kwargs = {}
    if annotate:
        annotations = []
        for n in df.index:
            for m in df.columns:
                annotations.append(
                    {
                        "text": str(df.loc[n, m]).replace("nan", ""),
                        "x": m,
                        "y": n,
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False,
                        "font": {
                            "color": annotation_text_color,
                            "size": annotation_fontsize,
                        },
                        "textangle": annotation_text_angle,
                    }
                )
        annotation_kwargs["annotations"] = annotations
    return annotation_kwargs


def _get_fig_title(periods):
    """Get title of correlation heatmap.
    Args:
        periods (int or list): The period or list of periods that correlations
            are calculated for.
    Returns:
        title (str): Title for the correlation heatmap that describes which periods
            the correlations have been calculated for.
    """
    if isinstance(periods, list) and len(periods) == 1:
        periods = periods[0]
    if isinstance(periods, list):
        title = f"Periods: {periods[0]}-{periods[-1]}"
    elif isinstance(periods, (int, float)):
        title = f"Period: {periods}"
    return title


def _get_heatmap_kwargs(heatmap_kwargs, colorscale, zmin, zmax, zmid):
    """Get kwargs to instantiate Heatmap object.
    Args:
        heatmap_kwargs (dct): Dictionary of key word arguments to pass to go.Heatmap().
        colorscale (str): Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        zmin (float): Lower bound to set on correlation color map. Default -1.
        zmax (float): Upper bound to set on correlation color map. Default 1.
        zmid (float): Midpoint to set on correlation color map. Default 0.

    Returns:
        default_heatmap_kwargs (dict): Dictionary to instantiate go.Heatmap.

    """
    default_heatmap_kwargs = {
        "colorscale": colorscale,
        "zmin": zmin,
        "zmax": zmax,
        "zmid": zmid,
    }
    if heatmap_kwargs:
        default_heatmap_kwargs.update(heatmap_kwargs)
    return default_heatmap_kwargs


def _process_axes_tick_args(args):
    if isinstance(args, (tuple, list)):
        args = {"x": args[0], "y": args[1]}
    return args
