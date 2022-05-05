import numpy as np
import pandas as pd
from plotly import graph_objects as go

from skillmodels.process_data import pre_process_data
from skillmodels.process_model import process_model


def plot_correlation_heatmap(
    corr,
    heatmap_kwargs=None,
    layout_kwargs=None,
    rounding=2,
    zmax=None,
    zmin=None,
    zmid=None,
    colorscale="RdBu_r",
    show_color_bar=True,
    show_diagonal=True,
    show_upper_triangle=True,
    trim_heatmap=False,
    annotate=True,
    annotation_fontsize=13,
    annotation_text_color="black",
    annotation_text_angle=0,
    axes_tick_fontsize=(12, 12),
    axes_tick_label_angle=(90, 0),
    axes_tick_label_color=("black", "black"),
):
    """Plot correlation heatmaps for factor measurements.

    Args:
        corr (DataFrame): Data frame of measurement or factor score correlations.
        heatmap_kwargs (dct): Dictionary of key word arguments to pass to go.Heatmap ().
            If None, the default kwargs defined in the function will be used.
        layout_kwargs (dct): Dictionary of key word arguments used to update layout of
            go.Figure object. If None, the default kwargs defined in the function will
            be used. Through layout_kwargs, you can edit figure properties such as
            - template
            - title
            - figsize
        rounding (int): Number of digits after the decimal point to round the
            correlation values to. Default 2.
        zmax (float ot NoneType): Upper bound to set on correlation color map. If None,
            is set to maximum absolute correlation value.
        zmin (float or NoneType): Lower bound to set on correlation color map. If None,
            is set to -zmax.
        zmid (float or NoneType): Midpoint to set on correlation color map. If None,
            is set to 0.
        colorscale (str): Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        show_color_bar (bool): A boolean variable for displaying heatmap colorbar.
            Default True.
        show_diagonal (bool): A boolean for displaying the correlations on the diagonal.
            Default False.
        show_upper_triangle (bool): A boolean for displaying upper triangular part
            of the correlation heatmap. Default False.

        The following arguments are processed into dictionaries or special plotly
        objects and passed to layout_kwargs. Defining them as additional arguments
        allows the user to pass values of desired properties without having to know
        how exactly plotly needs them to be passed as (e.g. as a nested dicitonary in
        casevof axes tick relevant arguments or as Annotation object in case of
        annotations).

        Same properties can be set via the argument layout_kwargs. In this case, values
        defined in layout_kwargs will overwrite values passed via the individual
        arguments.

        annotate (bool): If True, annotate the heatmap figure with correlation values.
            Default False.
        annotation_font_size (int): Font size of the annotation text. Default 13.
        annotation_font_color (str): Collor of the annotation text. Default 'black'.
        annotation_text_angle (float): The angle at which to rotate annotation text.
            Default 0.
        axes_tick_fontsize (list, tuple, other iterable or dict): Fontsize of axes
            ticks. Default (12,12)
        axes_tick_label_angle (list, tuple, other iterable or dict): Rotation angles of
            axes tick labels. Default (90,0).
        axes_tick_label_color (list, tuple, other iterable or dict): Colors of the axes
            tick labels. Default ('black', 'black').
    Returns:
        fig (plotly graph object): The figure with correlaiton heatmap.

    """
    corr = _process_corr_data_for_plotting(
        corr, rounding, show_upper_triangle, show_diagonal, trim_heatmap
    )
    heatmap_kwargs = _get_heatmap_kwargs(
        corr, heatmap_kwargs, colorscale, show_color_bar, zmax, zmin, zmid
    )
    layout_kwargs = _get_layout_kwargs(
        corr=corr,
        layout_kwargs=layout_kwargs,
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
        x=corr.columns.values,
        y=corr.index.values,
        **heatmap_kwargs,
    )
    fig = go.Figure(goh)
    fig.update_layout(**layout_kwargs)
    return fig


def get_measurements_corr(data, model_dict, factors, periods):
    """Get data frame with measurement correlations.

    Process data to retrieve measurements for each period and calculate correlations
    across period specific measurements.

    Args:
        data (pd.DataFrame): DataFrame with observed measurements.
        model_dict (dct): Dictionary of model attributes to be passed to process_model
            and extract measurements for each period.
        factors (list, str or NoneType): List of factors, to retrieve measurements for.
            If None, then calculate correlations of measurements of all factors.
        periods (int,float, list or NoneType): If int, the period within which to
            calculate measurement correlations. If a list, calculate correlations over
            periods. If None, calculate correlations across all periods.
    Returns:
        corr (DataFrame): DataFrame with measurement correlations.

    """
    data = data.copy(deep=True)
    model = process_model(model_dict)
    periods = _process_periods(periods, model)
    data = pre_process_data(data, periods)
    latent_factors, observed_factors = _process_factors(model, factors)
    update_info = model["update_info"]
    df = _get_measurement_data(
        data, update_info, periods, latent_factors, observed_factors
    )
    corr = df.corr()
    return corr


def get_scores_corr(data, model_dict, factors, periods):
    """Get data frame with correlations of factor scores.

    Process data to retrieve measurements for each period, standardize measurements
    to zero mean and unit standard deviation, take the mean of factor specific
    measurements in each period, and calculate correlations across those factor
    and period specific scores.

    The calculated scores coincide with factor scores for linear models.

    Args:
        data (pd.DataFrame): DataFrame with observed measurements.
        model_dict (dct): Dictionary of model attributes to be passed to process_model
            and extract measurements for each period.
        factors (list, str or NoneType): List of factors, to retrieve measurements for.
            If None, then calculate correlations of measurements of all factors.
        periods (int,float, list or NoneType): If int, the period within which to
            calculate measurement correlations. If a list, calculate correlations over
            periods. If None, calculate correlations across all periods.
    Returns:
        corr (DataFrame): DataFrame with score correlations.

    """
    data = data.copy(deep=True)
    model = process_model(model_dict)
    periods = _process_periods(periods, model)
    data = pre_process_data(data, periods)
    latent_factors, observed_factors = _process_factors(model, factors)
    update_info = model["update_info"]
    df = _get_quasi_factor_scores_data(
        data, update_info, periods, latent_factors, observed_factors
    )
    corr = df.corr()
    return corr


def _process_corr_data_for_plotting(
    corr, rounding, show_upper_triangle, show_diagonal, trim_heatmap
):
    """Apply mask and rounding to correlation DataFrame."""
    mask = _get_mask(corr, show_upper_triangle, show_diagonal)
    corr = corr.where(mask).round(rounding)
    if trim_heatmap:
        keeprows = mask.any(axis=1) & corr.notnull().any(axis="columns").to_numpy()
        mask = mask[keeprows]
        corr = corr[keeprows]
        keepcols = mask.any(axis=0) & corr.notnull().any(axis="index").to_numpy()
        mask = mask.T[keepcols].T
        corr = corr.T[keepcols].T
    return corr


def _get_mask(corr, show_upper_triangle, show_diagonal):
    """Get array to mask the correlation DataFrame."""
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.tril_indices_from(mask, k=-1)] = True
    if show_upper_triangle:
        mask[np.triu_indices_from(mask, k=1)] = True
    if show_diagonal:
        np.fill_diagonal(mask, True)
    return mask


def _get_measurement_data(data, update_info, periods, latent_factors, observed_factors):
    """Get data frame with factor measurements in each period, in wide format.

    For each factor, retrieve the data on measurements in each period and stack
    the data columns into a data frame.

    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (list): The list of periods that correlations are
            calculated for.
        latent_factors (list): List of latent factors the measurements of which
            correlations are calculated for.
        observed_factors (list): List of observed factors the measurements of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): Processed DataFrame to calculate correlations over.

    """
    if len(periods) == 1:
        period = periods[0]
        df = _get_measurement_data_for_single_period(
            data, update_info, period, latent_factors, observed_factors
        )
    else:
        df = _get_measurement_data_for_multiple_periods(
            data, update_info, periods, latent_factors, observed_factors
        )
    return df


def _get_measurement_data_for_single_period(
    data, update_info, period, latent_factors, observed_factors
):
    """Extract measurements of factors for the given period.

    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (int or float): The period to extract measurements for.
        latent_factors (list): List of latent factors the measurements of which
            correlations are calculated for.
        observed_factors (list): List of observed factors the measurements of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): DataFrame with measurements of factors for period 'period'.

    """

    period_info = update_info.loc[period].reset_index()
    measurements = []

    for fac in latent_factors:
        measurements += period_info.query(
            f"{fac} == True and purpose == 'measurement'"
        )["variable"].to_list()
    for fac in observed_factors:
        measurements.append(fac)
    df = data.query(f"{update_info.index.names[0]}=={period}")[measurements]
    return df


def _get_measurement_data_for_multiple_periods(
    data, update_info, periods, latent_factors, observed_factors
):
    """Extract measurements for factors for given periods.

    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (list): The periods to extract measurements for.
        latent_factors (list): List of latent factors the measurements of which
            correlations are calculated for.
        observed_factors (list): List of observed factors the measurements of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): DataFrame with measurements of factors in each period as
            columns.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _get_measurement_data_for_single_period(
                data, update_info, period, latent_factors, observed_factors
            )
            .add_suffix(f", {period}")
            .reset_index(drop=True)
        )
    df = pd.concat(to_concat, axis=1)
    return df


def _get_quasi_factor_scores_data(
    data, update_info, periods, latent_factors, observed_factors
):
    """Get data frame with summary information on factor measurements in each period.

    In each period, standardize factor measurements to zero mean and unit standard
    deviation, and for each factor take the average of all measurements as
    a summary statistics. The calculated scores coincide with factor scores for linear
    models.

    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (list): The list of periods that correlations are
            calculated for.
        latent_factors (list): List of latent factors the scores of which
            correlations are calculated for.
        observed_factors (list): List of observed factors the scores of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): Processed DataFrame to calculate correlations over.

    """
    if len(periods) == 1:
        period = periods[0]
        df = _get_quasi_factor_scores_data_for_single_period(
            data, update_info, period, latent_factors, observed_factors
        )
    else:
        df = _get_quasi_factor_scores_data_for_multiple_periods(
            data, update_info, periods, latent_factors, observed_factors
        )

    return df


def _get_quasi_factor_scores_data_for_single_period(
    data, update_info, period, latent_factors, observed_factors
):
    """Get frame with summary scores on factor measurements in a given period.

    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (list): The list of periods that correlations are
            calculated for.
        latent_factors (list): List of latent factors the scores of which
            correlations are calculated for.
        observed_factors (list): List of observed factors the scores of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): Processed DataFrame to calculate correlations over.

    """
    period_info = update_info.loc[period].reset_index()
    to_concat = []
    for factor in latent_factors:
        period_factor_measurements = period_info.query(
            f"{factor} == True and purpose == 'measurement'"
        )["variable"].to_list()
        df = data.query(f"{update_info.index.names[0]}=={period}")[
            period_factor_measurements
        ]
        df = (df - df.mean()) / df.std()
        sr = df.mean(axis=1)
        sr.name = f"{factor}"
        to_concat.append(sr)
    for factor in observed_factors:
        df = data.query(f"{update_info.index.names[0]}=={period}")[factor]
        to_concat.append(df)
    df = pd.concat(to_concat, axis=1)
    return df


def _get_quasi_factor_scores_data_for_multiple_periods(
    data, update_info, periods, latent_factors, observed_factors
):
    """Get frame with summary scores of factor measurements in a given period.

    Args:
        data (pd.DataFrame): Data with observable variables.
        update_info (pd.DataFrame): DataFrame with information on measurements
            for each factor in each model period.
        periods (list): The list of periods that correlations are
            calculated for.
        latent_factors (list): List of latent factors the scores of which
            correlations are calculated for.
        observed_factors (list): List of observed factors the scores of which
            correlations are calculated for.
    Returns:
        df (pd.DataFrame): Processed DataFrame to calculate correlations over.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _get_quasi_factor_scores_data_for_single_period(
                data, update_info, period, latent_factors, observed_factors
            )
            .add_suffix(f", {period}")
            .reset_index(drop=True)
        )
    df = pd.concat(to_concat, axis=1)
    return df


def _process_factors(model, factors):
    "Process factors to get a tuple of lists."
    if not factors:
        latent_factors = model["labels"]["latent_factors"]
        observed_factors = model["labels"]["observed_factors"]
    elif isinstance(factors, str):
        if factors in model["labels"]["latent_factors"]:
            latent_factors = [factors]
            observed_factors = []
        elif factors in model["labels"]["observed_factors"]:
            observed_factors = [factors]
            latent_factors = []
    else:
        observed_factors = []
        latent_factors = []
        for factor in factors:
            if factor in model["labels"]["latent_factors"]:
                latent_factors.append(factor)
            elif factor in model["labels"]["observed_factors"]:
                observed_factors.append(factor)
    return latent_factors, observed_factors


def _process_periods(periods, model):
    """Process periods to get a list."""
    if periods is None:
        periods = list(range(model["dimensions"]["n_periods"]))
    elif isinstance(periods, (int, float)):
        periods = [periods]
    return periods


def _get_layout_kwargs(
    corr,
    layout_kwargs,
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
        corr (DataFrame): The processed data frame with correlation coefficients.
        layout_kwargs (dct): Dictionary of keyword arguments used to update layout of
            go.Figure object.
        annotate (bool): Add annotations to the figure if True.
        annotation_font_size (int): Fontsize of the annotation text.
        annotation_font_color (str): Color of the annotation text.
        annotation_text_angle (float): The angle at which to rotate annotation text.
        axes_tick_fontsize(tuple,list or dict): Fontsizes of axes tick labels.
        axes_tick_label_angle(tuple,list or dict): The angle at which to rotate axes
            tick labels.
        axes_tick_label_color(tuple,list or dict): Collor of axes labels.
    Returns:
        default_layout_kwargs (dict): Dictionary to update figure layout.

    """
    default_layout_kwargs = {
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "template": "plotly_white",
        "yaxis_autorange": "reversed",
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
        for n in df.columns[::-1]:
            for m in df.index[::-1]:
                annotations.append(
                    {
                        "text": str(df.loc[m, n]).replace("nan", ""),
                        "x": n,
                        "y": m,
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


def _get_heatmap_kwargs(
    corr, heatmap_kwargs, colorscale, show_color_bar, zmax, zmin, zmid
):
    """Get kwargs to instantiate Heatmap object.

    Args:
        heatmap_kwargs (dct): Dictionary of key word arguments to pass to go.Heatmap().
        colorscale (str): Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        show_color_bar (bool): A boolean variable for displayin heatmap colorbar.
        zmax (float or None): Upper bound to set on correlation color map.
        zmin (float or None): Lower bound to set on correlation color map.
        zmid (float or None): Midpoint to set on correlation color map.

    Returns:
        default_heatmap_kwargs (dict): Dictionary of kwargs to instantiate go.Heatmap.

    """
    if zmax is None:
        zmax = np.abs(corr.to_numpy())[np.tril_indices_from(corr, k=-1)].max()
    if zmin is None:
        zmin = -zmax
    if zmid is None:
        zmid = 0
    default_heatmap_kwargs = {
        "colorscale": colorscale,
        "showscale": show_color_bar,
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
