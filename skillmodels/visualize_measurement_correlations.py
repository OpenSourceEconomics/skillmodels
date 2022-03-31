"""Visualize the correlations of measurement variables in a given period."""
import numpy as np
import pandas as pd
from plotly import graph_objects as go

from skillmodels.process_model import process_model


def visualize_measurement_correlations(
    periods,
    model_dict,
    data,
    *,
    return_type=None,
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
    annotate=False,
    show_title=True,
):
    """Plot correlation heatmaps for factor measurements.
    Args:
        period(int,float or list): If int, the period within which to calculate
            measurement correlations. If a list, calculate correlations over periods.
        model_dict(dct): Dictionary of model attributes to be passed to process_model
            and extract measurements for each period.
        data(pd.DataFrame): DataFrame with observed measurements.
        return_type(None, str or pathlib.Path): If None, return the figure object.
            Else, must be a path to save the figure to.
        factors(list): List of factors, whose measurement correlation to calculate. If
            the default value of None is passed, then calculate and plot correlations
            of all measurements.
        heatmap_kwargs(dct): Dictionary of key word arguments to pass to go.Heatmap().
            If None, the default kwargs defined in the function will be used.
        layout_kwargs(dct): Dictionary of key word arguments used to update layout of
            go.Figure object. If None, the default kwargs defined in the function will
            be used.
        rounding(int): Number of digits after the decimal point to round the
            correlation values to. Default 2.
        zmin (float): Lower bound to set on correlation color map. Default -1.
        zmax (float): Upper bound to set on correlation color map. Default 1.
        zmid(float): Midpoint to set on correlation color map. Default 0.
        colorscale(str): Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        show_diagonal(bool): A boolean for displaying the correlations on the diagonal.
            Default False.
        show_upper_triangular(bool): A boolean for displaying upper triangular part
            of the correlation heatmap. Default False.
        annotate(bool): if True, annotate the heatmap figure with correlation values.
        show_title(bool): if True, show figure title.
    Returns:
        fig(plotly figure): The figure with correlaiton heatmap.

    """
    data = data.copy(deep=True)
    model = process_model(model_dict)
    period_name = model["update_info"].index.names[0]
    corr = _get_correlation_matrix(
        data,
        model,
        periods,
        period_name,
        factors,
        rounding,
        show_diagonal,
        show_upper_triangle,
    )
    heatmap_kwargs = _get_heatmap_kwargs(heatmap_kwargs, colorscale, zmin, zmax, zmid)
    layout_kwargs = _get_layout_kwargs(
        corr,
        layout_kwargs,
        annotate,
        show_title,
        periods,
        period_name,
    )
    goh = go.Heatmap(
        z=corr,
        x=corr.index.values,
        y=corr.columns.values,
        **heatmap_kwargs,
    )
    fig = go.Figure(goh)
    fig.layout.update(**layout_kwargs)
    if not return_type:
        return fig
    else:
        fig.write_image(return_type)


def _get_correlation_matrix(
    data,
    model,
    periods,
    period_name,
    factors,
    rounding,
    show_diagonal,
    show_upper_triangle,
):
    data = _process_data_for_plotting(data, model, periods, period_name, factors)
    corr = data.corr().round(rounding)
    mask = _get_mask(corr, show_upper_triangle, show_diagonal)
    corr = corr.mask(mask)
    return corr


def _get_mask(corr, show_upper_triangle, show_diagonal):
    if not show_upper_triangle:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        if show_diagonal:
            np.fill_diagonal(mask, False)
    else:
        if not show_diagonal:
            mask = np.equal(corr, 1)
    return mask


def _process_data_for_plotting(data, model, periods, period_name, factors):
    """Process data for passing to heatmap plot.
    Args:
        data(pd.DataFrame): Data with observable variables.
        model(dict): Processed model_dict that contains information on measurements for
            each period.
        periods(int or list): The period or list of periods that correlations are
            calculated for.
        period_name(str): Name of the period variable in the data.
        factors(list or str): List of or a single factor the measurements of which
            correlations are calculated for.
    Returns:
        df(pd.DataFrame): Processed DataFrame to calculate correlations over.

    """
    if isinstance(periods, list) and len(periods) == 1:
        periods = periods[0]
    if isinstance(periods, (int, float)):
        df = _process_data_with_single_period(
            data, model, periods, period_name, factors
        )
    elif isinstance(periods, list):
        df = _process_data_with_multiple_periods(
            data, model, periods, period_name, factors
        )
    return df


def _process_data_with_single_period(data, model, period, period_name, factors):
    """Extract measurements of factors for the given period.
    Args:
        data(pd.DataFrame): Data with observable variables.
        model(dict): Processed model_dict that contains information on measurements
            for each period.
        periods(int or float): The period to extract measurements for.
        period_name(str): Name of the period variable in the data.
        factors(list or str): List of or a single factor the measurements of which
            correlations are calculated for.
    Returns:
        df(pd.DataFrame): DataFrame with measurements of factors for period 'period'.

    """

    period_info = model["update_info"].loc[period].reset_index()
    measurements = []
    if factors:
        if isinstance(factors, str):
            factors = [factors]
        for fac in factors:
            measurements += period_info.query(
                f"{fac} == True and purpose == 'measurement'"
            )["variable"].to_list()
    else:
        measurements = period_info.query(f"purpose == 'measurement'")[
            "variable"
        ].to_list()
    df = data.query(f"{period_name}=={period}")[measurements]
    return df


def _process_data_with_multiple_periods(data, model, periods, period_name, factors):
    """Extract measurements for factors for given periods.
    Args:
        data(pd.DataFrame): Data with observable variables.
        model(dict): Processed model_dict that contains information on measurements for
            each period.
        periods(list): The periods to extract measurements for.
        period_name(str): Name of the period variable in the data.
        factors(list or str): List of or a single factor the measurements of which
            correlations are calculated for.
    Returns:
        df(pd.DataFrame): DataFrame with measurements of factors in each period as
            columns.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _process_data_with_single_period(data, model, period, period_name, factors)
            .add_suffix(f"_{period}")
            .reset_index(drop=True)
        )
    df = pd.concat(to_concat, axis=1)
    return df


def _get_layout_kwargs(corr, layout_kwargs, annotate, show_title, periods, period_name):
    default_layout_kwargs = {
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "yaxis_autorange": "reversed",
        "template": "plotly_white",
    }

    if annotate:
        annotations = []
        for n, row in enumerate(corr.values):
            for m in range(len(row)):
                annotations.append(
                    {
                        "text": str(corr.values[n][m]).replace("nan", ""),
                        "x": corr.index[m],
                        "y": corr.columns[n],
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False,
                    }
                )
        default_layout_kwargs["annotations"] = annotations
    if show_title:
        title = _get_fig_title(periods, period_name)
        default_layout_kwargs["title"] = title
    if layout_kwargs:
        default_layout_kwargs.update(layout_kwargs)
    return default_layout_kwargs


def _get_fig_title(periods, period_name):
    """Get title of correlation heatmap.
    Args:
        periods(int or list): The period or list of periods that correlations
            are calculated for.
        period_name(str): Name of the period variable in the data.
    Returns:
        title(str): Title for the correlation heatmap that describes which periods
            the correlations have been calculated for.
    """
    if isinstance(periods, list) and len(periods) == 1:
        periods = periods[0]
    if isinstance(periods, list):
        title = f"{period_name}: {periods[0]}-{periods[-1]}"
    elif isinstance(periods, (int, float)):
        title = f"{period_name}: {periods}"
    return title


def _get_heatmap_kwargs(heatmap_kwargs, colorscale, zmin, zmax, zmid):
    default_heatmap_kwargs = {
        "colorscale": colorscale,
        "zmin": zmin,
        "zmax": zmax,
        "zmid": zmid,
    }
    if heatmap_kwargs:
        default_heatmap_kwargs.update(heatmap_kwargs)
    return default_heatmap_kwargs
