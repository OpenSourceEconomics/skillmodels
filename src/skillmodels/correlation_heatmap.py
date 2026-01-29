"""Functions for creating correlation heatmap visualizations."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from plotly import graph_objects as go

from skillmodels.model_spec import ModelSpec
from skillmodels.process_data import pre_process_data
from skillmodels.process_model import process_model
from skillmodels.types import ProcessedModel


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    heatmap_kwargs: dict[str, Any] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    rounding: int = 2,
    zmax: float | None = None,
    zmin: float | None = None,
    zmid: float | None = None,
    colorscale: str = "RdBu_r",
    *,
    show_color_bar: bool = True,
    show_diagonal: bool = True,
    show_upper_triangle: bool = True,
    trim_heatmap: bool = False,
    annotate: bool = True,
    annotation_fontsize: int = 13,
    annotation_text_color: str = "black",
    annotation_text_angle: float = 0,
    axes_tick_fontsize: tuple[int, int] = (12, 12),
    axes_tick_label_angle: tuple[float, float] = (90, 0),
    axes_tick_label_color: tuple[str, str] = ("black", "black"),
) -> go.Figure:
    """Plot correlation heatmaps for factor measurements.

    Args:
        corr: Data frame of measurement or factor score correlations.
        heatmap_kwargs: Dictionary of key word arguments to pass to go.Heatmap ().
            If None, the default kwargs defined in the function will be used.
        layout_kwargs: Dictionary of key word arguments used to update layout of
            go.Figure object. If None, the default kwargs defined in the function will
            be used. Through layout_kwargs, you can edit figure properties such as
            - template
            - title
            - figsize
        rounding: Number of digits after the decimal point to round the
            correlation values to. Default 2.
        zmax: Upper bound to set on correlation color map. If None,
            is set to maximum absolute correlation value.
        zmin: Lower bound to set on correlation color map. If None,
            is set to -zmax.
        zmid: Midpoint to set on correlation color map. If None,
            is set to 0.
        colorscale: Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        show_color_bar: A boolean variable for displaying heatmap colorbar.
            Default True.
        show_diagonal: A boolean for displaying the correlations on the diagonal.
            Default False.
        show_upper_triangle: A boolean for displaying upper triangular part
            of the correlation heatmap. Default False.
        trim_heatmap: If True, trim empty rows/columns from the heatmap.
            Default False.

        The following arguments are processed into dictionaries or special plotly
        objects and passed to layout_kwargs. Defining them as additional arguments
        allows the user to pass values of desired properties without having to know
        how exactly plotly needs them to be passed as (e.g. as a nested dicitonary in
        casevof axes tick relevant arguments or as Annotation object in case of
        annotations).

        Same properties can be set via the argument layout_kwargs. In this case, values
        defined in layout_kwargs will overwrite values passed via the individual
        arguments.

        annotate: If True, annotate the heatmap figure with correlation values.
            Default False.
        annotation_fontsize: Font size of the annotation text. Default 13.
        annotation_text_color: Color of the annotation text. Default 'black'.
        annotation_text_angle: The angle at which to rotate annotation text.
            Default 0.
        axes_tick_fontsize: Fontsize of axes
            ticks. Default (12,12)
        axes_tick_label_angle: Rotation angles of
            axes tick labels. Default (90,0).
        axes_tick_label_color: Colors of the axes
            tick labels. Default ('black', 'black').

    Returns:
        fig: The figure with correlaiton heatmap.

    """
    corr = _process_corr_data_for_plotting(
        corr=corr,
        rounding=rounding,
        show_upper_triangle=show_upper_triangle,
        show_diagonal=show_diagonal,
        trim_heatmap=trim_heatmap,
    )
    heatmap_kwargs = _get_heatmap_kwargs(
        corr=corr,
        heatmap_kwargs=heatmap_kwargs,
        colorscale=colorscale,
        show_color_bar=show_color_bar,
        zmax=zmax,
        zmin=zmin,
        zmid=zmid,
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


def get_measurements_corr(
    data: pd.DataFrame,
    model: dict | ModelSpec,
    factors: list[str] | tuple[str, ...] | str | None,
    periods: float | list[int] | None,
) -> pd.DataFrame:
    """Get data frame with measurement correlations.

    Process data to retrieve measurements for each period and calculate correlations
    across period specific measurements.

    Args:
        data: DataFrame with observed measurements.
        model: The model specification, either as a dict or ModelSpec instance.
            See: :ref:`model_specs`
        factors: List of factors, to retrieve measurements for.
            If None, then calculate correlations of measurements of all factors.
        periods: If int, the period within which to
            calculate measurement correlations. If a list, calculate correlations over
            periods. If None, calculate correlations across all periods. Note: Periods
            refer to originl periods, not the augmented periods.

    Returns:
        corr: DataFrame with measurement correlations.

    """
    data = data.copy(deep=True)
    processed_model = process_model(model)
    periods = _process_periods(periods=periods, model=processed_model)
    processed_data = pre_process_data(df=data, periods=periods)
    latent_factors, observed_factors = _process_factors(
        model=processed_model, factors=factors
    )
    update_info_by_period = _get_update_info_for_periods(processed_model)
    df = _get_measurement_data(
        data=processed_data,
        update_info_by_period=update_info_by_period,
        periods=periods,
        latent_factors=latent_factors,
        observed_factors=observed_factors,
    )
    return df.corr()


def get_quasi_scores_corr(
    data: pd.DataFrame,
    model: dict | ModelSpec,
    factors: list[str] | tuple[str, ...] | str | None,
    periods: float | list[int] | None,
) -> pd.DataFrame:
    """Get data frame with correlations of factor scores.

    Process data to retrieve measurements for each period, standardize measurements
    to zero mean and unit standard deviation, take the mean of factor specific
    measurements in each period, and calculate correlations across those factor
    and period specific scores.

    The calculated scores coincide with factor scores for linear models.

    Args:
        data: DataFrame with observed measurements.
        model: The model specification, either as a dict or ModelSpec instance.
            See: :ref:`model_specs`
        factors: List of factors, to retrieve measurements for.
            If None, then calculate correlations of measurements of all factors.
        periods: If int, the period within which to
            calculate measurement correlations. If a list, calculate correlations over
            periods. If None, calculate correlations across all periods.

    Returns:
        corr: DataFrame with score correlations.

    """
    data = data.copy(deep=True)
    processed_model = process_model(model)
    periods = _process_periods(periods=periods, model=processed_model)
    processed_data = pre_process_data(df=data, periods=periods)
    latent_factors, observed_factors = _process_factors(
        model=processed_model, factors=factors
    )
    update_info = _get_update_info_for_periods(processed_model)
    df = _get_quasi_factor_scores_data(
        data=processed_data,
        update_info_by_period=update_info,
        periods=periods,
        latent_factors=latent_factors,
        observed_factors=observed_factors,
    )
    return df.corr()


def get_scores_corr(
    data: pd.DataFrame,
    params: pd.DataFrame,
    model: dict | ModelSpec,
    factors: list[str] | tuple[str, ...] | str | None,
    periods: float | list[int] | None,
) -> pd.DataFrame:
    """Get data frame with correlations of factor scores.

    Process data to retrieve measurements for each period, standardize measurements
    using intercept and loadings, take the mean of factor specific  measurements in
    each period, and calculate correlations across those factor and period specific
    scores.

    Args:
        data: DataFrame with observed measurements.
        params: DataFrame with estimated model parameters
        model: The model specification, either as a dict or ModelSpec instance.
            See: :ref:`model_specs`
        factors: List of factors, to retrieve measurements for.
            If None, then calculate correlations of measurements of all factors.
        periods: If int, the period within which to
            calculate measurement correlations. If a list, calculate correlations over
            periods. If None, calculate correlations across all periods.

    Returns:
        corr: DataFrame with score correlations.

    """
    data = data.copy(deep=True)
    processed_model = process_model(model)
    periods = _process_periods(periods=periods, model=processed_model)
    processed_data = pre_process_data(df=data, periods=periods)
    latent_factors, observed_factors = _process_factors(
        model=processed_model, factors=factors
    )
    params = params.loc[["controls", "loadings"]]
    df = _get_factor_scores_data(
        data=processed_data,
        params=params,
        model=processed_model,
        periods=periods,
        latent_factors=latent_factors,
        observed_factors=observed_factors,
    )
    return df.corr()


def _process_corr_data_for_plotting(
    corr: pd.DataFrame,
    rounding: int,
    *,
    show_upper_triangle: bool,
    show_diagonal: bool,
    trim_heatmap: bool,
) -> pd.DataFrame:
    """Apply mask and rounding to correlation DataFrame."""
    mask = _get_mask(
        corr, show_upper_triangle=show_upper_triangle, show_diagonal=show_diagonal
    )
    corr = corr.where(mask).round(rounding)
    if trim_heatmap:
        keeprows = mask.any(axis=1) & corr.notna().any(axis="columns").to_numpy()
        mask = mask[keeprows]
        corr = corr[keeprows]
        keepcols = mask.any(axis=0) & corr.notna().any(axis="index").to_numpy()
        mask = mask.T[keepcols].T
        corr = corr.T[keepcols].T
    return corr


def _get_mask(
    corr: pd.DataFrame,
    *,
    show_upper_triangle: bool,
    show_diagonal: bool,
) -> NDArray[np.bool_]:
    """Get array to mask the correlation DataFrame."""
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.tril_indices_from(mask, k=-1)] = True
    if show_upper_triangle:
        mask[np.triu_indices_from(mask, k=1)] = True
    if show_diagonal:
        np.fill_diagonal(mask, val=True)
    return mask


def _get_update_info_for_periods(model: ProcessedModel) -> pd.DataFrame:
    """Return update_info with user-provided periods instead of augmented periods."""
    update_info = model.update_info.copy()

    # Replace period level with user-provided period using set_codes
    period_values = update_info.index.get_level_values("aug_period").map(
        model.labels.aug_periods_to_periods
    )
    update_info.index = update_info.index.set_codes(period_values, level="aug_period")  # ty: ignore[unresolved-attribute]
    update_info.index = update_info.index.set_names(["period", "variable"])

    # Group by period and variable, apply OR logic for boolean columns
    cols = [col for col in update_info.columns if col != "purpose"]
    agg_dict = dict.fromkeys(cols, "any")
    agg_dict["purpose"] = "first"

    return update_info.groupby(["period", "variable"]).agg(agg_dict)


def _get_measurement_data(
    data: pd.DataFrame,
    update_info_by_period: pd.DataFrame,
    periods: list[int],
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get data frame with factor measurements in each period, in wide format.

    For each factor, retrieve the data on measurements in each period and stack
    the data columns into a data frame.

    Args:
        data: Data with observable variables.
        update_info_by_period: DataFrame with information on measurements
            for each factor in each model period.
        periods: The list of periods that correlations are
            calculated for.
        latent_factors: List of latent factors the measurements of which
            correlations are calculated for.
        observed_factors: List of observed factors the measurements of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    if len(periods) == 1:
        period = periods[0]
        df = _get_measurement_data_for_single_period(
            data=data,
            update_info_by_period=update_info_by_period,
            period=period,
            latent_factors=latent_factors,
            observed_factors=observed_factors,
        )
    else:
        df = _get_measurement_data_for_multiple_periods(
            data=data,
            update_info_by_period=update_info_by_period,
            periods=periods,
            latent_factors=latent_factors,
            observed_factors=observed_factors,
        )
    return df


def _get_measurement_data_for_single_period(
    data: pd.DataFrame,
    update_info_by_period: pd.DataFrame,
    period: int,
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Extract measurements of factors for the given period.

    Args:
        data: Data with observable variables.
        update_info_by_period: DataFrame with information on measurements
            for each factor in each model period.
        period: The period to extract measurements for.
        latent_factors: List of latent factors the measurements of which
            correlations are calculated for.
        observed_factors: List of observed factors the measurements of which
            correlations are calculated for.

    Returns:
        df: DataFrame with measurements of factors for period 'period'.

    """
    period_info = update_info_by_period.loc[period].reset_index()
    measurements = []

    for fac in latent_factors:
        measurements += period_info.query(
            f"{fac} == True and purpose == 'measurement'",
        )["variable"].to_list()
    for fac in observed_factors:
        measurements.append(fac)
    return data.query(f"{update_info_by_period.index.names[0]}=={period}")[measurements]


def _get_measurement_data_for_multiple_periods(
    data: pd.DataFrame,
    update_info_by_period: pd.DataFrame,
    periods: list[int],
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Extract measurements for factors for given periods.

    Args:
        data: Data with observable variables.
        update_info_by_period: DataFrame with information on measurements
            for each factor in each user-provided period.
        periods: The periods to extract measurements for.
        latent_factors: List of latent factors the measurements of which
            correlations are calculated for.
        observed_factors: List of observed factors the measurements of which
            correlations are calculated for.

    Returns:
        df: DataFrame with measurements of factors in each period as
            columns.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _get_measurement_data_for_single_period(
                data=data,
                update_info_by_period=update_info_by_period,
                period=period,
                latent_factors=latent_factors,
                observed_factors=observed_factors,
            )
            .add_suffix(f", {period}")
            .reset_index(drop=True),
        )
    return pd.concat(to_concat, axis=1)


def _get_quasi_factor_scores_data(
    data: pd.DataFrame,
    update_info_by_period: pd.DataFrame,
    periods: list[int],
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get data frame with summary information on factor measurements in each period.

    In each period, standardize factor measurements to zero mean and unit standard
    deviation, and for each factor take the average of all measurements as
    a summary statistics. The calculated scores coincide with factor scores for linear
    models.

    Args:
        data: Data with observable variables.
        update_info_by_period: DataFrame with information on measurements
            for each factor in each model period.
        periods: The list of periods that correlations are
            calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    if len(periods) == 1:
        period = periods[0]
        df = _get_quasi_factor_scores_data_for_single_period(
            data=data,
            update_info_by_period=update_info_by_period,
            period=period,
            latent_factors=latent_factors,
            observed_factors=observed_factors,
        )
    else:
        df = _get_quasi_factor_scores_data_for_multiple_periods(
            data=data,
            update_info_by_period=update_info_by_period,
            periods=periods,
            latent_factors=latent_factors,
            observed_factors=observed_factors,
        )

    return df


def _get_quasi_factor_scores_data_for_single_period(
    data: pd.DataFrame,
    update_info_by_period: pd.DataFrame,
    period: int,
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get frame with summary scores on factor measurements in a given period.

    Args:
        data: Data with observable variables.
        update_info_by_period: DataFrame with information on measurements
            for each factor in each user-provided period.
        period: The period that correlations are calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    period_info = update_info_by_period.loc[period].reset_index()
    to_concat = []
    for factor in latent_factors:
        period_factor_measurements = period_info.query(
            f"{factor} == True and purpose == 'measurement'",
        )["variable"].to_list()
        df = data.query(f"{update_info_by_period.index.names[0]}=={period}")[
            period_factor_measurements
        ]
        df = (df - df.mean()) / df.std()
        sr = df.mean(axis=1)
        sr.name = f"{factor}"
        to_concat.append(sr)
    for factor in observed_factors:
        df = data.query(f"{update_info_by_period.index.names[0]}=={period}")[factor]
        to_concat.append(df)
    return pd.concat(to_concat, axis=1)


def _get_quasi_factor_scores_data_for_multiple_periods(
    data: pd.DataFrame,
    update_info_by_period: pd.DataFrame,
    periods: list[int],
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get frame with summary scores of factor measurements in a given period.

    Args:
        data: Data with observable variables.
        update_info_by_period: DataFrame with information on measurements
            for each factor in each user-provided period.
        periods: The list of periods that correlations are
            calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _get_quasi_factor_scores_data_for_single_period(
                data=data,
                update_info_by_period=update_info_by_period,
                period=period,
                latent_factors=latent_factors,
                observed_factors=observed_factors,
            )
            .add_suffix(f", {period}")
            .reset_index(drop=True),
        )
    return pd.concat(to_concat, axis=1)


def _get_factor_scores_data(
    data: pd.DataFrame,
    params: pd.DataFrame,
    model: ProcessedModel,
    periods: list[int],
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get data frame with factor scores in each period.

    In each period, standardize factor measurements to with estimated intercepts and
    loadings, and for each factor take the average of all measurements as
    a summary statistics.

    Args:
        data: Data with observable variables.
        params: Data frame with estimated measurement relevant
            model parameters.
        model: Processed model dict.
        periods: The list of periods that correlations are
            calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    if len(periods) == 1:
        period = periods[0]
        df = _get_factor_scores_data_for_single_period(
            data=data,
            params=params,
            model=model,
            period=period,
            latent_factors=latent_factors,
            observed_factors=observed_factors,
        )
    else:
        df = _get_factor_scores_data_for_multiple_periods(
            data=data,
            params=params,
            model=model,
            periods=periods,
            latent_factors=latent_factors,
            observed_factors=observed_factors,
        )

    return df


def _get_factor_scores_data_for_single_period(
    data: pd.DataFrame,
    params: pd.DataFrame,
    model: ProcessedModel,
    period: int,
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get frame with factor scores in a given period.

    Careful: When we have endogenous factors, *period* refers to the raw period, but the
    *params* are for the augmented periods. This function is the layer to abstract from
    augmented periods.

    Args:
        data: Data with observable variables.
        params: Data frame with estimated measurement relevant
            model parameters.
        model: Processed model dict.
        period: The period that correlations are calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    aug_periods = model.endogenous_factors_info.aug_periods_from_period(period)
    df = pd.concat(
        [
            _get_factor_scores_data_for_single_model_period(
                data=data,
                params=params,
                update_info=model.update_info,
                aug_period=ap,
                period=period,
                latent_factors=latent_factors,
                observed_factors=observed_factors,
            )
            for ap in aug_periods
        ],
        axis=0,
    )
    df = df.groupby("id").max()
    return df.set_index(
        pd.MultiIndex.from_tuples(
            [(idx, period) for idx in df.index], names=["id", "period"]
        )
    )


def _get_factor_scores_data_for_single_model_period(
    data: pd.DataFrame,
    params: pd.DataFrame,
    update_info: pd.DataFrame,
    aug_period: int,
    period: int,
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get frame with factor scores in a given model period.

    In this function, all calculations are at the augmented period level.

    Args:
        data: Data with observable variables.
        params: Data frame with estimated measurement relevant
        update_info: DataFrame with information on measurements
            for each factor in each model period.
        aug_period: The (augmented) period that correlations are calculated for.
        period: The (raw) period that correlations are calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.
    """
    if aug_period not in update_info.index:
        return pd.DataFrame()
    period_info = update_info.loc[aug_period].reset_index()
    params = params.query(f"aug_period=={aug_period}").droplevel("aug_period")
    loadings = params.loc["loadings"]["value"]
    intercepts = (
        params.loc["controls"].query("name2 == 'constant'").droplevel("name2")["value"]
    )
    loadings_count = loadings.astype(bool).groupby("name1").sum()
    leave_out_meas = loadings_count[loadings_count > 1].index.to_list()  # ty: ignore[unsupported-operator]
    to_concat = []
    for factor in latent_factors:
        period_factor_measurements = period_info.query(
            f"{factor} == True and purpose == 'measurement'",
        )["variable"].to_list()
        period_factor_measurements = [
            m for m in period_factor_measurements if m not in leave_out_meas
        ]
        df = data.query(f"period == {period}")[period_factor_measurements]
        for m in period_factor_measurements:
            df[m] = (df[m] - intercepts.loc[m]) / loadings.loc[(m, factor)]
        sr = df.mean(axis=1)
        sr.name = f"{factor}"
        to_concat.append(sr)
    for factor in observed_factors:
        df = data.query(f"period == {period}")[factor]
        to_concat.append(df)
    return pd.concat(to_concat, axis=1)


def _get_factor_scores_data_for_multiple_periods(
    data: pd.DataFrame,
    params: pd.DataFrame,
    model: ProcessedModel,
    periods: list[int],
    latent_factors: list[str] | tuple[str, ...],
    observed_factors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Get frame with factor scores in a given period.

    Args:
        data: Data with observable variables.
        params: Data frame with estimated model parameters.
        model: Processed model dict.
        periods: The list of periods that correlations are
            calculated for.
        latent_factors: List of latent factors the scores of which
            correlations are calculated for.
        observed_factors: List of observed factors the scores of which
            correlations are calculated for.

    Returns:
        df: Processed DataFrame to calculate correlations over.

    """
    to_concat = []
    for period in periods:
        to_concat.append(
            _get_factor_scores_data_for_single_period(
                data=data,
                params=params,
                model=model,
                period=period,
                latent_factors=latent_factors,
                observed_factors=observed_factors,
            )
            .add_suffix(f", {period}")
            .reset_index(drop=True),
        )
    return pd.concat(to_concat, axis=1)


def _process_factors(
    model: ProcessedModel,
    factors: list[str] | tuple[str, ...] | str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Process factors to get a tuple of tuples."""
    if not factors:
        latent_factors = model.labels.latent_factors
        observed_factors = model.labels.observed_factors
    elif isinstance(factors, str):
        if factors in model.labels.latent_factors:
            latent_factors = (factors,)
            observed_factors = ()
        elif factors in model.labels.observed_factors:
            observed_factors = (factors,)
            latent_factors = ()
    else:
        latent_factors = tuple(
            fac for fac in factors if fac in model.labels.latent_factors
        )
        observed_factors = tuple(
            fac for fac in factors if fac in model.labels.observed_factors
        )
    return latent_factors, observed_factors  # ty: ignore[possibly-unresolved-reference]


def _process_periods(
    periods: float | list[int] | None,
    model: ProcessedModel,
) -> list[int]:
    """Process periods to get a list."""
    if periods is None:
        return list(range(model.dimensions.n_periods))
    if isinstance(periods, int | float):
        return [int(periods)]
    return periods


def _get_layout_kwargs(
    corr: pd.DataFrame,
    layout_kwargs: dict[str, Any] | None,
    *,
    annotate: bool,
    annotation_fontsize: int,
    annotation_text_color: str,
    annotation_text_angle: float,
    axes_tick_fontsize: tuple[int, int],
    axes_tick_label_angle: tuple[float, float],
    axes_tick_label_color: tuple[str, str],
) -> dict[str, Any]:
    """Get kwargs to update figure layout.

    Args:
        corr: The processed data frame with correlation coefficients.
        layout_kwargs: Dictionary of keyword arguments used to update layout of
            go.Figure object.
        annotate: Add annotations to the figure if True.
        annotation_fontsize: Fontsize of the annotation text.
        annotation_text_color: Color of the annotation text.
        annotation_text_angle: The angle at which to rotate annotation text.
        axes_tick_fontsize(tuple,list or dict): Fontsizes of axes tick labels.
        axes_tick_label_angle(tuple,list or dict): The angle at which to rotate axes
            tick labels.
        axes_tick_label_color(tuple,list or dict): Color of axes labels.

    Returns:
        default_layout_kwargs: Dictionary to update figure layout.

    """
    default_layout_kwargs = {
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "template": "simple_white",
        "yaxis_autorange": "reversed",
    }
    default_layout_kwargs.update(
        _get_annotations(
            corr,
            annotate=annotate,
            annotation_fontsize=annotation_fontsize,
            annotation_text_color=annotation_text_color,
            annotation_text_angle=annotation_text_angle,
        ),
    )
    default_layout_kwargs.update(
        _get_axes_ticks_kwargs(
            axes_tick_fontsize=axes_tick_fontsize,
            axes_tick_label_angle=axes_tick_label_angle,
            axes_tick_label_color=axes_tick_label_color,
        ),
    )
    if layout_kwargs:
        default_layout_kwargs.update(layout_kwargs)
    return default_layout_kwargs


def _get_axes_ticks_kwargs(
    axes_tick_fontsize: tuple[int, int] | dict[str, int],
    axes_tick_label_angle: tuple[float, float] | dict[str, float],
    axes_tick_label_color: tuple[str, str] | dict[str, str],
) -> dict[str, Any]:
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
    df: pd.DataFrame,
    *,
    annotate: bool,
    annotation_fontsize: int,
    annotation_text_color: str,
    annotation_text_angle: float,
) -> dict[str, Any]:
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
                    },
                )
        annotation_kwargs["annotations"] = annotations
    return annotation_kwargs


def _get_heatmap_kwargs(
    corr: pd.DataFrame,
    heatmap_kwargs: dict[str, Any] | None,
    colorscale: str,
    *,
    show_color_bar: bool,
    zmax: float | None,
    zmin: float | None,
    zmid: float | None,
) -> dict[str, Any]:
    """Get kwargs to instantiate Heatmap object.

    Args:
        corr: Data frame with correlation coefficients.
        heatmap_kwargs: Dictionary of key word arguments to pass to go.Heatmap().
        colorscale: Name of the color palette to use in the heatmap.
            Default 'RdBu_r'.
        show_color_bar: A boolean variable for displaying heatmap colorbar.
        zmax: Upper bound to set on correlation color map.
        zmin: Lower bound to set on correlation color map.
        zmid: Midpoint to set on correlation color map.

    Returns:
        default_heatmap_kwargs: Dictionary of kwargs to instantiate go.Heatmap.

    """
    if zmax is None:
        corr_arr = corr.to_numpy()
        zmax = np.abs(corr_arr)[np.tril_indices_from(corr_arr, k=-1)].max()
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


def _process_axes_tick_args(
    args: tuple[Any, Any] | list[Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(args, tuple | list):
        args = {"x": args[0], "y": args[1]}
    return args
