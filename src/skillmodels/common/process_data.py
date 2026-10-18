"""Functions to process and prepare data for model estimation."""

import warnings
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.common.types import Anchoring, Labels


def process_data(
    df: pd.DataFrame,
    *,
    has_endogenous_factors: bool,
    labels: Labels,
    update_info: pd.DataFrame,
    anchoring_info: Anchoring,
    purpose: Literal["estimation", "anything", "simulation"] = "estimation",
) -> dict[str, Any]:
    """Process the data for estimation.

    Args:
        df: panel dataset in long format. It has a MultiIndex
            where the first level indicates the period and the second the individual.
        has_endogenous_factors: Whether the model includes endogenous factors.
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        anchoring_info: Information about anchoring. See :ref:`anchoring`
        purpose: Whether the data is used for
            estimation (default, includes measurement data) or not.

    Returns a dictionary with keys:
        measurements: Array of shape (n_updates, n_obs) with data on
            observed measurements. NaN if the measurement was not observed. Only
            returned if estimation==True
        controls: Array of shape (n_periods, n_obs, n_controls) with
            observed control variables for the measurement equations.
        observed_factors: Array of shape
            (n_periods, n_obs, n_observed_factors) with data on the observed factors.
            Where a factor listed in `labels.missing_observed_factor_moments` is
            missing, it holds the fill mean.
        missing_observed_factors: Names of the observed factors with fill moments
            that are missing somewhere in the data (estimation only; else empty).
        observed_factor_vars: Array of shape
            (n_periods, n_obs, len(missing_observed_factors)) with the fill
            variance where the factor is missing and 0 where it is observed.

    """
    df = pre_process_data(df=df, periods=labels.periods)
    df["constant"] = 1
    out = {}

    df = _add_copies_of_anchoring_outcome(df=df, anchoring_info=anchoring_info)
    if has_endogenous_factors:
        df = _augment_data_for_endogenous_factors(
            df=df, labels=labels, update_info=update_info
        )
    else:
        df.index = df.index.set_names(["id", "aug_period"])

    _check_data(df=df, update_info=update_info, labels=labels, purpose=purpose)
    n_obs = int(len(df) / len(labels.aug_periods))
    df = _handle_controls_with_missings(
        df=df, controls=labels.controls, update_info=update_info
    )
    out["controls"] = _generate_controls_array(df=df, labels=labels, n_obs=n_obs)
    missing = tuple(
        f
        for f in labels.observed_factors
        if purpose == "estimation"
        and f in labels.missing_observed_factor_moments
        and df[f].isna().any()
    )
    out["missing_observed_factors"] = missing
    out["observed_factor_vars"] = _generate_observed_factor_vars_array(
        df=df, labels=labels, n_obs=n_obs, factors=missing
    )
    for factor in missing:
        mean_col, _ = labels.missing_observed_factor_moments[factor]
        df[factor] = df[factor].fillna(df[mean_col])
    out["observed_factors"] = _generate_observed_factor_array(
        df=df, labels=labels, n_obs=n_obs
    )

    if purpose == "estimation":
        out["measurements"] = _generate_measurements_array(
            df=df, update_info=update_info, n_obs=n_obs
        )
    return out


def pre_process_data(
    df: pd.DataFrame,
    periods: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Balance panel data in long format, drop unnecessary periods and set index.

    Args:
        df: panel dataset in long format. It has a MultiIndex
            where the first level indicates the period and the second
            the individual.
        periods: The periods to keep in the balanced panel.

    Returns:
        balanced: balanced panel. It has a MultiIndex. The first
            enumerates individuals. The second level counts periods, starting at 0.

    """
    df = df.sort_index()
    df["__old_id__"] = df.index.get_level_values(0)
    df["__old_period__"] = df.index.get_level_values(1)

    # replace existing codes for periods and
    df.index.names = ["id", "period"]
    for level in [0, 1]:
        # df.index is a MultiIndex but typed as Index
        df.index = df.index.set_levels(range(len(df.index.levels[level])), level=level)  # ty: ignore[unresolved-attribute]

    # create new index
    ids = sorted(df.index.get_level_values("id").unique())
    new_index = pd.MultiIndex.from_product([ids, periods], names=["id", "period"])

    # set new index
    return df.reindex(new_index)


def _get_period_data_for_endogenous_factors(
    aug_period: int,
    period: int,
    df: pd.DataFrame,
    labels: Labels,
    update_info: pd.DataFrame,
) -> pd.DataFrame:
    meas = _get_period_measurements(update_info=update_info, aug_period=aug_period)
    controls = labels.controls
    observed = labels.observed_factors
    moment_cols = [
        col for cols in labels.missing_observed_factor_moments.values() for col in cols
    ]

    # Several factor mappings may share a physical moment column, or reuse a
    # column already selected above. Preserve its first position, not duplicates.
    columns = list(
        dict.fromkeys(
            [
                "id",
                *meas,
                *controls,
                *observed,
                *moment_cols,
                "period",
                "__old_id__",
                "__old_period__",
            ]
        )
    )
    out = df.query(f"period == {period}")[columns]
    out["aug_period"] = aug_period
    return out


def _augment_data_for_endogenous_factors(
    df: pd.DataFrame,
    labels: Labels,
    update_info: pd.DataFrame,
) -> pd.DataFrame:
    """Make room for endogenous factors by doubling up the periods.

    Endogeneity means that current states influence the factor. Typically, this comes
    as an investment equation. We make that look like a transition for skillmodels'
    internal machinery.

    """
    df = df.reset_index()
    # Make sure datset is balanced
    n_ids = df["id"].nunique()
    n_periods = df["period"].nunique()
    if n_ids * n_periods != df.shape[0]:
        raise ValueError("Dataset is not balanced: n_ids * n_periods != n_rows")
    if set(df["period"]) != set(labels.aug_periods_to_periods.values()):
        raise ValueError("Periods in data don't match expected periods")

    out = pd.concat(
        [
            _get_period_data_for_endogenous_factors(
                aug_period=aug_period,
                period=period,
                df=df,
                update_info=update_info,
                labels=labels,
            )
            for aug_period, period in labels.aug_periods_to_periods.items()
        ]
    )
    return out.set_index(["id", "aug_period"]).sort_index()


def _add_copies_of_anchoring_outcome(
    df: pd.DataFrame,
    anchoring_info: Anchoring,
) -> pd.DataFrame:
    df = df.copy()
    for factor in anchoring_info.factors:
        outcome = anchoring_info.outcomes[factor]
        df[f"{outcome}_{factor}"] = df[outcome]
    return df


def _check_data(  # noqa: C901
    df: pd.DataFrame,
    update_info: pd.DataFrame,
    labels: Labels,
    purpose: Literal["estimation", "anything", "simulation"],
) -> None:
    var_report = pd.DataFrame(index=update_info.index[:0], columns=["problem"])
    for aug_period in labels.aug_periods:
        period_data = df.query(f"aug_period == {aug_period}")
        for cont in labels.controls:
            if cont not in period_data.columns or period_data[cont].isna().all():
                var_report.loc[(aug_period, cont), "problem"] = "Variable is missing"

        if purpose == "estimation":
            for meas in _get_period_measurements(
                update_info=update_info, aug_period=aug_period
            ):
                if meas not in period_data.columns:
                    var_report.loc[(aug_period, meas), "problem"] = (
                        "Variable is missing"
                    )
                elif len(period_data[meas].dropna().unique()) == 1:
                    var_report.loc[(aug_period, meas), "problem"] = (
                        "Variable has no variance"
                    )

        for factor in labels.observed_factors:
            moments = (
                labels.missing_observed_factor_moments.get(factor)
                if purpose == "estimation"
                else None
            )
            problems = _observed_factor_problems(period_data, factor, moments)
            for var, problem in problems.items():
                var_report.loc[(aug_period, var), "problem"] = problem

    var_report = var_report.to_string() if len(var_report) > 0 else ""

    if var_report:
        raise ValueError(var_report)


def _observed_factor_problems(
    period_data: pd.DataFrame, factor: str, moments: tuple[str, str] | None
) -> dict[str, str]:
    """Map variables to problems with an observed factor and its fill moments.

    Without `moments`, any missing value is a problem. With `moments`, missing
    values need a non-missing fill mean and a non-negative fill variance.
    """
    if factor not in period_data.columns:
        return {factor: "Variable is missing"}
    missing = period_data[factor].isna()
    if moments is None:
        return {factor: "Variable has missings"} if missing.any() else {}
    absent = [col for col in moments if col not in period_data.columns]
    if absent:
        return dict.fromkeys(absent, "Variable is missing")
    mean, var = (period_data.loc[missing, col] for col in moments)
    if mean.isna().any() or var.isna().any():
        return {factor: f"Variable has missings without fill moments {moments}"}
    if (var < 0).any():
        return {
            factor: f"Variable has missings with negative fill variance in {moments[1]}"
        }
    return {}


def _handle_controls_with_missings(
    df: pd.DataFrame,
    controls: tuple[str, ...],
    update_info: pd.DataFrame,
) -> pd.DataFrame:
    aug_periods = update_info.index.get_level_values(0).unique().tolist()
    problematic_index = df.index[:0]
    for aug_period in aug_periods:
        period_data = df.query(f"aug_period == {aug_period}")
        control_data = period_data[list(controls)]
        meas_data = period_data[
            _get_period_measurements(update_info=update_info, aug_period=aug_period)
        ]
        problem = control_data.isna().any(axis=1) & meas_data.notna().any(axis=1)
        problematic_index = problematic_index.union(period_data[problem].index)

    if len(problematic_index) > 0:
        old_names = df.loc[problematic_index][["__old_id__", "__old_period__"]]
        msg = "Set measurements to NaN because there are NaNs in the controls for:\n{}"
        msg = msg.format(list(map(tuple, old_names.to_numpy().tolist())))
        warnings.warn(msg, stacklevel=2)
        df.loc[problematic_index] = np.nan
    return df


def _get_period_measurements(
    update_info: pd.DataFrame,
    aug_period: int,
) -> list[str]:
    if aug_period in update_info.index:
        measurements = list(update_info.loc[aug_period].index)
    else:
        measurements = []
    return measurements


def _generate_measurements_array(
    df: pd.DataFrame,
    update_info: pd.DataFrame,
    n_obs: int,
) -> Array:
    arr = np.zeros((len(update_info), n_obs))
    for k, (aug_period, var) in enumerate(update_info.index):
        arr[k] = df.query(f"aug_period == {aug_period}")[var].to_numpy()
    return jnp.array(arr, dtype="float32")


def _generate_controls_array(
    df: pd.DataFrame,
    labels: Labels,
    n_obs: int,
) -> Array:
    arr = np.zeros((len(labels.aug_periods), n_obs, len(labels.controls)))
    for aug_period in labels.aug_periods:
        arr[aug_period] = df.query(f"aug_period == {aug_period}")[
            list(labels.controls)
        ].to_numpy()
    return jnp.array(arr, dtype="float32")


def _generate_observed_factor_vars_array(
    df: pd.DataFrame,
    labels: Labels,
    n_obs: int,
    factors: tuple[str, ...],
) -> Array:
    arr = np.zeros((len(labels.aug_periods), n_obs, len(factors)))
    for aug_period in labels.aug_periods:
        period_data = df.query(f"aug_period == {aug_period}")
        for k, factor in enumerate(factors):
            var_col = labels.missing_observed_factor_moments[factor][1]
            arr[aug_period, :, k] = np.where(
                period_data[factor].isna(), period_data[var_col], 0.0
            )
    return jnp.array(arr, dtype="float32")


def _generate_observed_factor_array(
    df: pd.DataFrame,
    labels: Labels,
    n_obs: int,
) -> Array:
    arr = np.zeros((len(labels.aug_periods), n_obs, len(labels.observed_factors)))
    for aug_period in labels.aug_periods:
        arr[aug_period] = df.query(f"aug_period == {aug_period}")[
            list(labels.observed_factors)
        ].to_numpy()
    return jnp.array(arr, dtype="float32")
