import warnings
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

if TYPE_CHECKING:
    from skillmodels.types import Anchoring, Labels


def process_data(
    df: pd.DataFrame,
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
            Only returned if estimation==True

    """
    df = pre_process_data(df, labels.periods)
    df["constant"] = 1
    out = {}

    df = _add_copies_of_anchoring_outcome(df, anchoring_info)
    if has_endogenous_factors:
        df = _augment_data_for_endogenous_factors(df, labels, update_info)
    else:
        df.index = df.index.set_names(["id", "aug_period"])

    _check_data(df, update_info, labels, purpose=purpose)
    n_obs = int(len(df) / len(labels.aug_periods))
    df = _handle_controls_with_missings(df, labels.controls, update_info)
    out["controls"] = _generate_controls_array(df, labels, n_obs)
    out["observed_factors"] = _generate_observed_factor_array(df, labels, n_obs)

    if purpose == "estimation":
        out["measurements"] = _generate_measurements_array(df, update_info, n_obs)
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
    df = df.reindex(new_index)

    return df


def _get_period_data_for_endogenous_factors(
    aug_period: int,
    period: int,
    df: pd.DataFrame,
    labels: Labels,
    update_info: pd.DataFrame,
) -> pd.DataFrame:
    meas = _get_period_measurements(update_info, aug_period)
    controls = labels.controls
    observed = labels.observed_factors

    out = df.query(f"period == {period}")[
        [
            "id",
            *meas,
            *controls,
            *observed,
            "period",
            "__old_id__",
            "__old_period__",
        ]
    ]
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
    assert n_ids * n_periods == df.shape[0]
    assert set(df["period"]) == set(labels.aug_periods_to_periods.values())

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
        outcome = anchoring_info.outcomes[factor]  # ty: ignore[invalid-argument-type]
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
            for meas in _get_period_measurements(update_info, aug_period):
                if meas not in period_data.columns:
                    var_report.loc[(aug_period, meas), "problem"] = (
                        "Variable is missing"
                    )
                elif len(period_data[meas].dropna().unique()) == 1:
                    var_report.loc[(aug_period, meas), "problem"] = (
                        "Variable has no variance"
                    )

        for factor in labels.observed_factors:
            if factor not in period_data.columns:
                var_report.loc[(aug_period, factor), "problem"] = "Variable is missing"
            elif period_data[factor].isna().any():
                var_report.loc[(aug_period, factor), "problem"] = (
                    "Variable has missings"
                )

    var_report = var_report.to_string() if len(var_report) > 0 else ""

    if var_report:
        raise ValueError(var_report)


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
        meas_data = period_data[_get_period_measurements(update_info, aug_period)]
        problem = control_data.isna().any(axis=1) & meas_data.notna().any(axis=1)
        problematic_index = problematic_index.union(period_data[problem].index)

    if len(problematic_index) > 0:
        old_names = df.loc[problematic_index][["__old_id__", "__old_period__"]]
        msg = "Set measurements to NaN because there are NaNs in the controls for:\n{}"
        msg = msg.format(list(map(tuple, old_names.to_numpy().tolist())))
        warnings.warn(msg)
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
