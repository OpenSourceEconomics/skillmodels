import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from skillmodels.process_model import get_period_measurements


def _get_period_data_for_investments(
    period: int,
    period_raw: int,
    df: pd.DataFrame,
    update_info: pd.DataFrame,
    labels: dict[str, Any],
) -> pd.DataFrame:
    meas = get_period_measurements(update_info, period)
    controls = labels["controls"]
    observed = labels["observed_factors"]

    out = df.query(f"period_raw == {period_raw}")[
        [
            "id",
            *meas,
            *controls,
            *observed,
            "period_raw",
            "__old_id__",
            "__old_period__",
        ]
    ]
    out["period"] = period
    return out


def _augment_data_for_investments(df: pd.DataFrame, model: dict[str, Any]):
    """Make room for endogenous investments by doubling up the periods.

    Endogeneity of investments means that current states influence the

    """
    df = df.reset_index().rename(columns={"period": "period_raw"})
    # Make sure datset is balanced
    n_ids = df["id"].nunique()
    n_periods = df["period_raw"].nunique()
    assert n_ids * n_periods == df.shape[0]
    assert set(df["period_raw"]) == set(
        model["labels"]["periods_to_periods_raw"].values()
    )

    out = pd.concat(
        [
            _get_period_data_for_investments(
                period=period,
                period_raw=period_raw,
                df=df,
                update_info=model["update_info"],
                labels=model["labels"],
            )
            for period, period_raw in model["labels"]["periods_to_periods_raw"].items()
        ]
    )
    out = out.set_index(["id", "period"]).sort_index()
    return out


def process_data(
    df, has_investments, labels, update_info, anchoring_info, purpose="estimation"
):
    """Process the data for estimation.

    Args:
        df (DataFrame): panel dataset in long format. It has a MultiIndex
            where the first level indicates the period and the second the individual.
        has_investments (bool):
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        anchoring_qinfo (dict): Information about anchoring. See :ref:`anchoring`
        purpose (Literal["estimation", "anything"]): Whether the data is used for
            estimation (default, includes measurement data) or not.

    Returns a dictionary with keys:
        controls (jax.numpy.array): Array of shape (n_updates, n_obs) with data on
            observed measurements. NaN if the measurement was not observed. Only
            returned if estimation==True
        observed_factors (jax.numpy.array): Array of shape
            (n_periods, n_obs, n_controls) with observed control variables for the
            measurement equations.
        measurements (jax.numpy.array): Array of shape (n_periods, n_obs,
            n_observed_factors) with data on the observed factors.

    """
    df = pre_process_data(df, labels["periods_raw"])
    df["constant"] = 1
    out = {}

    if has_investments:
        df = _augment_data_for_investments(df, labels)
    else:
        df = _add_copies_of_anchoring_outcome(df, anchoring_info)
    _check_data(df, update_info, labels, purpose=purpose)
    n_obs = int(len(df) / len(labels["periods"]))
    df = _handle_controls_with_missings(df, labels["controls"], update_info)
    out["controls"] = _generate_controls_array(df, labels, n_obs)
    out["observed_factors"] = _generate_observed_factor_array(df, labels, n_obs)

    if purpose == "estimation":
        out["measurements"] = _generate_measurements_array(df, update_info, n_obs)
    return out


def pre_process_data(df, periods):
    """Balance panel data in long format, drop unnecessary periods and set index.

    Args:
        df (DataFrame): panel dataset in long format. It has a MultiIndex
            where the first level indicates the period and the second
            the individual.

    Returns:
        balanced (DataFrame): balanced panel. It has a MultiIndex. The first
            enumerates individuals. The second level counts periods, starting at 0.

    """
    df = df.sort_index()
    df["__old_id__"] = df.index.get_level_values(0)
    df["__old_period__"] = df.index.get_level_values(1)

    # replace existing codes for periods and
    df.index.names = ["id", "period"]
    for level in [0, 1]:
        df.index = df.index.set_levels(range(len(df.index.levels[level])), level=level)

    # create new index
    ids = sorted(df.index.get_level_values("id").unique())
    new_index = pd.MultiIndex.from_product([ids, periods], names=["id", "period"])

    # set new index
    df = df.reindex(new_index)

    return df


def _add_copies_of_anchoring_outcome(df, anchoring_info):
    df = df.copy()
    for factor in anchoring_info["factors"]:
        outcome = anchoring_info["outcomes"][factor]
        df[f"{outcome}_{factor}"] = df[outcome]
    return df


def _check_data(df, update_info, labels, purpose):  # noqa: C901
    var_report = pd.DataFrame(index=update_info.index[:0], columns=["problem"])
    for period in labels["periods"]:
        period_data = df.query(f"period == {period}")
        for cont in labels["controls"]:
            if cont not in period_data.columns or period_data[cont].isna().all():
                var_report.loc[(period, cont), "problem"] = "Variable is missing"

        if purpose == "estimation":
            for meas in get_period_measurements(update_info, period):
                if meas not in period_data.columns:
                    var_report.loc[(period, meas), "problem"] = "Variable is missing"
                elif len(period_data[meas].dropna().unique()) == 1:
                    var_report.loc[(period, meas), "problem"] = (
                        "Variable has no variance"
                    )

        for factor in labels["observed_factors"]:
            if factor not in period_data.columns:
                var_report.loc[(period, factor), "problem"] = "Variable is missing"
            elif period_data[factor].isna().any():
                var_report.loc[(period, factor), "problem"] = "Variable has missings"

    var_report = var_report.to_string() if len(var_report) > 0 else ""

    if var_report:
        raise ValueError(var_report)


def _handle_controls_with_missings(df, controls, update_info):
    periods = update_info.index.get_level_values(0).unique().tolist()
    problematic_index = df.index[:0]
    for period in periods:
        period_data = df.query(f"period == {period}")
        control_data = period_data[controls]
        meas_data = period_data[get_period_measurements(update_info, period)]
        problem = control_data.isna().any(axis=1) & meas_data.notna().any(axis=1)
        problematic_index = problematic_index.union(period_data[problem].index)

    if len(problematic_index) > 0:
        old_names = df.loc[problematic_index][["__old_id__", "__old_period__"]]
        msg = "Set measurements to NaN because there are NaNs in the controls for:\n{}"
        msg = msg.format(list(map(tuple, old_names.to_numpy().tolist())))
        warnings.warn(msg)
        df.loc[problematic_index] = np.nan
    return df


def _generate_measurements_array(df, update_info, n_obs):
    arr = np.zeros((len(update_info), n_obs))
    for k, (period, var) in enumerate(update_info.index):
        arr[k] = df.query(f"period == {period}")[var].to_numpy()
    return jnp.array(arr, dtype="float32")


def _generate_controls_array(df, labels, n_obs):
    arr = np.zeros((len(labels["periods"]), n_obs, len(labels["controls"])))
    for period in labels["periods"]:
        arr[period] = df.query(f"period == {period}")[labels["controls"]].to_numpy()
    return jnp.array(arr, dtype="float32")


def _generate_observed_factor_array(df, labels, n_obs):
    arr = np.zeros((len(labels["periods"]), n_obs, len(labels["observed_factors"])))
    for period in labels["periods"]:
        arr[period] = df.query(f"period == {period}")[
            labels["observed_factors"]
        ].to_numpy()
    return jnp.array(arr, dtype="float32")
