import numpy as np
import pandas as pd


def process_debug_data(debug_data, model):
    """Process the raw debug data into pandas objects that make visualization easy.

    Args:
        debug_data (dict): Dictionary containing the following entries (
        and potentially others which are not modified):
        - filtered_states (list): List of arrays. Each array has shape (n_obs,
            n_mixtures, n_states) and contains the filtered states after each Kalman
            update. The list has length n_updates.
        - initial_states (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states)
            with the state estimates before the first Kalman update.
        - residuals (list): List of arrays. Each array has shape (n_obs, n_mixtures)
            and contains the residuals of a Kalman update. The list has length
            n_updates.
        - residual_sds (list): List of arrays. Each array has shape (n_obs, n_mixtures)
            and contains the theoretical standard deviation of the residuals. The list
                has length n_updates.
        - all_contributions (jax.numpy.array): Array of shape (n_updates, n_obs) with
            the likelihood contributions per update and individual.

        model (dict): Processed model dictionary.

    Returns:
        dict: Dictionary with processed debug data. It has the following entries:

        - pre_update_states (pd.DataFrame): Tidy DataFrame with filtered states before
            each update. Columns are factor names, "mixture", "period", "measurement".
            and "id". "period" and "measurement" identify the next measurement that
            will be incorporated.
        - post_update_states (pd.DataFrame). As pre_update_states but "period" and
            "measurement" identify the last measurement that was incorporated.
        - filtered_states (pd.DataFrame). Tidy DataFrame with filtered states
            after the last update of each period. The columns are the factor names,
            "period" and "id"
        - state_ranges (dict): The keys are the names of the latent factors.
            The values are DataFrames with the columns "period", "minimum", "maximum".
            Note that this aggregates over mixture distributions.
        - residuals (pd.DataFrame): Tidy DataFrame with residuals of each Kalman update.
            Columns are "residual", "mixture", "period", "measurement" and "id".
            "period" and "measurement" identify the Kalman update to which the residual
            belongs.
        - residual_sds (pd.DataFrame): As residuals but containing the theoretical
            standard deviation of the corresponding residual.
        - all_contributions (pd.DataFrame): Tidy DataFrame with log likelihood
            contribution per individual and Kalman Update. The columns are
            "contribution", "period", "measurement" and "id". "period" and "measurement"
            identify the Kalman Update to which the likelihood contribution corresponds.

    """
    update_info = model["update_info"]
    factors = model["labels"]["factors"]

    pre_update_states = _create_pre_update_states(
        debug_data["initial_states"],
        debug_data["filtered_states"],
        factors,
        update_info,
    )

    post_update_states = _create_post_update_states(
        debug_data["filtered_states"], factors, update_info
    )

    filtered_states = _create_filtered_states(post_update_states, update_info)

    state_ranges = create_state_ranges(filtered_states, factors)

    residuals = _process_residuals(debug_data["residuals"], update_info)
    residual_sds = _process_residual_sds(debug_data["residual_sds"], update_info)

    all_contributions = _process_all_contributions(
        debug_data["all_contributions"], update_info
    )

    res = {
        "pre_update_states": pre_update_states,
        "post_update_states": post_update_states,
        "filtered_states": filtered_states,
        "state_ranges": state_ranges,
        "residuals": residuals,
        "residual_sds": residual_sds,
        "all_contributions": all_contributions,
    }

    for key in ["value", "contributions"]:
        if key in debug_data:
            res[key] = debug_data[key]

    return res


def _create_pre_update_states(initial_states, filtered_states, factors, update_info):
    to_concat = []
    df = _convert_state_array_to_df(initial_states, factors)
    period, meas = update_info.index[0]
    df["period"] = period
    df["measurement"] = meas
    df["id"] = np.arange(len(df))
    to_concat.append(df)

    for k, (period, meas) in enumerate(update_info.index[1:]):
        purpose = update_info.loc[(period, meas), "purpose"]
        # It is important to discard the states after anchoring updates and use the
        # states after the last measurement update for all anchoring updates.
        if purpose == "measurement":
            pos = k
        else:
            pos = _get_position_of_last_measurement_in_period(update_info, period)

        df = _convert_state_array_to_df(filtered_states[pos], factors)
        df["period"] = period
        df["measurement"] = meas
        df["id"] = np.arange(len(df))
        to_concat.append(df)

    pre_states = pd.concat(to_concat)

    return pre_states


def _create_post_update_states(filtered_states, factors, update_info):
    to_concat = []
    for (period, meas), data in zip(update_info.index, filtered_states):
        df = _convert_state_array_to_df(data, factors)
        df["period"] = period
        df["id"] = np.arange(len(df))
        df["measurement"] = meas
        to_concat.append(df)

    post_states = pd.concat(to_concat)

    return post_states


def _get_position_of_last_measurement_in_period(update_info, period):
    """Return position of the row in update info of the last measurement in period"""
    ind_tup = update_info.query(f"purpose == 'measurement' & period == {period}").index[
        -1
    ]
    sr = pd.Series(data=np.arange(len(update_info)), index=update_info.index)
    pos = sr.loc[ind_tup]
    return pos


def _convert_state_array_to_df(arr, factor_names):
    """Convert a 3d state array into a 2d DataFrame.

    Args:
        arr (np.ndarray): Array of shape (n_obs, n_mixtures, n_states)
        factor_names (list): Names of the latent factors.
    """
    n_obs, n_mixtures, n_states = arr.shape
    df = pd.DataFrame(data=arr.reshape(-1, n_states), columns=factor_names)
    df["mixture"] = np.full((n_obs, n_mixtures), np.arange(n_mixtures)).flatten()
    return df


def _create_filtered_states(post_update_states, update_info):
    periods = sorted(update_info.index.get_level_values("period").unique())
    to_concat = []
    for period in periods:
        last_measurement = update_info.query(
            f"purpose == 'measurement' & period == {period}"
        ).index[-1][1]
        to_concat.append(
            post_update_states.query(
                f"period == {period} & measurement == '{last_measurement}'"
            )
        )

    filtered_states = pd.concat(to_concat)
    filtered_states.drop(columns=["measurement"], inplace=True)
    return filtered_states


def create_state_ranges(filtered_states, factors):
    ranges = {}
    minima = filtered_states.groupby("period").min()
    maxima = filtered_states.groupby("period").max()
    for factor in factors:
        df = pd.concat([minima[factor], maxima[factor]], axis=1)
        df.columns = ["minimum", "maximum"]
        ranges[factor] = df
    return ranges


def _process_residuals(residuals, update_info):
    to_concat = []
    n_obs, n_mixtures = residuals[0].shape
    for (period, meas), data in zip(update_info.index, residuals):
        df = pd.DataFrame(data.reshape(-1, 1), columns=["residual"])
        df["mixture"] = np.full((n_obs, n_mixtures), np.arange(n_mixtures)).flatten()
        df["period"] = period
        df["id"] = np.arange(len(df))
        df["measurement"] = meas
        to_concat.append(df)
    return pd.concat(to_concat)


def _process_residual_sds(residual_sds, update_info):
    return _process_residuals(residual_sds, update_info)


def _process_all_contributions(all_contributions, update_info):
    to_concat = []
    for (period, meas), contribs in zip(update_info.index, all_contributions):
        df = pd.DataFrame(data=contribs.reshape(-1, 1), columns=["contribution"])
        df["measurement"] = meas
        df["period"] = period
        df["id"] = np.arange(len(df))
        to_concat.append(df)
    return pd.concat(to_concat)
