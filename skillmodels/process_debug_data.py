import numpy as np
import pandas as pd


def process_debug_data(debug_data, model):
    """Process the raw debug data into pandas objects that make visualization easy.

    Args:
        debug_data (dict): Dictionary containing the following entries (
        and potentially others which are not modified):
        - filtered_states (jax.numpy.array): Array of shape (n_updates, n_obs,
            n_mixtures, n_states) containing the filtered states after each Kalman
            update.
        - initial_states (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states)
            with the state estimates before the first Kalman update.
        - residuals (jax.numpy.array): Array of shape (n_updates, n_obs, n_mixtures)
            containing the residuals of a Kalman update.
        - residual_sds (jax.numpy.ndarray): Array of shape (n_updates, n_obs,
            n_mixtures) containing the theoretical standard deviation of the residuals.
        - all_contributions (jax.numpy.array): Array of shape (n_updates, n_obs) with
            the likelihood contributions per update and individual.
        - log_mixture_weights (jax.numpy.array): Array of shape (n_updates, n_obs,
            n_mixtures) containing the log mixture weights after each update.
        - initial_log_mixture_weights (jax.numpy.array): Array of shape (n_obs,
            n_mixtures) containing the log mixture weights before the first
            kalman update.

        model (dict): Processed model dictionary.

    Returns:
        dict: Dictionary with processed debug data. It has the following entries:

        - post_update_states (pd.DataFrame). As pre_update_states but "period" and
            "measurement" identify the last measurement that was incorporated.
        - filtered_states (pd.DataFrame). Tidy DataFrame with filtered states
            after the last update of each period. The columns are the factor names,
            "period" and "id". The filtered states are already aggregated over
            mixture distributions.
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
    factors = model["labels"]["latent_factors"]

    post_update_states = _create_post_update_states(
        debug_data["filtered_states"], factors, update_info
    )

    filtered_states = _create_filtered_states(
        filtered_states=debug_data["filtered_states"],
        log_mixture_weights=debug_data["log_mixture_weights"],
        update_info=update_info,
        factors=factors,
    )

    state_ranges = create_state_ranges(filtered_states, factors)

    residuals = _process_residuals(debug_data["residuals"], update_info)
    residual_sds = _process_residual_sds(debug_data["residual_sds"], update_info)

    all_contributions = _process_all_contributions(
        debug_data["all_contributions"], update_info
    )

    res = {
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


def _create_filtered_states(filtered_states, log_mixture_weights, update_info, factors):

    filtered_states = np.array(filtered_states)
    log_mixture_weights = np.array(log_mixture_weights)
    weights = np.exp(log_mixture_weights)

    agg_states = (filtered_states * weights.reshape(*weights.shape, 1)).sum(axis=-2)

    keep = []
    for i, (period, measurement) in enumerate(update_info.index):
        last_measurement = update_info.query(
            f"purpose == 'measurement' & period == {period}"
        ).index[-1][1]

        if measurement == last_measurement:
            keep.append(i)

    to_concat = []
    for period, i in enumerate(keep):
        df = pd.DataFrame(data=agg_states[i], columns=factors)
        df["period"] = period
        df["id"] = np.arange(len(df))
        to_concat.append(df)

    filtered_states = pd.concat(to_concat)

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
