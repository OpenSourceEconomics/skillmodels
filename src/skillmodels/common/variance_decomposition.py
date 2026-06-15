"""Functions for variance decomposition of measurements.

Decomposes variance of observed measurements into signal (factor variance) and
noise (measurement error) components following Cunha, Heckman, Schennach (2010),
Section 4.2.2.
"""

from collections.abc import Mapping

import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import DIAGNOSTICS_CONF
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model


@beartype(conf=DIAGNOSTICS_CONF)
def decompose_measurement_variance(
    model_spec: ModelSpec,
    params: pd.DataFrame,
    *,
    filtered_states: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose measurement variance into signal and noise components.

    Based on Section 4.2.2 "The Empirical Importance of Measurement Error" of
    Cunha, Heckman, Schennach (2010).

    For each measurement y with loading L on factor F and measurement error sd:
        Var(y) = L^2 * Var(F) + sd^2

    This function computes:
        - fraction_signal = L^2 * Var(F) / Var(y)
        - fraction_noise = sd^2 / Var(y)

    Args:
        model_spec: The model specification.
        params: DataFrame with estimated model parameters.
        filtered_states: DataFrame with one column per latent factor plus a
            "period" column. The caller is responsible for producing this
            via the estimator they used (CHS:
            ``get_individual_states(...)["anchored_states"]["states"]``;
            AF: ``get_af_posterior_states(...)``; AMN:
            ``get_amn_posterior_states(...)``). Anchored states are
            preferable when available; if not, unanchored states still
            give a valid decomposition of the latent variance.

    Return:
        DataFrame indexed by (period, measurement, factor) with columns:
        - loading: The factor loading (L)
        - factor_variance: Var(F) for that period
        - meas_sd: The measurement error standard deviation
        - fraction_signal: Fraction of variance due to factor (signal)
        - fraction_noise: Fraction of variance due to measurement error (noise)
        - signal_to_noise_ratio: Ratio of signal variance to noise variance

    References:
        Cunha, F., Heckman, J. J., & Schennach, S. M. (2010). Estimating the
        Technology of Cognitive and Noncognitive Skill Formation. Econometrica,
        78(3), 883-931. https://doi.org/10.3982/ECTA6551

    """
    processed_model = process_model(model_spec)
    return _compute_variance_decomposition(
        filtered_states=filtered_states,
        params=params,
        aug_periods_to_periods=processed_model.labels.aug_periods_to_periods,
    )


def _compute_variance_decomposition(
    filtered_states: pd.DataFrame,
    params: pd.DataFrame,
    aug_periods_to_periods: Mapping[int, int],
) -> pd.DataFrame:
    """Compute variance decomposition from filtered states and parameters.

    Args:
        filtered_states: DataFrame with filtered states, must have columns for
            each factor plus "period" and "id".
        params: DataFrame with model parameters indexed by
            (category, aug_period, name1, name2).
        aug_periods_to_periods: Mapping from aug_period to period.

    Returns:
        DataFrame with variance decomposition results.

    """
    # Build reverse mapping: period → aug_period (pick first aug_period per period)
    periods_to_aug_periods = {}
    for ap, p in aug_periods_to_periods.items():
        if p not in periods_to_aug_periods:
            periods_to_aug_periods[p] = ap

    # Add aug_period column for internal merges with params
    filtered_states = filtered_states.copy()
    filtered_states["aug_period"] = filtered_states["period"].map(
        periods_to_aug_periods
    )

    # Compute factor variances by period
    periods = filtered_states["aug_period"].unique()
    factor_cols = [
        c
        for c in filtered_states.columns
        if c not in ("aug_period", "period", "id", "mixture")
    ]

    factor_variances = {}
    for period in periods:
        period_data = filtered_states[filtered_states["aug_period"] == period]
        factor_variances[period] = period_data[factor_cols].var()

    variance_df = pd.DataFrame.from_dict(factor_variances, orient="index")
    variance_df = variance_df.melt(
        var_name="factor", value_name="factor_variance", ignore_index=False
    ).reset_index(names="aug_period")

    # Extract loadings (non-zero only). The params index uses either
    # `aug_period` (CHS, internal) or `period` (AF / AMN, public) as the
    # second level name; normalize both to `aug_period` so the merge
    # below is symmetric across estimators.
    loadings_df = params.loc["loadings"].reset_index()
    loadings_df = loadings_df[loadings_df["value"] != 0].copy()
    loadings_df = loadings_df.rename(
        columns={
            "name1": "measurement",
            "name2": "factor",
            "value": "loading",
            "period": "aug_period",
        }
    )

    # Merge loadings with factor variances
    merged = loadings_df.merge(
        variance_df,
        on=["aug_period", "factor"],
    )

    # Extract measurement standard deviations
    meas_sds_df = params.loc["meas_sds"].reset_index()
    meas_sds_df = meas_sds_df.rename(
        columns={"name1": "measurement", "value": "meas_sd", "period": "aug_period"}
    )
    meas_sds_df = meas_sds_df[["aug_period", "measurement", "meas_sd"]]

    # Merge with measurement SDs
    merged = merged.merge(meas_sds_df, on=["aug_period", "measurement"])

    # Validate measurement standard deviations
    if (merged["meas_sd"] <= 0).any():
        bad = merged.loc[merged["meas_sd"] <= 0, "meas_sd"]
        raise ValueError(
            f"meas_sd must be positive for variance decomposition, "
            f"got non-positive values:\n{bad}",
        )

    # Compute variance decomposition
    # Total variance of measurement: Var(y) = L^2 * Var(F) + sd^2
    signal_var = merged["loading"] ** 2 * merged["factor_variance"]
    noise_var = merged["meas_sd"] ** 2
    total_var = signal_var + noise_var

    merged["fraction_signal"] = signal_var / total_var
    merged["fraction_noise"] = noise_var / total_var
    merged["signal_to_noise_ratio"] = signal_var / noise_var

    # Map aug_period → period for the public API
    merged["period"] = merged["aug_period"].map(aug_periods_to_periods)

    # Set index and select columns
    return merged.set_index(["period", "measurement", "factor"])[
        [
            "loading",
            "factor_variance",
            "meas_sd",
            "fraction_signal",
            "fraction_noise",
            "signal_to_noise_ratio",
        ]
    ]


@beartype(conf=DIAGNOSTICS_CONF)
def summarize_measurement_reliability(
    variance_decomposition: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize measurement reliability across periods.

    Args:
        variance_decomposition: Output from decompose_measurement_variance.

    Returns:
        DataFrame indexed by measurement with summary statistics:
        - mean_signal: Average fraction of variance due to signal
        - min_signal: Minimum fraction due to signal
        - max_signal: Maximum fraction due to signal

    """
    # Reset index to access columns
    df = variance_decomposition.reset_index()

    summary = (
        df.groupby("measurement")["fraction_signal"]
        .agg(["mean", "min", "max"])
        .rename(
            columns={"mean": "mean_signal", "min": "min_signal", "max": "max_signal"}
        )
    )

    return summary.sort_values("mean_signal", ascending=False)
