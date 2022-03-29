import pandas as pd


def create_dataset_with_variance_decomposition(filtered_states, params):
    """Calculate variance decomposition.
    Variance is decomposed into the measurement error and the signal.
    Below the function calculation is based on section 4.2.2.The Empirical Importance of
    Measurement Error of CHS paper.(Cuhna, et al. 2010, 907).
    Article location:
    https://www.econometricsociety.org/publications/econometrica/2010/05/01/estimating-technology-cognitive-and-noncognitive-skill
    Args:
        params (pandas.DataFrame): DataFrame with model parameters.
        filtered_states (pandas.DataFrame): Tidy DataFrame with filtered states.

    Returns:
        data_variance_decomposition (pd.DataFrame): Dataset with a decomposed variance.
    """

    periods = filtered_states.period.unique()
    var = {}
    for period in periods:
        data_period = filtered_states.query(f"period == {period}")
        data_cleaned = data_period.drop(columns=["period", "id", "mixture"])
        var[period] = data_cleaned.var()
    variance_df = pd.DataFrame.from_dict(var, orient="index")
    variance_df = (
        variance_df.stack()
        .reset_index(drop=False)
        .rename(
            columns={"level_0": "period", "level_1": "name2", 0: "variance of factor"}
        )
    )
    loadings_df = params.loc[("loadings")].reset_index()
    loadings_df = loadings_df[loadings_df["value"] != 0]
    merged_df = pd.merge(loadings_df, variance_df, on=["period", "name2"])
    meas_sds_df = params.loc[("meas_sds")].reset_index()
    meas_sds_df = meas_sds_df.drop(columns=["name2"])
    merged_df = pd.merge(merged_df, meas_sds_df, on=["period", "name1"])
    merged_df = merged_df.rename(columns={"value_x": "loadings", "value_y": "meas_sds"})
    denominator = (
        merged_df["meas_sds"] ** 2
        + merged_df["loadings"] ** 2 * merged_df["variance of factor"]
    )

    merged_df["fraction due to meas error"] = merged_df["meas_sds"] ** 2 / denominator
    merged_df["fraction due to factor var"] = (
        merged_df["loadings"] ** 2 * merged_df["variance of factor"] / denominator
    )
    data_variance_decomposition = merged_df.set_index(["period", "name1", "name2"])

    return data_variance_decomposition
