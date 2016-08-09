"""Contains functions that are needed for the WA estimator."""

import numpy as np


def loadings_from_covs(data, normalization):
    """Factor loadings of measurements of one factor in one period.

    Calculate the factor loadings of all measurements of one factor in one
    period as average of ratios of covariances. For this to be possible, at
    least three  measurement variables have to be available in the dataset.

    Args:
        data (DataFrame): pandas DataFrame with the measurement data for one
            factor in one period.
        normalization (list): The first value is the name of a normalized
            measurement, the second is the value it is normalized to.

    Returns:
        list of factor loadings in the order of measurements in the dataset.

    """
    measurements = list(data.columns)
    nmeas = len(measurements)
    assert nmeas >= 3, (
        'For covariance based factor loading estimation 3 or more '
        'measurements are needed.')

    correct_length = len(normalization) == 2
    correct_types = type(normalization[0] == str) and \
        type(normalization[1]) in (int, float)

    assert correct_length and correct_types, (
        'A valid normalization is a list of length two, the first entry is a '
        'string and the second entry is a number.')

    cov = data.cov()
    load_norm, load_norm_val = normalization

    loadings = []
    long_loadings = []

    for m in measurements:
        if m == load_norm:
            long_loadings.append(load_norm_val)
        else:
            estimates = []
            for m_prime in measurements:
                if m_prime not in [m, load_norm]:
                    nominator = cov.loc[m, m_prime]
                    denominator = load_norm_val * cov.loc[load_norm, m_prime]
                    estimates.append(nominator / denominator)
            loadings.append(np.mean(estimates))
            long_loadings.append(np.meant(estimates))

    return loadings, long_loadings


def intercepts_from_means(data, loadings, normalization=None):
    if normalization is None:
        intercept_list = list(data.mean())
    else:
        intercept_list = []
        intercept_norm, intercept_norm_val = normalization
        measurements = data.columns
        for m, meas in enumerate(measurements):
            if meas == intercept_norm:
                intercept_norm_loading = loadings[m]

        estimated_factor_mean = \
            (data[intercept_norm].mean() - intercept_norm_val) \
            / intercept_norm_loading

        for m, meas in enumerate(measurements):
            if meas != intercept_norm:
                mu = data[meas].mean() - loadings[m] * estimated_factor_mean
                intercept_list.append(mu)

    return intercept_list

