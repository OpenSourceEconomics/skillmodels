"""Contains functions that are needed for the WA estimator."""

import numpy as np


def loadings_from_covs(data, normalization, storage_df):
    """Factor loadings of measurements of one factor in the first.

    Calculate the factor loadings of all measurements of one factor in the
    first period as average of ratios of covariances. For this to be possible,
    at least three  measurement variables have to be available in the dataset.
    The result is stored in storage_df

    Args:
        data (DataFrame): pandas DataFrame with the measurement data for one
            factor in one period.
        normalization (list): The first value is the name of a normalized
            measurement, the second is the value it is normalized to.
        storage_df (DataFrame): DataFrame in which the results are stored

    """
    t = 0
    measurements = list(data.columns)
    nmeas = len(measurements)
    assert nmeas >= 3, (
        'For covariance based factor loading estimation 3 or more '
        'measurements are needed.')

    cov = data.cov()
    load_norm, load_norm_val = normalization

    for m in measurements:
        if m != load_norm:
            estimates = []
            for m_prime in measurements:
                if m_prime not in [m, load_norm]:
                    nominator = load_norm_val * cov.loc[m, m_prime]
                    denominator = cov.loc[load_norm, m_prime]
                    estimates.append(nominator / denominator)
            storage_df.loc[(t, m), 'loadings'] = np.mean(estimates)


def intercepts_from_means(data, normalization, storage_df, mean_list):
    """Calculate intercepts and factor means for 1 factor in the first period.

    If the normalization list is not empty, it is assumed that the factor
    mean is not normalized and has to be estimated. In this case, the factor
    mean is calculated first and appended to the mean_list. Later the
    non-normalized intercepts are calculated and stored in storage_df.

    Args:
        data (DataFrame): pandas DataFrame with the measurement data for one
            factor in one period.
        normalization (list): The first value is the name of a normalized
            measurement, the second is the value it is normalized to.
        storage_df (DataFrame): DataFrame in which the results are stored
        mean_list (list): a list to which the estimated mean is appended

    """
    t = 0
    measurements = list(data.columns)
    if len(normalization) == 0:
        for meas in measurements:
            storage_df.loc[(t, meas), 'intercepts'] = data[meas].mean()
    else:
        intercept_norm, intercept_norm_val = normalization
        loading = storage_df.loc[(t, intercept_norm), 'loadings']
        estimated_factor_mean = \
            (data[intercept_norm].mean() - intercept_norm_val) / loading
        mean_list.append(estimated_factor_mean)

        for m, meas in enumerate(measurements):
            if meas != intercept_norm:
                loading = storage_df.loc[(t, meas), 'loadings']
                storage_df.loc[(t, meas), 'intercepts'] = \
                    data[meas].mean() - loading * estimated_factor_mean


def initial_cov_matrix(data, storage_df, measurements_per_factor):
    """Estimate initial cov matrix of factors from covs of measurements."""
    meas_cov = data.cov()
    factor_covs = []

    loadings = storage_df.loc[0, 'loadings']
    scaled_meas_cov = meas_cov.divide(
        loadings, axis=0).divide(loadings, axis=1)

    # set diagonal elements to NaN
    for i in scaled_meas_cov.index:
        scaled_meas_cov.loc[i, i] = np.nan

    factors = sorted(list(measurements_per_factor.keys()))

    for f1, factor1 in enumerate(factors):
        measurements1 = measurements_per_factor[factor1]
        for f2, factor2 in enumerate(factors):
            measurements2 = measurements_per_factor[factor2]
            if f2 >= f1:
                relevant = scaled_meas_cov.loc[measurements1, measurements2]
                factor_covs.append(relevant.mean().mean())

    return np.array(factor_covs)


def residual_measurements(data, loadings, intercepts):
    df = (data - intercepts) / loadings
    df.columns = ['{}_resid'.format(col) for col in df.columns]
    return df


def iv_reg(y, x, z, fit_method='2sls'):
    """Estimate a linear-in-parameters instrumental variable equation via GMM.

    args:
        y (np.ndarray): array of length n, dependent variable.
        x (np.ndarray): array of shape [n, k], original explanatory variables
        z (np.ndarray): array of shape [n, >=k], the instruments. Instruments
            have to include exogenous variables that are already in x.
        fit_method (str): takes the values `2sls' or `optimal'. `Optimal' is
            computationally  expensive but uses a more efficient weight matrix.
            The default is `2sls'.

    returns:
        beta (np.ndarray): array of length k with the estimated parameters

    All input arrays must not contain NaNs and constants must be included
    explicitly in x and z.

    """
    nobs, k_prime = z.shape
    w = np.linalg.pinv(np.dot(z.T, z) / nobs)
    beta = _iv_math(y, x, z, w)

    if fit_method == 'optimal':
        u_squared = (y - np.dot(x, beta)) ** 2
        outerprod = z.reshape(nobs, 1, k_prime) * z.reshape(nobs, k_prime, 1)
        s = (u_squared.reshape(nobs, 1, 1) * outerprod).sum(axis=0) / nobs
        w = np.linalg.pinv(s)
        beta = _iv_math(y, x, z, w)

    return beta


def _iv_math(y, x, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.linalg.pinv(np.dot(helper, xTz.T))
    y_part = helper.dot(z.T.dot(y))
    beta = inverse_part.dot(y_part)

    return beta











