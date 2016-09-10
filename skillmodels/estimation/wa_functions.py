"""Contains functions that are needed for the WA estimator."""

import numpy as np
import pandas as pd


def loadings_from_covs(data, normalization):
    """Factor loadings of measurements of one factor in the first.

    Calculate the factor loadings of all measurements of one factor in the
    first period as average of ratios of covariances. For this to be possible,
    at least three  measurement variables have to be available in the dataset.

    Args:
        data (DataFrame): pandas DataFrame with the measurement data for one
            factor in one period.
        normalization (list): The first value is the name of a normalized
            measurement, the second is the value it is normalized to.

    Returns:
        loadings (Series): pandas Series with estimated factor loadings

    """
    measurements = list(data.columns)
    nmeas = len(measurements)
    assert nmeas >= 3, (
        'For covariance based factor loading estimation 3 or more '
        'measurements are needed.')

    cov = data.cov()
    load_norm, load_norm_val = normalization
    loadings = pd.Series(index=measurements, name='loadings')

    for m in measurements:
        if m != load_norm:
            estimates = []
            for m_prime in measurements:
                if m_prime not in [m, load_norm]:
                    nominator = load_norm_val * cov.loc[m, m_prime]
                    denominator = cov.loc[load_norm, m_prime]
                    estimates.append(nominator / denominator)
            loadings[m] = np.mean(estimates)
        else:
            loadings[m] = load_norm_val
    return loadings


def intercepts_from_means(data, normalization, loadings):
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
        loadings (Series): pandas Series with estimated factor loadings

    Returns:
        intercepts (Series): pandas Series with estimated measurement
            intercepts
        factor mean: The estimated factor mean if a intercept was normalized
            or None
    """
    measurements = list(data.columns)

    if len(normalization) == 0:
        intercepts = data.mean()
        factor_mean = None
    else:
        intercepts = pd.Series(index=measurements, name='intercepts')
        intercept_norm, intercept_norm_val = normalization
        loading = loadings[intercept_norm]
        factor_mean = \
            (data[intercept_norm].mean() - intercept_norm_val) / loading

        for m, meas in enumerate(measurements):
            if meas != intercept_norm:
                loading = loadings[meas]
                intercepts[meas] = \
                    data[meas].mean() - loading * factor_mean
            else:
                intercepts[meas] = intercept_norm_val
    return intercepts, factor_mean


def prepend_index_level(df, to_prepend):
    df = df.copy(deep=True)
    df.index = pd.MultiIndex.from_tuples(
        [(to_prepend, x) for x in df.index])
    return df


def initial_meas_coeffs(y_data, measurements, normalizations):
    """Dataframe of loadings and intercepts for all factors in initial period.

    Args:
        y_data (DataFrame): pandas DataFrame with measurement data of initial
            period.
        measurements (dictionary): the keys are the factors. The values are
            nested lits with one sublist of measurement names for each period.
        normalizations (dictionary): the keys are the factors. the values are
            dictionaries with 'intercepts' and 'loadings' as keys. Their values
            are lists of lists with one sublist for each period. Each sublist
            contains the name of the normalized measurements as first entry and
            the value to which it is normalized as second entry.

    Returns:
        meas_coeffs (DataFrame): DataFrame with loadings and intercepts of the
            initial period.
        X_zero (np.ndarray): numpy array with initial factor means.

    """
    factors = sorted(list(measurements.keys()))
    to_concat = []
    X_zero = []
    for f, factor in enumerate(factors):
        meas_list = measurements[factor][0]
        data = y_data[meas_list]
        norminfo_load = normalizations[factor]['loadings'][0]

        loadings = loadings_from_covs(data, norminfo_load)
        norminfo_intercept = normalizations[factor]['intercepts'][0]
        intercepts, factor_mean = intercepts_from_means(
            data, norminfo_intercept, loadings)
        to_concat.append(pd.concat([loadings, intercepts], axis=1))
        X_zero.append(factor_mean)

    meas_coeffs = pd.concat(to_concat)
    return meas_coeffs, np.array(X_zero)


def initial_cov_matrix(data, storage_df, measurements):
    """
    Initial covariance matrix of latent factors from measurement covariances.

    Args:
        data (DataFrame): DataFrame of the measurements of the initial period.
        storage_df (DataFrame): DataFrame of loadings and intercepts.
        measurements (dictionary): the keys are the factors. The values are
            nested lits with one sublist of measurement names for each period.

    Returns:
        factor_covs (np.ndarray): 1d array with the upper triangular
            elements of the covariance matrix.
    """
    t = 0
    factors = sorted(list(measurements.keys()))
    measurements = {factor: measurements[factor][t] for factor in factors}

    meas_cov = data.cov()
    factor_covs = []

    loadings = storage_df.loc[0, 'loadings']
    scaled_meas_cov = meas_cov.divide(
        loadings, axis=0).divide(loadings, axis=1)

    # set diagonal elements to NaN
    for i in scaled_meas_cov.index:
        scaled_meas_cov.loc[i, i] = np.nan

    for f1, factor1 in enumerate(factors):
        measurements1 = measurements[factor1]
        for f2, factor2 in enumerate(factors):
            measurements2 = measurements[factor2]
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
    w = _iv_gmm_weights(z)
    beta = _iv_math(y, x, z, w)

    if fit_method == 'optimal':
        u = y - np.dot(x, beta)
        w = _iv_gmm_weights(z, u)
        beta = _iv_math(y, x, z, w)

    return beta


def _iv_math(y, x, z, w):
    xTz = x.T.dot(z)
    helper = xTz.dot(w)
    inverse_part = np.linalg.pinv(np.dot(helper, xTz.T))
    y_part = helper.dot(z.T.dot(y))
    beta = inverse_part.dot(y_part)

    return beta


def _iv_gmm_weights(z, u=None):
    nobs, k_prime = z.shape
    if u is None:
        w = np.linalg.pinv(np.dot(z.T, z) / nobs)
    else:
        u_squared = u ** 2
        outerprod = z.reshape(nobs, 1, k_prime) * z.reshape(nobs, k_prime, 1)
        s = (u_squared.reshape(nobs, 1, 1) * outerprod).sum(axis=0) / nobs
        w = np.linalg.pinv(s)
    return w
