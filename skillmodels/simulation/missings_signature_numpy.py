import numpy as np
from numpy.random import binomial


def add_missings(
    data, meas_names, control_names, share, serial_corr=0.0, within_period_corr=0.0
):
    """Add np.nans to data.

    nans are only added to measurements, not to control variables or factors.

    The function does not modify data in place. (create a new one)

    Args:
        data (pd.DataFrame): contains the observable part of a simulated dataset
        share (float): this share of observations is set to np.nan
        serial_corr (float, optional): serial correlation between missings of
            one individual
        within_period_corr (float, optional): within period correlation of
            missings of one individual

    Returns:
        data_with_missings
    """
    # note: should generate either {0,1} in a loop from correlated bivariate bernoullis or
    # from multivariate bernoulli given  the covariance  matrix.
    # questions:
    # 1. the number of missing values evenly distributed between individuals?
    # 2.

    # marginal probability of getting nan:
    p = (
        share
        * (np.size(data[control_names]) + np.size(data[meas_names]))
        / np.size(data[meas_names])
    )
    # serial transition probabilities
    p_t_10 = p * (1 - serial_corr)
    p_t_11 = p + serial_corr * (1 - p)
    # between measurements transition probabilities
    p_m_10 = p * (1 - within_period_corr)
    p_m_11 = p + within_period_corr * (1 - p)

    data_with_missings = data.copy()
    data_interim = data[meas_names].values.copy()
    data_interim = data_interim.reshape(
        len(set(data.index)), int(len(data) / len(set(data.index))), len(meas_names)
    )
    #num_missing_val = np.count_nonzero(np.isnan(data_interim))
    #replaced_share = num_missing_val / (
     #   data[meas_names].size + data[control_names].size
    #)

    while np.count_nonzero(np.isnan(data_interim))/(
                        data[meas_names].size + data[control_names].size
                    )<share:#replaced_share < share:
        for i in range(len(data_interim)):  # alternatively randomly choose individual??
            ind_data = data_interim[i]

            if binomial(1, p) == 1:
                ind_data[0, 0] = np.nan
            for m in range(1, len(meas_names)):   
                if np.isnan(ind_data[0, m - 1]):
                        if binomial(1, p_m_11) == 1:
                            ind_data[0, m] = np.nan
                else:
                        if binomial(1, p_m_10) == 1:
                            ind_data[0, m] = np.nan
            for t in range(1, len(ind_data)):
                if np.isnan(ind_data[t - 1, 0]):
                    if binomial(1, p_t_11) == 1:
                        ind_data[t, 0] = np.nan
                else:
                    if binomial(1, p_t_10) == 1:
                        ind_data[t, 0] = np.nan
                for m in range(1, len(meas_names)):
                    if np.isnan(ind_data[t, m - 1]):
                        if binomial(1, p_m_11) == 1:
                            ind_data[t, m] = np.nan
                    else:
                        if binomial(1, p_m_10) == 1:
                            ind_data[t, m] = np.nan

                    #num_missing_val = np.count_nonzero(np.isnan(data_interim))
                    #replaced_share = num_missing_val / (
                     #   data[meas_names].size + data[control_names].size
                    #)
    data_with_missings[meas_names] = data_interim.reshape(data[meas_names].shape)

    return data_with_missings
