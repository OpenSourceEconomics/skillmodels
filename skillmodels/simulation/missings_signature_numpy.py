import numpy as np
import pandas as pd
from numpy.random import binomial


def add_missings(data, meas_names, share, serial_corr=0.0, within_period_corr=0.0):
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
    p = 0.6#share * np.size(data.values) / np.size(data[meas_names].values)
    # serial transition probabilities
    p_t_10 = p * (1 - serial_corr)
    p_t_11 = p + serial_corr * (1 - p)

    # between measurements transition probabilities
    p_m_10 = p * (1 - within_period_corr)
    p_m_11 = p + within_period_corr * (1 - p)
    #data_with_missings = [np.zeros((int(len(data)/len(set(data.index))),len(meas_names)))]*len(set(data.index))
    data_with_missings = data[meas_names].values.copy()
    data_with_missings=data_with_missings.reshape(len(set(data.index)),int(len(data)/len(set(data.index))),len(meas_names))
    num_missing_val = np.count_nonzero(np.isnan(data_with_missings))
    replaced_share = num_missing_val / data_with_missings.size
    
    while replaced_share < share:
        for i in range(len(data_with_missings)):  # alternatively randomly choose individual??
            ind_data = data_with_missings[i]

            if binomial(1, p) == 1:
                ind_data[0,0] = np.nan
            for t in range(1, len(ind_data)):
                if np.isnan(ind_data[t - 1,0]):
                    if binomial(1, p_t_11) == 1:
                        ind_data[t,0] = np.nan
                else:
                    if binomial(1, p_t_10) == 1:
                        ind_data[t,0] = np.nan
                for m in range(1, len(meas_names)):
                    if np.isnan(ind_data[t,m-1]):
                        if binomial(1, p_m_11) == 1:
                            ind_data[t,m] = np.nan
                    else:
                        if binomial(1, p_m_10) == 1:
                            ind_data[t,m] = np.nan

                    num_missing_val = np.count_nonzero(np.isnan(data_with_missings))
                    replaced_share = num_missing_val / data.size
    return data_with_missings
