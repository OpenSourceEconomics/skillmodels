import numpy as np
from numpy.random import binomial


def add_missings(data, meas_names, p, q):
    """Add np.nans to data.

    nans are only added to measurements, not to control variables or factors.

    The function does not modify data in place. (create a new one)

    Args:
        data (pd.DataFrame): contains the observable part of a simulated dataset
        meas_names (list): list of strings of names of each measurement variable
        p (float): probability of a measurement to become missing
        q (float): probability of a measurement to remain missing in the next period
       
    Returns:
        data_with_missings
    """
    # note: should generate either {0,1} in a loop from correlated bivariate bernoullis or
    # from multivariate bernoulli given  the covariance  matrix.
    # questions:
    # 1. the number of missing values evenly distributed between individuals?
    # 2.

    # marginal probability of getting nan:

    nmeas = len(meas_names)
    data_with_missings = data.copy()
    data_interim = data[meas_names].values.copy()
    data_interim = data_interim.reshape(
        len(set(data.index)), int(len(data) / len(set(data.index))), len(meas_names)
    )
    for i in range(len(data_interim)):
        ind_data = data_interim[i]
        s = binomial(1, p, nmeas)
        ind_data[0, np.where(s == 1)] = np.nan
        for t in range(1, len(ind_data)):
            i_nan = np.isnan(ind_data[t - 1])
            prob = q * i_nan + p * (1 - i_nan)
            s_m = binomial(1, prob)
            ind_data[t, np.where(s_m == 1)] = np.nan
    data_with_missings[meas_names] = data_interim.reshape(data[meas_names].shape)

    return data_with_missings
