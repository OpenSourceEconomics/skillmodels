"""Define policy functions for endogeneity correction.

The correction method is the one described in section 4.2.5 of the CHS paper,
i.e. the one used in the empirical part of the paper.

each function takes the following arguments:
    * sigma_points: A 2d array with the sigma points
    * psi: a parameter vector for the policy function
    * endog_position: the position of the endogenous factor in the
      alphabetically ordered factor list

.. Note:: It is possible that this module changes quite heavily when I optimize
    the predict step for speed.

"""

import numpy as np


def linear(sigma_points, psi, endog_position):
    return np.dot(sigma_points, psi)
