"""Contains transition functions and corresponding helper functions.

Not all functions and helper functions will be used in all estimators. But if
you contribute a transition function you should also add the helper functions
for all estimators.

Below the signature and purpose of a transition function and its helper
functions is explained with a transition function called example_func:

**example_func(** *sigma_points, coeffs, included_positions* **)**:

    The actual transition function.

    Args:
        * sigma_points: 2d array of sigma_points or states being transformed
        * coeffs: vector with coefficients specific to this transition function
          If the coeffs include an intercept term (e.g. the log of a TFP term),
          this has to be the FIRST or LAST element of coeffs.
        * included_positions: the positions of the factors that are
          included in the transition equation

    Returns
        * 1d array


**index_tuples_example_func(** *factor, included_factors, period* **)**:

    A list of index tuples for the params of the transition function.

    Each index tuple contains four entries:

    - 'trans' (fix)
    - period
    - factor
    - 'some-name'

"""
import numpy as np
from numba import jit

# =============================================================================
# linear_with_constant
# =============================================================================


def linear_with_constant(sigma_points, coeffs, included_positions):
    nfac = sigma_points.shape[1]
    coeff_vec = np.zeros(nfac)
    for p, pos in enumerate(included_positions):
        coeff_vec[pos] = coeffs[p]
    without_constant = np.dot(sigma_points, coeff_vec)
    return coeffs[-1] + without_constant


def index_tuples_linear_with_constant(factor, included_factors, period):
    ind_tups = []
    for incl_fac in included_factors:
        ind_tups.append(("trans", period, factor, f"lincoeff-{incl_fac}"))
    ind_tups.append(("trans", period, factor, "lincoeff-constant"))
    return ind_tups


# =============================================================================
# constant
# =============================================================================


def constant(sigma_points, coeffs, included_positions):
    return sigma_points[:, included_positions[0]]


def index_tuples_constant(factor, included_factors, period):
    return []


# =============================================================================
# log_ces function, (KLS-Verion)
# =============================================================================


def log_ces(sigma_points, coeffs, included_positions):
    nfac = sigma_points.shape[1]
    phi = coeffs[-1]
    gammas = np.zeros(nfac)
    for p, pos in enumerate(included_positions):
        gammas[pos] = coeffs[p]

    exponents = sigma_points * phi
    x = np.exp(exponents)
    x = np.dot(x, gammas)
    scaling_factor = 1 / phi
    result = scaling_factor * np.log(x)

    return result


def index_tuples_log_ces(factor, included_factors, period):
    ind_tups = []
    for incl_fac in included_factors:
        ind_tups.append(("trans", period, factor, f"gamma-{incl_fac}"))
    ind_tups.append(("trans", period, factor, "phi"))
    return ind_tups


def constraints_log_ces(factor, included_factors, period):
    ind_tups = index_tuples_log_ces(factor, included_factors, period)
    loc = ind_tups[:-1]
    return {"loc": loc, "type": "probability"}


# =============================================================================
# translog function, (Non-KLS-Version)
# =============================================================================


@jit
def translog(sigma_points, coeffs, included_positions):
    # the coeffs will be parsed as follows:
    # last entry = TFP term
    # first len(included_position) entries = coefficients for the factors
    # rest = coefficients for interaction terms (including squared terms)
    long_side, nfac = sigma_points.shape
    result_array = np.zeros(long_side)
    nr_included = len(included_positions)
    for i in range(long_side):
        # TFP term is additive in logs
        res = coeffs[-1]
        next_coeff = nr_included
        for p, pos1 in enumerate(included_positions):
            # counter for coefficients
            # the factor held fix during the inner loop
            fac = sigma_points[i, pos1]
            # add the factor term
            res += coeffs[p] * fac
            # add the interaction terms
            for pos2 in included_positions[p:]:
                res += coeffs[next_coeff] * fac * sigma_points[i, pos2]
                next_coeff += 1
        result_array[i] = res
    return result_array


def index_tuples_translog(factor, included_factors, period):
    ind_tups = []
    for i_fac in included_factors:
        ind_tups.append(("trans", period, factor, f"translog-{i_fac}"))

    for i, i_fac1 in enumerate(included_factors):
        for i_fac2 in included_factors[i:]:
            if i_fac1 == i_fac2:
                ind_tups.append(("trans", period, factor, f"translog-{i_fac1}-squared"))
            else:
                ind_tups.append(
                    ("trans", period, factor, f"translog-{i_fac1}-{i_fac2}")
                )
    ind_tups.append(("trans", period, factor, "translog-tfp"))
    return ind_tups
