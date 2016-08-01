"""Contains transition functions and corresponding helper functions.

Each transition function has to transform a 2d array of flattened sigma points
of the form [nemf * nind * nsigma, nfac] and should be optimized for speed.

It will be called with the following arguments:

    * sigma_points: the array of sigma_points that is being transformed
    * coeffs: a vector with coefficients specific to this transition function
    * included_positions: the positions of the factors that are
      included in the transition equation

Moreover for each transition function the following auxiliary functions should
be implemented:

    * mandatory:
        * nr_coeffs_NAME(included_factors, params_type)
    * optional:
        * transform_params_NAME(params, included_factors, out)
        * bounds_Name(included_factors)
        * coeff_names_NAME(included_factors, params_type, factor, stage)
        * start_values_NAME(factor, included_factors)

NAME has to be replaced by the transition function's name.
The naming of the auxiliary functions is very important as other code relies
on it. Since most of the helper functions are optional the code won't raise an
error if a function is not found because of a wrong name.

.. Note:: It is possible that this module changes quite heavily when I optimize
    the predict step for speed.

"""
import numpy as np
from numba import jit

# =============================================================================
# linear
# =============================================================================


def linear(sigma_points, coeffs, included_positions):
    nfac = sigma_points.shape[1]
    coeff_vec = np.zeros(nfac)
    for p, pos in enumerate(included_positions):
        coeff_vec[pos] = coeffs[p]
    return np.dot(sigma_points, coeff_vec)


def nr_coeffs_linear(included_factors, params_type):
    return len(included_factors)


def coeff_names_linear(included_factors, params_type, factor, stage):
    fs = 'lincoeff__{}__{}__{}'
    return [fs.format(stage, factor, i_fac) for i_fac in included_factors]


# =============================================================================
# constant
# =============================================================================


def constant(sigma_points, coeffs, included_positions):
    return sigma_points[:, included_positions[0]]


def nr_coeffs_constant(included_factors, params_type):
    return 0

# =============================================================================
# ar1
# =============================================================================


def ar1(sigma_points, coeffs, included_positions):
    return sigma_points[:, included_positions[0]] * coeffs[0]


def nr_coeffs_ar1(included_factors, params_type):
    return 1


def coeff_names_ar1(included_factors, params_type, factor, stage):
    return ['ar1_coeff__{}__{}__{}'.format(stage, factor, factor)]


# =============================================================================
# log_ces (KLS-Verion)
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


def nr_coeffs_log_ces(included_factors, params_type):
    if params_type == 'short':
        return len(included_factors)
    else:
        return len(included_factors) + 1


def transform_coeffs_log_ces(coeffs, included_factors, out):
    out[:-2] = coeffs[:-1]
    out[-2] = 1
    out /= np.sum(out[:-1])
    out[-1] = coeffs[-1]


def bounds_log_ces(included_factors):
    lb = np.zeros(len(included_factors), dtype=object)
    lb[-1] = None

    ub = np.zeros(len(included_factors), dtype=object)
    ub[:] = None
    return lb, ub


def coeff_names_log_ces(included_factors, params_type, factor, stage):
    fs1 = 'gamma__{}__{}__{}'
    fs2 = 'phi__{}__{}__Phi'

    names = [
        fs1.format(stage, factor, i_fac) for i_fac in included_factors[:-1]]
    if params_type == 'long':
        names.append(fs1.format(stage, factor, included_factors[-1]))
    names.append(fs2.format(stage, factor))

    return names

# =============================================================================
# translog (Non-KLS-Version)
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


def nr_coeffs_translog(included_factors, params_type):
    nfac = len(included_factors)
    return 1 + 0.5 * nfac * (nfac + 3)


def coeff_names_translog(included_factors, params_type, factor, stage):
    names = ['translog__{}__{}__{}'.format(stage, factor, i_fac)
             for i_fac in included_factors]

    for i, i_fac1 in enumerate(included_factors):
        for i_fac2 in included_factors[i:]:
            if i_fac1 == i_fac2:
                names.append('translog__{}__{}__{}-squared'.format(
                    stage, factor, i_fac1))
            else:
                names.append('translog__{}__{}__{}-{}'.format(
                    stage, factor, i_fac1, i_fac2))
    names.append('translog__{}__{}__TFP'.format(stage, factor))
    return names
