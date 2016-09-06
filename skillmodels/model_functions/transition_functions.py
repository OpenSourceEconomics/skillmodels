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
import pandas as pd

replace_constant = ' - 1 + constant'

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


def iv_formula_linear(x_list, z_list):
    x_formula = ' + '.join(x_list) + replace_constant
    z_formula = ' + '.join(flatten_nested_list(z_list)) + replace_constant
    return x_formula, z_formula


def model_coeffs_from_iv_coeffs_linear(
        iv_coeffs, loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=0):

    return general_model_coeffs_from_iv_coeffs(
        iv_coeffs=iv_coeffs, iv_intercept_position=-1,
        has_trans_intercept=False, loading_norminfo=loading_norminfo,
        intercept_norminfo=intercept_norminfo, coeff_sum_value=coeff_sum_value,
        trans_intercept_value=trans_intercept_value)


# =============================================================================
# constant
# =============================================================================


def constant(sigma_points, coeffs, included_positions):
    return sigma_points[:, included_positions[0]]


def nr_coeffs_constant(included_factors, params_type):
    return 0


def iv_formula_constant(x_list, z_list):
    raise NotImplementedError


def model_coeffs_from_iv_coeffs_constant(
        iv_coeffs, loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=0):
    raise NotImplementedError


# =============================================================================
# ar1
# =============================================================================


def ar1(sigma_points, coeffs, included_positions):
    return sigma_points[:, included_positions[0]] * coeffs[0]


def nr_coeffs_ar1(included_factors, params_type):
    return 1


def coeff_names_ar1(included_factors, params_type, factor, stage):
    return ['ar1_coeff__{}__{}__{}'.format(stage, factor, factor)]


def iv_formula_ar1(x_list, z_list):
    assert len(x_list) == 1, (
        'Only one factor can be included in the ar1 transition equation')
    return iv_formula_linear(x_list, z_list)


def model_coeffs_from_iv_coeffs_ar1(
        iv_coeffs, loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=0):

    return general_model_coeffs_from_iv_coeffs(
        iv_coeffs=iv_coeffs, iv_intercept_position=-1,
        has_trans_intercept=False, loading_norminfo=loading_norminfo,
        intercept_norminfo=intercept_norminfo, coeff_sum_value=coeff_sum_value,
        trans_intercept_value=trans_intercept_value)

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


def iv_formula_log_ces(x_list, z_list):
    raise NotImplementedError(
        'The log_ces function would lead to an IV equation that is not linear '
        ' in parameters that cannot be estimated with closed form estimators. '
        ' It is not and will not be implemented as part of the WA estimator.')


def model_coeffs_from_iv_coeffs_log_ces(
        iv_coeffs, loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=0):

    raise NotImplementedError(
        'The log_ces function would lead to an IV equation that is not linear '
        ' in parameters that cannot be estimated with closed form estimators. '
        ' It is not and will not be implemented as part of the WA estimator.')


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


def iv_formula_translog(x_list, z_list):

    raise NotImplementedError(
        'The squared terms in the general translog function will need special '
        'treatment in the WA estimator and is therefore not yet implemented.')


def model_coeffs_from_iv_coeffs_translog(
        iv_coeffs, loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=0):
    raise NotImplementedError(
        'The log_ces function would lead to an IV equation that is not linear '
        ' in parameters that cannot be estimated with closed form estimators. '
        ' It is not and will not be implemented as part of the WA estimator.')

# =============================================================================
# translog without square terms
# =============================================================================


def no_squares_translog(sigma_points, coeffs, included_positions):
    # the coeffs will be parsed as follows:
    # last entry = TFP term
    # first len(included_position) entries = coefficients for the factors
    # rest = coefficients for interaction terms (excluding squared terms)
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
            for pos2 in included_positions[p + 1:]:
                res += coeffs[next_coeff] * fac * sigma_points[i, pos2]
                next_coeff += 1
        result_array[i] = res
    return result_array


def nr_coeffs_no_squares_translog(included_factors, params_type):
    nfac = len(included_factors)
    return nr_coeffs_translog(included_factors, params_type) - nfac


def coeff_names_no_squares_translog(
        included_factors, params_type, factor, stage):
    names = ['translog__{}__{}__{}'.format(stage, factor, i_fac)
             for i_fac in included_factors]

    for i, i_fac1 in enumerate(included_factors):
        for i_fac2 in included_factors[i + 1:]:
            names.append('translog__{}__{}__{}-{}'.format(
                stage, factor, i_fac1, i_fac2))
    names.append('translog__{}__{}__TFP'.format(stage, factor))
    return names


def iv_formula_no_squares_translog(x_list, z_list):
    x_polynomials = x_patsy_polynomials(x_list, squares=False)
    x_formula = ' + '.join(x_list + x_polynomials) + replace_constant
    z_polynomials = z_patsy_polynomials(z_list, squares=False)
    flat_z = flatten_nested_list(z_list)
    z_formula = ' + '.join(flat_z + z_polynomials) + replace_constant

    return x_formula, z_formula


def model_coeffs_from_iv_coeffs_no_squares_translog(
        iv_coeffs, loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=None):

    return general_model_coeffs_from_iv_coeffs(
        iv_coeffs=iv_coeffs, iv_intercept_position=-1,
        has_trans_intercept=True, loading_norminfo=loading_norminfo,
        intercept_norminfo=intercept_norminfo, coeff_sum_value=coeff_sum_value,
        trans_intercept_value=trans_intercept_value)


# =============================================================================
# helper functions
# =============================================================================


def x_patsy_polynomials(varlist, squares=True, interactions=True):
    polynomials = []
    for i, x1 in enumerate(varlist):
        if squares is True:
            polynomials.append('np.square({})'.format(x1))
        if interactions is True:
            for x2 in varlist[i + 1:]:
                polynomials.append('{}:{}'.format(x1, x2))
    return polynomials


def z_patsy_polynomials(varlist, squares=True, interactions=True):
    polynomials = []
    if squares is True:
        for sublist in varlist:
            polynomials += x_patsy_polynomials(sublist)
    if interactions is True:
        for i, sublist1 in enumerate(varlist):
            for m1 in sublist1:
                for sublist2 in varlist[i + 1:]:
                    for m2 in sublist2:
                        polynomials.append('{}:{}'.format(m1, m2))
    return polynomials


def flatten_nested_list(nested_list):
    return [x for sublist in nested_list for x in sublist]


def general_model_coeffs_from_iv_coeffs(
        iv_coeffs, iv_intercept_position, has_trans_intercept,
        loading_norminfo=None, intercept_norminfo=None,
        coeff_sum_value=None, trans_intercept_value=None):

    # assert statements
    to_check = [coeff_sum_value, loading_norminfo]
    assert None in to_check, ('')
    assert to_check != [None, None], ('')

    to_check = [trans_intercept_value, intercept_norminfo]
    assert None in to_check, ('')
    assert to_check != [None, None], ('')

    assert iv_intercept_position in [0, -1], ('')

    iv_coeffs = iv_coeffs.groupby(level=0).mean()
    meas_coeffs = pd.DataFrame(data=0, index=iv_coeffs.index,
                               columns=['loadings', 'intercepts'])

    if iv_intercept_position == 0:
        all_but_intercept = slice(1, len(iv_coeffs))
    else:
        all_but_intercept = slice(0, -1)

    # get coeff sum
    if coeff_sum_value is None:
        load_norm_y, load_norm_val = loading_norminfo
        iv_sum = iv_coeffs.loc[load_norm_y].values[all_but_intercept].sum()
        coeff_sum_value = iv_sum / load_norm_val

    # calculate all lambdas
    for y_variable in iv_coeffs.index:
        iv_sum = iv_coeffs.loc[y_variable].values[all_but_intercept].sum()
        meas_coeffs.loc[y_variable, 'loadings'] = iv_sum / coeff_sum_value

    # get trans intercept
    if trans_intercept_value is None:    # TFP term is free

        intercept_norm_y, intercept_norm_val = intercept_norminfo
        intercept_coeff = list(iv_coeffs.loc[intercept_norm_y])[iv_intercept_position]
        corresponding_loading = meas_coeffs.loc[intercept_norm_y, 'loadings']
        trans_intercept_value = \
            (intercept_coeff - intercept_norm_val) / corresponding_loading

    # calculate all mus
    for y_variable in iv_coeffs.index:
        intercept_coeff = list(iv_coeffs.loc[y_variable])[iv_intercept_position]
        corresponding_loading = meas_coeffs.loc[y_variable, 'loadings']
        meas_coeffs.loc[y_variable, 'intercepts'] = \
            intercept_coeff - corresponding_loading * trans_intercept_value

    gamma_coeffs = iv_coeffs[iv_coeffs.columns[all_but_intercept]]
    next_period_loadings = meas_coeffs['loadings']
    gammas = gamma_coeffs.divide(next_period_loadings, axis=0).mean().values

    if iv_intercept_position == 0:
        gammas = np.hstack([trans_intercept_value, gammas])
    else:
        gammas = np.hstack([gammas, trans_intercept_value])

    return meas_coeffs, gammas
