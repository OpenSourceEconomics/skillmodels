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

For each transition function, the following auxiliary functions must
be implemented:

**nr_coeffs_example_func(** *included_factors, params_type* **)**:

    Determines the number of estimated coefficients.

    Args:
        * included_factors: a list with the name of factors that are included
          in the right hand side of the transition equation.
        * params_type: see :ref:`params_type`

    Returns:
        nr_coeffs: integer value

**iv_formula_example_func(** *x_list, z_list* **)**:

    Translates a transition function into patsy formulas that can be used to
    construct the arrays for an iv regression. This is only used in the WA
    estimator.

    The formula strings MUST contain '- 1' to disable the automatic inclusion
    of a intercept term by patsy. If an intercept is needed in the IV equation,
    the variable 'constant' has to be the FIRST or LAST term in the formula.

    Args:
        * x_list: a list with names of variables that are used to form the
          independent variables of the iv equation. Does not yet contain
          squares or interaction terms. Adding those is the purpose of the
          formula.

        * z_list: a list of lists with one sublist for each included factor.
          Each sublist contains the variables used to form the instruments.
          The sublists do not yet contain squares or interaction terms.
          Adding those is the purpose of the formula.

    Returns:
        * x_formula: a valid patsy formula to construct a DesignMatrix of
          independent variables out of a DataFrame that contains the variables
          from x_list.
        * z_formula: a valid patsy formula to construct a DesignMatrix of
          instruments out of a DataFrame that contains the variables from
          z_list.

**model_coeffs_from_iv_coeffs_example_func(**
*iv_coeffs, loading_norminfo, intercept_norminfo, coeff_sum_value,
trans_intercept_value* **)**:

    Calculates the true transition function parameters as well as next period
    measurement system parameters (intercepts and loadings) from the iv
    estimates, given that enough restrictions or normalizations for
    identification are provided. The restrictions and normalizations are
    provided by all but the first argument of the function.

    This is the most complicated helper function of all, but for most
    linear-in-parameters functions it will only be a call to
    general_model_coeffs_from_iv_coeffs with suitable arguments.

    Args:
        * iv_coeffs: DataFrame with IV-regression estimates. The columns are
          just delta0, delta1, ...; The index is a MultiIndex. The first level
          are the names of the dependent variables (next period measurements)
          of the estimated equations. The second level consists of integers
          that identify the different combinations of independent variables.
        * loading_norminfo: None or a list of length 2 of the form
          [normalized_variable_name, norm_value]
        * intercept_norminfo: same as loading norminfo.
        * coeff_sum_value: None or a float that specifies the sum of the
          transition equation parameters, excluding the transition intercept.
          For KLS functions this is usually a transition function property. For
          non KLS functions it can be identified from the first period in a
          development stage.
        * trans_intercept_value: analogous to coeff_sum_value, but for the
          intercept of the transition equation.

    Returns
        * meas_coeffs: DataFrame with next-period loadings and intercepts.
          The index are the names of the dependent variable of the regression.
        * gammas: numpy array with estimated transition function coefficients.
        * newly_identified_coeff_sum_value: None if a coeff_sum_value was
          given; else, the coeff_sum_value that was identified.
        * newly_identified_trans_intercept_value: analogous.


Moreover, for each transition function, the following auxiliary functions can
be implemented:

**transform_coeffs_example_func(** *coeffs, included_factors, direction, out* **)**

    Transform parameters from params_type 'short' to 'long'. See
    :ref:`params_type` for details. Only needed for CHS estimator and only for
    functions that need transformation of parameters.

    Args:
        * coeffs: 1d array of parameters
        * included_factors: list of names of included factors
        * direction: takes values 'short_to_long' and 'long_to_short'
        * out: numpy array in which the result is stored

**bounds_example_func(** *included_factors* **)**

    Generate a list of bounds for the estimated parameters of the transition
    equation. Only needed for CHS and only if the parameters need bounds.

    Args:
        * included_factors: list of names of included factors

    Returns:
        * lower_bound: 1d array with bounds for the estimated parameters.
          Takes the value None if no lower bound is needed.
        * upper_bound: analogous to lower bound.

**coeff_names_example_func(**
*included_factors, params_type, factor, stage* **)**

    List of names for the estimated parameters.
    Optional but highly recommended.


The naming of the auxiliary functions is very important as other code relies
on it. Since most of the helper functions are optional the code won't raise an
error if a function is not found because of a wrong name.

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
        coeff_sum_value=None, trans_intercept_value=None):

    return general_model_coeffs_from_iv_coeffs(
        iv_coeffs=iv_coeffs, iv_intercept_position=-1,
        has_trans_intercept=False, loading_norminfo=loading_norminfo,
        intercept_norminfo=intercept_norminfo, coeff_sum_value=coeff_sum_value,
        trans_intercept_value=trans_intercept_value)


def output_has_known_scale_linear():
    return False


def output_has_known_location_linear():
    return True

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


def output_has_known_scale_constant():
    return True


def output_has_known_location_constant():
    return True

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
        coeff_sum_value=None, trans_intercept_value=None):

    return general_model_coeffs_from_iv_coeffs(
        iv_coeffs=iv_coeffs, iv_intercept_position=-1,
        has_trans_intercept=False, loading_norminfo=loading_norminfo,
        intercept_norminfo=intercept_norminfo, coeff_sum_value=coeff_sum_value,
        trans_intercept_value=trans_intercept_value)


def output_has_known_scale_ar1():
    return False


def output_has_known_location_ar1():
    return True

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


def transform_coeffs_log_ces(coeffs, included_factors, direction, out):
    if direction == 'short_to_long':
        out[:-2] = coeffs[:-1]
        out[-2] = 1
        out /= np.sum(out[:-1])
        out[-1] = coeffs[-1]
    else:
        out[:-1] = coeffs[:-2] / coeffs[-2]
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


def output_has_known_scale_log_ces():
    return True


def output_has_known_location_log_ces():
    return True
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


def output_has_known_scale_translog():
    return False


def output_has_known_location_translog():
    return False

# =============================================================================
# translog without square terms
# =============================================================================


@jit
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


def output_has_known_scale_no_squares_translog():
    return False


def output_has_known_location_no_squares_translog():
    return False


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
    """Calculate model coeffs from iv coeffs for most transition functions.

    Many linear-in-parameters transition functions have very similar ways of
    calculating the model parameters from iv parameters in the wa estimator.
    The few differences can be influenced by the arguments:

    Args:
        iv_coeffs (DataFrame): see example above.
        iv_intercept_position (int): The iv intercept either must be the
            first (intercept_position=0) or the last (intercept_position=-1)
            IV parameter. It has to coincide with the (relative) position
            of the intercept in the coeff vector of the transition function
            if the transition function has a intercept.
        has_trans_intercept (bool): specifies if the transition function has
            a intercept. This is a transition function property. Do not
            confound it with the question whether the transition equation
            intercept is normalized or known in some periods.
        loading_norminfo (list): see example above
        intercept_norminfo (list): see example above
        coeff_sum_value (float):see example above
        trans_intercept_value (float): see example above

    Returns:
        **meas_coeffs**: see example above

        **gammas**: see example above

        **newly_identified_coeff_sum_value**: see example above

        **newly_identified_trans_intercept_value**: see example above

    """
    # assert statements
    to_check = [coeff_sum_value, loading_norminfo]
    assert None in to_check, ('Overidentified scale')
    assert to_check != [None, None], ('Underidentified scale')

    to_check = [trans_intercept_value, intercept_norminfo]
    assert None in to_check, ('Overidentified location')
    if has_trans_intercept:
        assert to_check != [None, None], ('Underidentified location')

    assert iv_intercept_position in [0, -1], ('')

    if has_trans_intercept is False:
        assert trans_intercept_value is None, ('')

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
        newly_identified_coeff_sum_value = coeff_sum_value
    else:
        newly_identified_coeff_sum_value = None

    # calculate all lambdas
    for y_variable in iv_coeffs.index:
        iv_sum = iv_coeffs.loc[y_variable].values[all_but_intercept].sum()
        meas_coeffs.loc[y_variable, 'loadings'] = iv_sum / coeff_sum_value

    # get trans intercept
    if has_trans_intercept is False:
        trans_intercept_value = 0
        newly_identified_trans_intercept_value = None
    elif trans_intercept_value is None:
        intercept_norm_y, intercept_norm_val = intercept_norminfo
        intercept_coeff = list(
            iv_coeffs.loc[intercept_norm_y])[iv_intercept_position]
        corresponding_loading = meas_coeffs.loc[intercept_norm_y, 'loadings']
        trans_intercept_value = \
            (intercept_coeff - intercept_norm_val) / corresponding_loading
        newly_identified_trans_intercept_value = trans_intercept_value
    else:
        newly_identified_trans_intercept_value = None

    # calculate all mus
    for y_variable in iv_coeffs.index:
        intercept_coeff = list(
            iv_coeffs.loc[y_variable])[iv_intercept_position]
        corresponding_loading = meas_coeffs.loc[y_variable, 'loadings']
        meas_coeffs.loc[y_variable, 'intercepts'] = \
            intercept_coeff - corresponding_loading * trans_intercept_value

    gamma_coeffs = iv_coeffs[iv_coeffs.columns[all_but_intercept]]
    next_period_loadings = meas_coeffs['loadings']
    gammas = gamma_coeffs.divide(next_period_loadings, axis=0).mean().values

    if has_trans_intercept is True:
        if iv_intercept_position == 0:
            gammas = np.hstack([trans_intercept_value, gammas])
        else:
            gammas = np.hstack([gammas, trans_intercept_value])

    return meas_coeffs, gammas, newly_identified_coeff_sum_value, \
        newly_identified_trans_intercept_value
