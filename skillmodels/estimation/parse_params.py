"""Parse the params into quantities for the likelihood function.

params is a one dimensional numpy array with the free parameters of the model.
In each evaluation of the likelihood it has to be parsed into the quantities
described in :ref:`params_and_quants`.

Speed is relevant throughout this module. Therefore everything that does not
depend on the parameter vector but only on the model specifications has
already been done in SkillModel.

"""
import skillmodels.model_functions.transition_functions as tf
import numpy as np
from numpy.core.umath_tests import matrix_multiply
from numpy.linalg import cholesky


def _map_params_to_deltas(params, initial, params_slice, boo,
                          replacements=None):
    """Map paremeters from params to delta for each period."""
    for t, delta in enumerate(initial):
        delta[boo[t]] = params[params_slice[t]]
    if replacements is not None:
        for (t_put, pos_put), (t_take, pos_take) in replacements:
            initial[t_put][pos_put, :] = initial[t_take][pos_take, :]


def _map_params_to_psi(params, initial, params_slice, boo):
    """Map parameters from params to psi."""
    initial[boo] = params[params_slice]


def _map_params_to_H(params, initial, params_slice, boo, psi_bool_for_H=None,
                     psi=None, arr1=None, endog_position=None,
                     initial_copy=None, replacements=None):
    """Map parameters from params to H."""
    # the initial H has to be restored if endog_correction is True
    if psi is not None:
        initial[:] = initial_copy

    # fill H with the new parameters
    initial[boo] = params[params_slice]

    # make replacements
    if replacements is not None:
        for put_position, take_position in replacements:
            initial[put_position, :] = initial[take_position, :]

    # make some transformations with psi in the case of endog correction
    if psi is not None:
        arr1[:] = initial[psi_bool_for_H, endog_position].reshape(arr1.shape)
        initial[psi_bool_for_H, endog_position] = 0
        initial[psi_bool_for_H] += np.multiply(arr1, psi)


def _map_params_to_R(params, initial, params_slice, boo, square_root_filters,
                     replacements=None):
    """Map parameters from params to R."""
    if square_root_filters is False:
        initial[boo] = params[params_slice]
    else:
        initial[boo] = np.sqrt(params[params_slice])

    if replacements is not None:
        for put_position, take_position in replacements:
            initial[put_position] = initial[take_position]


def _map_params_to_Q(params, initial, params_slice, boo, replacements=None):
    """Map parameters from params to Q."""
    initial[boo] = params[params_slice]
    if replacements is not None:
        for put_position, take_from_position in replacements:
            initial[put_position] = initial[take_from_position]


def _map_params_to_X_zero(
        params, initial, filler, params_slice, replacements=None):
    """Map parameters from params to X_zero."""
    filler[:] = params[params_slice].reshape(filler.shape)
    if replacements is not None:
        for add_to_position, take_from_position in replacements:
            filler[add_to_position] += filler[take_from_position]

    initial[:] = filler


def _map_params_to_W_zero(params, initial, params_slice):
    """Map parameters from params to W_zero."""
    initial[:] = params[params_slice]


def _map_params_to_P_zero(params, params_type, initial, params_slice, filler,
                          boo, cholesky_of_P_zero,
                          square_root_filters):
    """Map parameters from params to P_zero."""
    # write params in filler
    filler[:] = 0
    filler[boo] = params[params_slice]

    # transform the filler
    if params_type == 'short' or cholesky_of_P_zero is True:
        if square_root_filters is False:
            # make chol_t to not chol
            filler = matrix_multiply(
                np.transpose(filler, axes=(0, 2, 1)), filler)
    else:
        # make not_chol to not_chol (as covariance matrices are symmetric,
        # only half of its off-diagonal elements have to be estimated. here the
        # lower triangle is filled with he transpose of the upper triangle.)
        for i in range(len(filler)):
            filler[i] += (filler[i] - np.diag(np.diagonal(filler[i]))).T

        if square_root_filters is True:
            # make not_chol to chol_t
            filler = np.transpose(cholesky(filler), axes=(0, 2, 1))

    if square_root_filters is False:
        initial[:] = filler
    else:
        initial[:, :, 1:, 1:] = filler


def _map_params_to_trans_coeffs(params, initial, params_slice,
                                transform_funcs=None, included_factors=None,
                                direction='short_to_long'):
    """Map parameters from params to trans_coeffs."""
    for f, coeffs in enumerate(initial):
        func = transform_funcs[f] if transform_funcs is not None else None
        if func is None:
            for s in range(len(coeffs)):
                coeffs[s] = params[params_slice[f][s]]
        else:
            for s in range(len(coeffs)):
                getattr(tf, func)(
                    params[params_slice[f][s]], included_factors[f],
                    direction, out=coeffs[s])


def parse_params(params, deltas_args, H_args, R_args, Q_args, P_zero_args,
                 trans_coeffs_args, X_zero_args=None, W_zero_args=None,
                 psi_args=None):
    """Parse params into the quantities that depend on it.

    All quantities are updated in place. The order is important in some cases.
    For example, H has to be updated after psi (if psi is used at all).

    The arguments of this function are generated in the CHSModel class.

    """
    _map_params_to_deltas(params, **deltas_args)
    if psi_args is not None:
        _map_params_to_psi(params, **psi_args)
    _map_params_to_H(params, **H_args)
    _map_params_to_R(params, **R_args)
    _map_params_to_Q(params, **Q_args)
    if X_zero_args is not None:
        _map_params_to_X_zero(params, **X_zero_args)
    if W_zero_args is not None:
        _map_params_to_W_zero(params, **W_zero_args)
    _map_params_to_P_zero(params, **P_zero_args)
    _map_params_to_trans_coeffs(params, **trans_coeffs_args)


def restore_unestimated_quantities(X_zero=None, X_zero_value=None,
                                   W_zero=None, W_zero_value=None):
    """Restore X_zero and W_zero for the next evaluation of the likelihood."""
    if X_zero is not None:
        X_zero[:] = X_zero_value
    if W_zero is not None:
        W_zero[:] = W_zero_value


def transform_params_for_X_zero(params_for_X_zero, filler, direction,
                                replacements=None):
    if replacements is None:
        return params_for_X_zero
    elif direction == 'short_to_long':
        filler[:] = params_for_X_zero.reshape(filler.shape)
        for add_to_pos, take_from_pos in replacements:
            filler[add_to_pos] += filler[take_from_pos]
        return filler.flatten()
    else:
        filler[:] = params_for_X_zero.reshape(filler.shape)
        for subtract_from_pos, take_from_pos in reversed(replacements):
            filler[subtract_from_pos] -= filler[take_from_pos]
        return filler.flatten()


def transform_params_for_P_zero(params_for_P_zero, filler, boo,
                                estimate_cholesky_of_P_zero, direction):

    filler[:] = 0
    if estimate_cholesky_of_P_zero is True:
        return params_for_P_zero
    elif direction == 'short_to_long':
        filler[boo] = params_for_P_zero
        filler = matrix_multiply(
            np.transpose(filler, axes=(0, 2, 1)), filler)
        return filler[boo]
    else:
        filler[boo] = params_for_P_zero
        for i in range(len(filler)):
            filler[i] += (filler[i] - np.diag(np.diagonal(filler[i]))).T
        filler = np.transpose(cholesky(filler), axes=(0, 2, 1))
        return filler[boo]


def transform_params_for_trans_coeffs(
        params, initial, params_slice, transform_funcs,
        included_factors, direction):

    _map_params_to_trans_coeffs(
        params, initial, params_slice, transform_funcs, included_factors,
        direction)

    transformed = []
    for f, sl in enumerate(params_slice):
        for s in range(len(sl)):
            if s == 0 or sl[s] != sl[s - 1]:
                transformed += list(initial[f][s, :])
    return np.array(transformed)





