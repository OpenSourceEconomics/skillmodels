"""Parse the params into quantities for the likelihood function."""
import numpy as np
from estimagic.optimization.utilities import cov_params_to_matrix
import pandas as pd


def parse_params(params, initial_quantities, factors, square_root_filters):
    """Parse params into the quantities that depend on it."""
    if isinstance(params, pd.DataFrame):
        params = params['value']
    _map_params_to_delta(params, initial_quantities['delta'])
    _map_params_to_h(params, initial_quantities['h'])
    _map_params_to_r(params, initial_quantities['r'], square_root_filters)
    _map_params_to_q(params, initial_quantities['q'])
    if 'x' in initial_quantities:
        _map_params_to_x(params, initial_quantities['x'])
    if 'w' in initial_quantities:
        _map_params_to_w(params, initial_quantities['w'])
    _map_params_to_p(params, initial_quantities['p'], square_root_filters)
    _map_params_to_trans_coeffs(params, initial_quantities['trans_coeffs'], factors)


def _map_params_to_delta(params, initial):
    for period, delta in enumerate(initial):
        delta[:] = params.loc['delta', period].to_numpy().reshape(delta.shape)
    return initial


def _map_params_to_h(params, initial):
    initial[:] = params.loc['h'].to_numpy().reshape(initial.shape)


def _map_params_to_r(params, initial, square_root_filters):
    if square_root_filters is True:
        initial[:] = np.sqrt(params.loc['r'].to_numpy())
    else:
        initial[:] = params.loc['r'].to_numpy()


def _map_params_to_q(params, initial):
    for period in range(len(initial)):
        initial[period] = np.diag(params.loc['q', period].to_numpy())


def _map_params_to_x(params, initial):
    nobs, nemf, nfac = initial.shape
    initial[:] = params.loc['x'].to_numpy().reshape(nemf, nfac)


def _map_params_to_p(params, initial, square_root_filters):
    nobs, nemf, nfac, _ = initial.shape
    nfac = nfac - 1 if square_root_filters is True else nfac

    filler = np.zeros((nemf, nfac, nfac))
    for emf in range(nemf):
        filler[emf] = cov_params_to_matrix(params.loc['p', 0, emf].to_numpy())

    if square_root_filters is True:
        filler = np.transpose(np.linalg.cholesky(filler), axes=(0, 2, 1))
        initial[:, :, 1:, 1:] = filler
    else:
        initial[:] = filler


def _map_params_to_w(params, initial):
    initial[:] = params.loc['w'].to_numpy()


def _map_params_to_trans_coeffs(params, initial, factors):
    """Map parameters from params to trans_coeffs."""
    relevant_periods = range(len(initial[0]))

    for factor, init in zip(factors, initial):
        for period in relevant_periods:
            if factor in params.loc['trans', period]:
                init[period] = params.loc['trans', period, factor].to_numpy()


def restore_unestimated_quantities(x=None, x_value=None, w=None, w_value=None):
    """Restore x and w for the next evaluation of the likelihood."""
    if x is not None:
        x[:] = x_value
    if w is not None:
        w[:] = w_value
