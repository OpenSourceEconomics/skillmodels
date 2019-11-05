"""Parse the params into quantities for the likelihood function."""
import warnings

import numpy as np
import pandas as pd
from estimagic.optimization.utilities import cov_params_to_matrix


def parse_params(params, initial_quantities, factors, parsing_info):
    """Parse params into the quantities that depend on it."""
    if isinstance(params, pd.DataFrame):
        params = params["value"]

    params_vec = params.to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        _map_params_to_delta(params_vec, initial_quantities["delta"], parsing_info)
        _map_params_to_loading(params_vec, initial_quantities["loading"], parsing_info)
        _map_params_to_meas_sd(params_vec, initial_quantities["meas_sd"], parsing_info)
        _map_params_to_shock_variance(
            params_vec, initial_quantities["shock_variance"], parsing_info
        )
        _map_params_to_initial_mean(
            params_vec, initial_quantities["initial_mean"], parsing_info
        )
        _map_params_to_mixture_weight(
            params_vec, initial_quantities["mixture_weight"], parsing_info
        )
        _map_params_to_initial_cov(
            params_vec, initial_quantities["initial_cov"], parsing_info
        )
        _map_params_to_trans_coeffs(
            params_vec, initial_quantities["trans_coeffs"], factors, parsing_info
        )


def _map_params_to_delta(params_vec, initial, parsing_info):
    info = parsing_info["delta"]
    for period, delta in enumerate(initial):
        delta[:] = params_vec[info[period]].reshape(delta.shape)
    return initial


def _map_params_to_loading(params_vec, initial, parsing_info):
    initial[:] = params_vec[parsing_info["loading"]].reshape(initial.shape)


def _map_params_to_meas_sd(params_vec, initial, parsing_info):
    initial[:] = np.sqrt(params_vec[parsing_info["meas_sd"]])


def _map_params_to_shock_variance(params_vec, initial, parsing_info):
    info = parsing_info["shock_variance"]
    for period in range(len(initial)):
        initial[period] = np.diag(params_vec[info[period]])


def _map_params_to_initial_mean(params_vec, initial, parsing_info):
    nobs, nmixtures, nfac = initial.shape
    initial[:] = params_vec[parsing_info["initial_mean"]].reshape(nmixtures, nfac)


def _map_params_to_initial_cov(params_vec, initial, parsing_info):
    info = parsing_info["initial_cov"]
    nobs, nmixtures, nfac, _ = initial.shape
    nfac = nfac - 1

    filler = np.zeros((nmixtures, nfac, nfac))
    for emf in range(nmixtures):
        filler[emf] = cov_params_to_matrix(params_vec[info[emf]])

    filler = np.transpose(np.linalg.cholesky(filler), axes=(0, 2, 1))
    initial[:, :, 1:, 1:] = filler


def _map_params_to_mixture_weight(params_vec, initial, parsing_info):
    initial[:] = params_vec[parsing_info["mixture_weight"]]


def _map_params_to_trans_coeffs(params_vec, initial, factors, parsing_info):
    """Map parameters from params to trans_coeffs."""
    relevant_periods = range(len(initial[0]))
    info = parsing_info["trans_coeffs"]

    for f, init in zip(range(len(factors)), initial):
        for period in relevant_periods:
            sl = info[period][f]
            if sl.start != sl.stop:
                init[period] = params_vec[sl]
