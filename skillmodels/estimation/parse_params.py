"""Parse the params into quantities for the likelihood function."""
import warnings

import numpy as np
import pandas as pd
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import robust_cholesky


def parse_params(params, initial_quantities, factors, parsing_info):
    """Parse params into the quantities that depend on it."""
    if isinstance(params, pd.DataFrame):
        params = params["value"]

    params_vec = params.to_numpy()

    iq = initial_quantities

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        _map_params_to_control_coeffs(params_vec, iq["control_coeffs"], parsing_info)
        _map_params_to_loading(params_vec, iq["loading"], parsing_info)
        _map_params_to_meas_sd(params_vec, iq["meas_sd"], parsing_info)
        _map_params_to_shock_sd(params_vec, iq["shock_sd"], parsing_info)
        _map_params_to_initial_mean(params_vec, iq["initial_mean"], parsing_info)
        _map_params_to_mixture_weight(params_vec, iq["mixture_weight"], parsing_info)
        _map_params_to_initial_cov(params_vec, iq["initial_cov"], parsing_info)
        _map_params_to_trans_coeffs(
            params_vec, iq["trans_coeffs"], factors, parsing_info
        )

        if "anchoring_loading" in iq:
            _update_anchoring_loadings(
                iq["loading"], iq["anchoring_loading"], parsing_info
            )


def _map_params_to_control_coeffs(params_vec, initial, parsing_info):
    info = parsing_info["control_coeffs"]
    for period, coeff in enumerate(initial):
        coeff[:] = params_vec[info[period]].reshape(coeff.shape)
    return initial


def _map_params_to_loading(params_vec, initial, parsing_info):
    initial[:] = params_vec[parsing_info["loading"]].reshape(initial.shape)


def _map_params_to_meas_sd(params_vec, initial, parsing_info):
    initial[:] = np.sqrt(params_vec[parsing_info["meas_sd"]])


def _map_params_to_shock_sd(params_vec, initial, parsing_info):
    info = parsing_info["shock_sd"]
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
        cov = cov_params_to_matrix(params_vec[info[emf]])
        chol = robust_cholesky(cov)
        filler[emf] = chol.T

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


def _update_anchoring_loadings(loading, anchoring_loadings, parsing_info):
    mask = parsing_info["anchoring_mask"]
    anchoring_loadings[:] = loading[mask].reshape(anchoring_loadings.shape)
