"""Parse the params into quantities for the likelihood function."""
import warnings

import numpy as np
import pandas as pd
from estimagic.optimization.utilities import cov_params_to_matrix


def parse_params(params, initial_quantities, factors):
    """Parse params into the quantities that depend on it."""
    if isinstance(params, pd.DataFrame):
        params = params["value"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        _map_params_to_delta(params, initial_quantities["delta"])
        _map_params_to_loading(params, initial_quantities["loading"])
        _map_params_to_meas_sd(params, initial_quantities["meas_sd"])
        _map_params_to_shock_variance(params, initial_quantities["shock_variance"])
        _map_params_to_initial_mean(params, initial_quantities["initial_mean"])
        _map_params_to_mixture_weight(params, initial_quantities["mixture_weight"])
        _map_params_to_initial_cov(params, initial_quantities["initial_cov"])
        _map_params_to_trans_coeffs(params, initial_quantities["trans_coeffs"], factors)


def _map_params_to_delta(params, initial):
    for period, delta in enumerate(initial):
        delta[:] = params.loc["delta", period].to_numpy().reshape(delta.shape)
    return initial


def _map_params_to_loading(params, initial):
    initial[:] = params.loc["loading"].to_numpy().reshape(initial.shape)


def _map_params_to_meas_sd(params, initial):
    initial[:] = np.sqrt(params.loc["meas_sd"].to_numpy())


def _map_params_to_shock_variance(params, initial):
    for period in range(len(initial)):
        initial[period] = np.diag(params.loc["shock_variance", period].to_numpy())


def _map_params_to_initial_mean(params, initial):
    nobs, nmixtures, nfac = initial.shape
    initial[:] = params.loc["initial_mean"].to_numpy().reshape(nmixtures, nfac)


def _map_params_to_initial_cov(params, initial):
    nobs, nmixtures, nfac, _ = initial.shape
    nfac = nfac - 1

    filler = np.zeros((nmixtures, nfac, nfac))
    for emf in range(nmixtures):
        filler[emf] = cov_params_to_matrix(
            params.loc["initial_cov", 0, f"mixture_{emf}"].to_numpy()
        )

    filler = np.transpose(np.linalg.cholesky(filler), axes=(0, 2, 1))
    initial[:, :, 1:, 1:] = filler


def _map_params_to_mixture_weight(params, initial):
    initial[:] = params.loc["mixture_weight"].to_numpy()


def _map_params_to_trans_coeffs(params, initial, factors):
    """Map parameters from params to trans_coeffs."""
    relevant_periods = range(len(initial[0]))

    for factor, init in zip(factors, initial):
        for period in relevant_periods:
            if factor in params.loc["trans", period]:
                init[period] = params.loc["trans", period, factor].to_numpy()
