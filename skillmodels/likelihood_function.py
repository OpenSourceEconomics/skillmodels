import functools

import jax.numpy as jnp
from jax.ops import index
from jax.ops import index_update

from skillmodels.kalman_filters import calculate_sigma_scaling_factor_and_weights
from skillmodels.kalman_filters import kalman_predict
from skillmodels.kalman_filters import kalman_update
from skillmodels.params_index import params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.parse_params import parse_params
from skillmodels.process_data import process_data_for_estimation
from skillmodels.process_model import process_model


def get_optimization_functions(
    model_dict, data, aggregation="sum", additional_data=False
):
    """Create a likelihood function and its first derivative.

    The resulting functions take an estimagic-style params DataFrame as only argument.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        dataset (DataFrame): datset in long format. see :ref:`basic_usage`.
        aggregation (str): One of "sum" and "mean" and None. Default: "sum".
        additional_data (bool): If true, the likelihood function returns a tuple where
            the first element is the aggregated log likelihood and the second a
            DataFrame with all likelihood contributions (i.e. disaggregated by
            individual and measurement).
        gradient_type (str): (One of "grad", "value_and_grad").

    Returns:
        dict: Dictionary of functions for likelihood optimization. Has the entries:
            "log_likelihood", "gradient", "value_and_gradient", "hessian"

    """
    pass


def get_log_likelihood_contributions_func(model_dict, data):
    model = process_model(model_dict)
    p_index = params_index(model["update_info"], model["labels"], model["dimensions"])
    parsing_info = create_parsing_info(
        p_index, model["update_info"], model["labels"], model["dimensions"]
    )
    measurements, controls, anchoring_variables = process_data_for_estimation(
        data, model["labels"], model["update_info"], model["anchoring"]
    )

    sigma_scaling_factor, sigma_weights = calculate_sigma_scaling_factor_and_weights(
        model["dimensions"]["n_states"], model["options"]["sigma_points_scale"]
    )

    partialed = functools.partial(
        _log_likelihood_contributions_jax,
        parsing_info=parsing_info,
        update_info=model["update_info"],
        measurements=measurements,
        controls=controls,
        transition_functions=model["transition_functions"],
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        anchoring_variables=anchoring_variables,
        dimensions=model["dimensions"],
        labels=model["labels"],
    )

    return partialed


def get_likelihood_jacobian(model_dict, data):
    raise NotImplementedError


def get_likelihood_hessian(model_dict, data):
    raise NotImplementedError


def _log_likelihood_contributions_jax(
    params,
    parsing_info,
    update_info,
    measurements,
    controls,
    transition_functions,
    sigma_scaling_factor,
    sigma_weights,
    anchoring_variables,
    dimensions,
    labels,
):
    """Log likelihood contributions per individual and update.

    This function is jax-differentiable and jax-jittable as long as all but the first
    argument are marked as static.

    Args:
        params (jax.numpy.array): 1d array with model parameters.
        parsing_info (dict): Contains information how to parse parameter vector.
        update_info (pandas.DataFrame): Contains information about number of updates in
            each period and purpose of each update.
        measurements (jax.numpy.array): Array of shape (n_updates, n_obs) with data on
            observed measurements. NaN if the measurement was not observed.
        controls (jax.numpy.array): Array of shape (n_periods, n_obs, n_controls)
            with observed control variables for the measurement equations.
        transition_functions (tuple): tuple of tuples where the first element is the
            name of the transition function and the second the actual transition
            function. Order is important and corresponds to the latent
            factors in alphabetical order.
        sigma_scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights (jax.numpy.array): 1d array of length n_sigma with non-negative
            sigma weights.
        anchoring_variables (jax.numpy.array): Array of shape (n_periods, n_obs, n_fac)
            with anchoring outcomes. Can be 0 for unanchored factors or if no centering
            is desired.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        jnp.array: Array of shape (n_obs, n_updates) log likelihood contributions per
            individual and update.

    """
    n_obs = measurements.shape[1]
    states, upper_chols, log_mixture_weights, pardict = parse_params(
        params, parsing_info, dimensions, labels, n_obs
    )
    n_updates = len(update_info)
    loglikes = jnp.zeros((n_updates, n_obs))

    k = 0
    for t in labels["periods"]:
        nmeas = len(update_info.loc[t])
        for _j in range(nmeas):
            purpose = update_info.iloc[k]["purpose"]
            new_states, new_upper_chols, new_weights, loglikes_k = kalman_update(
                states,
                upper_chols,
                pardict["loadings"][k],
                pardict["controls"][k],
                pardict["meas_sds"][k],
                measurements[k],
                controls[t],
                log_mixture_weights,
            )
            loglikes = index_update(loglikes, index[k], loglikes_k)
            log_mixture_weights = new_weights
            if purpose == "measurement":
                states, upper_chols = new_states, new_upper_chols

            k += 1

        if t != labels["periods"][-1]:
            states, upper_chols = kalman_predict(
                states,
                upper_chols,
                sigma_scaling_factor,
                sigma_weights,
                transition_functions,
                pardict["transition"][t],
                pardict["shock_sds"][t],
                pardict["anchoring_scaling_factors"][t : t + 2],
                anchoring_variables[t : t + 2],
            )

    return loglikes
