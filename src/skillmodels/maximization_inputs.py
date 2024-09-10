import functools

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import skillmodels.likelihood_function as lf
import skillmodels.likelihood_function_debug as lfd
from skillmodels.constraints import add_bounds, get_constraints
from skillmodels.kalman_filters import calculate_sigma_scaling_factor_and_weights
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.process_data import process_data
from skillmodels.process_debug_data import process_debug_data
from skillmodels.process_model import process_model

jax.config.update("jax_enable_x64", False)  # noqa: FBT003


def get_maximization_inputs(model_dict, data):
    """Create inputs for optimagic's maximize function.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        data (DataFrame): dataset in long format.

    Returns a dictionary with keys:
        loglike (function): A jax jitted function that takes an optimagic-style
            params dataframe as only input and returns a dict with entries:
            - "value": The scalar log likelihood
            - "contributions": An array with the log likelihood per observation
        debug_loglike (function): Similar to loglike, with the following differences:
            - It is not jitted and thus faster on the first call and debuggable
            - It will add intermediate results as additional entries in the returned
              dictionary. Those can be used for debugging and plotting.
        gradient (function): The gradient of the scalar log likelihood
            function with respect to the parameters.
        loglike_and_gradient (function): Combination of loglike and
            loglike_gradient that is faster than calling the two functions separately.
        constraints (list): List of optimagic constraints that are implied by the
            model specification.
        params_template (pd.DataFrame): Parameter DataFrame with correct index and
            bounds but with empty value column.

    """
    model = process_model(model_dict)
    p_index = get_params_index(
        model["update_info"],
        model["labels"],
        model["dimensions"],
        model["transition_info"],
    )

    parsing_info = create_parsing_info(
        p_index,
        model["update_info"],
        model["labels"],
        model["anchoring"],
    )
    measurements, controls, observed_factors = process_data(
        data,
        model["labels"],
        model["update_info"],
        model["anchoring"],
    )

    sigma_scaling_factor, sigma_weights = calculate_sigma_scaling_factor_and_weights(
        model["dimensions"]["n_latent_factors"],
        model["estimation_options"]["sigma_points_scale"],
    )

    partialed_get_jnp_params_vec = functools.partial(
        _get_jnp_params_vec,
        target_index=p_index,
    )

    partialed_loglikes = {}
    for n, fun in {
        "ll": lf.log_likelihood,
        "llo": lf.log_likelihood_obs,
        "debug_ll": lfd.log_likelihood,
    }.items():
        partialed_loglikes[n] = _partial_some_log_likelihood(
            fun=fun,
            parsing_info=parsing_info,
            measurements=measurements,
            controls=controls,
            observed_factors=observed_factors,
            model=model,
            sigma_weights=sigma_weights,
            sigma_scaling_factor=sigma_scaling_factor,
        )

    _jitted_loglike = jax.jit(partialed_loglikes["ll"])
    _jitted_loglikeobs = jax.jit(partialed_loglikes["llo"])
    _gradient = jax.jit(jax.grad(partialed_loglikes["ll"]))

    def loglike(params):
        params_vec = partialed_get_jnp_params_vec(params)
        return float(_jitted_loglike(params_vec))

    def loglikeobs(params):
        params_vec = partialed_get_jnp_params_vec(params)
        return _to_numpy(_jitted_loglikeobs(params_vec))

    def loglike_and_gradient(params):
        params_vec = partialed_get_jnp_params_vec(params)
        crit = float(_jitted_loglike(params_vec))
        grad = _to_numpy(_gradient(params_vec))
        return crit, grad

    def debug_loglike(params):
        params_vec = partialed_get_jnp_params_vec(params)
        jax_output = partialed_loglikes["debug_ll"](params_vec)
        tmp = _to_numpy(jax_output)
        tmp["value"] = float(tmp["value"])
        return process_debug_data(debug_data=tmp, model=model)

    constr = get_constraints(
        dimensions=model["dimensions"],
        labels=model["labels"],
        anchoring_info=model["anchoring"],
        update_info=model["update_info"],
        normalizations=model["normalizations"],
    )

    params_template = pd.DataFrame(columns=["value"], index=p_index)
    params_template = add_bounds(
        params_template,
        model["estimation_options"]["bounds_distance"],
    )

    out = {
        "loglike": loglike,
        "loglikeobs": loglikeobs,
        "debug_loglike": debug_loglike,
        "loglike_and_gradient": loglike_and_gradient,
        "constraints": constr,
        "params_template": params_template,
    }

    return out


def _partial_some_log_likelihood(
    fun,
    parsing_info,
    measurements,
    controls,
    observed_factors,
    model,
    sigma_weights,
    sigma_scaling_factor,
):
    update_info = model["update_info"]
    is_measurement_iteration = (update_info["purpose"] == "measurement").to_numpy()
    _periods = pd.Series(update_info.index.get_level_values("period").to_numpy())
    is_predict_iteration = ((_periods - _periods.shift(-1)) == -1).to_numpy()
    last_period = model["labels"]["periods"][-1]
    # iteration_to_period is used as an indexer to loop over arrays of different lengths
    # in a jax.lax.scan. It needs to work for arrays of length n_periods and not raise
    # IndexErrors on tracer arrays of length n_periods - 1 (i.e. n_transitions).
    # To achieve that, we replace the last period by -1.
    iteration_to_period = _periods.replace(last_period, -1).to_numpy()

    return functools.partial(
        fun,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        transition_func=model["transition_info"]["func"],
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        dimensions=model["dimensions"],
        labels=model["labels"],
        estimation_options=model["estimation_options"],
        is_measurement_iteration=is_measurement_iteration,
        is_predict_iteration=is_predict_iteration,
        iteration_to_period=iteration_to_period,
        observed_factors=observed_factors,
    )


def _to_numpy(obj):
    if isinstance(obj, dict):
        res = {}
        for key, value in obj.items():
            if np.isscalar(value):
                res[key] = value
            else:
                res[key] = np.array(value)

    elif np.isscalar(obj):
        res = obj
    else:
        res = np.array(obj)

    return res


def _get_jnp_params_vec(params, target_index):
    if set(params.index) != set(target_index):
        additional_entries = params.index.difference(target_index).tolist()
        missing_entries = target_index.difference(params.index).tolist()
        msg = "Invalid params DataFrame. "
        if additional_entries:
            msg += f"Your params have additional entries: {additional_entries}. "
        if missing_entries:
            msg += f"Your params have missing entries: {missing_entries}. "
        raise ValueError(msg)

    vec = jnp.array(params.reindex(target_index)["value"].to_numpy())
    return vec
