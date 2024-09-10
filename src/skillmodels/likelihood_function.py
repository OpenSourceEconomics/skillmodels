import functools

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import skillmodels.likelihood_function_debug as lfd
from skillmodels.clipping import soft_clipping
from skillmodels.constraints import add_bounds, get_constraints
from skillmodels.kalman_filters import (
    calculate_sigma_scaling_factor_and_weights,
    kalman_predict,
    kalman_update,
)
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info, parse_params
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
        "ll": _log_likelihood_jax,
        "llo": _log_likelihood_obs_jax,
        "debug_ll": lfd._log_likelihood_jax,
    }.items():
        partialed_loglikes[n] = _partial_some_log_likelihood_jax(
            fun=fun,
            parsing_info=parsing_info,
            measurements=measurements,
            controls=controls,
            observed_factors=observed_factors,
            model=model,
            sigma_weights=sigma_weights,
            sigma_scaling_factor=sigma_scaling_factor,
        )

    def loglike(params):
        params_vec = partialed_get_jnp_params_vec(params)
        return float(partialed_loglikes["ll"](params_vec))

    def loglikeobs(params):
        params_vec = partialed_get_jnp_params_vec(params)
        return _to_numpy(partialed_loglikes["llo"](params_vec))

    def loglike_and_gradient(params):
        params_vec = partialed_get_jnp_params_vec(params)
        crit = float(partialed_loglikes["ll"](params_vec))
        grad = _to_numpy(jax.grad(partialed_loglikes["ll"])(params_vec))
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


def _partial_some_log_likelihood_jax(
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


def _log_likelihood_obs_jax(
    params,
    parsing_info,
    measurements,
    controls,
    transition_func,
    sigma_scaling_factor,
    sigma_weights,
    dimensions,
    labels,
    estimation_options,
    is_measurement_iteration,
    is_predict_iteration,
    iteration_to_period,
    observed_factors,
):
    """Log likelihood of a skill formation model.

    This function is jax-differentiable and jax-jittable as long as all but the first
    argument are marked as static.

    The function returns both a tuple (float, dict). The first entry is the aggregated
    log likelihood value. The second additional information like the log likelihood
    contribution of each individual. Note that the dict also contains the aggregated
    value. Returning that value separately is only needed to calculate a gradient
    with Jax.

    Args:
        params (jax.numpy.array): 1d array with model parameters.
        parsing_info (dict): Contains information how to parse parameter vector.
        update_info (pandas.DataFrame): Contains information about number of updates in
            each period and purpose of each update.
        measurements (jax.numpy.array): Array of shape (n_updates, n_obs) with data on
            observed measurements. NaN if the measurement was not observed.
        controls (jax.numpy.array): Array of shape (n_periods, n_obs, n_controls)
            with observed control variables for the measurement equations.
        transition_func (Callable): The transition function.
        sigma_scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights (jax.numpy.array): 1d array of length n_sigma with non-negative
            sigma weights.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        observed_factors (jax.numpy.array): Array of shape (n_periods, n_obs,
            n_observed_factors) with data on the observed factors.

    Returns:
        jnp.array: 1d array of length N, the aggregated log likelihood.

    """
    n_obs = measurements.shape[1]
    states, upper_chols, log_mixture_weights, pardict = parse_params(
        params,
        parsing_info,
        dimensions,
        labels,
        n_obs,
    )

    carry = {
        "states": states,
        "upper_chols": upper_chols,
        "log_mixture_weights": log_mixture_weights,
    }

    loop_args = {
        "period": iteration_to_period,
        "loadings": pardict["loadings"],
        "control_params": pardict["controls"],
        "meas_sds": pardict["meas_sds"],
        "measurements": measurements,
        "is_measurement_iteration": is_measurement_iteration,
        "is_predict_iteration": is_predict_iteration,
    }

    _body = functools.partial(
        _scan_body,
        controls=controls,
        pardict=pardict,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        transition_func=transition_func,
        observed_factors=observed_factors,
    )

    static_out = jax.lax.scan(_body, carry, loop_args)[1]

    # clip contributions before aggregation to preserve as much information as
    # possible.
    return soft_clipping(
        arr=static_out["loglikes"],
        lower=estimation_options["clipping_lower_bound"],
        upper=estimation_options["clipping_upper_bound"],
        lower_hardness=estimation_options["clipping_lower_hardness"],
        upper_hardness=estimation_options["clipping_upper_hardness"],
    ).sum(axis=0)


def _log_likelihood_jax(
    params,
    parsing_info,
    measurements,
    controls,
    transition_func,
    sigma_scaling_factor,
    sigma_weights,
    dimensions,
    labels,
    estimation_options,
    is_measurement_iteration,
    is_predict_iteration,
    iteration_to_period,
    observed_factors,
):
    return _log_likelihood_obs_jax(
        params=params,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        transition_func=transition_func,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        dimensions=dimensions,
        labels=labels,
        estimation_options=estimation_options,
        is_measurement_iteration=is_measurement_iteration,
        is_predict_iteration=is_predict_iteration,
        iteration_to_period=iteration_to_period,
        observed_factors=observed_factors,
    ).sum()


def _scan_body(
    carry,
    loop_args,
    controls,
    pardict,
    sigma_scaling_factor,
    sigma_weights,
    transition_func,
    observed_factors,
):
    # ==================================================================================
    # create arguments needed for update
    # ==================================================================================
    t = loop_args["period"]
    states = carry["states"]
    upper_chols = carry["upper_chols"]
    log_mixture_weights = carry["log_mixture_weights"]

    update_kwargs = {
        "states": states,
        "upper_chols": upper_chols,
        "loadings": loop_args["loadings"],
        "control_params": loop_args["control_params"],
        "meas_sd": loop_args["meas_sds"],
        "measurements": loop_args["measurements"],
        "controls": controls[t],
        "log_mixture_weights": log_mixture_weights,
    }

    # ==================================================================================
    # do a measurement or anchoring update
    # ==================================================================================
    states, upper_chols, log_mixture_weights, loglikes = jax.lax.cond(
        loop_args["is_measurement_iteration"],
        functools.partial(_one_arg_measurement_update),
        functools.partial(_one_arg_anchoring_update),
        update_kwargs,
    )

    # ==================================================================================
    # create arguments needed for predict step
    # ==================================================================================
    predict_kwargs = {
        "states": states,
        "upper_chols": upper_chols,
        "sigma_scaling_factor": sigma_scaling_factor,
        "sigma_weights": sigma_weights,
        "trans_coeffs": {k: arr[t] for k, arr in pardict["transition"].items()},
        "shock_sds": pardict["shock_sds"][t],
        "anchoring_scaling_factors": pardict["anchoring_scaling_factors"][
            jnp.array([t, t + 1])
        ],
        "anchoring_constants": pardict["anchoring_constants"][jnp.array([t, t + 1])],
        "observed_factors": observed_factors[t],
    }

    fixed_kwargs = {"transition_func": transition_func}

    # ==================================================================================
    # Do a predict step or a do-nothing fake predict step
    # ==================================================================================
    states, upper_chols, filtered_states = jax.lax.cond(
        loop_args["is_predict_iteration"],
        functools.partial(_one_arg_predict, **fixed_kwargs),
        functools.partial(_one_arg_no_predict, **fixed_kwargs),
        predict_kwargs,
    )

    new_state = {
        "states": states,
        "upper_chols": upper_chols,
        "log_mixture_weights": log_mixture_weights,
    }

    static_out = {"loglikes": loglikes, "states": filtered_states}
    return new_state, static_out


def _one_arg_measurement_update(kwargs):
    out = kalman_update(**kwargs)
    return out


def _one_arg_anchoring_update(kwargs):
    _, _, new_log_mixture_weights, new_loglikes = kalman_update(**kwargs)
    out = (
        kwargs["states"],
        kwargs["upper_chols"],
        new_log_mixture_weights,
        new_loglikes,
    )
    return out


def _one_arg_no_predict(kwargs, transition_func):  # noqa: ARG001
    """Just return the states cond chols without any changes."""
    return kwargs["states"], kwargs["upper_chols"], kwargs["states"]


def _one_arg_predict(kwargs, transition_func):
    """Do a predict step but also return the input states as filtered states."""
    new_states, new_upper_chols = kalman_predict(
        **kwargs,
        transition_func=transition_func,
    )
    return new_states, new_upper_chols, kwargs["states"]


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
