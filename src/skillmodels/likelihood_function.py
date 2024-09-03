import functools

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import config, lax

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

config.update("jax_enable_x64", True)


def get_maximization_inputs(model_dict, data, jacobian_type="jacrev"):
    """Create inputs for estimagic's maximize function.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        data (DataFrame): dataset in long format.

    Returns a dictionary with keys:
        loglike (function): A jax jitted function that takes an estimagic-style
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
        constraints (list): List of estimagic constraints that are implied by the
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

    update_info = model["update_info"]
    is_measurement_iteration = (update_info["purpose"] == "measurement").to_numpy()
    _periods = pd.Series(update_info.index.get_level_values("period").to_numpy())
    is_predict_iteration = ((_periods - _periods.shift(-1)) == -1).to_numpy()
    last_period = model["labels"]["periods"][-1]
    # iteration_to_period is used as an indexer to loop over arrays of different lengths
    # in a lax.scan. It needs to work for arrays of length n_periods and not raise
    # IndexErrors on tracer arrays of length n_periods - 1 (i.e. n_transitions).
    # To achieve that, we replace the last period by -1.
    iteration_to_period = _periods.replace(last_period, -1).to_numpy()

    _base_loglike = functools.partial(
        _log_likelihood_jax,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        transition_info=model["transition_info"],
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

    partialed_process_debug_data = functools.partial(process_debug_data, model=model)

    partialed_get_jnp_params_vec = functools.partial(
        _get_jnp_params_vec,
        target_index=p_index,
    )

    _debug_loglike = functools.partial(_base_loglike, debug=True)

    _loglike = functools.partial(_base_loglike, debug=False)

    _jitted_loglike = jax.jit(_loglike)
    _gradient = jax.jit(jax.grad(_loglike, has_aux=True))
    _jacobian = jax.jit(getattr(jax, jacobian_type)(_loglike, has_aux=True))

    def debug_loglike(params):
        params_vec = partialed_get_jnp_params_vec(params)
        jax_output = _debug_loglike(params_vec)[1]
        if jax_output["contributions"].dtype != "float64":
            raise TypeError
        numpy_output = _to_numpy(jax_output)
        numpy_output["value"] = float(numpy_output["value"])
        numpy_output = partialed_process_debug_data(numpy_output)
        return numpy_output

    def loglike(params):
        params_vec = partialed_get_jnp_params_vec(params)
        jax_output = _jitted_loglike(params_vec)[1]
        numpy_output = _to_numpy(jax_output)
        numpy_output["value"] = float(numpy_output["value"])
        return numpy_output["contributions"]

    def gradient(params):
        params_vec = partialed_get_jnp_params_vec(params)
        jax_output = _gradient(params_vec)[0]
        return _to_numpy(jax_output)

    def jacobian(params):
        params_vec = partialed_get_jnp_params_vec(params)
        jax_output = _jacobian(params_vec)[0]
        return _to_numpy(jax_output)

    def loglike_and_gradient(params):
        params_vec = partialed_get_jnp_params_vec(params)
        jax_grad, jax_crit = _gradient(params_vec)
        numpy_grad = _to_numpy(jax_grad)
        crit = float(jax_crit["value"])
        return crit, numpy_grad

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
        "debug_loglike": debug_loglike,
        "gradient": gradient,
        "jacobian": jacobian,
        "loglike_and_gradient": loglike_and_gradient,
        "constraints": constr,
        "params_template": params_template,
    }

    return out


def _log_likelihood_jax(
    params,
    parsing_info,
    measurements,
    controls,
    transition_info,
    sigma_scaling_factor,
    sigma_weights,
    dimensions,
    labels,
    estimation_options,
    is_measurement_iteration,
    is_predict_iteration,
    iteration_to_period,
    debug,
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
        transition_info (dict): Dict with the entries "func" (the actual transition
            function) and "columns" (a dictionary mapping factors that are needed
            as individual columns to positions in the factor array).
        sigma_scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights (jax.numpy.array): 1d array of length n_sigma with non-negative
            sigma weights.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        debug (bool): Boolean flag. If True, more intermediate results are returned
        observed_factors (jax.numpy.array): Array of shape (n_periods, n_obs,
            n_observed_factors) with data on the observed factors.

    Returns:
        jnp.array: 1d array of length 1, the aggregated log likelihood.
        dict: Additional data, containing log likelihood contribution of each Kalman
            update potentially if ``debug`` is ``True`` additional information like
            the filtered states.

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
        transition_info=transition_info,
        observed_factors=observed_factors,
        debug=debug,
    )

    carry, static_out = lax.scan(_body, carry, loop_args)
    loglikes = static_out["loglikes"]

    # clip contributions before aggregation to preserve as much information as
    # possible.
    clipped = soft_clipping(
        arr=loglikes,
        lower=estimation_options["clipping_lower_bound"],
        upper=estimation_options["clipping_upper_bound"],
        lower_hardness=estimation_options["clipping_lower_hardness"],
        upper_hardness=estimation_options["clipping_upper_hardness"],
    )

    value = clipped.sum()

    additional_data = {
        # used for scalar optimization, thus has to be clipped
        "value": value,
        # can be used for sum-structure optimizers, thus has to be clipped
        "contributions": clipped.sum(axis=0),
    }

    if debug:
        additional_data["all_contributions"] = loglikes
        additional_data["residuals"] = static_out["residuals"]
        additional_data["residual_sds"] = static_out["residual_sds"]

        initial_states, _, initial_log_mixture_weights, _ = parse_params(
            params,
            parsing_info,
            dimensions,
            labels,
            n_obs,
        )
        additional_data["initial_states"] = initial_states
        additional_data["initial_log_mixture_weights"] = initial_log_mixture_weights

        additional_data["filtered_states"] = static_out["states"]
        additional_data["log_mixture_weights"] = static_out["log_mixture_weights"]

    return value, additional_data


def _scan_body(
    carry,
    loop_args,
    controls,
    pardict,
    sigma_scaling_factor,
    sigma_weights,
    transition_info,
    observed_factors,
    debug,
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
    states, upper_chols, log_mixture_weights, loglikes, info = lax.cond(
        loop_args["is_measurement_iteration"],
        functools.partial(_one_arg_measurement_update, debug=debug),
        functools.partial(_one_arg_anchoring_update, debug=debug),
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

    fixed_kwargs = {"transition_info": transition_info}

    # ==================================================================================
    # Do a predict step or a do-nothing fake predict step
    # ==================================================================================
    states, upper_chols, filtered_states = lax.cond(
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

    static_out = {"loglikes": loglikes, **info, "states": filtered_states}
    return new_state, static_out


def _one_arg_measurement_update(kwargs, debug):
    out = kalman_update(**kwargs, debug=debug)
    return out


def _one_arg_anchoring_update(kwargs, debug):
    _, _, new_log_mixture_weights, new_loglikes, debug_info = kalman_update(
        **kwargs,
        debug=debug,
    )
    out = (
        kwargs["states"],
        kwargs["upper_chols"],
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )
    return out


def _one_arg_no_predict(kwargs, transition_info):
    """Just return the states cond chols without any changes."""
    return kwargs["states"], kwargs["upper_chols"], kwargs["states"]


def _one_arg_predict(kwargs, transition_info):
    """Do a predict step but also return the input states as filtered states."""
    new_states, new_upper_chols = kalman_predict(
        **kwargs,
        transition_info=transition_info,
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