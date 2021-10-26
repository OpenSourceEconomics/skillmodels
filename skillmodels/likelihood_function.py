import functools

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import config
from jax.ops import index
from jax.ops import index_update
from jax import lax

from skillmodels.constraints import add_bounds
from skillmodels.constraints import get_constraints
from skillmodels.kalman_filters import calculate_sigma_scaling_factor_and_weights
from skillmodels.kalman_filters import kalman_predict
from skillmodels.kalman_filters import kalman_update
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.parse_params import parse_params
from skillmodels.process_data import process_data_for_estimation
from skillmodels.process_debug_data import process_debug_data
from skillmodels.process_model import process_model

config.update("jax_enable_x64", True)


def get_maximization_inputs(model_dict, data):
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
        model["update_info"], model["labels"], model["dimensions"]
    )

    parsing_info = create_parsing_info(
        p_index, model["update_info"], model["labels"], model["anchoring"]
    )
    measurements, controls, = process_data_for_estimation(
        data, model["labels"], model["update_info"], model["anchoring"]
    )

    not_missing_arr = jnp.isfinite(measurements)
    # not_missing needs to be a list of 1d jax arrays, not a 2d jax array. Otherwise
    # selecting one row of the 2d array generates a new array which causes a problem
    # in jax jitted functions. I made an issue for what I think is the underlying
    # problem (https://github.com/google/jax/issues/4471) and hope it will be resolved
    # soon.
    not_missing_list = [row for row in not_missing_arr]  # noqa

    sigma_scaling_factor, sigma_weights = calculate_sigma_scaling_factor_and_weights(
        model["dimensions"]["n_states"],
        model["estimation_options"]["sigma_points_scale"],
    )

    update_info = model["update_info"]
    is_measurement_iteration = (update_info["purpose"] == "measurement").to_numpy()
    _periods = pd.Series(update_info.index.get_level_values("period").to_numpy())
    is_predict_iteration = ((_periods - _periods.shift(-1)) == -1).to_numpy()
    last_period = model["labels"]["periods"][-1]
    iteration_to_period = _periods.replace(last_period, -1).to_numpy()

    _base_loglike = functools.partial(
        _log_likelihood_jax,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        transition_functions=model["transition_functions"],
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        dimensions=model["dimensions"],
        labels=model["labels"],
        estimation_options=model["estimation_options"],
        not_missing_arr=not_missing_arr,
        is_measurement_iteration=is_measurement_iteration,
        is_predict_iteration=is_predict_iteration,
        iteration_to_period=iteration_to_period,
    )

    partialed_process_debug_data = functools.partial(process_debug_data, model=model)

    _debug_loglike = functools.partial(_base_loglike, debug=True)

    _loglike = functools.partial(_base_loglike, debug=False)

    _jitted_loglike = jax.jit(_loglike)
    _gradient = jax.grad(_jitted_loglike, has_aux=True)

    def debug_loglike(params):
        params_vec = jnp.array(params["value"].to_numpy())
        jax_output = _debug_loglike(params_vec)[1]
        if jax_output["contributions"].dtype != "float64":
            raise TypeError()
        numpy_output = _to_numpy(jax_output)
        # numpy_output["value"] = float(numpy_output["value"])
        # numpy_output = partialed_process_debug_data(numpy_output)
        return numpy_output

    def loglike(params):
        params_vec = jnp.array(params["value"].to_numpy())
        jax_output = _jitted_loglike(params_vec)[1]
        numpy_output = _to_numpy(jax_output)
        numpy_output["value"] = float(numpy_output["value"])
        return numpy_output

    def gradient(params):
        params_vec = jnp.array(params["value"].to_numpy())
        jax_output = _gradient(params_vec)[0]
        return _to_numpy(jax_output)

    def loglike_and_gradient(params):
        params_vec = jnp.array(params["value"].to_numpy())
        jax_grad, jax_crit = _gradient(params_vec)
        numpy_grad = _to_numpy(jax_grad)
        numpy_crit = _to_numpy(jax_crit)
        numpy_crit["value"] = float(numpy_crit["value"])
        return numpy_crit, numpy_grad

    constr = get_constraints(
        dimensions=model["dimensions"],
        labels=model["labels"],
        anchoring_info=model["anchoring"],
        update_info=model["update_info"],
        normalizations=model["normalizations"],
    )

    params_template = pd.DataFrame(columns=["value"], index=p_index)
    params_template = add_bounds(
        params_template, model["estimation_options"]["bounds_distance"]
    )

    out = {
        "loglike": loglike,
        "debug_loglike": debug_loglike,
        "gradient": gradient,
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
    transition_functions,
    sigma_scaling_factor,
    sigma_weights,
    dimensions,
    labels,
    estimation_options,
    not_missing_arr,
    is_measurement_iteration,
    is_predict_iteration,
    iteration_to_period,
    debug,
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
        transition_functions (tuple): tuple of tuples where the first element is the
            name of the transition function and the second the actual transition
            function. Order is important and corresponds to the latent
            factors in alphabetical order.
        sigma_scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights (jax.numpy.array): 1d array of length n_sigma with non-negative
            sigma weights.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        not_missing (jax.numpy.array): Array with same shape as measurements that is
            True where measurements are not missing.
        debug (bool): Boolean flag. If True, more intermediate results are returned

    Returns:
        jnp.array: 1d array of length 1, the aggregated log likelihood.
        dict: Additional data, containing log likelihood contribution of each Kalman
            update potentially if ``debug`` is ``True`` additional information like
            the filtered states.

    """
    n_obs = measurements.shape[1]
    states, upper_chols, log_mixture_weights, pardict = parse_params(
        params, parsing_info, dimensions, labels, n_obs
    )

    state = {
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
        # "not_missing": not_missing,
        "is_measurement_iteration": is_measurement_iteration,
        "is_predict_iteration": is_predict_iteration,
    }

    _body = functools.partial(
        _scan_body,
        controls=controls,
        pardict=pardict,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        transition_functions=transition_functions,
        debug=debug,
        not_missing_arr=not_missing_arr,
    )

    # ====
    loglike_list = []
    for k in range(len(iteration_to_period)):
        la = {key: val[k] for key, val in loop_args.items()}

        state, static_out = _body(state, la)
        loglike_list.append(static_out["loglikes"])

    loglikes = jnp.vstack(loglike_list)
    # ===
    state, static_out = lax.scan(_body, state, loop_args)

    breakpoint()



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

    return value, additional_data


def _scan_body(
    state,
    loop_args,
    controls,
    pardict,
    sigma_scaling_factor,
    sigma_weights,
    transition_functions,
    debug,
    not_missing_arr,
):
    t = loop_args["period"]
    states = state["states"]
    upper_chols = state["upper_chols"]
    log_mixture_weights = state["log_mixture_weights"]

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

    fixed_kwargs = {
        "not_missing": not_missing_arr[t],
        "debug": debug,
    }

    states, upper_chols, log_mixture_weights, loglikes, info = lax.cond(
        loop_args["is_measurement_iteration"],
        functools.partial(_one_arg_measurement_update, **fixed_kwargs),
        functools.partial(_one_arg_anchoring_update, **fixed_kwargs),
        update_kwargs,
    )

    # predict_kwargs = {
    #     "states": states,
    #     "upper_chols": upper_chols,
    #     "sigma_scaling_factor": sigma_scaling_factor,
    #     "sigma_weights": sigma_weights,
    #     # "transition_functions": transition_functions,
    #     "trans_coeffs": pardict["transition"][t],
    #     "shock_sds": pardict["shock_sds"][t],
    #     "anchoring_scaling_factors": pardict["anchoring_scaling_factors"][t : t + 2],
    #     "anchoring_constants": pardict["anchoring_constants"][t : t + 2],
    # }

    # fixed_kwargs = {"transition_functions": transition_functions}

    # states, upper_chols = lax.cond(
    #     loop_args["is_predict_iteration"],
    #     functools.partial(_one_arg_predict, **fixed_kwargs),
    #     functools.partial(_one_arg_no_predict, **fixed_kwargs),
    #     predict_kwargs,
    # )

    new_state = {
        "states": states,
        "upper_chols": upper_chols,
        "log_mixture_weights": log_mixture_weights,
    }

    static_out = {"loglikes": loglikes, **info}
    return new_state, static_out


def _one_arg_measurement_update(kwargs, not_missing, debug):
    out = kalman_update(**kwargs, not_missing=not_missing, debug=debug)
    return out


def _one_arg_anchoring_update(kwargs, not_missing, debug):
    _, _, new_log_mixture_weights, new_loglikes, debug_info = kalman_update(**kwargs, not_missing=not_missing, debug=debug)
    out = (kwargs["states"], kwargs["upper_chols"], new_log_mixture_weights, new_loglikes, debug_info)
    return out


def _one_arg_no_predict(kwargs, transition_functions):
    return kwargs["states"], kwargs["upper_chols"]


def _one_arg_predict(kwargs, transition_functions):
    out = kalman_predict(**kwargs, transition_functions=transition_functions)
    return out


def _log_likelihood_jax_old(
    params,
    parsing_info,
    measurements,
    controls,
    transition_functions,
    sigma_scaling_factor,
    sigma_weights,
    dimensions,
    labels,
    estimation_options,
    not_missing,
    is_measurement_iteration,
    is_predict_iteration,
    iteration_to_period,
    debug,
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
        transition_functions (tuple): tuple of tuples where the first element is the
            name of the transition function and the second the actual transition
            function. Order is important and corresponds to the latent
            factors in alphabetical order.
        sigma_scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights (jax.numpy.array): 1d array of length n_sigma with non-negative
            sigma weights.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        not_missing (jax.numpy.array): Array with same shape as measurements that is
            True where measurements are not missing.
        debug (bool): Boolean flag. If True, more intermediate results are returned

    Returns:
        jnp.array: 1d array of length 1, the aggregated log likelihood.
        dict: Additional data, containing log likelihood contribution of each Kalman
            update potentially if ``debug`` is ``True`` additional information like
            the filtered states.

    """
    n_obs = measurements.shape[1]
    states, upper_chols, log_mixture_weights, pardict = parse_params(
        params, parsing_info, dimensions, labels, n_obs
    )
    n_updates = len(iteration_to_period)
    loglikes = jnp.zeros((n_updates, n_obs))
    debug_infos = []
    states_history = []

    for k, t in enumerate(iteration_to_period):

        new_states, new_upper_chols, new_weights, loglikes_k, info = kalman_update(
            states,
            upper_chols,
            pardict["loadings"][k],
            pardict["controls"][k],
            pardict["meas_sds"][k],
            measurements[k],
            controls[t],
            log_mixture_weights,
            not_missing[k],
            debug,
        )
        if debug:
            states_history.append(new_states)

        loglikes = index_update(loglikes, index[k], loglikes_k)
        log_mixture_weights = new_weights
        if is_measurement_iteration[k]:
            states, upper_chols = new_states, new_upper_chols

        debug_infos.append(info)

        if is_predict_iteration[k]:
            states, upper_chols = kalman_predict(
                states,
                upper_chols,
                sigma_scaling_factor,
                sigma_weights,
                transition_functions,
                pardict["transition"][t],
                pardict["shock_sds"][t],
                pardict["anchoring_scaling_factors"][t : t + 2],
                pardict["anchoring_constants"][t : t + 2],
            )

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
        additional_data["residuals"] = [info["residuals"] for info in debug_infos]
        additional_data["residual_sds"] = [info["residual_sds"] for info in debug_infos]

        initial_states, *_ = parse_params(
            params, parsing_info, dimensions, labels, n_obs
        )
        additional_data["initial_states"] = initial_states

        additional_data["filtered_states"] = states_history

    return value, additional_data


def soft_clipping(arr, lower=None, upper=None, lower_hardness=1, upper_hardness=1):
    """Clip values in an array elementwise using a soft maximum to avoid kinks.

    Clipping from below is taking a maximum between two values. Clipping
    from above is taking a minimum, but it can be rewritten as taking a maximum after
    switching the signs.

    To smooth out the kinks introduced by normal clipping, we first rewrite all clipping
    operations to taking maxima. Then we replace the normal maximum by the soft maximum.

    For background on the soft maximum check out this
    `article by John Cook: <https://www.johndcook.com/soft_maximum.pdf>`_

    Note that contrary to the name, the soft maximum can be calculated using
    ``scipy.special.logsumexp``. ``scipy.special.softmax`` is the gradient of
    ``scipy.special.logsumexp``.


    Args:
        arr (jax.numpy.array): Array that is clipped elementwise.
        lower (float): The value at which the array is clipped from below.
        upper (float): The value at which the array is clipped from above.
        lower_hardness (float): Scaling factor that is applied inside the soft maximum.
            High values imply a closer approximation of the real maximum.
        upper_hardness (float): Scaling factor that is applied inside the soft maximum.
            High values imply a closer approximation of the real maximum.

    """
    shape = arr.shape
    flat = arr.flatten()
    dim = len(flat)
    if lower is not None:
        helper = jnp.column_stack([flat, jnp.full(dim, lower)])
        flat = (
            jax.scipy.special.logsumexp(lower_hardness * helper, axis=1)
            / lower_hardness
        )
    if upper is not None:
        helper = jnp.column_stack([-flat, jnp.full(dim, -upper)])
        flat = (
            -jax.scipy.special.logsumexp(upper_hardness * helper, axis=1)
            / upper_hardness
        )
    return flat.reshape(shape)


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
