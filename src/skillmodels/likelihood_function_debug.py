import functools
from collections.abc import Callable  # noqa: TC003
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from skillmodels.clipping import soft_clipping
from skillmodels.kalman_filters import kalman_predict
from skillmodels.kalman_filters_debug import kalman_update
from skillmodels.parse_params import parse_params
from skillmodels.types import Dimensions, EstimationOptions, Labels  # noqa: TC001


def log_likelihood(
    params: Array,
    parsing_info: dict[str, Any],
    measurements: Array,
    controls: Array,
    transition_func: Callable[..., Array],
    sigma_scaling_factor: float,
    sigma_weights: Array,
    dimensions: Dimensions,
    labels: Labels,
    estimation_options: EstimationOptions,
    is_measurement_iteration: Array,
    is_predict_iteration: Array,
    iteration_to_period: Array,
    observed_factors: Array,
) -> dict[str, Any]:
    """Log likelihood of a skill formation model, returning debug data on top.

    This function is jax-differentiable and jax-jittable as long as all but the first
    argument are marked as static.

    Args:
        params: 1d array with model parameters.
        parsing_info: Contains information how to parse parameter vector.
        measurements: Array of shape (n_updates, n_obs) with data on observed
            measurements. NaN if the measurement was not observed.
        controls: Array of shape (n_periods, n_obs, n_controls) with observed
            control variables for the measurement equations.
        transition_func: The transition function.
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart.
        sigma_weights: 1d array of length n_sigma with non-negative sigma weights.
        dimensions: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures.
        labels: Labels for the model quantities like factors, periods, controls,
            stagemap and stages.
        estimation_options: Options for estimation including clipping bounds.
        is_measurement_iteration: Boolean array indicating which iterations are
            measurement updates.
        is_predict_iteration: Boolean array indicating which iterations are predict
            steps.
        iteration_to_period: Array mapping iteration index to period.
        observed_factors: Array of shape (n_periods, n_obs, n_observed_factors) with
            data on the observed factors.

    Returns:
        All data relevant for debugging, e.g. the log likelihood contribution of
        each Kalman update and additional information like the filtered states.

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
    clipped = soft_clipping(
        arr=static_out["loglikes"],
        lower=estimation_options.clipping_lower_bound,
        upper=estimation_options.clipping_upper_bound,
        lower_hardness=estimation_options.clipping_lower_hardness,
        upper_hardness=estimation_options.clipping_upper_hardness,
    )

    value = clipped.sum()

    out = {
        # used for scalar optimization, thus has to be clipped
        "value": value,
        # can be used for sum-structure optimizers, thus has to be clipped
        "contributions": clipped.sum(axis=0),
    }

    out["all_contributions"] = static_out["loglikes"]
    out["residuals"] = static_out["residuals"]
    out["residual_sds"] = static_out["residual_sds"]

    initial_states, _, initial_log_mixture_weights, _ = parse_params(
        params,
        parsing_info,
        dimensions,
        labels,
        n_obs,
    )
    out["initial_states"] = initial_states
    out["initial_log_mixture_weights"] = initial_log_mixture_weights

    out["filtered_states"] = static_out["states"]
    out["log_mixture_weights"] = static_out["log_mixture_weights"]

    return out


def _scan_body(
    carry: dict[str, Array],
    loop_args: dict[str, Array],
    controls: Array,
    pardict: dict[str, Any],
    sigma_scaling_factor: float,
    sigma_weights: Array,
    transition_func: Callable[..., Array],
    observed_factors: Array,
) -> tuple[dict[str, Array], dict[str, Any]]:
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
    states, upper_chols, log_mixture_weights, loglikes, info = jax.lax.cond(
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

    static_out = {"loglikes": loglikes, **info, "states": filtered_states}
    return new_state, static_out


def _one_arg_measurement_update(
    kwargs: dict[str, Any],
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    out = kalman_update(**kwargs)
    return out


def _one_arg_anchoring_update(
    kwargs: dict[str, Any],
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    _, _, new_log_mixture_weights, new_loglikes, debug_info = kalman_update(**kwargs)
    out = (
        kwargs["states"],
        kwargs["upper_chols"],
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )
    return out


def _one_arg_no_predict(
    kwargs: dict[str, Any],
    transition_func: Callable[..., Array],  # noqa: ARG001
) -> tuple[Array, Array, Array]:
    """Just return the states cond chols without any changes."""
    return kwargs["states"], kwargs["upper_chols"], kwargs["states"]


def _one_arg_predict(
    kwargs: dict[str, Any],
    transition_func: Callable[..., Array],
) -> tuple[Array, Array, Array]:
    """Do a predict step but also return the input states as filtered states."""
    new_states, new_upper_chols = kalman_predict(
        transition_func,
        **kwargs,
    )
    return new_states, new_upper_chols, kwargs["states"]
