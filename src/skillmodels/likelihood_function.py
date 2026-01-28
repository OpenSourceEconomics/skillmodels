"""Log-likelihood function for latent factor models."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from skillmodels.clipping import soft_clipping
from skillmodels.kalman_filters import (
    kalman_predict,
    kalman_update,
)
from skillmodels.parse_params import parse_params
from skillmodels.types import (
    Dimensions,
    EstimationOptions,
    Labels,
    ParsedParams,
    ParsingInfo,
)


def log_likelihood(
    params: Array,
    parsing_info: ParsingInfo,
    measurements: Array,
    controls: Array,
    transition_func: Callable,
    sigma_scaling_factor: float,
    sigma_weights: Array,
    dimensions: Dimensions,
    labels: Labels,
    estimation_options: EstimationOptions,
    is_measurement_iteration: Array,
    is_predict_iteration: Array,
    iteration_to_period: Array,
    observed_factors: Array,
) -> Array:
    """Aggregated log likelihood of a skill formation model.

    Wrapper around log_likelihood_obs that sums contributions across observations.

    Args:
        params: 1d array with model parameters.
        parsing_info: Contains information how to parse parameter vector.
        measurements: Array of shape (n_updates, n_obs) with data on
            observed measurements. NaN if the measurement was not observed.
        controls: Array of shape (n_periods, n_obs, n_controls)
            with observed control variables for the measurement equations.
        transition_func: The transition function.
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points.
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
        Scalar aggregated log likelihood.

    """
    return log_likelihood_obs(
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


def log_likelihood_obs(
    params: Array,
    parsing_info: ParsingInfo,
    measurements: Array,
    controls: Array,
    transition_func: Callable,
    sigma_scaling_factor: float,
    sigma_weights: Array,
    dimensions: Dimensions,
    labels: Labels,
    estimation_options: EstimationOptions,
    is_measurement_iteration: Array,
    is_predict_iteration: Array,
    iteration_to_period: Array,
    observed_factors: Array,
) -> Array:
    """Log likelihood of a skill formation model.

    This function is jax-differentiable and jax-jittable as long as all but the first
    argument are marked as static.

    The function returns both a tuple (float, dict). The first entry is the aggregated
    log likelihood value. The second additional information like the log likelihood
    contribution of each individual. Note that the dict also contains the aggregated
    value. Returning that value separately is only needed to calculate a gradient
    with Jax.

    Args:
        params: 1d array with model parameters.
        parsing_info: Contains information how to parse parameter vector.
        measurements: Array of shape (n_updates, n_obs) with data on
            observed measurements. NaN if the measurement was not observed.
        controls: Array of shape (n_periods, n_obs, n_controls)
            with observed control variables for the measurement equations.
        transition_func: The transition function.
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights: 1d array of length n_sigma with non-negative
            sigma weights.
        dimensions: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        estimation_options: Options for estimation including clipping bounds.
        is_measurement_iteration: Boolean array indicating which
            iterations are measurement updates.
        is_predict_iteration: Boolean array indicating which
            iterations are predict steps.
        iteration_to_period: Array mapping iteration index to period.
        observed_factors: Array of shape (n_periods, n_obs,
            n_observed_factors) with data on the observed factors.

    Returns:
        jnp.array: 1d array of length N, the aggregated log likelihood.

    """
    n_obs = measurements.shape[1]
    states, upper_chols, log_mixture_weights, parsed_params = parse_params(
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
        "loadings": parsed_params.loadings,
        "control_params": parsed_params.controls,
        "meas_sds": parsed_params.meas_sds,
        "measurements": measurements,
        "is_measurement_iteration": is_measurement_iteration,
        "is_predict_iteration": is_predict_iteration,
    }

    _body = functools.partial(
        _scan_body,
        controls=controls,
        parsed_params=parsed_params,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        transition_func=transition_func,
        observed_factors=observed_factors,
    )
    _body = jax.checkpoint(_body, prevent_cse=False)
    static_out = jax.lax.scan(_body, carry, loop_args, unroll=False)[1]

    # clip contributions before aggregation to preserve as much information as
    # possible.
    return soft_clipping(
        arr=static_out["loglikes"],
        lower=estimation_options.clipping_lower_bound,
        upper=estimation_options.clipping_upper_bound,
        lower_hardness=estimation_options.clipping_lower_hardness,
        upper_hardness=estimation_options.clipping_upper_hardness,
    ).sum(axis=0)


def _scan_body(
    carry: dict[str, Array],
    loop_args: dict[str, Array],
    controls: Array,
    parsed_params: ParsedParams,
    sigma_scaling_factor: float,
    sigma_weights: Array,
    transition_func: Callable,
    observed_factors: Array,
) -> tuple[dict[str, Array], dict[str, Array]]:
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
        "trans_coeffs": {k: arr[t] for k, arr in parsed_params.transition.items()},
        "shock_sds": parsed_params.shock_sds[t],
        "anchoring_scaling_factors": parsed_params.anchoring_scaling_factors[
            jnp.array([t, t + 1])
        ],
        "anchoring_constants": parsed_params.anchoring_constants[jnp.array([t, t + 1])],
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


def _one_arg_measurement_update(
    kwargs: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
    return kalman_update(**kwargs)


def _one_arg_anchoring_update(
    kwargs: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
    _, _, new_log_mixture_weights, new_loglikes = kalman_update(**kwargs)
    return (
        kwargs["states"],
        kwargs["upper_chols"],
        new_log_mixture_weights,
        new_loglikes,
    )


def _one_arg_no_predict(
    kwargs: dict[str, Any],
    transition_func: Callable,  # noqa: ARG001
) -> tuple[Array, Array, Array]:
    """Just return the states cond chols without any changes."""
    return kwargs["states"], kwargs["upper_chols"], kwargs["states"]


def _one_arg_predict(
    kwargs: dict[str, Any],
    transition_func: Callable,
) -> tuple[Array, Array, Array]:
    """Do a predict step but also return the input states as filtered states."""
    new_states, new_upper_chols = kalman_predict(
        transition_func,
        **kwargs,
    )
    return new_states, new_upper_chols, kwargs["states"]
