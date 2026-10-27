"""Debug version of log-likelihood function that returns intermediate results."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from skillmodels.chs.clipping import soft_clipping
from skillmodels.chs.halfline import (
    LIMITED_LOOP_KEYS,
    LimitedUpdates,
    kalman_update_limited,
)
from skillmodels.chs.kalman_filters_debug import kalman_update
from skillmodels.chs.likelihood import (
    gaussian_update_kwargs,
    get_limited_loop_args,
    get_observation_masks,
    predict_if_observed_later,
)
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.parse_params import parse_params
from skillmodels.common.types import (
    Dimensions,
    Labels,
    ParsedParams,
    ParsingInfo,
)


def log_likelihood(
    params: Array,
    parsing_info: ParsingInfo,
    measurements: Array | np.ndarray,
    controls: Array,
    predict_func: Callable[..., tuple[Array, Array]],
    sigma_scaling_factor: float | Array,
    sigma_weights: Array,
    dimensions: Dimensions,
    labels: Labels,
    chs_estimation_options: CHSEstimationOptions,
    is_measurement_iteration: Array | np.ndarray,
    is_predict_iteration: Array | np.ndarray,
    iteration_to_period: Array | np.ndarray,
    observed_factors: Array | np.ndarray,
    observed_factor_vars: Array | np.ndarray,
    events: Array | None = None,
    limited_updates: LimitedUpdates | None = None,
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
        predict_func: Callable that performs the predict step.
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart.
        sigma_weights: 1d array of length n_sigma with non-negative sigma weights.
        dimensions: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures.
        labels: Labels for the model quantities like factors, periods, controls,
            stagemap and stages.
        chs_estimation_options: Options for estimation including clipping bounds.
        is_measurement_iteration: Boolean array indicating which iterations are
            measurement updates.
        is_predict_iteration: Boolean array indicating which iterations are predict
            steps.
        iteration_to_period: Array mapping iteration index to period.
        observed_factors: Array of shape (n_periods, n_obs, n_observed_factors) with
            data on the observed factors.
        observed_factor_vars: Array of shape (n_periods, n_obs, n_missing) with
            the variance of each possibly-missing observed factor (0 where it is
            observed).
        events: Array of shape (n_updates, n_obs) with the `ObservationEvent` of
            each measurement, or None if all measurements are Gaussian.
        limited_updates: Per-update families and thresholds, or None if all
            measurements are Gaussian.

    Returns:
        All data relevant for debugging, e.g. the log likelihood contribution of
        each Kalman update and additional information like the filtered states.

    """
    n_obs = measurements.shape[1]
    states, upper_chols, log_mixture_weights, parsed_params = parse_params(
        params=params,
        parsing_info=parsing_info,
        dimensions=dimensions,
        labels=labels,
        n_obs=n_obs,
    )

    carry = {
        "states": states,
        "upper_chols": upper_chols,
        "log_mixture_weights": log_mixture_weights,
    }

    observed_later, observed_from_period = get_observation_masks(
        measurements, iteration_to_period
    )
    loop_args = {
        "period": iteration_to_period,
        "loadings": parsed_params.loadings,
        "control_params": parsed_params.controls,
        "meas_sds": parsed_params.meas_sds,
        "measurements": measurements,
        "is_measurement_iteration": is_measurement_iteration,
        "is_predict_iteration": is_predict_iteration,
        "observed_later": observed_later,
        **get_limited_loop_args(
            events=events,
            limited_updates=limited_updates,
            meas_sds=parsed_params.meas_sds,
        ),
    }

    _body = functools.partial(
        _scan_body,
        controls=controls,
        parsed_params=parsed_params,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        predict_func=predict_func,
        observed_factors=observed_factors,
        observed_factor_vars=observed_factor_vars,
    )

    static_out = jax.lax.scan(_body, carry, loop_args)[1]

    # clip contributions before aggregation to preserve as much information as
    # possible.
    clipped = soft_clipping(
        arr=static_out["loglikes"],
        lower=chs_estimation_options.clipping_lower_bound,
        upper=chs_estimation_options.clipping_upper_bound,
        lower_hardness=chs_estimation_options.clipping_lower_hardness,
        upper_hardness=chs_estimation_options.clipping_upper_hardness,
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
        params=params,
        parsing_info=parsing_info,
        dimensions=dimensions,
        labels=labels,
        n_obs=n_obs,
    )
    out["initial_states"] = initial_states
    out["initial_log_mixture_weights"] = initial_log_mixture_weights

    out["filtered_states"] = jnp.where(
        observed_from_period[..., None, None], static_out["states"], jnp.nan
    )
    out["log_mixture_weights"] = static_out["log_mixture_weights"]

    return out


def _scan_body(
    carry: dict[str, Array],
    loop_args: dict[str, Array | np.ndarray],
    controls: Array,
    parsed_params: ParsedParams,
    sigma_scaling_factor: float | Array,
    sigma_weights: Array,
    predict_func: Callable[..., tuple[Array, Array]],
    observed_factors: Array | np.ndarray,
    observed_factor_vars: Array | np.ndarray,
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
        **{k: loop_args[k] for k in LIMITED_LOOP_KEYS if k in loop_args},
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
        "trans_coeffs": {k: arr[t] for k, arr in parsed_params.transition.items()},
        "shock_sds": parsed_params.shock_sds[t],
        "anchoring_scaling_factors": parsed_params.anchoring_scaling_factors[
            jnp.array([t, t + 1])
        ],
        "anchoring_constants": parsed_params.anchoring_constants[jnp.array([t, t + 1])],
        "observed_factors": observed_factors[t],
        "observed_factor_vars": observed_factor_vars[t],
    }

    # ==================================================================================
    # Do a predict step for persons observed later; freeze all others
    # ==================================================================================
    filtered_states = states
    states, upper_chols = predict_if_observed_later(
        predict_kwargs=predict_kwargs,
        observed_later=loop_args["observed_later"],
        is_predict_iteration=loop_args["is_predict_iteration"],
        predict_func=predict_func,
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
    if "events" not in kwargs:
        return kalman_update(**kwargs)
    gaussian_kwargs = gaussian_update_kwargs(kwargs)
    limited_kwargs = {k: v for k, v in kwargs.items() if k != "is_limited"}
    return jax.lax.cond(
        kwargs["is_limited"],
        lambda: kalman_update_limited(**limited_kwargs),
        lambda: kalman_update(**gaussian_kwargs),
    )


def _one_arg_anchoring_update(
    kwargs: dict[str, Any],
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    _, _, new_log_mixture_weights, new_loglikes, debug_info = kalman_update(
        **gaussian_update_kwargs(kwargs)
    )
    return (
        kwargs["states"],
        kwargs["upper_chols"],
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )
