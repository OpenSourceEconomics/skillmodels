"""Log-likelihood function for latent factor models."""

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
    get_cut_table,
    limited_update_from_loop_kwargs,
)
from skillmodels.chs.kalman_filters import kalman_update
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
    predict_func: Callable,
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
        predict_func: Callable that performs the predict step. Either
            `kalman_predict` (partialed with `transition_func`) or
            `linear_kalman_predict` (partialed with factor metadata).
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points.
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
        Scalar aggregated log likelihood.

    """
    return log_likelihood_obs(
        params=params,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        predict_func=predict_func,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        dimensions=dimensions,
        labels=labels,
        chs_estimation_options=chs_estimation_options,
        is_measurement_iteration=is_measurement_iteration,
        is_predict_iteration=is_predict_iteration,
        iteration_to_period=iteration_to_period,
        observed_factors=observed_factors,
        observed_factor_vars=observed_factor_vars,
        events=events,
        limited_updates=limited_updates,
    ).sum()


def log_likelihood_obs(
    params: Array,
    parsing_info: ParsingInfo,
    measurements: Array | np.ndarray,
    controls: Array,
    predict_func: Callable,
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
        predict_func: Callable that performs the predict step. Either
            `kalman_predict` (partialed with `transition_func`) or
            `linear_kalman_predict` (partialed with factor metadata).
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights: 1d array of length n_sigma with non-negative
            sigma weights.
        dimensions: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        chs_estimation_options: Options for estimation including clipping bounds.
        is_measurement_iteration: Boolean array indicating which
            iterations are measurement updates.
        is_predict_iteration: Boolean array indicating which
            iterations are predict steps.
        iteration_to_period: Array mapping iteration index to period.
        observed_factors: Array of shape (n_periods, n_obs,
            n_observed_factors) with data on the observed factors.
        observed_factor_vars: Array of shape (n_periods, n_obs, n_missing) with
            the variance of each possibly-missing observed factor (0 where it is
            observed).
        events: Array of shape (n_updates, n_obs) with the `ObservationEvent` of
            each measurement, or None if all measurements are Gaussian.
        limited_updates: Per-update families and thresholds, or None if all
            measurements are Gaussian.

    Returns:
        jnp.array: 1d array of length N, the aggregated log likelihood.

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

    loop_args = {
        "period": iteration_to_period,
        "loadings": parsed_params.loadings,
        "control_params": parsed_params.controls,
        "meas_sds": parsed_params.meas_sds,
        "measurements": measurements,
        "is_measurement_iteration": is_measurement_iteration,
        "is_predict_iteration": is_predict_iteration,
        "observed_later": get_observation_masks(measurements, iteration_to_period)[0],
        **get_limited_loop_args(
            events=events,
            limited_updates=limited_updates,
            meas_sds=parsed_params.meas_sds,
            params=params,
        ),
    }

    _body = functools.partial(
        _scan_body,
        has_intervals=limited_updates is not None and limited_updates.has_intervals,
        controls=controls,
        parsed_params=parsed_params,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        predict_func=predict_func,
        observed_factors=observed_factors,
        observed_factor_vars=observed_factor_vars,
    )
    _body = jax.checkpoint(_body, prevent_cse=False)
    static_out = jax.lax.scan(_body, carry, loop_args, unroll=False)[1]

    # clip contributions before aggregation to preserve as much information as
    # possible.
    return soft_clipping(
        arr=static_out["loglikes"],
        lower=chs_estimation_options.clipping_lower_bound,
        upper=chs_estimation_options.clipping_upper_bound,
        lower_hardness=chs_estimation_options.clipping_lower_hardness,
        upper_hardness=chs_estimation_options.clipping_upper_hardness,
    ).sum(axis=0)


def get_observation_masks(
    measurements: Array | np.ndarray,
    iteration_to_period: Array | np.ndarray,
) -> tuple[Array, Array]:
    """Return masks of persons with an observed measurement after each update.

    Args:
        measurements: Array of shape (n_updates, n_obs). NaN if not observed.
        iteration_to_period: Array of length n_updates mapping each update to its
            period. Updates of one period are contiguous.

    Returns:
        - observed_later: Array of shape (n_updates, n_obs). True if the person has
          an observed measurement in a later update. Only these persons are
          predicted after the update.
        - observed_from_period: Array of shape (n_updates, n_obs). True if the
          person has an observed measurement in the period of the update or in a
          later period. Filtered states are reported only for these persons.

    """
    observed = jnp.isfinite(measurements)
    observed_from = jnp.flip(jnp.cumsum(jnp.flip(observed, axis=0), axis=0) > 0, axis=0)
    observed_later = jnp.concatenate(
        [observed_from[1:], jnp.zeros_like(observed_from[:1])], axis=0
    )
    periods = jnp.asarray(iteration_to_period)
    is_period_start = jnp.concatenate([jnp.array([True]), periods[1:] != periods[:-1]])
    period_start = jax.lax.cummax(
        jnp.where(is_period_start, jnp.arange(len(periods)), 0), axis=0
    )
    return observed_later, observed_from[period_start]


def predict_if_observed_later(
    *,
    predict_kwargs: dict[str, Any],
    observed_later: Array | np.ndarray,
    is_predict_iteration: Array | np.ndarray,
    predict_func: Callable,
) -> tuple[Array, Array]:
    """Predict the persons with a later observation; keep all others unchanged.

    After the last observed measurement of a person, states and covariances stay at
    their last filtered values. The predict step runs on placeholder inputs (zero
    states, identity covariance) for these persons and its result is discarded.
    The placeholders keep the predict step and its derivatives finite, which a
    covariance that is predicted over many unobserved periods does not do.

    Args:
        predict_kwargs: Keyword arguments of `predict_func`, with `states` of shape
            (n_obs, n_mixtures, n_states) and `upper_chols` of shape
            (n_obs, n_mixtures, n_states, n_states).
        observed_later: Boolean array of length n_obs.
        is_predict_iteration: Whether a predict step follows this update.
        predict_func: The predict function.

    Returns:
        new_states: Same shape as `predict_kwargs["states"]`.
        new_upper_chols: Same shape as `predict_kwargs["upper_chols"]`.

    """
    states = predict_kwargs["states"]
    upper_chols = predict_kwargs["upper_chols"]
    keep_states = observed_later.reshape(-1, 1, 1)
    keep_chols = observed_later.reshape(-1, 1, 1, 1)
    safe_kwargs = {
        **predict_kwargs,
        "states": jnp.where(keep_states, states, 0.0),
        "upper_chols": jnp.where(
            keep_chols, upper_chols, jnp.eye(states.shape[-1], dtype=upper_chols.dtype)
        ),
    }
    new_states, new_upper_chols = jax.lax.cond(
        is_predict_iteration,
        lambda kwargs: predict_func(**kwargs),
        lambda kwargs: (kwargs["states"], kwargs["upper_chols"]),
        safe_kwargs,
    )
    return (
        jnp.where(keep_states, new_states, states),
        jnp.where(keep_chols, new_upper_chols, upper_chols),
    )


def _scan_body(
    carry: dict[str, Array],
    loop_args: dict[str, Array],
    controls: Array,
    parsed_params: ParsedParams,
    sigma_scaling_factor: float | Array,
    sigma_weights: Array,
    predict_func: Callable,
    observed_factors: Array | np.ndarray,
    observed_factor_vars: Array | np.ndarray,
    *,
    has_intervals: bool = False,
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
        **{k: loop_args[k] for k in LIMITED_LOOP_KEYS if k in loop_args},
    }

    # ==================================================================================
    # do a measurement or anchoring update
    # ==================================================================================
    if "is_limited" in loop_args:
        states, upper_chols, log_mixture_weights, loglikes = _mixed_model_update(
            update_kwargs,
            is_measurement_iteration=loop_args["is_measurement_iteration"],
            has_intervals=has_intervals,
        )
    else:
        states, upper_chols, log_mixture_weights, loglikes = jax.lax.cond(
            loop_args["is_measurement_iteration"],
            kalman_update_from_kwargs,
            _one_arg_anchoring_update,
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

    static_out = {"loglikes": loglikes, "states": filtered_states}
    return new_state, static_out


def get_limited_loop_args(
    events: Array | None,
    limited_updates: LimitedUpdates | None,
    meas_sds: Array,
    params: Array,
) -> dict[str, Array]:
    """Return the per-update scan inputs of limited measurements.

    Empty without limited measurements, so an all-Gaussian model runs the plain
    Gaussian scan. Otherwise the `meas_sds` entry replaces the parsed one with a
    unit error SD for probit and ordered-probit updates, and `cut_table` holds the
    cut values of each update row with the current cutpoints.
    """
    if events is None or limited_updates is None:
        return {}
    return {
        "events": events,
        "is_limited": jnp.asarray(limited_updates.is_limited),
        "cut_table": get_cut_table(limited_updates, params),
        "codes_from_values": jnp.asarray(limited_updates.codes_from_values),
        "meas_sds": jnp.where(
            jnp.asarray(limited_updates.is_unit_scale), 1.0, meas_sds
        ),
    }


def _mixed_model_update(
    kwargs: dict[str, Array],
    *,
    is_measurement_iteration: Array,
    has_intervals: bool,
) -> tuple[Array, Array, Array, Array]:
    """Run the anchoring, Gaussian, or limited update of a model with limited measures.

    One three-way `lax.switch` selects the update; the limited branch is
    rematerialized (`jax.checkpoint`). Under reverse-mode differentiation inside the
    scan, the residuals of all branches are merged and the branches not taken are
    zero-filled at every iteration; rematerializing the limited branch reduces its
    residuals to its inputs, so a Gaussian update pays almost nothing for the
    limited branch.
    """

    def limited(kw: dict[str, Array]) -> tuple[Array, Array, Array, Array]:
        update = limited_update_from_loop_kwargs(kw, has_intervals=has_intervals)
        return update[0], update[1], update[2], update[3]

    index = jnp.where(
        is_measurement_iteration, 1 + kwargs["is_limited"].astype(jnp.int32), 0
    )
    return jax.lax.switch(
        index,
        [
            _one_arg_anchoring_update,
            kalman_update_from_kwargs,
            jax.checkpoint(limited),
        ],
        kwargs,
    )


def kalman_update_from_kwargs(
    kwargs: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
    """Run the Gaussian Kalman update on update arguments of one scan iteration."""
    return kalman_update(**gaussian_update_kwargs(kwargs))


def _one_arg_anchoring_update(
    kwargs: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
    _, _, new_log_mixture_weights, new_loglikes = kalman_update(
        **gaussian_update_kwargs(kwargs)
    )
    return (
        kwargs["states"],
        kwargs["upper_chols"],
        new_log_mixture_weights,
        new_loglikes,
    )


def gaussian_update_kwargs(kwargs: dict[str, Array]) -> dict[str, Array]:
    """Return the update arguments without the limited-measurement inputs."""
    return {k: v for k, v in kwargs.items() if k not in LIMITED_LOOP_KEYS}
