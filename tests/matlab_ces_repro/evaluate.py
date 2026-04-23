"""Evaluate skillmodels' AF log-likelihood at a given parameter vector.

This mirrors the setup in ``skillmodels.af.initial_period`` and
``skillmodels.af.transition_period`` up to building the jitted likelihood
but stops short of running the optimizer. It lets tests score MATLAB-
translated parameters under skillmodels' own likelihood so we can ask
"does MATLAB's optimum give a higher likelihood than ours?" without
having to run a second optimisation pass.

Public entry points:

- ``evaluate_af_initial_loglike(model_spec, period_0_data, params_df,
  af_options, observed_factor_values)`` → scalar ``log L``.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.af.batching import auto_n_obs_per_batch
from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.initial_period import (
    _build_loading_mask,
    _get_ordered_measures,
)
from skillmodels.af.likelihood import (
    af_loglike_initial,
    af_loglike_transition,
    create_loglike_and_gradient,
)
from skillmodels.af.params import (
    get_initial_period_params_index,
    get_measurements_per_factor,
    get_normalizations_for_period,
    get_transition_period_params_index,
)
from skillmodels.af.transition_period import (
    _extract_prev_measurement_params,
    _get_raw_transition_functions,
    _prepare_transition_inputs,
)
from skillmodels.af.types import AFEstimationOptions, ConditionalDistribution
from skillmodels.model_spec import ModelSpec
from skillmodels.process_model import process_model


def evaluate_af_initial_loglike(
    *,
    model_spec: ModelSpec,
    measurements: Array,
    controls: Array,
    params_df: pd.DataFrame,
    af_options: AFEstimationOptions,
    observed_factors: tuple[str, ...] = (),
    observed_factor_values: Array | None = None,
) -> float:
    """Return ``-neg_log_likelihood`` i.e. the log-likelihood per observation.

    Args:
        model_spec: The AF model spec.
        measurements: Shape ``(n_obs, n_measures)`` period-0 measurement
            values.
        controls: Shape ``(n_obs, n_controls)`` period-0 control values.
        params_df: Full parameter DataFrame with the initial-period
            MultiIndex produced by ``get_initial_period_params_index``.
            Must have a ``"value"`` column.
        af_options: AF options (uses the same Halton count as the
            estimator would).
        observed_factors: Names of observed factors in the initial joint.
        observed_factor_values: Shape ``(n_obs, n_observed_factors)`` of
            observed factor values.

    Return:
        Average log-likelihood per observation (matches what the estimator
        reports as ``AFPeriodResult.loglikelihood``).
    """
    processed_model = process_model(model_spec)
    n_components = af_options.n_mixture_components
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    n_obs_factors = len(observed_factors)

    obs_values = (
        observed_factor_values
        if observed_factor_values is not None
        else jnp.zeros((measurements.shape[0], 0))
    )

    measurements_p0 = get_measurements_per_factor(model_spec.factors, period=0)
    reconstructed_factors = tuple(
        f for f in factors if not model_spec.factors[f].has_initial_distribution
    )
    state_latent_factors = tuple(f for f in factors if f not in reconstructed_factors)
    n_state_latent = len(state_latent_factors)
    n_joint = n_state_latent + n_obs_factors
    params_index = get_initial_period_params_index(
        n_mixture_components=n_components,
        latent_factors=factors,
        measurements_period_0=measurements_p0,
        controls=controls_names,
        observed_factors=observed_factors,
        reconstructed_factors=reconstructed_factors,
    )
    # Sanity check that the caller-supplied params_df matches the AF index.
    if not params_df.index.equals(params_index):
        msg = (
            "params_df has a different MultiIndex than the AF initial-period "
            "index. Build it via get_initial_period_params_index."
        )
        raise ValueError(msg)
    # Unused but kept as a lookup in case future calls need it.
    _ = get_normalizations_for_period(model_spec.factors, period=0)

    measurements_p0_filtered = {
        f: m for f, m in measurements_p0.items() if f in state_latent_factors
    }
    all_measures_full = _get_ordered_measures(measurements_p0)
    all_measures = _get_ordered_measures(measurements_p0_filtered)
    if len(all_measures) != len(all_measures_full):
        col_indices = jnp.array(
            [all_measures_full.index(m) for m in all_measures], dtype=jnp.int32
        )
        measurements = measurements[:, col_indices]
    loading_mask = _build_loading_mask(
        all_measures, state_latent_factors, measurements_p0_filtered
    )
    nodes, weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_state_latent,
    )

    n_obs_per_batch = af_options.n_obs_per_batch
    if n_obs_per_batch is None:
        n_obs_per_batch = auto_n_obs_per_batch(
            n_obs=int(measurements.shape[0]),
            n_halton_points=af_options.n_halton_points,
            n_halton_points_shock=af_options.n_halton_points_shock,
            n_latent=n_joint,
            n_endogenous=0,
        )

    loglike_kwargs = {
        "n_factors": n_joint,
        "n_latent_factors": n_state_latent,
        "n_mixture_components": n_components,
        "n_measures": len(all_measures),
        "n_controls": len(controls_names),
        "measurements": measurements,
        "controls": controls,
        "observed_factor_values": obs_values,
        "loading_mask": jnp.array(loading_mask),
        "nodes": nodes,
        "weights": weights,
        "stability_floor": af_options.stability_floor,
        "n_obs_per_batch": n_obs_per_batch,
    }

    loglike_and_grad = create_loglike_and_gradient(af_loglike_initial, **loglike_kwargs)

    params_array = jnp.array(params_df["value"].to_numpy(dtype=np.float64))
    neg_ll, _grad = loglike_and_grad(params_array)
    return -float(neg_ll)


def evaluate_af_transition_loglike(
    *,
    model_spec: ModelSpec,
    period: int,
    measurements: Array,
    controls: Array,
    prev_measurements: Array,
    prev_controls: Array,
    prev_period_params: pd.DataFrame,
    prev_distribution: ConditionalDistribution,
    params_df: pd.DataFrame,
    af_options: AFEstimationOptions,
    endogenous_factors: tuple[str, ...] = (),
    observed_factors: tuple[str, ...] = (),
    observed_factor_data: Array | None = None,
) -> float:
    """Return the log-likelihood at a supplied transition-period params vector.

    Mirrors the setup in ``estimate_transition_period`` but evaluates the
    jitted likelihood once instead of running an optimizer.
    """
    processed_model = process_model(model_spec)
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls

    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    all_measures = _get_ordered_measures(measurements_pt)

    transition_info = processed_model.transition_info
    state_factors = tuple(f for f in factors if f not in endogenous_factors)
    n_state = len(state_factors)
    n_endog = len(endogenous_factors)
    shock_factors = tuple(
        f for f in state_factors if model_spec.factors[f].has_production_shock
    )
    n_shock = len(shock_factors)
    shock_factor_indices = jnp.array(
        [state_factors.index(f) for f in shock_factors], dtype=jnp.int32
    )

    params_index = get_transition_period_params_index(
        period=period,
        latent_factors=state_factors,
        transition_info=transition_info,
        measurements_at_period=measurements_pt,
        controls=controls_names,
        endogenous_factors=endogenous_factors,
        observed_factors=observed_factors,
        shock_factors=shock_factors,
    )
    if not params_df.index.equals(params_index):
        msg = (
            "params_df has a different MultiIndex than the transition-period "
            f"index for period {period}. Build it via "
            "get_transition_period_params_index."
        )
        raise ValueError(msg)

    loading_mask = _build_loading_mask(all_measures, factors, measurements_pt)

    joint_dim = n_state + n_shock + n_endog
    joint_nodes, joint_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        joint_dim,
    )

    prev_dist_arrays, total_n_transition_params = _prepare_transition_inputs(
        prev_distribution,
        transition_info,
        state_factors,
        measurements.shape[0],
    )

    raw_funcs = _get_raw_transition_functions(model_spec, state_factors)
    param_counts = tuple(len(transition_info.param_names[f]) for f in state_factors)

    def combined_transition(full_states: Array, params: Array) -> Array:
        result = jnp.zeros(n_state)
        p_idx = 0
        for i in range(n_state):
            n_p = param_counts[i]
            factor_params = params[p_idx : p_idx + n_p]
            result = result.at[i].set(  # noqa: PD008
                raw_funcs[i](full_states, factor_params)
            )
            p_idx += n_p
        return result

    n_inv_eq_params_per = 1 + n_state + len(observed_factors) if n_endog > 0 else 0
    total_n_inv_params = n_endog * n_inv_eq_params_per

    n_obs_fac = len(observed_factors)
    obs_factor_values = (
        observed_factor_data
        if observed_factor_data is not None
        else jnp.zeros((measurements.shape[0], n_obs_fac))
    )

    prev_meas_info = _extract_prev_measurement_params(
        prev_period_params, model_spec, factors, period - 1
    )

    n_obs_per_batch = af_options.n_obs_per_batch
    if n_obs_per_batch is None:
        n_obs_per_batch = auto_n_obs_per_batch(
            n_obs=int(measurements.shape[0]),
            n_halton_points=af_options.n_halton_points,
            n_halton_points_shock=af_options.n_halton_points_shock,
            n_latent=n_state,
            n_endogenous=n_endog,
        )

    loglike_kwargs = {
        "n_state_factors": n_state,
        "n_endogenous_factors": n_endog,
        "n_shock_factors": n_shock,
        "shock_factor_indices": shock_factor_indices,
        "n_measures": len(all_measures),
        "n_controls": len(controls_names),
        "measurements": measurements,
        "controls": controls,
        "loading_mask": jnp.array(loading_mask),
        "prev_measurements": prev_measurements,
        "prev_controls": prev_controls,
        "prev_loading_mask": prev_meas_info["loading_mask"],
        "prev_control_params": prev_meas_info["control_params"],
        "prev_loadings_flat": prev_meas_info["loadings_flat"],
        "prev_meas_sds": prev_meas_info["meas_sds"],
        "prev_distribution": prev_dist_arrays,
        "joint_nodes": joint_nodes,
        "joint_weights": joint_weights,
        "transition_func": combined_transition,
        "total_n_transition_params": total_n_transition_params,
        "total_n_inv_params": total_n_inv_params,
        "n_inv_eq_params_per": n_inv_eq_params_per,
        "observed_factor_values": obs_factor_values,
        "stability_floor": af_options.stability_floor,
        "n_obs_per_batch": n_obs_per_batch,
    }

    loglike_and_grad = create_loglike_and_gradient(
        af_loglike_transition, **loglike_kwargs
    )
    params_array = jnp.array(params_df["value"].to_numpy(dtype=np.float64))
    neg_ll, _grad = loglike_and_grad(params_array)
    return -float(neg_ll)
