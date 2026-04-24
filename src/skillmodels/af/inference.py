"""Asymptotic standard errors for the AF estimator.

Compute the block-diagonal version of the Newey-McFadden sandwich
covariance for a sequential M-estimator:

    V_t = A_tt^{-1} Omega_tt A_tt^{-T} / n

for each period ``t``, where

- ``Omega_tt = (1/n) sum_i g_{ti} g_{ti}^T`` is the outer product of
  period-``t`` per-individual scores (own parameters only).
- ``A_tt`` is the Hessian of the period-``t`` negative-mean
  log-likelihood with respect to its own parameters.

This ignores cross-period terms in ``Omega`` and ``A``, so standard errors
for parameters at period ``t >= 1`` are a **lower bound** on the true
asymptotic SE. They do not propagate plug-in uncertainty from
``theta_{<t}``. See ``docs/superpowers/specs/2026-04-23-af-standard-
errors-design.md`` for the full formulation and the planned cross-period
extension.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.af.batching import auto_n_obs_per_batch
from skillmodels.af.estimate import _extract_period_data
from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.initial_period import (
    _build_loading_mask,
    _get_ordered_measures,
)
from skillmodels.af.likelihood import (
    af_per_obs_loglike_initial,
    af_per_obs_loglike_transition,
)
from skillmodels.af.params import (
    build_optimagic_inputs,
    get_measurements_per_factor,
)
from skillmodels.af.transition_period import (
    _extract_prev_measurement_params,
    _get_raw_transition_functions,
    _prepare_transition_inputs,
)
from skillmodels.af.types import (
    AFEstimationOptions,
    AFEstimationResult,
    ConditionalDistribution,
)
from skillmodels.constraints import FixedConstraintWithValue
from skillmodels.process_model import process_model


@dataclass(frozen=True)
class AFInferenceResult:
    """Asymptotic inference result for the AF estimator."""

    standard_errors: pd.Series
    """Standard errors indexed by ``all_params.index``.

    Fixed-parameter entries are set to zero. Later-period entries use the
    block-diagonal sandwich and are therefore a lower bound on the true
    asymptotic SE (see module docstring).
    """

    vcov: pd.DataFrame
    """Full variance-covariance matrix; rows and columns share
    ``all_params.index``. Off-diagonal cross-period entries are zero in
    the current block-diagonal implementation.
    """

    period_results: tuple[AFPeriodInferenceResult, ...]
    """Per-period inference components, in period order."""


@dataclass(frozen=True)
class AFPeriodInferenceResult:
    """Per-period components of the sandwich inference."""

    period: int
    """Calendar period index."""

    free_param_locs: tuple[tuple[Any, ...], ...]
    """MultiIndex locations of the free (unpinned) parameters used for
    this period's sandwich, in the same order as ``score_matrix`` columns.
    """

    score_matrix: Array
    """Per-observation score matrix, shape ``(n_obs, n_free)``. Row ``i``
    holds ``d log L_{it} / d theta_t`` for individual ``i`` at the
    estimated parameters.
    """

    information_matrix: Array
    """Estimated information matrix ``A_tt``, shape ``(n_free, n_free)``.
    Computed as the Hessian of the scalar negative-mean log-likelihood
    at the estimated parameters.
    """

    score_outer_product: Array
    """Estimated ``Omega_tt = score_matrix.T @ score_matrix / n_obs``,
    shape ``(n_free, n_free)``.
    """

    vcov: Array
    """Period-``t`` own-param variance-covariance matrix, shape
    ``(n_free, n_free)``; equals ``A^{-1} Omega A^{-T} / n_obs``.
    """


def compute_af_standard_errors(
    result: AFEstimationResult,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
) -> AFInferenceResult:
    """Compute asymptotic standard errors for an AF estimate.

    Use the block-diagonal Newey-McFadden sandwich: for each period,
    compute ``V_t = A_tt^{-1} Omega_tt A_tt^{-T} / n`` from own-period
    scores and Hessian. Cross-period terms are ignored; see the module
    docstring.

    Args:
        result: Output of ``estimate_af``.
        data: The dataset used for estimation (long format, same index
            layout as passed to ``estimate_af``).
        af_options: Options used at estimation time. Pass the same
            instance used to fit ``result``; defaults are acceptable if
            options were default at estimation time.

    Return:
        ``AFInferenceResult`` with standard errors, variance-covariance
        matrix, and per-period components.

    """
    if af_options is None:
        af_options = AFEstimationOptions()

    jax.config.update("jax_enable_x64", val=True)

    model_spec = result.model_spec
    processed_model = process_model(model_spec)

    n_periods = processed_model.dimensions.n_periods
    latent_factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    observed_factors = processed_model.labels.observed_factors

    endog_info = processed_model.endogenous_factors_info
    endogenous_factors = tuple(
        f
        for f in latent_factors
        if f in endog_info.factor_info and endog_info.factor_info[f].is_endogenous
    )

    period_data = _extract_period_data(
        data,
        n_periods,
        latent_factors,
        controls_names,
        model_spec,
        observed_factors=observed_factors,
    )

    period_inference: list[AFPeriodInferenceResult] = []
    prev_cond_dists: tuple[ConditionalDistribution | None, ...] = (
        None,
        *result.conditional_distributions[:-1],
    )
    for period_result, prev_cond_dist in zip(
        result.period_results,
        prev_cond_dists,
        strict=False,
    ):
        t = period_result.period
        if t == 0:
            inference = _inference_for_initial_period(
                period_result_params=period_result.params,
                model_spec=model_spec,
                processed_model=processed_model,
                af_options=af_options,
                data_at_period=period_data[0],
                observed_factors=observed_factors,
            )
        else:
            assert prev_cond_dist is not None  # noqa: S101
            prev_period_params = result.period_results[t - 1].params
            inference = _inference_for_transition_period(
                period=t,
                period_result_params=period_result.params,
                prev_period_params=prev_period_params,
                prev_cond_dist=prev_cond_dist,
                model_spec=model_spec,
                processed_model=processed_model,
                af_options=af_options,
                data_at_period=period_data[t],
                prev_data_at_period=period_data[t - 1],
                endogenous_factors=endogenous_factors,
                observed_factors=observed_factors,
            )
        period_inference.append(inference)

    standard_errors, vcov = _assemble_full_vcov(
        result.all_params,
        period_inference,
    )

    return AFInferenceResult(
        standard_errors=standard_errors,
        vcov=vcov,
        period_results=tuple(period_inference),
    )


def _inference_for_initial_period(
    *,
    period_result_params: pd.DataFrame,
    model_spec: Any,  # noqa: ANN401
    processed_model: Any,  # noqa: ANN401
    af_options: AFEstimationOptions,
    data_at_period: Mapping[str, Array],
    observed_factors: tuple[str, ...],
) -> AFPeriodInferenceResult:
    """Compute per-period sandwich for the initial period."""
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    n_components = af_options.n_mixture_components

    reconstructed_factors = tuple(
        f for f in factors if not model_spec.factors[f].has_initial_distribution
    )
    state_latent_factors = tuple(f for f in factors if f not in reconstructed_factors)
    n_state_latent = len(state_latent_factors)
    n_obs_factors = len(observed_factors)
    n_joint = n_state_latent + n_obs_factors

    measurements_p0 = get_measurements_per_factor(model_spec.factors, period=0)
    measurements_p0_filtered = {
        f: m for f, m in measurements_p0.items() if f in state_latent_factors
    }
    all_measures_full = _get_ordered_measures(measurements_p0)
    all_measures = _get_ordered_measures(measurements_p0_filtered)

    measurements = data_at_period["measurements"]
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

    obs_values = data_at_period.get(
        "observed_factors",
        jnp.zeros((int(measurements.shape[0]), 0)),
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
        "controls": data_at_period["controls"],
        "observed_factor_values": obs_values,
        "loading_mask": jnp.array(loading_mask),
        "nodes": nodes,
        "weights": weights,
        "stability_floor": af_options.stability_floor,
        "n_obs_per_batch": n_obs_per_batch,
    }

    return _sandwich_from_loglike(
        params_df=period_result_params,
        period=0,
        per_obs_loglike_fn=af_per_obs_loglike_initial,
        loglike_kwargs=loglike_kwargs,
    )


def _inference_for_transition_period(
    *,
    period: int,
    period_result_params: pd.DataFrame,
    prev_period_params: pd.DataFrame,
    prev_cond_dist: ConditionalDistribution,
    model_spec: Any,  # noqa: ANN401
    processed_model: Any,  # noqa: ANN401
    af_options: AFEstimationOptions,
    data_at_period: Mapping[str, Array],
    prev_data_at_period: Mapping[str, Array],
    endogenous_factors: tuple[str, ...],
    observed_factors: tuple[str, ...],
) -> AFPeriodInferenceResult:
    """Compute per-period sandwich for a transition period."""
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
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

    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    all_measures = _get_ordered_measures(measurements_pt)
    loading_mask = _build_loading_mask(all_measures, factors, measurements_pt)

    joint_dim = n_state + n_shock + n_endog
    joint_nodes, joint_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        joint_dim,
    )

    measurements = data_at_period["measurements"]
    controls = data_at_period["controls"]
    prev_measurements = prev_data_at_period["measurements"]
    prev_controls = prev_data_at_period["controls"]

    prev_dist_arrays, total_n_transition_params = _prepare_transition_inputs(
        prev_cond_dist,
        transition_info,
        state_factors,
        int(measurements.shape[0]),
    )

    raw_funcs = _get_raw_transition_functions(model_spec, state_factors)
    param_counts = tuple(len(transition_info.param_names[f]) for f in state_factors)

    def combined_transition(full_states: Array, params: Array) -> Array:
        result = jnp.zeros(n_state)
        p_idx = 0
        for i in range(n_state):
            n_p = param_counts[i]
            factor_params = params[p_idx : p_idx + n_p]
            result = result.at[i].set(raw_funcs[i](full_states, factor_params))  # noqa: PD008
            p_idx += n_p
        return result

    n_inv_eq_params_per = 1 + n_state + len(observed_factors) if n_endog > 0 else 0
    total_n_inv_params = n_endog * n_inv_eq_params_per

    obs_factor_values = prev_data_at_period.get(
        "observed_factors",
        jnp.zeros((int(measurements.shape[0]), len(observed_factors))),
    )

    prev_meas_info = _extract_prev_measurement_params(
        prev_period_params,
        model_spec,
        factors,
        period - 1,
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

    return _sandwich_from_loglike(
        params_df=period_result_params,
        period=period,
        per_obs_loglike_fn=af_per_obs_loglike_transition,
        loglike_kwargs=loglike_kwargs,
    )


def _sandwich_from_loglike(
    *,
    params_df: pd.DataFrame,
    period: int,
    per_obs_loglike_fn: Callable[..., Array],
    loglike_kwargs: Mapping[str, Any],
) -> AFPeriodInferenceResult:
    """Compute the block-diagonal sandwich for a single period.

    Identify free (unpinned) parameters from ``params_df`` via the same
    logic used at estimation time, then compute the per-obs score
    matrix by ``jax.jacfwd``, the Hessian of the negative-mean
    log-likelihood, and the sandwich ``V = A^{-1} Omega A^{-T} / n``.
    """
    _full_params_df, fixed_constraints = build_optimagic_inputs(params_df, None)
    fixed_locs: set[Any] = set()
    for constraint in fixed_constraints:
        if isinstance(constraint, FixedConstraintWithValue):
            loc = constraint.loc
            fixed_locs.add(tuple(loc) if isinstance(loc, tuple) else loc)
    # Simplex-constrained parameters (mixture_weights) cannot be treated
    # as unconstrained for the sandwich; their Hessian along the simplex
    # direction is degenerate. Drop them from the free set in Phase 1;
    # their SE is reported as zero. A delta-method treatment on
    # reparameterized log-odds is a follow-up.
    all_locs = list(params_df.index)
    free_positions = [
        i
        for i, loc in enumerate(all_locs)
        if tuple(loc) not in fixed_locs and loc[0] != "mixture_weights"
    ]
    free_positions_array = jnp.array(free_positions, dtype=jnp.int32)

    flat_values = jnp.array(params_df["value"].to_numpy())

    def per_obs_loglike_full(flat_params: Array) -> Array:
        return per_obs_loglike_fn(flat_params, **loglike_kwargs)

    def neg_mean_loglike_full(flat_params: Array) -> Array:
        return -jnp.mean(per_obs_loglike_full(flat_params))

    jac_full = jax.jacfwd(per_obs_loglike_full)(flat_values)
    hess_full = jax.hessian(neg_mean_loglike_full)(flat_values)

    score_matrix = jac_full[:, free_positions_array]
    information_matrix = hess_full[free_positions_array][:, free_positions_array]

    n_obs = int(score_matrix.shape[0])
    omega = score_matrix.T @ score_matrix / n_obs

    a_inv = jnp.linalg.inv(information_matrix)
    vcov_period = a_inv @ omega @ a_inv.T / n_obs

    return AFPeriodInferenceResult(
        period=period,
        free_param_locs=tuple(tuple(all_locs[i]) for i in free_positions),
        score_matrix=score_matrix,
        information_matrix=information_matrix,
        score_outer_product=omega,
        vcov=vcov_period,
    )


def _assemble_full_vcov(
    all_params: pd.DataFrame,
    period_inference: list[AFPeriodInferenceResult],
) -> tuple[pd.Series, pd.DataFrame]:
    """Assemble per-period variance-covariance blocks onto the full params index.

    Returns:
        Tuple ``(standard_errors, vcov)``. ``standard_errors`` is a
        Series indexed by ``all_params.index``; fixed entries are zero.
        ``vcov`` is a square DataFrame with the same index on rows and
        columns.

    """
    index = all_params.index
    size = len(index)

    vcov_values = np.zeros((size, size))
    pos_lookup = {tuple(loc): i for i, loc in enumerate(index)}

    for period_res in period_inference:
        block_vcov = np.array(period_res.vcov)
        positions = [pos_lookup[loc] for loc in period_res.free_param_locs]
        positions_arr = np.array(positions, dtype=np.int64)
        vcov_values[positions_arr[:, None], positions_arr[None, :]] = block_vcov

    standard_errors = pd.Series(
        np.sqrt(np.clip(np.diag(vcov_values), 0.0, None)),
        index=index,
        name="standard_error",
    )
    vcov_df = pd.DataFrame(vcov_values, index=index, columns=index)
    return standard_errors, vcov_df


__all__ = [
    "AFInferenceResult",
    "AFPeriodInferenceResult",
    "compute_af_standard_errors",
]
