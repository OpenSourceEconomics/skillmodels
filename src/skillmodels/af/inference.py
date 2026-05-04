"""Asymptotic standard errors for the AF estimator.

Implement the Newey-McFadden (1994, ch. 6) sandwich covariance for a
sequential M-estimator. Let ``theta = (theta_0, ..., theta_{T-1})`` be
the stacked parameter vector and let

    g_{ti}(theta) = d log L_{it} / d theta_t

be individual ``i``'s period-``t`` own-parameter score. Stack per
individual: ``g_i in R^{P_total}``. Then

    Omega_{ts} = (1/n) sum_i g_{ti} g_{si}^T
    A_{ts}     = (1/n) sum_i d g_{ti} / d theta_s
    V_hat      = A^{-1} Omega A^{-T} / n_obs

``A`` is block lower triangular because period ``t``'s likelihood does
not depend on ``theta_{>t}``. The off-diagonal blocks of ``A`` and
``Omega`` are what make this sandwich differ from the naive
per-period block-diagonal version — they propagate the plug-in
uncertainty from earlier periods.

Two computation modes:

- ``method="full_sandwich"`` (default): compute the full cross-period
  sandwich by reconstructing ``prev_distribution`` and
  ``prev_meas_info`` as JAX-differentiable functions of earlier-period
  parameters. Asymptotically correct for the AF sequential estimator.
- ``method="block_diagonal"``: compute only the diagonal blocks
  ``V_t = A_tt^{-1} Omega_tt A_tt^{-T} / n_obs``. Cheaper, but SEs for
  periods ``t >= 1`` are a lower bound on the true asymptotic SE.

Memory: the Hessian is computed via ``jax.hessian`` (forward-over-reverse).
The ``n_obs_per_batch`` memory contract that ``_map_over_obs`` promises
for a single reverse-mode pass does NOT bound the Hessian tape: the outer
jacobian materialises the full gradient of length ``n_obs``, so peak
memory scales with ``n_params * n_obs`` regardless of ``n_obs_per_batch``.
For very large models the Hessian path may OOM where estimation did not;
switch to ``method="block_diagonal"`` or reduce ``n_halton_points`` to
mitigate.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

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
    _parse_initial_params,
    _parse_transition_params,
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
from skillmodels.model_spec import ModelSpec
from skillmodels.process_model import process_model
from skillmodels.types import ProcessedModel


@dataclass(frozen=True)
class AFInferenceResult:
    """Asymptotic inference result for the AF estimator."""

    standard_errors: pd.Series
    """Standard errors indexed by ``all_params.index``.

    Fixed-parameter entries are set to zero. In ``block_diagonal`` mode,
    period-``t`` entries for ``t >= 1`` are a lower bound on the true
    asymptotic SE; in ``full_sandwich`` mode they are asymptotically
    correct.
    """

    vcov: pd.DataFrame
    """Full variance-covariance matrix; rows and columns share
    ``all_params.index``. In ``block_diagonal`` mode off-diagonal
    cross-period entries are zero; in ``full_sandwich`` they are the
    actual cross-period covariances.
    """

    period_results: tuple[AFPeriodInferenceResult, ...]
    """Per-period inference components, in period order."""

    method: str
    """Which method produced the result (``"full_sandwich"`` or
    ``"block_diagonal"``).
    """


@dataclass(frozen=True)
class AFPeriodInferenceResult:
    """Per-period components of the sandwich inference."""

    period: int
    """Calendar period index."""

    free_param_locs: tuple[tuple[Any, ...], ...]
    """MultiIndex locations of the free (unpinned, non-simplex) parameters
    used for this period's own-param score columns, in the same order as
    ``score_matrix`` columns.
    """

    score_matrix: Array
    """Per-observation own-parameter score matrix, shape
    ``(n_obs, n_free_own)``. Row ``i`` holds
    ``d log L_{it} / d theta_t`` for individual ``i`` at the estimated
    parameters.
    """

    information_matrix: Array
    """Estimated diagonal-block information matrix ``A_tt``,
    shape ``(n_free_own, n_free_own)``. Hessian of the scalar negative
    mean log-likelihood restricted to period-``t`` own parameters.
    """

    score_outer_product: Array
    """Estimated ``Omega_tt = score_matrix.T @ score_matrix / n_obs``,
    shape ``(n_free_own, n_free_own)``.
    """

    vcov: Array
    """Own-parameter block of the variance-covariance matrix,
    shape ``(n_free_own, n_free_own)``. In ``block_diagonal`` mode
    this equals ``A_tt^{-1} Omega_tt A_tt^{-T} / n_obs``; in
    ``full_sandwich`` it is the corresponding diagonal block of the
    full sandwich (which also accounts for cross-period uncertainty).
    """


def compute_af_standard_errors(
    result: AFEstimationResult,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
    method: Literal["full_sandwich", "block_diagonal"] = "full_sandwich",
) -> AFInferenceResult:
    """Compute asymptotic standard errors for an AF estimate.

    Args:
        result: Output of ``estimate_af``.
        data: The dataset used for estimation (long format, same index
            layout as passed to ``estimate_af``).
        af_options: Options used at estimation time. Pass the same
            instance used to fit ``result``; defaults are acceptable if
            options were default at estimation time.
        method: ``"full_sandwich"`` computes the asymptotically correct
            Newey-McFadden sandwich, propagating plug-in uncertainty
            through the ``prev_distribution`` and ``prev_meas_info``
            chain. ``"block_diagonal"`` computes only the diagonal
            blocks and is faster but underestimates SEs for periods
            ``t >= 1``.

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

    metas = _build_period_metas(
        result=result,
        period_data=period_data,
        model_spec=model_spec,
        processed_model=processed_model,
        af_options=af_options,
        observed_factors=observed_factors,
        endogenous_factors=endogenous_factors,
    )

    full_free_block: _FreeVcovBlock | None
    if method == "block_diagonal":
        period_inference = _compute_block_diagonal_sandwich(result, metas)
        full_free_block = None
    elif method == "full_sandwich":
        period_inference, full_free_block = _compute_full_sandwich(result, metas)
    else:
        msg = f"Unknown method: {method!r}"
        raise ValueError(msg)

    standard_errors, vcov = _assemble_full_vcov(
        result.all_params,
        period_inference,
        full_free_block=full_free_block,
    )

    return AFInferenceResult(
        standard_errors=standard_errors,
        vcov=vcov,
        period_results=tuple(period_inference),
        method=method,
    )


@dataclass(frozen=True)
class _FreeVcovBlock:
    """Internal carrier for the full cross-period free-parameter vcov."""

    free_param_locs: tuple[tuple[Any, ...], ...]
    vcov: Array


# ---------------------------------------------------------------------------
# Period metadata: all the static info we need for both sandwich modes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PeriodMeta:
    """Precomputed static metadata for one period's likelihood.

    Pure-Python dataclass; JAX arrays live in ``loglike_kwargs`` and
    ``propagation``.
    """

    period: int
    is_initial: bool
    slice_start: int
    slice_stop: int
    params_df: pd.DataFrame
    loglike_kwargs: MappingProxyType[str, Any]
    """Keyword arguments forwarded to ``af_per_obs_loglike_initial`` (if
    ``is_initial``) or ``af_per_obs_loglike_transition`` otherwise.
    """
    parse_kwargs: MappingProxyType[str, Any]
    """Keyword arguments forwarded to ``_parse_initial_params`` or
    ``_parse_transition_params`` respectively. Used by the Phase 2 chain.
    """
    n_components: int
    n_factors_joint: int
    """Joint factor count in the initial mixture (state_latent + observed).
    Only meaningful for the initial period; zero otherwise.
    """
    n_state: int
    """State-factor count (``n_state_latent`` in the initial period;
    ``n_state_factors`` in transition periods).
    """
    n_endog: int
    n_shock: int
    n_observed_factors: int
    state_factor_indices_in_joint: tuple[int, ...]
    """Integer positions within the joint factor vector at which state
    factors live (the complement is observed factors). Used to marginalise
    the joint cond-dist to its state-factor sub-block.
    """
    propagation: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Extra JAX-pure bits for propagation of the conditional distribution
    through this period's transition. Only populated for transition
    periods. Keys: ``state_nodes``, ``state_weights``,
    ``combined_transition``, ``obs_factor_values``.
    """


def _build_period_metas(
    *,
    result: AFEstimationResult,
    period_data: dict[int, dict[str, Array]],
    model_spec: ModelSpec,
    processed_model: ProcessedModel,
    af_options: AFEstimationOptions,
    observed_factors: tuple[str, ...],
    endogenous_factors: tuple[str, ...],
) -> tuple[_PeriodMeta, ...]:
    """Build per-period metadata objects for both inference modes."""
    metas: list[_PeriodMeta] = []
    offset = 0
    for period_result in result.period_results:
        t = period_result.period
        params_df = period_result.params
        length = len(params_df)

        if t == 0:
            meta = _build_initial_period_meta(
                period_result_params=params_df,
                slice_start=offset,
                slice_stop=offset + length,
                model_spec=model_spec,
                processed_model=processed_model,
                af_options=af_options,
                data_at_period=period_data[0],
                observed_factors=observed_factors,
            )
        else:
            prev_period_params = result.period_results[t - 1].params
            prev_cond_dist = result.conditional_distributions[t - 1]
            meta = _build_transition_period_meta(
                period=t,
                period_result_params=params_df,
                slice_start=offset,
                slice_stop=offset + length,
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
        metas.append(meta)
        offset += length
    return tuple(metas)


def _build_initial_period_meta(
    *,
    period_result_params: pd.DataFrame,
    slice_start: int,
    slice_stop: int,
    model_spec: ModelSpec,
    processed_model: ProcessedModel,
    af_options: AFEstimationOptions,
    data_at_period: Mapping[str, Array],
    observed_factors: tuple[str, ...],
) -> _PeriodMeta:
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
    state_factor_indices_in_joint = tuple(range(n_state_latent))

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
        af_options.n_halton_points, n_state_latent
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

    parse_kwargs = {
        "n_factors": n_joint,
        "n_mixture_components": n_components,
        "n_measures": len(all_measures),
        "n_controls": len(controls_names),
    }

    return _PeriodMeta(
        period=0,
        is_initial=True,
        slice_start=slice_start,
        slice_stop=slice_stop,
        params_df=period_result_params,
        loglike_kwargs=MappingProxyType(loglike_kwargs),
        parse_kwargs=MappingProxyType(parse_kwargs),
        n_components=n_components,
        n_factors_joint=n_joint,
        n_state=n_state_latent,
        n_endog=0,
        n_shock=0,
        n_observed_factors=n_obs_factors,
        state_factor_indices_in_joint=state_factor_indices_in_joint,
        propagation=MappingProxyType({}),
    )


def _build_transition_period_meta(
    *,
    period: int,
    period_result_params: pd.DataFrame,
    slice_start: int,
    slice_stop: int,
    prev_period_params: pd.DataFrame,
    prev_cond_dist: ConditionalDistribution,
    model_spec: ModelSpec,
    processed_model: ProcessedModel,
    af_options: AFEstimationOptions,
    data_at_period: Mapping[str, Array],
    prev_data_at_period: Mapping[str, Array],
    endogenous_factors: tuple[str, ...],
    observed_factors: tuple[str, ...],
) -> _PeriodMeta:
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
        af_options.n_halton_points, joint_dim
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
        out = jnp.zeros(n_state)
        p_idx = 0
        for i in range(n_state):
            n_p = param_counts[i]
            factor_params = params[p_idx : p_idx + n_p]
            out = out.at[i].set(raw_funcs[i](full_states, factor_params))  # noqa: PD008
            p_idx += n_p
        return out

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

    parse_kwargs = {
        "n_state_factors": n_state,
        "n_endogenous_factors": n_endog,
        "n_measures": len(all_measures),
        "n_controls": len(controls_names),
        "total_n_transition_params": total_n_transition_params,
        "total_n_inv_params": total_n_inv_params,
        "n_inv_eq_params_per": n_inv_eq_params_per,
        "n_shock_factors": n_shock,
    }

    # For propagating the cond-dist forward to the next period: marginal
    # state grid (same convention as ``_update_conditional_distribution``).
    propagation_nodes, propagation_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points, n_state
    )

    propagation = {
        "state_nodes": propagation_nodes,
        "state_weights": propagation_weights,
        "combined_transition": combined_transition,
        "obs_factor_values": obs_factor_values,
        "shock_factor_indices": shock_factor_indices,
    }

    return _PeriodMeta(
        period=period,
        is_initial=False,
        slice_start=slice_start,
        slice_stop=slice_stop,
        params_df=period_result_params,
        loglike_kwargs=MappingProxyType(loglike_kwargs),
        parse_kwargs=MappingProxyType(parse_kwargs),
        n_components=len(prev_cond_dist.components),
        n_factors_joint=0,
        n_state=n_state,
        n_endog=n_endog,
        n_shock=n_shock,
        n_observed_factors=len(observed_factors),
        state_factor_indices_in_joint=tuple(range(n_state)),
        propagation=MappingProxyType(propagation),
    )


# ---------------------------------------------------------------------------
# Free-parameter bookkeeping.
# ---------------------------------------------------------------------------


def _free_positions_for_period(
    params_df: pd.DataFrame,
) -> tuple[list[int], list[tuple[Any, ...]]]:
    """Return positions and locs of free (unpinned, non-simplex) params."""
    _, fixed_constraints = build_optimagic_inputs(params_df, None)
    fixed_locs: set[Any] = set()
    for constraint in fixed_constraints:
        if isinstance(constraint, FixedConstraintWithValue):
            loc = constraint.loc
            fixed_locs.add(tuple(loc) if isinstance(loc, tuple) else loc)

    all_locs = list(params_df.index)
    positions: list[int] = []
    locs: list[tuple[Any, ...]] = []
    for i, loc in enumerate(all_locs):
        loc_t = tuple(loc)
        if loc_t in fixed_locs or loc[0] == "mixture_weights":
            continue
        positions.append(i)
        locs.append(loc_t)
    return positions, locs


# ---------------------------------------------------------------------------
# Block-diagonal sandwich (Phase 1 behaviour).
# ---------------------------------------------------------------------------


def _compute_block_diagonal_sandwich(
    _result: AFEstimationResult,
    metas: tuple[_PeriodMeta, ...],
) -> list[AFPeriodInferenceResult]:
    """Compute per-period block-diagonal sandwich ignoring cross-period terms."""
    results: list[AFPeriodInferenceResult] = []
    for meta in metas:
        per_obs_fn = (
            af_per_obs_loglike_initial
            if meta.is_initial
            else af_per_obs_loglike_transition
        )
        inference = _block_diagonal_sandwich_single(
            meta=meta,
            per_obs_loglike_fn=per_obs_fn,
        )
        results.append(inference)
    return results


def _block_diagonal_sandwich_single(
    *,
    meta: _PeriodMeta,
    per_obs_loglike_fn: Callable[..., Array],
) -> AFPeriodInferenceResult:
    """Compute V_t = A_tt^{-1} Omega_tt A_tt^{-T} / n for one period only."""
    positions, locs = _free_positions_for_period(meta.params_df)
    free_positions_array = jnp.array(positions, dtype=jnp.int32)
    flat_values = jnp.array(meta.params_df["value"].to_numpy())
    kwargs = dict(meta.loglike_kwargs)

    def per_obs_loglike_full(flat_params: Array) -> Array:
        return per_obs_loglike_fn(flat_params, **kwargs)

    def neg_mean_loglike_full(flat_params: Array) -> Array:
        return -jnp.mean(per_obs_loglike_full(flat_params))

    jac_full = jax.jacfwd(per_obs_loglike_full)(flat_values)
    hess_full = jax.hessian(neg_mean_loglike_full)(flat_values)

    score_matrix = jac_full[:, free_positions_array]
    information_matrix = hess_full[free_positions_array][:, free_positions_array]
    n_obs = int(score_matrix.shape[0])
    omega = score_matrix.T @ score_matrix / n_obs
    # Use the Moore-Penrose pseudoinverse: the user's `fixed_params` argument
    # to `estimate_af` pins parameter values via FixedConstraintWithValue, but
    # the bounds-relaxation in `build_optimagic_inputs` strips those rows of
    # their lb==ub markers, so `_free_positions_for_period` cannot detect them
    # here. The resulting information matrix is rank-deficient (zero rows on
    # the pinned coordinates), and `inv` produces NaN that propagates to every
    # diagonal entry of the vcov. `pinv` returns zero on the null-space
    # directions instead, so identifiable parameters retain their correct SE
    # while pinned parameters get SE 0 (rendered as "—" by downstream display).
    a_inv = jnp.linalg.pinv(information_matrix, hermitian=True)
    vcov_period = a_inv @ omega @ a_inv.T / n_obs

    return AFPeriodInferenceResult(
        period=meta.period,
        free_param_locs=tuple(locs),
        score_matrix=score_matrix,
        information_matrix=information_matrix,
        score_outer_product=omega,
        vcov=vcov_period,
    )


# ---------------------------------------------------------------------------
# Full cross-period sandwich (Phase 2).
#
# Reconstruct ``prev_distribution`` and ``prev_meas_info`` as JAX-pure
# functions of a single concatenated ``flat_super`` parameter vector, so
# ``jax.jacfwd`` captures the full chain of dependencies.
# ---------------------------------------------------------------------------


def _build_initial_state_cond_dist_jax(
    flat_params_0: Array,
    meta: _PeriodMeta,
) -> tuple[Array, Array, Array]:
    """JAX-pure state-factor marginal of the initial conditional dist.

    Returns ``(state_means, state_chols, mixture_weights)``.
    """
    parsed = _parse_initial_params(
        flat_params_0,
        meta.parse_kwargs["n_factors"],
        meta.parse_kwargs["n_mixture_components"],
        meta.parse_kwargs["n_measures"],
        meta.parse_kwargs["n_controls"],
    )
    joint_means = parsed["mixture_means"]
    joint_chols = parsed["mixture_chol_covs"]
    mixture_weights = parsed["mixture_weights"]

    if meta.n_state == meta.n_factors_joint:
        return joint_means, joint_chols, mixture_weights

    state_idx = jnp.asarray(meta.state_factor_indices_in_joint, dtype=jnp.int32)
    joint_covs = joint_chols @ jnp.swapaxes(joint_chols, -1, -2)
    sub_covs = joint_covs[:, state_idx[:, None], state_idx[None, :]]
    state_chols = jnp.linalg.cholesky(sub_covs + 1e-10 * jnp.eye(meta.n_state))
    state_means = joint_means[:, state_idx]
    return state_means, state_chols, mixture_weights


def _propagate_cond_dist_jax(
    prev_means: Array,
    prev_chols: Array,
    flat_params_t: Array,
    meta: _PeriodMeta,
) -> tuple[Array, Array]:
    """Propagate a mixture through period ``t``'s transition.

    Mirrors the estimation-time logic of ``_update_conditional_distribution``
    and ``_compute_mean_investment`` but operates purely on JAX arrays.
    """
    parsed = _parse_transition_params(
        flat_params_t,
        meta.parse_kwargs["n_state_factors"],
        meta.parse_kwargs["n_endogenous_factors"],
        meta.parse_kwargs["n_measures"],
        meta.parse_kwargs["n_controls"],
        meta.parse_kwargs["total_n_transition_params"],
        meta.parse_kwargs["total_n_inv_params"],
        meta.parse_kwargs["n_inv_eq_params_per"],
        n_shock_factors=meta.parse_kwargs["n_shock_factors"],
    )
    trans_params = parsed["transition_params"]
    shock_sds = parsed["shock_sds"]
    inv_eq_params = parsed["inv_eq_params"]

    n_endog = meta.n_endog
    n_state = meta.n_state
    n_obs_factors = meta.n_observed_factors
    n_per = 1 + n_state + n_obs_factors if n_endog > 0 else 0

    obs_values = meta.propagation["obs_factor_values"]
    obs_mean = (
        jnp.mean(obs_values, axis=0)
        if obs_values.shape[0] > 0
        else jnp.zeros(n_obs_factors)
    )

    prior_mean_first = prev_means[0]
    if n_endog == 0:
        mean_inv = jnp.zeros(0)
    else:
        beta_matrix = inv_eq_params.reshape(n_endog, n_per)
        state_part = beta_matrix[:, 1 : 1 + n_state] @ prior_mean_first
        obs_part = (
            beta_matrix[:, 1 + n_state :] @ obs_mean
            if n_obs_factors > 0
            else jnp.zeros(n_endog)
        )
        mean_inv = beta_matrix[:, 0] + state_part + obs_part

    combined_transition = meta.propagation["combined_transition"]
    state_nodes = meta.propagation["state_nodes"]
    state_weights = meta.propagation["state_weights"]
    shock_factor_indices = meta.propagation["shock_factor_indices"]

    shock_diag = (
        jnp.zeros(n_state).at[shock_factor_indices].set(shock_sds**2)  # noqa: PD008
    )

    def state_only_transition(state_vals: Array, trans_p: Array) -> Array:
        full = jnp.concatenate([state_vals, mean_inv, obs_mean])
        return combined_transition(full, trans_p)

    def per_component(mean_k: Array, chol_k: Array) -> tuple[Array, Array]:
        theta_samples = mean_k[None, :] + state_nodes @ chol_k.T
        propagated = jax.vmap(state_only_transition, in_axes=(0, None))(
            theta_samples, trans_params
        )
        new_mean = jnp.sum(state_weights[:, None] * propagated, axis=0)
        centered = propagated - new_mean[None, :]
        new_cov = jnp.einsum(
            "q,qi,qj->ij", state_weights, centered, centered
        ) + jnp.diag(shock_diag)
        new_chol = jnp.linalg.cholesky(new_cov + 1e-8 * jnp.eye(n_state))
        return new_mean, new_chol

    new_means, new_chols = jax.vmap(per_component)(prev_means, prev_chols)
    return new_means, new_chols


def _extract_prev_meas_info_jax(
    flat_params_prev: Array,
    meta: _PeriodMeta,
) -> dict[str, Array]:
    """JAX-pure extraction of ``prev_meas_info`` from a period's flat params."""
    if meta.is_initial:
        parsed = _parse_initial_params(
            flat_params_prev,
            meta.parse_kwargs["n_factors"],
            meta.parse_kwargs["n_mixture_components"],
            meta.parse_kwargs["n_measures"],
            meta.parse_kwargs["n_controls"],
        )
        return {
            "loadings_flat": parsed["loadings"],
            "control_params": parsed["control_params"],
            "meas_sds": parsed["meas_sds"],
        }
    parsed = _parse_transition_params(
        flat_params_prev,
        meta.parse_kwargs["n_state_factors"],
        meta.parse_kwargs["n_endogenous_factors"],
        meta.parse_kwargs["n_measures"],
        meta.parse_kwargs["n_controls"],
        meta.parse_kwargs["total_n_transition_params"],
        meta.parse_kwargs["total_n_inv_params"],
        meta.parse_kwargs["n_inv_eq_params_per"],
        n_shock_factors=meta.parse_kwargs["n_shock_factors"],
    )
    return {
        "loadings_flat": parsed["loadings_flat"],
        "control_params": parsed["control_params"],
        "meas_sds": parsed["meas_sds"],
    }


def _build_prev_dist_arrays(
    flat_super: Array,
    target_t: int,
    metas: tuple[_PeriodMeta, ...],
    cond_weights_override: Array | None = None,
) -> dict[str, Array]:
    """Chain period 0 -> ... -> t-1 to produce prev_dist_arrays for period t.

    When the propagated distribution carries individual-level
    ``conditional_weights`` (e.g. posterior weights from a Bayes update),
    pass them via ``cond_weights_override`` — otherwise the chain falls
    back to the mixture-weights broadcast, which matches the estimation
    path's default in ``_prepare_transition_inputs``.
    """
    meta0 = metas[0]
    flat_params_0 = flat_super[meta0.slice_start : meta0.slice_stop]
    state_means, state_chols, mixture_weights = _build_initial_state_cond_dist_jax(
        flat_params_0, meta0
    )

    for s in range(1, target_t):
        meta_s = metas[s]
        flat_params_s = flat_super[meta_s.slice_start : meta_s.slice_stop]
        state_means, state_chols = _propagate_cond_dist_jax(
            state_means, state_chols, flat_params_s, meta_s
        )

    if cond_weights_override is not None:
        cond_weights = cond_weights_override
    else:
        meta_target = metas[target_t]
        n_obs = int(meta_target.loglike_kwargs["measurements"].shape[0])
        n_components = metas[0].n_components
        cond_weights = jnp.broadcast_to(mixture_weights[None, :], (n_obs, n_components))
    return {
        "cond_weights": cond_weights,
        "means": state_means,
        "chol_covs": state_chols,
    }


def _period_t_per_obs_loglike_full(
    flat_super: Array,
    t: int,
    metas: tuple[_PeriodMeta, ...],
) -> Array:
    """Per-obs loglike for period ``t`` as a function of the full flat vector."""
    meta_t = metas[t]
    flat_params_t = flat_super[meta_t.slice_start : meta_t.slice_stop]
    if meta_t.is_initial:
        return af_per_obs_loglike_initial(flat_params_t, **meta_t.loglike_kwargs)

    # Reuse the baked cond_weights from the meta (it was built via the same
    # ``_prepare_transition_inputs`` path as estimation and already honours
    # any stored ``conditional_weights``; when ``conditional_weights`` is
    # ``None`` it is a broadcast of the initial-period mixture weights).
    stored_cond_weights = meta_t.loglike_kwargs["prev_distribution"]["cond_weights"]
    prev_dist_arrays = _build_prev_dist_arrays(
        flat_super, t, metas, cond_weights_override=stored_cond_weights
    )
    meta_prev = metas[t - 1]
    flat_params_prev = flat_super[meta_prev.slice_start : meta_prev.slice_stop]
    prev_meas = _extract_prev_meas_info_jax(flat_params_prev, meta_prev)

    kwargs = dict(meta_t.loglike_kwargs)
    kwargs["prev_distribution"] = prev_dist_arrays
    kwargs["prev_loadings_flat"] = prev_meas["loadings_flat"]
    kwargs["prev_control_params"] = prev_meas["control_params"]
    kwargs["prev_meas_sds"] = prev_meas["meas_sds"]
    return af_per_obs_loglike_transition(flat_params_t, **kwargs)


def _compute_full_sandwich(
    result: AFEstimationResult,
    metas: tuple[_PeriodMeta, ...],
) -> tuple[list[AFPeriodInferenceResult], _FreeVcovBlock]:
    """Compute the full cross-period Newey-McFadden sandwich."""
    # Concatenated estimated parameter vector.
    flat_super = jnp.concatenate(
        [jnp.array(pr.params["value"].to_numpy()) for pr in result.period_results]
    )
    p_total = int(flat_super.shape[0])

    # Free-positions global to flat_super, plus per-period own-param positions.
    free_positions_global: list[int] = []
    period_own_global: list[jnp.ndarray] = []
    period_locs: list[tuple[tuple[Any, ...], ...]] = []
    for meta in metas:
        positions, locs = _free_positions_for_period(meta.params_df)
        global_positions = [meta.slice_start + p for p in positions]
        free_positions_global.extend(global_positions)
        period_own_global.append(jnp.array(global_positions, dtype=jnp.int32))
        period_locs.append(tuple(locs))
    free_positions_array = jnp.array(free_positions_global, dtype=jnp.int32)

    # Per-period full Jacobians and own-period score blocks.
    score_matrices_full: list[Array] = []  # (n_obs_t, p_total) each
    hessian_blocks_full: list[Array] = []  # (p_total, p_total) each

    for t, _ in enumerate(metas):

        def _per_obs_t(fs: Array, t_fixed: int = t) -> Array:
            return _period_t_per_obs_loglike_full(fs, t_fixed, metas)

        def _neg_mean_t(fs: Array, t_fixed: int = t) -> Array:
            return -jnp.mean(_per_obs_t(fs, t_fixed))

        score_matrices_full.append(jax.jacfwd(_per_obs_t)(flat_super))
        hessian_blocks_full.append(jax.hessian(_neg_mean_t)(flat_super))

    # Assemble Omega: stacked per-individual score has non-zero entries only
    # in each period's own-parameter columns. Accumulate
    # G = sum_t indicator_cols * S_t, then Omega = G.T G / n_obs.
    # Panel is assumed balanced; we use the n_obs of period 0.
    n_obs = int(metas[0].loglike_kwargs["measurements"].shape[0])
    stacked_scores = jnp.zeros((n_obs, p_total))
    for t, own_idx in enumerate(period_own_global):
        stacked_scores = stacked_scores.at[:, own_idx].add(  # noqa: PD008
            score_matrices_full[t][:, own_idx]
        )
    omega_full = stacked_scores.T @ stacked_scores / n_obs

    # Assemble A: row-block t gets the Hessian's own-param rows.
    a_full = jnp.zeros((p_total, p_total))
    for t, own_idx in enumerate(period_own_global):
        a_full = a_full.at[own_idx, :].set(  # noqa: PD008
            hessian_blocks_full[t][own_idx, :]
        )

    # Restrict to free positions only.
    omega_free = omega_full[free_positions_array][:, free_positions_array]
    a_free = a_full[free_positions_array][:, free_positions_array]

    # See comment on `pinv` in `_block_diagonal_sandwich_single`: the user's
    # `fixed_params` are stripped of their lb==ub markers in
    # `build_optimagic_inputs`, so the free-position set unavoidably contains
    # rows for pinned parameters whose Hessian rows are zero. `pinv` keeps the
    # vcov finite by zeroing out the null-space directions instead of
    # propagating NaN through `inv`.
    # Unlike the block-diagonal case, `a_free` here is *not* symmetric:
    # period-t rows are drawn from period-t's Hessian, which has zero
    # entries in later-period columns but non-zero entries in earlier
    # ones (period-t LL depends on period-(t-1) params via the
    # propagated conditional distribution). So we must NOT pass
    # `hermitian=True`, which would route through `eigh` and silently
    # symmetrise the input.
    a_inv = jnp.linalg.pinv(a_free)
    v_free = a_inv @ omega_free @ a_inv.T / n_obs

    # Build per-period inference results, restoring the block-diagonal
    # components that users commonly inspect.
    results: list[AFPeriodInferenceResult] = []
    cumulative_own_in_free = 0
    v_free_np = np.array(v_free)
    stacked_np = np.array(stacked_scores)
    a_full_np = np.array(a_full)
    for t, meta in enumerate(metas):
        own_global = np.array(period_own_global[t])
        n_own = int(own_global.shape[0])
        # Where are these own params in the free array?
        own_in_free_slice = slice(
            cumulative_own_in_free, cumulative_own_in_free + n_own
        )
        cumulative_own_in_free += n_own
        vcov_block = v_free_np[own_in_free_slice, own_in_free_slice]
        score_block = stacked_np[:, own_global]
        info_block = a_full_np[np.ix_(own_global, own_global)]
        omega_block = score_block.T @ score_block / n_obs
        results.append(
            AFPeriodInferenceResult(
                period=meta.period,
                free_param_locs=period_locs[t],
                score_matrix=jnp.asarray(score_block),
                information_matrix=jnp.asarray(info_block),
                score_outer_product=jnp.asarray(omega_block),
                vcov=jnp.asarray(vcov_block),
            )
        )

    full_free_block = _FreeVcovBlock(
        free_param_locs=tuple(loc for locs in period_locs for loc in locs),
        vcov=v_free,
    )
    return results, full_free_block


# ---------------------------------------------------------------------------
# Assembly back onto the params MultiIndex.
# ---------------------------------------------------------------------------


def _assemble_full_vcov(
    all_params: pd.DataFrame,
    period_inference: list[AFPeriodInferenceResult],
    full_free_block: _FreeVcovBlock | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Assemble per-period (and possibly full cross-period) vcov onto params index.

    When ``full_free_block`` is provided, the cross-period free-parameter
    vcov is written in first (so off-diagonal entries come from the full
    sandwich). Otherwise the per-period block-diagonal entries are used.
    """
    index = all_params.index
    size = len(index)

    vcov_values = np.zeros((size, size))
    pos_lookup = {tuple(loc): i for i, loc in enumerate(index)}

    if full_free_block is not None:
        block_vcov = np.array(full_free_block.vcov)
        positions = [pos_lookup[loc] for loc in full_free_block.free_param_locs]
        positions_arr = np.array(positions, dtype=np.int64)
        vcov_values[positions_arr[:, None], positions_arr[None, :]] = block_vcov
    else:
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


@dataclass(frozen=True)
class AFBootstrapResult:
    """Score-resampling bootstrap result for the AF estimator."""

    standard_errors: pd.Series
    """Bootstrap standard errors indexed by ``all_params.index``.

    SEs are the empirical standard deviation across bootstrap replicates
    of each parameter's one-step Newton shift from the point estimate.
    Fixed-parameter and constrained-direction entries are reported as
    zero (or NaN where the period's information matrix is singular on
    that direction).
    """

    replicate_params: pd.DataFrame
    """``(n_boot, n_params)`` DataFrame of bootstrap parameter draws.

    Each row is ``theta_hat + delta_b`` where ``delta_b = -A^{-1} *
    bar_g_b``, ``bar_g_b`` is the mean per-cluster score in bootstrap
    replicate ``b``, and ``A`` is the period's information matrix at
    the optimum. Columns share ``all_params.index``; pinned-parameter
    columns are constant at the point estimate.
    """

    n_clusters: int
    """Number of caseids resampled per replicate (= number of unique
    caseids in the data).
    """

    n_boot: int
    """Number of bootstrap replicates drawn."""


def compute_af_bootstrap_se(
    result: AFEstimationResult,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> AFBootstrapResult:
    """Score-resampling cluster bootstrap for the AF estimator.

    Computes per-observation scores once at the point estimate, then for
    each replicate resamples caseids with replacement, averages their
    scores, and applies a one-step Newton update from the optimum:

        theta_b = theta_hat - A_t^{-1} * bar_g_b

    where ``A_t`` is the period-``t`` information matrix (same one used
    by ``compute_af_standard_errors(method="block_diagonal")``) and
    ``bar_g_b`` is the bootstrap-averaged per-obs score restricted to
    period-``t`` free parameters. Each AF period is resampled
    independently — the same caseids would be redrawn jointly, but the
    block-diagonal information matrix makes the periods' shifts
    decouple, and we report only own-block bootstrap SEs.

    This is the "score bootstrap" of e.g. Kline & Santos (2012); it
    avoids re-estimating the model B times. For ``B = 10000`` and
    ``n_caseids = 1500``, the bootstrap step takes seconds rather than
    days.

    Args:
        result: Output of ``estimate_af``.
        data: The dataset used for estimation; the caseid level of its
            MultiIndex defines the bootstrap clusters.
        af_options: Options used at estimation time.
        n_boot: Number of bootstrap replicates.
        seed: Seed for the resampling RNG.

    Return:
        ``AFBootstrapResult`` with bootstrap SEs (per-period block) and
        the full replicate-by-parameter DataFrame.

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

    metas = _build_period_metas(
        result=result,
        period_data=period_data,
        model_spec=model_spec,
        processed_model=processed_model,
        af_options=af_options,
        observed_factors=observed_factors,
        endogenous_factors=endogenous_factors,
    )

    # Use the existing block-diagonal scaffolding to get per-period score
    # matrices and information matrices at the optimum.
    period_inference = _compute_block_diagonal_sandwich(result, metas)

    # Resample once per period: each AF period sees one observation per
    # caseid, so caseid-level resampling reduces to row-level resampling
    # of the (n_caseids, n_free_params) score matrix.
    rng = np.random.default_rng(seed)
    all_params = result.all_params
    replicate_values = np.tile(all_params["value"].to_numpy()[None, :], (n_boot, 1))

    pos_lookup = {tuple(loc): i for i, loc in enumerate(all_params.index)}

    n_clusters = int(metas[0].loglike_kwargs["measurements"].shape[0])

    for period_res in period_inference:
        score = np.array(period_res.score_matrix)  # (n, n_free_own)
        info = np.array(period_res.information_matrix)
        # Use pinv for the same null-space-tolerant reasons as
        # `_block_diagonal_sandwich_single`.
        a_inv = np.linalg.pinv(info)

        # Draw indices for all replicates at once: (n_boot, n_clusters).
        idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        # mean_score[b, p] = (1/n) * sum_i score[idx[b, i], p]
        # Use einsum-friendly path: gather then mean over the cluster axis.
        mean_score = score[idx].mean(axis=1)  # (n_boot, n_free_own)
        delta = -mean_score @ a_inv.T  # (n_boot, n_free_own); one-step shift

        # Place delta back into the global parameter columns.
        global_cols = np.array(
            [pos_lookup[loc] for loc in period_res.free_param_locs],
            dtype=np.int64,
        )
        replicate_values[:, global_cols] += delta

    replicate_params = pd.DataFrame(
        replicate_values,
        columns=all_params.index,
    )
    standard_errors = pd.Series(
        replicate_params.std(axis=0, ddof=1).to_numpy(),
        index=all_params.index,
        name="bootstrap_se",
    )
    return AFBootstrapResult(
        standard_errors=standard_errors,
        replicate_params=replicate_params,
        n_clusters=n_clusters,
        n_boot=n_boot,
    )


__all__ = [
    "AFBootstrapResult",
    "AFInferenceResult",
    "AFPeriodInferenceResult",
    "compute_af_bootstrap_se",
    "compute_af_standard_errors",
]
