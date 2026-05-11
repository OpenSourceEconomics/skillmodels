"""Score-bootstrap standard errors for the AF estimator.

Implements the score bootstrap procedure prescribed in Antweiler &
Freyberger (2025) §4.2 (inspired by Armstrong, Bertanha & Hong 2014).
The AF estimator is a sequential multi-step MLE; its asymptotic variance
includes terms that propagate the estimation uncertainty of earlier
steps, which makes the analytical sandwich

    V = A^{-1} Omega A^{-T} / n

incorrect when computed without those cross-step terms. AF §4.2 puts
this directly:

    "this asymptotic variance is incorrect because it ignores the
    estimation errors of tau_{t-1}, ..., tau_1, which is the second
    term in the expansion above. To account for those, we would have
    to calculate ... which is very difficult because the likelihood is
    (partly) simulated and not available in closed form. To avoid
    these calculations, we use a score bootstrap procedure inspired by
    Armstrong, Bertanha, and Hong (2014)."

This module exposes a single inference entry point,
:func:`compute_af_standard_errors`, which implements that score
bootstrap. It avoids re-estimating the model B times: per-observation
scores are computed once at the optimum, then for each of ``n_boot``
replicates we resample caseids with replacement, average their scores,
and take a one-step Newton update from the optimum. The empirical
standard deviation of the resulting parameter draws is the bootstrap
standard error.

"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

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
    ChainLink,
    ConditionalDistribution,
)
from skillmodels.common.constraints import FixedConstraintWithValue
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.types import ProcessedModel


@dataclass(frozen=True)
class AFInferenceResult:
    """Score-bootstrap inference result for the AF estimator.

    See :func:`compute_af_standard_errors` for the procedure (AF 2025
    §4.2 / Armstrong-Bertanha-Hong 2014).
    """

    standard_errors: pd.Series
    """Bootstrap standard errors indexed by ``all_params.index``.

    SEs are the empirical standard deviation across bootstrap
    replicates of each parameter's one-step Newton shift from the
    point estimate. Fixed-parameter and constrained-direction entries
    are reported as zero (or NaN where the period's information matrix
    is singular on that direction).
    """

    vcov: pd.DataFrame
    """Variance-covariance matrix, rows and columns share
    ``all_params.index``. Computed from
    ``replicate_params.cov(ddof=1)`` so SEs and vcov are internally
    consistent.
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


def compute_af_standard_errors(
    result: AFEstimationResult,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> AFInferenceResult:
    """Score-resampling cluster bootstrap for the AF estimator.

    Implements Antweiler & Freyberger (2025) §4.2 (Armstrong-Bertanha-Hong
    score bootstrap). Per-observation scores are computed once at the
    optimum; for each of ``n_boot`` replicates we resample caseids with
    replacement, average the resampled scores, and apply a one-step
    Newton update from the optimum:

        theta_b = theta_hat - A_t^{-1} * bar_g_b

    where ``A_t`` is the period-``t`` information matrix and
    ``bar_g_b`` is the bootstrap-averaged per-obs score restricted to
    period-``t`` free parameters. Periods are resampled independently
    — joint resampling would couple periods through the
    block-diagonal information matrix the same way separate draws do,
    so we report own-block bootstrap SEs.

    The analytical Newey-McFadden sandwich is **not** provided: as AF
    §4.2 notes, the closed-form variance ignores estimation error in
    the previous-period nuisance parameters tau_{t-1}, ..., tau_1, so
    it is incorrect for any t >= 1. The score bootstrap captures this
    propagation.

    For ``n_boot=10000`` and ``n_caseids=1500`` this typically takes
    seconds rather than days (no re-estimation per replicate).

    Args:
        result: Output of ``estimate_af``.
        data: The dataset used for estimation; the caseid level of its
            MultiIndex defines the bootstrap clusters.
        af_options: Options used at estimation time.
        n_boot: Number of bootstrap replicates.
        seed: Seed for the resampling RNG.

    Return:
        :class:`AFInferenceResult` with bootstrap SEs, vcov computed
        from the replicate distribution, and the full
        replicate-by-parameter DataFrame.

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

    # Precompute per-period score and information matrices at the
    # optimum. The bootstrap then resamples score rows (caseids) and
    # applies a one-step Newton update; no re-estimation per replicate.
    period_score_info = _compute_block_diagonal_sandwich(result, metas)

    rng = np.random.default_rng(seed)
    all_params = result.all_params
    replicate_values = np.tile(all_params["value"].to_numpy()[None, :], (n_boot, 1))

    pos_lookup = {tuple(loc): i for i, loc in enumerate(all_params.index)}

    n_clusters = int(metas[0].loglike_kwargs["measurements"].shape[0])

    for period_res in period_score_info:
        score = np.array(period_res.score_matrix)  # (n, n_free_own)
        info = np.array(period_res.information_matrix)
        # Use pinv for the same null-space-tolerant reasons as
        # ``_block_diagonal_sandwich_single``.
        a_inv = np.linalg.pinv(info)

        idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        mean_score = score[idx].mean(axis=1)  # (n_boot, n_free_own)
        delta = -mean_score @ a_inv.T  # (n_boot, n_free_own); one-step shift

        global_cols = np.array(
            [pos_lookup[loc] for loc in period_res.free_param_locs],
            dtype=np.int64,
        )
        replicate_values[:, global_cols] += delta

    replicate_params = pd.DataFrame(replicate_values, columns=all_params.index)
    standard_errors = pd.Series(
        replicate_params.std(axis=0, ddof=1).to_numpy(),
        index=all_params.index,
        name="standard_error",
    )
    # Variance-covariance from the replicate distribution. Pinned-parameter
    # rows/columns are zero (constant column → zero variance/covariance).
    vcov_values = replicate_params.cov(ddof=1).to_numpy()
    vcov = pd.DataFrame(
        vcov_values,
        index=all_params.index,
        columns=all_params.index,
    )

    return AFInferenceResult(
        standard_errors=standard_errors,
        vcov=vcov,
        replicate_params=replicate_params,
        n_clusters=n_clusters,
        n_boot=n_boot,
    )


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
    target_idx_in_joint: tuple[int, ...] = ()
    """Initial-period only: positions of the *target* state factors (the
    ones whose marginal we want carry-over samples for) within
    `joint_factors`. Differs from ``state_factor_indices_in_joint`` when
    the joint includes an endogenous factor with ``has_initial_distribution=True``
    that should be excluded from the carry-over.
    """
    obs_idx_in_joint: tuple[int, ...] = ()
    """Initial-period only: positions of observed factors within
    `joint_factors`. Empty for transition-period metas.
    """
    propagation: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Extra JAX-pure bits for propagation of the conditional distribution
    through this period's transition. Only populated for transition
    periods. Keys: ``joint_nodes``, ``combined_transition``,
    ``obs_factor_values``, ``shock_factor_indices``.
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
                endogenous_factors=endogenous_factors,
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
    endogenous_factors: tuple[str, ...] = (),
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

    # Target factors for the carry-over sample = state_latent minus
    # endogenous (matches what `estimate_initial_period` does in the
    # estimation path).
    joint_factors = state_latent_factors + observed_factors
    target_factors = tuple(
        f for f in state_latent_factors if f not in endogenous_factors
    )
    target_idx_in_joint = tuple(joint_factors.index(f) for f in target_factors)
    obs_idx_in_joint = tuple(joint_factors.index(f) for f in observed_factors)
    n_state_target = len(target_factors)

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
        n_state=n_state_target,
        n_endog=0,
        n_shock=0,
        n_observed_factors=n_obs_factors,
        state_factor_indices_in_joint=state_factor_indices_in_joint,
        target_idx_in_joint=target_idx_in_joint,
        obs_idx_in_joint=obs_idx_in_joint,
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
    state_factor_indices_in_latent = jnp.array(
        [factors.index(f) for f in state_factors], dtype=jnp.int32
    )

    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    all_measures = _get_ordered_measures(measurements_pt)
    loading_mask = _build_loading_mask(all_measures, factors, measurements_pt)

    # Match transition_period.py: a single joint Halton at every step
    # covers (z_state for theta_0) + (n_chain) prior chain shocks
    # (z_inv, z_P) + current step's (z_inv, z_P).
    n_chain = period - 1
    z_block = n_shock + n_endog
    joint_dim = n_state + n_chain * z_block + z_block
    joint_nodes, joint_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points, joint_dim, seed=period
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

    raw_funcs = _get_raw_transition_functions(
        model_spec,
        state_factors,
        all_factors=processed_model.labels.all_factors,
        param_names=transition_info.param_names,
    )
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

    chain_links = prev_cond_dist.chain_links
    if len(chain_links) == 0:
        obs_factor_values_chain = jnp.zeros(
            (int(measurements.shape[0]), 0, len(observed_factors))
        )
    else:
        obs_factor_values_chain = jnp.stack(
            [link.obs_factor_values for link in chain_links], axis=1
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
        "state_factor_indices_in_latent": state_factor_indices_in_latent,
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
        "chain_links": chain_links,
        "obs_factor_values_chain": obs_factor_values_chain,
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


class _PeriodScoreInfo(NamedTuple):
    """Per-period score and information matrices at the optimum.

    Internal carrier used by :func:`compute_af_standard_errors` to feed
    the score bootstrap. Not part of the public API.
    """

    period: int
    free_param_locs: tuple[tuple[Any, ...], ...]
    score_matrix: Array
    information_matrix: Array


def _compute_block_diagonal_sandwich(
    _result: AFEstimationResult,
    metas: tuple[_PeriodMeta, ...],
) -> list[_PeriodScoreInfo]:
    """Compute per-period score and information matrices for the bootstrap."""
    results: list[_PeriodScoreInfo] = []
    for meta in metas:
        per_obs_fn = (
            af_per_obs_loglike_initial
            if meta.is_initial
            else af_per_obs_loglike_transition
        )
        info = _block_diagonal_sandwich_single(
            meta=meta,
            per_obs_loglike_fn=per_obs_fn,
        )
        results.append(info)
    return results


def _block_diagonal_sandwich_single(
    *,
    meta: _PeriodMeta,
    per_obs_loglike_fn: Callable[..., Array],
) -> _PeriodScoreInfo:
    """Compute the per-period score matrix and information matrix at theta_hat.

    These feed the score bootstrap. The information matrix is the
    Hessian of the scalar negative-mean log-likelihood; the score
    matrix has one row per caseid and one column per free parameter.
    """
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

    return _PeriodScoreInfo(
        period=meta.period,
        free_param_locs=tuple(locs),
        score_matrix=score_matrix,
        information_matrix=information_matrix,
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
) -> tuple[Array, Array, Array, Array]:
    """JAX-pure analytical reconstruction of the period-0 conditional payload.

    Mirrors ``initial_period._extract_conditional_distribution``: parse
    initial-period params, compute per-component / per-obs Schur-conditional
    means and per-component Cholesky factors. Returns the inputs the
    transition likelihood needs to rebuild θ_0 from a joint Halton inside
    its integrand (no chained-sample materialisation here).

    Return:
        Tuple of (cond_means, cond_chols, log_unnorms, mixture_weights):
          * cond_means: (n_components, n_obs, n_state)
          * cond_chols: (n_components, n_state, n_state)
          * log_unnorms: (n_components, n_obs); softmaxes to per-obs Bayes
            posterior mixture weights when observed factors are present.
          * mixture_weights: (n_components,) prior mixture weights.
    """
    parsed = _parse_initial_params(
        flat_params_0,
        meta.parse_kwargs["n_factors"],
        meta.parse_kwargs["n_mixture_components"],
        meta.parse_kwargs["n_measures"],
        meta.parse_kwargs["n_controls"],
    )
    joint_means = parsed["mixture_means"]  # (K, n_joint)
    joint_chols = parsed["mixture_chol_covs"]  # (K, n_joint, n_joint)
    mixture_weights = parsed["mixture_weights"]

    obs_values = meta.loglike_kwargs["observed_factor_values"]
    n_obs = int(obs_values.shape[0])
    n_obs_factors = meta.n_observed_factors
    n_state = meta.n_state
    target_idx = jnp.asarray(meta.target_idx_in_joint, dtype=jnp.int32)

    if n_obs_factors == 0:

        def _per_component(
            joint_mean: Array, joint_chol: Array
        ) -> tuple[Array, Array, Array]:
            joint_cov = joint_chol @ joint_chol.T
            mu_t = joint_mean[target_idx]
            cov_tt = joint_cov[target_idx[:, None], target_idx[None, :]]
            sub_chol = jnp.linalg.cholesky(cov_tt + 1e-10 * jnp.eye(n_state))
            cond_mean = jnp.broadcast_to(mu_t[None, :], (n_obs, n_state))
            log_unnorm = jnp.zeros(n_obs)
            return cond_mean, sub_chol, log_unnorm

        cond_means, cond_chols, log_unnorms = jax.vmap(_per_component)(
            joint_means, joint_chols
        )
        log_unnorms = log_unnorms + jnp.log(mixture_weights + 1e-300)[:, None]
    else:
        obs_idx = jnp.asarray(meta.obs_idx_in_joint, dtype=jnp.int32)

        def _per_component(
            joint_mean: Array, joint_chol: Array
        ) -> tuple[Array, Array, Array]:
            joint_cov = joint_chol @ joint_chol.T
            mu_t = joint_mean[target_idx]
            mu_y = joint_mean[obs_idx]
            cov_tt = joint_cov[target_idx[:, None], target_idx[None, :]]
            cov_ty = joint_cov[target_idx[:, None], obs_idx[None, :]]
            cov_yy = joint_cov[obs_idx[:, None], obs_idx[None, :]]
            chol_yy = jnp.linalg.cholesky(cov_yy)
            solve_tt = jax.scipy.linalg.cho_solve((chol_yy, True), cov_ty.T)
            cond_cov = cov_tt - cov_ty @ solve_tt + 1e-10 * jnp.eye(n_state)
            cond_chol = jnp.linalg.cholesky(cond_cov)

            def _per_obs(y_i: Array) -> tuple[Array, Array]:
                alpha = jax.scipy.linalg.cho_solve((chol_yy, True), y_i - mu_y)
                cond_mean = mu_t + cov_ty @ alpha
                k = y_i.shape[0]
                sol = jax.scipy.linalg.solve_triangular(chol_yy, y_i - mu_y, lower=True)
                log_marg = (
                    -0.5 * k * jnp.log(2 * jnp.pi)
                    - jnp.sum(jnp.log(jnp.diag(chol_yy)))
                    - 0.5 * jnp.dot(sol, sol)
                )
                return cond_mean, log_marg

            cond_means_per_obs, log_margs = jax.vmap(_per_obs)(obs_values)
            return cond_means_per_obs, cond_chol, log_margs

        cond_means, cond_chols, log_marg_y = jax.vmap(_per_component)(
            joint_means, joint_chols
        )
        log_unnorms = log_marg_y + jnp.log(mixture_weights + 1e-300)[:, None]

    return cond_means, cond_chols, log_unnorms, mixture_weights


def _extract_chain_link_jax(
    flat_params_t: Array,
    meta: _PeriodMeta,
) -> ChainLink:
    """JAX-pure construction of a ChainLink from period ``t``'s flat params.

    Mirrors ``transition_period._build_chain_link`` but parses the flat
    params directly so the chain link's leaves are differentiable
    components of ``flat_super``. Used by the inference sandwich code to
    rebuild the chained sample on-demand inside the period-`t` likelihood,
    keeping the autodiff DAG intact across periods.
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
    return ChainLink(
        period=meta.period,
        transition_func=meta.propagation["combined_transition"],
        transition_params=parsed["transition_params"],
        shock_sds=parsed["shock_sds"],
        shock_factor_indices=meta.propagation["shock_factor_indices"],
        inv_eq_params=parsed["inv_eq_params"],
        inv_sds=parsed["inv_sds"],
        n_inv_eq_params_per=meta.parse_kwargs["n_inv_eq_params_per"],
        obs_factor_values=meta.propagation["obs_factor_values"],
    )


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
) -> tuple[dict[str, Array], tuple[ChainLink, ...], Array]:
    """Build the period-0 conditional payload and chain history for period ``t``.

    Replaces the previous static-sample carry-over with the joint-Halton
    chain rebuild contract: the period-`t` likelihood expects
    ``prev_dist_arrays`` (with cond_weights / cond_means / cond_chols),
    a tuple of `ChainLink`s for the prior transition steps, and a
    per-obs ``obs_factor_values_chain`` tensor. The chain rebuild
    happens inside the integrand from a single joint Halton design.

    The autodiff DAG flows through each `ChainLink`'s leaves
    (transition_params, shock_sds, inv_eq_params, inv_sds) which are
    parsed from `flat_super`'s per-period slices.
    """
    meta0 = metas[0]
    flat_params_0 = flat_super[meta0.slice_start : meta0.slice_stop]
    cond_means, cond_chols, log_unnorms, mixture_weights = (
        _build_initial_state_cond_dist_jax(flat_params_0, meta0)
    )

    chain_links: list[ChainLink] = []
    for s in range(1, target_t):
        meta_s = metas[s]
        flat_params_s = flat_super[meta_s.slice_start : meta_s.slice_stop]
        chain_links.append(_extract_chain_link_jax(flat_params_s, meta_s))

    if cond_weights_override is not None:
        cond_weights = cond_weights_override
    elif meta0.n_observed_factors > 0:
        cond_weights = jax.nn.softmax(log_unnorms, axis=0).T
    else:
        meta_target = metas[target_t]
        n_obs = int(meta_target.loglike_kwargs["measurements"].shape[0])
        n_components = metas[0].n_components
        cond_weights = jnp.broadcast_to(mixture_weights[None, :], (n_obs, n_components))

    prev_dist_arrays = {
        "cond_weights": cond_weights,
        "cond_means": cond_means,
        "cond_chols": cond_chols,
    }

    # Per-obs observed factor values at each chain link's source period.
    meta_target = metas[target_t]
    n_obs = int(meta_target.loglike_kwargs["measurements"].shape[0])
    n_obs_factors = meta0.n_observed_factors
    if not chain_links:
        obs_factor_values_chain = jnp.zeros((n_obs, 0, n_obs_factors))
    else:
        obs_factor_values_chain = jnp.stack(
            [link.obs_factor_values for link in chain_links], axis=1
        )

    return prev_dist_arrays, tuple(chain_links), obs_factor_values_chain


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
    prev_dist_arrays, chain_links, obs_factor_values_chain = _build_prev_dist_arrays(
        flat_super, t, metas, cond_weights_override=stored_cond_weights
    )
    meta_prev = metas[t - 1]
    flat_params_prev = flat_super[meta_prev.slice_start : meta_prev.slice_stop]
    prev_meas = _extract_prev_meas_info_jax(flat_params_prev, meta_prev)

    kwargs = dict(meta_t.loglike_kwargs)
    kwargs["prev_distribution"] = prev_dist_arrays
    kwargs["chain_links"] = chain_links
    kwargs["obs_factor_values_chain"] = obs_factor_values_chain
    kwargs["prev_loadings_flat"] = prev_meas["loadings_flat"]
    kwargs["prev_control_params"] = prev_meas["control_params"]
    kwargs["prev_meas_sds"] = prev_meas["meas_sds"]
    return af_per_obs_loglike_transition(flat_params_t, **kwargs)


__all__ = [
    "AFInferenceResult",
    "compute_af_standard_errors",
]
