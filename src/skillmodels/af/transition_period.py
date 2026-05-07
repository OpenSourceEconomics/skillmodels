"""Step t (t >= 1) of the AF estimator: transition period estimation.

Estimate transition function parameters and measurement system parameters
using Halton quadrature over the latent factor distribution from the
previous period.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
from jax import Array

from skillmodels.af.batching import auto_n_obs_per_batch
from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.initial_period import _build_loading_mask, _get_ordered_measures
from skillmodels.af.likelihood import af_loglike_transition, create_loglike_and_gradient
from skillmodels.af.params import (
    apply_fixed_params,
    apply_start_params,
    build_optimagic_inputs,
    create_af_params_template,
    get_measurements_per_factor,
    get_normalizations_for_period,
    get_transition_period_params_index,
)
from skillmodels.af.types import (
    AFEstimationOptions,
    AFPeriodResult,
    ChainLink,
    ConditionalDistribution,
    MixtureComponent,
)
from skillmodels.model_spec import ModelSpec
from skillmodels.types import ProcessedModel, TransitionInfo


def estimate_transition_period(
    period: int,
    model_spec: ModelSpec,
    processed_model: ProcessedModel,
    measurements: Array,
    controls: Array,
    prev_measurements: Array,
    prev_controls: Array,
    prev_period_params: pd.DataFrame,
    prev_distribution: ConditionalDistribution,
    af_options: AFEstimationOptions,
    endogenous_factors: tuple[str, ...] = (),
    observed_factors: tuple[str, ...] = (),
    observed_factor_data: Array | None = None,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
) -> tuple[AFPeriodResult, ConditionalDistribution]:
    """Estimate a transition period (Step t, t >= 1) of the AF procedure.

    Given the estimated distribution of latent factors from previous periods,
    estimate the transition function parameters and measurement system
    parameters for the current period via MLE with Halton quadrature.

    Args:
        period: Calendar period index (t >= 1).
        model_spec: Model specification.
        processed_model: Processed model from `process_model()`.
        measurements: Shape (n_obs, n_measures), period t measurement values.
        controls: Shape (n_obs, n_controls), period t control values.
        prev_measurements: Shape (n_obs, n_prev_measures), period t-1 measurements.
        prev_controls: Shape (n_obs, n_prev_controls), period t-1 controls.
        prev_period_params: Estimated params DataFrame from period t-1.
        prev_distribution: Estimated conditional distribution from period t-1.
        af_options: AF estimation options.
        endogenous_factors: Names of endogenous (investment) factors.
        observed_factors: Names of observed (non-latent) factors.
        observed_factor_data: Shape (n_obs, n_obs_factors), observed factor
            values. Required when `observed_factors` is non-empty.
        start_params: Optional starting values. Matching index entries
            override heuristic defaults.
        fixed_params: Optional DataFrame with a "value" column pinning
            specified parameters (value + bounds both clamped to the value).

    Return:
        Tuple of (AFPeriodResult, ConditionalDistribution) where the
        distribution represents f(theta_t | data_{0:t}).

    """
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls

    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    all_measures = _get_ordered_measures(measurements_pt)

    # Get transition function info
    # For now, use the first non-constant factor's transition for the combined function
    transition_info = processed_model.transition_info

    # Separate state factors from endogenous for the parameter index
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
    # Indices of the state factors within the full latent-factor ordering.
    # `prev_full_loadings` has columns in `factors` order (state +
    # endogenous, possibly interleaved); the prev-meas factor restricts to
    # state-factor columns to mirror MATLAB's likelihood_12 (which omits
    # period-(t-1) inv measurements from the chained-sample importance
    # weight). Build the mapping explicitly rather than relying on
    # state-before-endogenous ordering.
    state_factor_indices_in_latent = jnp.array(
        [factors.index(f) for f in state_factors], dtype=jnp.int32
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
    normalizations = get_normalizations_for_period(model_spec.factors, period=period)
    params_template = create_af_params_template(
        params_index,
        normalizations,
        period=period,
    )

    params_template = _initialize_transition_params(
        params_template,
        measurements,
        start_params,
        fixed_params,
    )

    # Collect transition function constraints (only for state factors' transitions)
    transition_constraints = _collect_transition_constraints(
        transition_info,
        state_factors,
        processed_model.labels.all_factors,
        period,
    )

    _seed_probability_start_values(
        params_template, transition_constraints, fixed_params
    )

    # Build loading mask
    loading_mask = _build_loading_mask(all_measures, factors, measurements_pt)

    # JOINT Halton design covering ALL randomness needed at this step,
    # mirroring MATLAB's `create_nodes_weights_01/12`. The chained sample
    # θ_0 → θ_{period-1} is rebuilt on-demand inside the integrand from
    # this single joint sequence (see `_rebuild_chain_at_period` in
    # `af/likelihood.py` and the obsidian note
    # `sigma-prod-collapse-2026-05-07.md` for why this matters).
    #
    # Layout of joint_nodes[j]:
    #   [:n_state]                              -- z_state for θ_0
    #   for s in 0..period-2:                   -- prior chain steps
    #       [n_state+s*zb : n_state+s*zb+n_shock]    -- z_P at period s+1
    #       [...n_shock+n_endog]                      -- z_inv at period s+1
    #   [tail: n_shock]                         -- z_P at current step (period)
    #   [tail: n_endog]                         -- z_inv at current step (period)
    #
    # Seed the Halton design with the period index. Each step draws an
    # independent low-discrepancy sequence; the joint structure within a
    # step delivers proper quasi-uniform 3D+ coverage (vs. the previous
    # split scheme which paired two independent sequences at the same j).
    n_chain = period - 1  # number of prior transition steps already estimated
    z_block = n_shock + n_endog
    joint_dim = n_state + n_chain * z_block + z_block
    joint_nodes, joint_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        joint_dim,
        seed=period,
    )

    prev_dist_arrays, total_n_transition_params = _prepare_transition_inputs(
        prev_distribution,
        transition_info,
        state_factors,
        measurements.shape[0],
    )

    # Build combined transition from raw transition functions.
    # Only state factors have transitions; endogenous factors use the investment eq.
    raw_funcs = _get_raw_transition_functions(model_spec, state_factors)
    param_counts = tuple(len(transition_info.param_names[f]) for f in state_factors)

    def combined_transition(
        full_states: Array,
        params: Array,
    ) -> Array:
        """Apply per-factor transitions."""
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

    # Count investment equation params (per endogenous factor: intercept + state + obs)
    n_inv_eq_params_per = 1 + n_state + len(observed_factors) if n_endog > 0 else 0
    total_n_inv_params = n_endog * n_inv_eq_params_per

    # Observed factor values for investment equation (from previous period)
    n_obs_fac = len(observed_factors)
    obs_factor_values = (
        observed_factor_data
        if observed_factor_data is not None
        else jnp.zeros((measurements.shape[0], n_obs_fac))
    )

    # Carry forward chain links from prior transition steps for the
    # joint-Halton chain rebuild. The period-0→1 step has chain_links == ().
    chain_links = prev_distribution.chain_links

    # Per-obs observed factors at the source period of each chain link
    # (period 0 for link 0, period 1 for link 1, ...). Stack across
    # links into shape (n_obs, n_chain, n_obs_factors). Each ChainLink
    # already carries its own period's `obs_factor_values` internally;
    # extract them here in obs-major order to match the per-obs map in
    # `_transition_loglike_per_obs`.
    if len(chain_links) == 0:
        obs_factor_values_chain = jnp.zeros((measurements.shape[0], 0, n_obs_fac))
    else:
        obs_factor_values_chain = jnp.stack(
            [link.obs_factor_values for link in chain_links], axis=1
        )

    result_params, opt_res = _run_transition_optimization(
        params_template=params_template,
        prev_period_params=prev_period_params,
        model_spec=model_spec,
        factors=factors,
        period=period,
        n_state=n_state,
        n_endog=n_endog,
        n_shock=n_shock,
        shock_factor_indices=shock_factor_indices,
        state_factor_indices_in_latent=state_factor_indices_in_latent,
        all_measures=all_measures,
        controls_names=controls_names,
        measurements=measurements,
        controls=controls,
        prev_measurements=prev_measurements,
        prev_controls=prev_controls,
        loading_mask=loading_mask,
        prev_dist_arrays=prev_dist_arrays,
        chain_links=chain_links,
        obs_factor_values_chain=obs_factor_values_chain,
        joint_nodes=joint_nodes,
        joint_weights=joint_weights,
        combined_transition=combined_transition,
        total_n_transition_params=total_n_transition_params,
        total_n_inv_params=total_n_inv_params,
        n_inv_eq_params_per=n_inv_eq_params_per,
        obs_factor_values=obs_factor_values,
        af_options=af_options,
        transition_constraints=transition_constraints,
        fixed_params=fixed_params,
    )

    # Build the next ChainLink from the just-fitted period parameters and
    # append it to the chain history. Future transition steps will replay
    # this link as part of their joint-Halton chain rebuild.
    new_link = _build_chain_link(
        period=period,
        result_params=result_params,
        combined_transition=combined_transition,
        shock_factor_indices=shock_factor_indices,
        n_inv_eq_params_per=n_inv_eq_params_per,
        obs_factor_values=obs_factor_values,
    )
    new_chain_links = (*chain_links, new_link)

    # Build the importance-sample SUMMARY (mean, chol_cov per component)
    # for posterior-state extraction. This path is no longer load-bearing
    # for the transition likelihood (rebuilt on-demand from joint Halton),
    # but `posterior_states.py` still consumes the per-component summary
    # statistics derived from the chained sample.
    updated_dist = _update_conditional_distribution(
        prev_distribution=prev_distribution,
        result_params=result_params,
        combined_transition=combined_transition,
        joint_nodes=joint_nodes,
        n_state=n_state,
        n_endog=n_endog,
        n_shock=n_shock,
        shock_factor_indices=shock_factor_indices,
        observed_factor_values=obs_factor_values,
        n_observed_factors=len(observed_factors),
    )
    # Carry the accumulated chain history forward.
    updated_dist = _replace_chain_links(updated_dist, new_chain_links)

    period_result = AFPeriodResult(
        period=period,
        params=result_params,
        loglikelihood=-float(opt_res.fun),
        success=bool(opt_res.success),
        optimize_result=opt_res,
    )

    return period_result, updated_dist


def _run_transition_optimization(
    *,
    params_template: pd.DataFrame,
    prev_period_params: pd.DataFrame,
    model_spec: ModelSpec,
    factors: tuple[str, ...],
    period: int,
    n_state: int,
    n_endog: int,
    n_shock: int,
    shock_factor_indices: Array,
    state_factor_indices_in_latent: Array,
    all_measures: list[str],
    controls_names: tuple[str, ...],
    measurements: Array,
    controls: Array,
    prev_measurements: Array,
    prev_controls: Array,
    loading_mask: np.ndarray,
    prev_dist_arrays: dict[str, Array],
    chain_links: tuple[ChainLink, ...],
    obs_factor_values_chain: Array,
    joint_nodes: Array,
    joint_weights: Array,
    combined_transition: Callable,
    total_n_transition_params: int,
    total_n_inv_params: int,
    n_inv_eq_params_per: int,
    obs_factor_values: Array,
    af_options: AFEstimationOptions,
    transition_constraints: list[om.constraints.Constraint],
    fixed_params: pd.DataFrame | None,
) -> tuple[pd.DataFrame, om.OptimizeResult]:
    """Build likelihood, run the optimizer, and return updated params.

    Handle the mechanical optimization setup: construct the log-likelihood
    keyword arguments, create the jitted value-and-gradient function, build
    the params DataFrame + constraint list, and call `om.minimize`.

    Return:
        Tuple of (result_params DataFrame, OptimizeResult).

    """
    full_params_df, fixed_constraints = build_optimagic_inputs(
        params_template, fixed_params
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

    loglike_and_grad = create_loglike_and_gradient(
        af_loglike_transition,
        **loglike_kwargs,
    )

    def fun(params_df: pd.DataFrame) -> float:
        val, _grad = loglike_and_grad(jnp.array(params_df["value"].to_numpy()))
        return float(val)

    def fun_and_jac(params_df: pd.DataFrame) -> tuple[float, np.ndarray]:
        val, grad = loglike_and_grad(jnp.array(params_df["value"].to_numpy()))
        return float(val), np.array(grad)

    combined_constraints = list(transition_constraints) + list(fixed_constraints)

    opt_res = om.minimize(
        fun=fun,
        params=full_params_df[["value"]],
        algorithm=af_options.optimizer_algorithm,
        bounds=om.Bounds(
            lower=full_params_df["lower_bound"],
            upper=full_params_df["upper_bound"],
        ),
        constraints=combined_constraints or None,
        fun_and_jac=fun_and_jac,
        **dict(af_options.optimizer_options),
    )

    result_params = params_template.copy()
    result_params["value"] = opt_res.params["value"].to_numpy()

    return result_params, opt_res


def _collect_transition_constraints(
    transition_info: TransitionInfo,
    factors: tuple[str, ...],
    all_factors: tuple[str, ...],
    period: int,
) -> list[om.constraints.Constraint]:
    """Collect transition function constraints for the AF optimizer.

    Look for `constraints_{function_name}()` in `transition_functions.py`,
    mirroring how CHS collects them in `constraints.py`.
    """
    import skillmodels.transition_functions as tf_mod  # noqa: PLC0415

    constraints: list[om.constraints.Constraint] = []
    for factor in factors:
        if factor not in transition_info.function_names:
            continue
        fname = transition_info.function_names[factor]
        constraint_fn = getattr(tf_mod, f"constraints_{fname}", None)
        if constraint_fn is not None:
            constraints.append(
                constraint_fn(
                    factor=factor,
                    factors=all_factors,
                    aug_period=period - 1,
                )
            )
    return constraints


def _extract_prev_measurement_params(
    prev_params: pd.DataFrame,
    model_spec: ModelSpec,
    factors: tuple[str, ...],
    prev_period: int,
) -> dict[str, Array]:
    """Extract estimated measurement params from the previous period.

    These are used as fixed (known) values when conditioning the transition
    likelihood on individual-specific previous-period data.
    """
    measurements_prev = get_measurements_per_factor(
        model_spec.factors, period=prev_period
    )
    all_prev_measures = _get_ordered_measures(measurements_prev)
    loading_mask = _build_loading_mask(all_prev_measures, factors, measurements_prev)

    # Extract loadings (packed, in order of the mask)
    loadings_list = []
    for mi, meas in enumerate(all_prev_measures):
        for fi, factor in enumerate(factors):
            if loading_mask[mi, fi]:
                loc = ("loadings", prev_period, meas, factor)
                if loc in prev_params.index:
                    loadings_list.append(
                        float(prev_params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
                    )

    # Extract control params
    ctrl_entries = prev_params.loc[
        prev_params.index.get_level_values("category") == "controls"
    ]
    ctrl_names = (
        sorted(set(ctrl_entries.index.get_level_values("name2")))
        if len(ctrl_entries) > 0
        else ["constant"]
    )
    ctrl_params_list = _collect_ctrl_params(
        prev_params,
        all_prev_measures,
        ctrl_names,
        prev_period,
    )
    control_params = jnp.array(ctrl_params_list).reshape(
        len(all_prev_measures), len(ctrl_names)
    )

    # Extract measurement SDs
    meas_sds_list = []
    for meas in all_prev_measures:
        loc = ("meas_sds", prev_period, meas, "-")
        if loc in prev_params.index:
            meas_sds_list.append(
                float(prev_params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
            )

    return {
        "loading_mask": jnp.array(loading_mask),
        "loadings_flat": jnp.array(loadings_list),
        "control_params": control_params,
        "meas_sds": jnp.array(meas_sds_list),
    }


def _collect_ctrl_params(
    prev_params: pd.DataFrame,
    measures: list[str],
    ctrl_names: list[str],
    prev_period: int,
) -> list[float]:
    """Collect control parameter values from the previous period's estimate."""
    result = []
    for meas in measures:
        for ctrl in ctrl_names:
            loc = ("controls", prev_period, meas, ctrl)
            if loc in prev_params.index:
                result.append(
                    float(prev_params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
                )
            else:
                result.append(0.0)
    return result


def _get_raw_transition_functions(
    model_spec: ModelSpec,
    factors: tuple[str, ...],
) -> tuple[Callable, ...]:
    """Get the raw (non-vmapped) transition functions for each factor.

    These are the simple `(states, params) -> scalar` callables from
    `transition_functions.py`, suitable for use inside JIT-compiled code.
    """
    import skillmodels.transition_functions as tf_mod  # noqa: PLC0415

    funcs: list[Callable] = []
    for factor in factors:
        spec = model_spec.factors[factor]
        tf = spec.transition_function
        if isinstance(tf, str):
            funcs.append(getattr(tf_mod, tf))
        elif callable(tf):
            funcs.append(tf)
        else:
            msg = f"Factor '{factor}': no transition function specified."
            raise TypeError(msg)
    return tuple(funcs)


def _prepare_transition_inputs(
    prev_distribution: ConditionalDistribution,
    transition_info: TransitionInfo,
    factors: tuple[str, ...],
    n_obs: int,
) -> tuple[dict[str, Array], int]:
    """Pack the period-0 conditional distribution payload for the likelihood.

    Returns a dict the transition likelihood reads to seed its on-demand
    chain rebuild from a joint Halton draw. The chain is rebuilt fresh at
    every likelihood call from the period-0 cond_means/cond_chols plus
    the carried `chain_links` (handled separately); no static
    chained-sample carry-over is consumed here.

    Return:
        Tuple of (prev_dist_arrays dict, n_transition_params). The dict
        contains keys "cond_weights" (per-obs Bayes-posterior mixture
        weights), "cond_means" (per-component, per-obs Schur-conditional
        means at period 0), and "cond_chols" (per-component
        Schur-conditional Cholesky factors at period 0).

    """
    n_components = len(prev_distribution.components)

    if prev_distribution.conditional_weights is not None:
        cond_weights = prev_distribution.conditional_weights
    else:
        cond_weights = jnp.broadcast_to(
            prev_distribution.mixture_weights[None, :],
            (n_obs, n_components),
        )

    if prev_distribution.cond_means is None or prev_distribution.cond_chols is None:
        msg = (
            "prev_distribution must carry cond_means and cond_chols (the "
            "period-0 Schur-conditional payload). Initial period must be "
            "estimated before any transition step."
        )
        raise ValueError(msg)

    prev_dist_arrays = {
        "cond_weights": cond_weights,
        "cond_means": prev_distribution.cond_means,
        "cond_chols": prev_distribution.cond_chols,
    }

    total_n_transition_params = sum(
        len(transition_info.param_names[f])
        for f in factors
        if f in transition_info.param_names
    )

    return prev_dist_arrays, total_n_transition_params


def _seed_probability_start_values(
    params_template: pd.DataFrame,
    transition_constraints: list[om.constraints.Constraint],
    fixed_params: pd.DataFrame | None,
) -> None:
    """Seed start values for probability-constrained selectors.

    Distribute ``1 - sum(fixed_values)`` uniformly over the unfixed entries
    so the simplex sums to one before optimization.
    """
    fixed_loc = set(fixed_params.index) if fixed_params is not None else set()
    for constr in transition_constraints:
        if not isinstance(constr, om.ProbabilityConstraint):
            continue
        prob_idx = constr.selector(params_template[["value"]]).index
        fixed_mask = prob_idx.isin(fixed_loc)
        fixed_sum = (
            float(params_template.loc[prob_idx[fixed_mask], "value"].sum())
            if fixed_mask.any()
            else 0.0
        )
        free_prob_idx = prob_idx[~fixed_mask]
        if len(free_prob_idx) > 0:
            params_template.loc[free_prob_idx, "value"] = (1.0 - fixed_sum) / len(
                free_prob_idx
            )


def _initialize_transition_params(
    params_template: pd.DataFrame,
    measurements: Array,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Initialize transition period parameters with reasonable defaults.

    If `start_params` is provided, matching entries override the defaults.
    If `fixed_params` is provided, matching entries are pinned (value +
    bounds clamped).
    """
    params = params_template.copy()
    meas_np = np.array(measurements)

    # Transition params: small values (near identity)
    trans_mask = params.index.get_level_values("category") == "transition"
    for idx in params.index[trans_mask]:
        if params.loc[idx, "lower_bound"] != params.loc[idx, "upper_bound"]:
            # Set linear terms close to identity
            params.loc[idx, "value"] = 0.5

    # Shock SDs: moderate
    shock_mask = params.index.get_level_values("category") == "shock_sds"
    params.loc[shock_mask, "value"] = 0.5

    # Measurement SDs from data
    sd_mask = params.index.get_level_values("category") == "meas_sds"
    for i, idx in enumerate(params.index[sd_mask]):
        if i < meas_np.shape[1]:
            obs_sd = float(np.nanstd(meas_np[:, i]))
            params.loc[idx, "value"] = max(obs_sd * 0.5, 0.01)

    # Loadings to 1.0 where free
    load_mask = params.index.get_level_values("category") == "loadings"
    for idx in params.index[load_mask]:
        if params.loc[idx, "lower_bound"] != params.loc[idx, "upper_bound"]:
            params.loc[idx, "value"] = 1.0

    if start_params is not None:
        apply_start_params(params, start_params)

    if fixed_params is not None:
        apply_fixed_params(params, fixed_params)

    return params


def _replace_chain_links(
    cond_dist: ConditionalDistribution,
    chain_links: tuple[ChainLink, ...],
) -> ConditionalDistribution:
    """Return a new ConditionalDistribution with `chain_links` replaced.

    Used by `estimate_transition_period` to carry the accumulated chain
    history forward (one extra `ChainLink` per estimated transition).
    """
    return ConditionalDistribution(
        mixture_weights=cond_dist.mixture_weights,
        components=cond_dist.components,
        samples_per_component=cond_dist.samples_per_component,
        conditional_weights=cond_dist.conditional_weights,
        cond_means=cond_dist.cond_means,
        cond_chols=cond_dist.cond_chols,
        chain_links=chain_links,
    )


def _build_chain_link(
    *,
    period: int,
    result_params: pd.DataFrame,
    combined_transition: Callable,
    shock_factor_indices: Array,
    n_inv_eq_params_per: int,
    obs_factor_values: Array,
) -> ChainLink:
    """Pack a freshly-fitted period's parameters into a ChainLink.

    The resulting `ChainLink` is appended to the carried `chain_links` so
    that downstream transition periods can replay this period inside their
    joint-Halton chain rebuild (see `_rebuild_chain_at_period`).
    """
    transition_mask = result_params.index.get_level_values("category") == "transition"
    transition_params = jnp.array(
        result_params.loc[transition_mask, "value"].to_numpy()
    )

    shock_mask = result_params.index.get_level_values("category") == "shock_sds"
    shock_sds = jnp.array(result_params.loc[shock_mask, "value"].to_numpy())

    inv_eq_mask = result_params.index.get_level_values("category") == "investment_eq"
    inv_eq_params = jnp.array(result_params.loc[inv_eq_mask, "value"].to_numpy())

    inv_sd_mask = result_params.index.get_level_values("category") == "investment_sds"
    inv_sds = jnp.array(result_params.loc[inv_sd_mask, "value"].to_numpy())

    return ChainLink(
        period=period,
        transition_func=combined_transition,
        transition_params=transition_params,
        shock_sds=shock_sds,
        shock_factor_indices=shock_factor_indices,
        inv_eq_params=inv_eq_params,
        inv_sds=inv_sds,
        n_inv_eq_params_per=n_inv_eq_params_per,
        obs_factor_values=obs_factor_values,
    )


def _update_conditional_distribution(
    prev_distribution: ConditionalDistribution,
    result_params: pd.DataFrame,
    combined_transition: Callable,
    joint_nodes: Array,
    n_state: int,
    n_endog: int,
    n_shock: int,
    shock_factor_indices: Array,
    observed_factor_values: Array,
    n_observed_factors: int,
) -> ConditionalDistribution:
    """Build the next-period importance sample by chaining forward.

    For each mixture component l, each Halton index j, and each observation
    i:

    1. ``theta_prev = prev_samples[l][j, i, :]`` (no fresh draw).
    2. ``inv = beta_0 + beta_state @ theta_prev + beta_obs @ Y_i +
        sigma_inv * z_inv[j]`` (current-period investment equation,
        evaluated at the just-estimated parameters and the same z_inv that
        the period-t likelihood used).
    3. ``theta_t = transition(full_prev_with_obs, trans_params) +
        sigma_prod * z_prod[j]``.

    The result is a per-component array of shape
    ``(n_halton, n_obs, n_state)`` which we hand to the next period's
    likelihood. Per-component summary stats (mean, chol_cov) are computed
    from each new sample for use by `posterior_states` and `inference`.

    This mirrors MATLAB's `create_nodes_weights_12` style: the previous
    period's Halton-driven samples are propagated through the just-fitted
    chain, and that chained sample becomes the next period's importance
    distribution.
    """
    # Extract estimated transition params, shock SDs, investment-equation
    # params, and investment-shock SDs.
    trans_mask = result_params.index.get_level_values("category") == "transition"
    shock_mask = result_params.index.get_level_values("category") == "shock_sds"
    inv_eq_mask = result_params.index.get_level_values("category") == "investment_eq"
    inv_sd_mask = result_params.index.get_level_values("category") == "investment_sds"

    trans_params = jnp.array(result_params.loc[trans_mask, "value"].to_numpy())
    shock_sds = jnp.array(result_params.loc[shock_mask, "value"].to_numpy())
    inv_eq_params = (
        jnp.array(result_params.loc[inv_eq_mask, "value"].to_numpy())
        if inv_eq_mask.any()
        else jnp.zeros(0)
    )
    inv_sds = (
        jnp.array(result_params.loc[inv_sd_mask, "value"].to_numpy())
        if inv_sd_mask.any()
        else jnp.zeros(0)
    )

    n_per_inv_eq = 1 + n_state + n_observed_factors if n_endog > 0 else 0

    n_halton = joint_nodes.shape[0]
    # The joint Halton design now has a larger dimension than just the
    # current step's shocks (it also covers the chain rebuild's z_state
    # and prior-step shocks; see `estimate_transition_period`). The
    # current-step shocks live in the LAST `n_shock + n_endog` columns.
    z_block_curr = n_shock + n_endog

    def _chain_one_component(prev_sample: Array) -> Array:
        """Map (j, i) -> theta_t given prev_sample (n_halton, n_obs, n_state)."""

        def _at_node(j_idx: int, i_idx: int) -> Array:
            theta_prev = prev_sample[j_idx, i_idx]
            obs_y = (
                observed_factor_values[i_idx]
                if n_observed_factors > 0
                else jnp.zeros(0)
            )
            z_at_j_full = joint_nodes[j_idx]
            z_at_j = z_at_j_full[-z_block_curr:]
            z_shock = z_at_j[:n_shock]
            z_inv_shock = z_at_j[n_shock:]

            # Investment equation at the just-estimated params.
            inv = jnp.zeros(n_endog)
            for k in range(n_endog):
                beta = inv_eq_params[k * n_per_inv_eq : (k + 1) * n_per_inv_eq]
                intercept = beta[0]
                state_coeffs = beta[1 : 1 + n_state]
                obs_coeffs = beta[1 + n_state :]
                inv_k = (
                    intercept
                    + jnp.dot(state_coeffs, theta_prev)
                    + jnp.dot(obs_coeffs, obs_y)
                    + inv_sds[k] * z_inv_shock[k]
                )
                inv = inv.at[k].set(inv_k)  # noqa: PD008

            full_prev_with_obs = jnp.concatenate([theta_prev, inv, obs_y])
            state_shock_contrib = (
                jnp.zeros(n_state)  # noqa: PD008
                .at[shock_factor_indices]
                .set(shock_sds * z_shock)
            )
            return combined_transition(full_prev_with_obs, trans_params) + (
                state_shock_contrib
            )

        n_obs = prev_sample.shape[1]
        return jax.vmap(
            jax.vmap(_at_node, in_axes=(None, 0)),
            in_axes=(0, None),
        )(jnp.arange(n_halton), jnp.arange(n_obs))

    new_samples_per_component: list[Array] = []
    new_components: list[MixtureComponent] = []
    for prev_sample in prev_distribution.samples_per_component:
        new_sample = _chain_one_component(prev_sample)
        new_samples_per_component.append(new_sample)
        # Summary stats: per-Halton mean across obs for posterior_states
        # consumption. (Mean is also taken across obs to give a population-
        # level summary; the actual likelihood uses the per-obs sample.)
        flat = new_sample.reshape(-1, n_state)
        new_mean = jnp.mean(flat, axis=0)
        centered = flat - new_mean[None, :]
        new_cov = (centered.T @ centered) / flat.shape[0] + 1e-8 * jnp.eye(n_state)
        new_chol = jnp.linalg.cholesky(new_cov)
        new_components.append(MixtureComponent(mean=new_mean, chol_cov=new_chol))

    return ConditionalDistribution(
        mixture_weights=prev_distribution.mixture_weights,
        components=tuple(new_components),
        samples_per_component=tuple(new_samples_per_component),
        conditional_weights=prev_distribution.conditional_weights,
        # Carry the period-0 Schur conditional payload AND the chain
        # history forward; downstream transition steps replay the chain
        # from period 0, not from this period's chained samples.
        cond_means=prev_distribution.cond_means,
        cond_chols=prev_distribution.cond_chols,
        chain_links=prev_distribution.chain_links,
    )
