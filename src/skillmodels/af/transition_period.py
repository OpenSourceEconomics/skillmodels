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

from skillmodels.af.halton import (
    create_halton_nodes_and_weights,
    create_shock_nodes_and_weights,
)
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

    params_index = get_transition_period_params_index(
        period=period,
        latent_factors=state_factors,
        transition_info=transition_info,
        measurements_at_period=measurements_pt,
        controls=controls_names,
        endogenous_factors=endogenous_factors,
        observed_factors=observed_factors,
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

    # Halton quadrature nodes for factor integration
    # State nodes cover only state factors (conditional distribution dimension)
    state_nodes, state_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_state,
    )
    shock_nodes, shock_weights = create_shock_nodes_and_weights(
        af_options.n_halton_points_shock,
        n_state,
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

    # Investment shock nodes (separate from production shocks)
    if n_endog > 0:
        inv_shock_nodes, inv_shock_weights = create_halton_nodes_and_weights(
            af_options.n_halton_points_shock,
            n_endog,
            seed=99,
        )
    else:
        inv_shock_nodes = jnp.zeros((1, 0))
        inv_shock_weights = jnp.ones(1)

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

    result_params, opt_res = _run_transition_optimization(
        params_template=params_template,
        prev_period_params=prev_period_params,
        model_spec=model_spec,
        factors=factors,
        period=period,
        n_state=n_state,
        n_endog=n_endog,
        all_measures=all_measures,
        controls_names=controls_names,
        measurements=measurements,
        controls=controls,
        prev_measurements=prev_measurements,
        prev_controls=prev_controls,
        loading_mask=loading_mask,
        prev_dist_arrays=prev_dist_arrays,
        state_nodes=state_nodes,
        state_weights=state_weights,
        shock_nodes=shock_nodes,
        shock_weights=shock_weights,
        inv_shock_nodes=inv_shock_nodes,
        inv_shock_weights=inv_shock_weights,
        combined_transition=combined_transition,
        total_n_transition_params=total_n_transition_params,
        total_n_inv_params=total_n_inv_params,
        n_inv_eq_params_per=n_inv_eq_params_per,
        obs_factor_values=obs_factor_values,
        af_options=af_options,
        transition_constraints=transition_constraints,
        fixed_params=fixed_params,
    )

    # Create a state-only transition wrapper for distribution propagation.
    # Uses mean investment (from investment eq at prior mean) and observed values.
    prior_mean = prev_distribution.components[0].mean
    mean_inv = _compute_mean_investment(
        prior_mean,
        obs_factor_values,
        result_params,
        n_endog,
        n_state,
        len(observed_factors),
    )

    def state_only_transition(
        state_factors_val: Array,
        params: Array,
    ) -> Array:
        """Transition wrapper using mean investment + mean observed."""
        mean_obs = jnp.mean(obs_factor_values, axis=0)
        full = jnp.concatenate([state_factors_val, mean_inv, mean_obs])
        return combined_transition(full, params)

    updated_dist = _update_conditional_distribution(
        prev_distribution=prev_distribution,
        result_params=result_params,
        combined_transition=state_only_transition,
        state_nodes=state_nodes,
        state_weights=state_weights,
        n_factors=n_state,
    )

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
    all_measures: list[str],
    controls_names: tuple[str, ...],
    measurements: Array,
    controls: Array,
    prev_measurements: Array,
    prev_controls: Array,
    loading_mask: np.ndarray,
    prev_dist_arrays: dict[str, Array],
    state_nodes: Array,
    state_weights: Array,
    shock_nodes: Array,
    shock_weights: Array,
    inv_shock_nodes: Array,
    inv_shock_weights: Array,
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

    loglike_kwargs = {
        "n_state_factors": n_state,
        "n_endogenous_factors": n_endog,
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
        "state_nodes": state_nodes,
        "state_weights": state_weights,
        "shock_nodes": shock_nodes,
        "shock_weights": shock_weights,
        "inv_shock_nodes": inv_shock_nodes,
        "inv_shock_weights": inv_shock_weights,
        "transition_func": combined_transition,
        "total_n_transition_params": total_n_transition_params,
        "total_n_inv_params": total_n_inv_params,
        "n_inv_eq_params_per": n_inv_eq_params_per,
        "observed_factor_values": obs_factor_values,
        "stability_floor": af_options.stability_floor,
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


def _compute_mean_investment(
    state_mean: Array,
    obs_factor_values: Array,
    result_params: pd.DataFrame,
    n_endog: int,
    n_state: int,
    n_obs_factors: int,
) -> Array:
    """Compute mean investment at the prior state mean (no shock)."""
    if n_endog == 0:
        return jnp.zeros(0)
    inv_eq_mask = result_params.index.get_level_values("category") == "investment_eq"
    inv_eq_vals = jnp.array(result_params.loc[inv_eq_mask, "value"].to_numpy())
    n_per = 1 + n_state + n_obs_factors
    # Use population mean of observed factor values
    obs_mean = (
        jnp.mean(obs_factor_values, axis=0)
        if obs_factor_values.shape[0] > 0
        else jnp.zeros(n_obs_factors)
    )
    result = jnp.zeros(n_endog)
    for j in range(n_endog):
        beta = inv_eq_vals[j * n_per : (j + 1) * n_per]
        inv_j = beta[0] + jnp.dot(beta[1 : 1 + n_state], state_mean)
        if n_obs_factors > 0:
            inv_j = inv_j + jnp.dot(beta[1 + n_state :], obs_mean)
        result = result.at[j].set(inv_j)  # noqa: PD008
    return result


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
    """Prepare distribution arrays and count transition params.

    Convert the previous-period conditional distribution into JAX arrays
    for the likelihood, and compute the maximum number of transition
    parameters across all factors.

    Return:
        Tuple of (prev_dist_arrays dict, n_transition_params).

    """
    n_components = len(prev_distribution.components)
    means = jnp.stack([c.mean for c in prev_distribution.components])
    chol_covs = jnp.stack([c.chol_cov for c in prev_distribution.components])

    if prev_distribution.conditional_weights is not None:
        cond_weights = prev_distribution.conditional_weights
    else:
        cond_weights = jnp.broadcast_to(
            prev_distribution.mixture_weights[None, :],
            (n_obs, n_components),
        )

    prev_dist_arrays = {
        "cond_weights": cond_weights,
        "means": means,
        "chol_covs": chol_covs,
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


def _update_conditional_distribution(
    prev_distribution: ConditionalDistribution,
    result_params: pd.DataFrame,
    combined_transition: Callable,
    state_nodes: Array,
    state_weights: Array,
    n_factors: int,
) -> ConditionalDistribution:
    """Propagate the conditional distribution through the transition function.

    Use quadrature-based moment matching: for each mixture component, sample
    the previous distribution at quadrature nodes, propagate through the
    transition function, and compute the new mean and covariance.

    """
    # Extract estimated transition params and shock SDs
    trans_mask = result_params.index.get_level_values("category") == "transition"
    shock_mask = result_params.index.get_level_values("category") == "shock_sds"

    trans_params = jnp.array(result_params.loc[trans_mask, "value"].to_numpy())
    shock_sds = jnp.array(result_params.loc[shock_mask, "value"].to_numpy())

    new_components: list[MixtureComponent] = []
    for component in prev_distribution.components:
        # Sample previous distribution at quadrature nodes
        # theta_{t-1} = mu + L @ z_q for each node z_q
        theta_samples = (
            component.mean[None, :] + state_nodes @ component.chol_cov.T
        )  # (n_nodes, n_factors)

        # Propagate each sample through transition function
        propagated = jax.vmap(combined_transition, in_axes=(0, None))(
            theta_samples, trans_params
        )  # (n_nodes, n_factors)

        # Moment matching: compute weighted mean and covariance
        new_mean = jnp.sum(state_weights[:, None] * propagated, axis=0)  # (n_factors,)

        centered = propagated - new_mean[None, :]
        new_cov = jnp.einsum(
            "q,qi,qj->ij", state_weights, centered, centered
        ) + jnp.diag(shock_sds**2)

        # Cholesky factorization of new covariance
        new_chol = jnp.linalg.cholesky(new_cov + 1e-8 * jnp.eye(n_factors))

        new_components.append(MixtureComponent(mean=new_mean, chol_cov=new_chol))

    return ConditionalDistribution(
        mixture_weights=prev_distribution.mixture_weights,
        components=tuple(new_components),
        conditional_weights=prev_distribution.conditional_weights,
    )
