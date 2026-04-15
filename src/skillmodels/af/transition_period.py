"""Step t (t >= 1) of the AF estimator: transition period estimation.

Estimate transition function parameters and measurement system parameters
using Halton quadrature over the latent factor distribution from the
previous period.
"""

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
    create_af_params_template,
    get_free_mask,
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
    prev_distribution: ConditionalDistribution,
    af_options: AFEstimationOptions,
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
        prev_distribution: Estimated conditional distribution from period t-1.
        af_options: AF estimation options.

    Return:
        Tuple of (AFPeriodResult, ConditionalDistribution) where the
        distribution represents f(theta_t | data_{0:t}).

    """
    n_factors = processed_model.dimensions.n_latent_factors
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls

    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    all_measures = _get_ordered_measures(measurements_pt)

    # Get transition function info
    # For now, use the first non-constant factor's transition for the combined function
    transition_info = processed_model.transition_info

    params_index = get_transition_period_params_index(
        period=period,
        latent_factors=factors,
        transition_info=transition_info,
        measurements_at_period=measurements_pt,
        controls=controls_names,
    )
    normalizations = get_normalizations_for_period(model_spec.factors, period=period)
    params_template = create_af_params_template(
        params_index,
        normalizations,
        period=period,
    )

    # Initialize transition params to reasonable defaults
    params_template = _initialize_transition_params(params_template, measurements)

    # Build loading mask
    loading_mask = _build_loading_mask(all_measures, factors, measurements_pt)

    # Halton quadrature nodes for factor integration
    state_nodes, state_weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_factors,
    )
    shock_nodes, shock_weights = create_shock_nodes_and_weights(
        af_options.n_halton_points_shock,
        n_factors,
    )

    prev_dist_arrays, n_transition_params = _prepare_transition_inputs(
        prev_distribution,
        transition_info,
        factors,
        measurements.shape[0],
    )

    # Build combined transition function that applies each factor's function
    def combined_transition(states: Array, params: Array) -> Array:
        """Apply per-factor transition functions."""
        result = jnp.zeros_like(states[:n_factors])
        p_idx = 0
        for i, factor in enumerate(factors):
            func = transition_info.individual_functions[factor]
            n_p = len(transition_info.param_names[factor])
            factor_params = params[p_idx : p_idx + n_p]
            result = result.at[i].set(func(states, factor_params))  # noqa: PD008
            p_idx += n_p
        return result

    # Set up optimization
    free_mask_np = get_free_mask(params_template)
    free_mask = jnp.array(free_mask_np)
    all_params_init = jnp.array(params_template["value"].to_numpy())

    loglike_kwargs = {
        "all_params": all_params_init,
        "free_mask": free_mask,
        "n_state_factors": n_factors,
        "n_measures": len(all_measures),
        "n_controls": len(controls_names),
        "measurements": measurements,
        "controls": controls,
        "loading_mask": jnp.array(loading_mask),
        "prev_distribution": prev_dist_arrays,
        "state_nodes": state_nodes,
        "state_weights": state_weights,
        "shock_nodes": shock_nodes,
        "shock_weights": shock_weights,
        "transition_func": combined_transition,
        "n_transition_params": n_transition_params,
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

    free_index = params_template.index[free_mask_np]
    free_params_df = pd.DataFrame(
        {
            "value": params_template.loc[free_index, "value"].to_numpy(),
            "lower_bound": params_template.loc[free_index, "lower_bound"].to_numpy(),
            "upper_bound": params_template.loc[free_index, "upper_bound"].to_numpy(),
        },
        index=free_index,
    )

    opt_res = om.minimize(
        fun=fun,
        params=free_params_df[["value"]],
        algorithm=af_options.optimizer_algorithm,
        bounds=om.Bounds(
            lower=free_params_df["lower_bound"],
            upper=free_params_df["upper_bound"],
        ),
        fun_and_jac=fun_and_jac,
        **dict(af_options.optimizer_options),
    )

    result_params = params_template.copy()
    result_params.loc[free_index, "value"] = opt_res.params["value"].to_numpy()

    # Update conditional distribution for the next period
    # For now, propagate the previous distribution (proper update with
    # production function will be implemented in a refinement pass)
    updated_dist = _update_conditional_distribution(
        prev_distribution=prev_distribution,
        result_params=result_params,
        _transition_info=transition_info,
        _factors=factors,
        _n_factors=n_factors,
    )

    period_result = AFPeriodResult(
        period=period,
        params=result_params,
        loglikelihood=-float(opt_res.fun),
        success=bool(opt_res.success),
        optimize_result=opt_res,
    )

    return period_result, updated_dist


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

    n_transition_params = 0
    for factor in factors:
        if factor in transition_info.param_names:
            n_tp = len(transition_info.param_names[factor])
            n_transition_params = max(n_transition_params, n_tp)

    return prev_dist_arrays, n_transition_params


def _initialize_transition_params(
    params_template: pd.DataFrame,
    measurements: Array,
) -> pd.DataFrame:
    """Initialize transition period parameters with reasonable defaults."""
    params = params_template.copy()
    meas_np = np.array(measurements)

    # Transition params: small values (near identity)
    trans_mask = params.index.get_level_values("category") == "transition"
    for idx in params.index[trans_mask]:
        if params.loc[idx, "lower_bound"] != params.loc[idx, "upper_bound"]:
            # Set linear terms close to identity
            params.loc[idx, "value"] = 0.1

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

    return params


def _update_conditional_distribution(
    prev_distribution: ConditionalDistribution,
    result_params: pd.DataFrame,
    _transition_info: TransitionInfo,
    _factors: tuple[str, ...],
    _n_factors: int,
) -> ConditionalDistribution:
    """Update the conditional distribution for the next period.

    Apply the estimated transition function to propagate the distribution
    forward. For the MVP, this uses a simple mean propagation; a full
    implementation would integrate over the production function.
    """
    # Extract estimated shock SDs
    shock_mask = result_params.index.get_level_values("category") == "shock_sds"

    shock_sds = jnp.array(result_params.loc[shock_mask, "value"].to_numpy())

    # For each mixture component, propagate the mean through the transition
    # and inflate the covariance by the shock variance
    new_components: list[MixtureComponent] = []
    for component in prev_distribution.components:
        # Simple propagation: mean stays (transition is applied in likelihood),
        # covariance grows by shock variance
        new_cov_diag = jnp.diag(component.chol_cov) ** 2 + shock_sds**2
        new_chol = jnp.diag(jnp.sqrt(new_cov_diag))

        new_components.append(
            MixtureComponent(
                mean=component.mean,
                chol_cov=new_chol,
            )
        )

    return ConditionalDistribution(
        mixture_weights=prev_distribution.mixture_weights,
        components=tuple(new_components),
        conditional_weights=prev_distribution.conditional_weights,
    )
