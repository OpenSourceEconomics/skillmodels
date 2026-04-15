"""JAX-based likelihood functions for AF estimation.

All functions are JAX-compatible (jittable, differentiable via jax.grad).
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def af_loglike_initial(
    free_params: Array,
    *,
    all_params: Array,
    free_mask: Array,
    n_factors: int,
    n_mixture_components: int,
    n_measures: int,
    n_controls: int,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    nodes: Array,
    weights: Array,
    stability_floor: float,
) -> Array:
    """Negative log-likelihood for the initial period (Step 0).

    Integrate over latent factors using Halton quadrature:

        L_i = sum_q w_q * [sum_l pi_l * N(z_q | mu_l, Sigma_l)]
              * prod_m N(Z_{0,m,i} | c_m + lambda_m' z_q, sigma_{eps,m}^2)

    where q indexes quadrature nodes, l indexes mixture components, and
    m indexes measurements.

    Args:
        free_params: Free (non-fixed) parameter values.
        all_params: Full parameter vector with fixed values pre-filled.
        free_mask: Boolean mask, True for free parameters.
        n_factors: Number of latent factors.
        n_mixture_components: Number of mixture components.
        n_measures: Number of measurement variables in period 0.
        n_controls: Number of control variables (including constant).
        measurements: Shape (n_obs, n_measures), observed measurements.
        controls: Shape (n_obs, n_controls), control variable values.
        loading_mask: Shape (n_measures, n_factors), True where loading exists.
        nodes: Shape (n_nodes, n_factors), standard normal quadrature nodes.
        weights: Shape (n_nodes,), quadrature weights.
        stability_floor: Small constant added for numerical stability.

    Return:
        Scalar negative log-likelihood.

    """
    params = all_params.at[free_mask].set(free_params)

    parsed = _parse_initial_params(
        params,
        n_factors,
        n_mixture_components,
        n_measures,
        n_controls,
    )

    # Evaluate likelihood per observation
    log_likes = _initial_loglike_per_obs(
        mixture_weights=parsed["mixture_weights"],
        mixture_means=parsed["mixture_means"],
        mixture_chol_covs=parsed["mixture_chol_covs"],
        control_params=parsed["control_params"],
        loadings=parsed["loadings"],
        meas_sds=parsed["meas_sds"],
        measurements=measurements,
        controls=controls,
        loading_mask=loading_mask,
        nodes=nodes,
        weights=weights,
        stability_floor=stability_floor,
    )

    return -jnp.mean(log_likes)


def _parse_initial_params(
    params: Array,
    n_factors: int,
    n_mixture_components: int,
    n_measures: int,
    n_controls: int,
) -> dict[str, Array]:
    """Parse flat parameter vector into structured initial-period params."""
    idx = 0

    # Mixture weights
    mixture_weights = params[idx : idx + n_mixture_components]
    mixture_weights = mixture_weights / mixture_weights.sum()
    idx += n_mixture_components

    # Mixture means: (n_components, n_factors)
    n_mean = n_mixture_components * n_factors
    mixture_means = params[idx : idx + n_mean].reshape(n_mixture_components, n_factors)
    idx += n_mean

    # Mixture Cholesky covariances: (n_components, n_factors, n_factors) lower tri
    n_chol = n_factors * (n_factors + 1) // 2
    mixture_chol_covs = jnp.zeros((n_mixture_components, n_factors, n_factors))
    for m in range(n_mixture_components):
        chol_flat = params[idx : idx + n_chol]
        idx += n_chol
        chol = jnp.zeros((n_factors, n_factors))
        chol = chol.at[jnp.tril_indices(n_factors)].set(chol_flat)
        mixture_chol_covs = mixture_chol_covs.at[m].set(chol)

    # Control params: (n_measures, n_controls)
    n_ctrl = n_measures * n_controls
    control_params = params[idx : idx + n_ctrl].reshape(n_measures, n_controls)
    idx += n_ctrl

    # Loadings: (n_measures, n_factors) -- sparse, packed
    n_loadings = int(params.shape[0]) - idx - n_measures
    loadings_flat = params[idx : idx + n_loadings]
    idx += n_loadings

    # Measurement SDs
    meas_sds = params[idx : idx + n_measures]

    return {
        "mixture_weights": mixture_weights,
        "mixture_means": mixture_means,
        "mixture_chol_covs": mixture_chol_covs,
        "control_params": control_params,
        "loadings_flat": loadings_flat,
        "loadings": loadings_flat,  # Will be expanded using loading_mask
        "meas_sds": meas_sds,
    }


def _initial_loglike_per_obs(
    *,
    mixture_weights: Array,
    mixture_means: Array,
    mixture_chol_covs: Array,
    control_params: Array,
    loadings: Array,
    meas_sds: Array,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    nodes: Array,
    weights: Array,
    stability_floor: float,
) -> Array:
    """Compute log-likelihood for each observation at the initial period.

    Return:
        Shape (n_obs,) log-likelihood per observation.

    """
    # Expand loadings into full matrix using mask
    n_measures, n_factors = loading_mask.shape
    full_loadings = jnp.zeros((n_measures, n_factors))
    full_loadings = full_loadings.at[loading_mask].set(loadings)

    # Control contribution: (n_obs, n_measures)
    control_contrib = controls @ control_params.T

    # Residuals before factor contribution: (n_obs, n_measures)
    residuals_base = measurements - control_contrib

    def _single_obs_loglike(residual_base: Array) -> Array:
        """Log-likelihood for a single observation, integrated over factors."""
        return _integrate_initial_single_obs(
            residual_base=residual_base,
            full_loadings=full_loadings,
            meas_sds=meas_sds,
            mixture_weights=mixture_weights,
            mixture_means=mixture_means,
            mixture_chol_covs=mixture_chol_covs,
            nodes=nodes,
            weights=weights,
            stability_floor=stability_floor,
        )

    return jax.vmap(_single_obs_loglike)(residuals_base)


def _integrate_initial_single_obs(
    *,
    residual_base: Array,
    full_loadings: Array,
    meas_sds: Array,
    mixture_weights: Array,
    mixture_means: Array,
    mixture_chol_covs: Array,
    nodes: Array,
    weights: Array,
    stability_floor: float,
) -> Array:
    """Quadrature integration for one observation at the initial period.

    For each quadrature node z_q and mixture component l::

        theta_q,l = mu_l + L_l @ z_q
        kernel = pi_l * N(theta_q,l | mu_l, Sigma_l)
                 * prod_m N(obs_m | loading_m' theta_q,l, sd_m^2)

    Since z_q is standard normal and we transform
    theta = mu_l + L_l @ z_q, the density of the mixture at theta is
    already accounted for by the quadrature (importance sampling with
    the mixture as proposal). So we just need::

        kernel = sum_l pi_l * |L_l|
                 * prod_m N(obs_m | loading_m' (mu_l + L_l @ z_q),
                            sd_m^2)

    But with Halton nodes from N(0,I), the correct formula is::

        L_i = sum_q w_q * sum_l pi_l
              * prod_m N(residual_m
                         - loading_m' (mu_l + L_l z_q), 0, sd_m)

    """
    n_components = mixture_weights.shape[0]

    def _node_contribution(z_q: Array) -> Array:
        """Contribution from one quadrature node."""
        total = jnp.array(0.0)

        for l_idx in range(n_components):
            # Transform node to factor space for component l
            theta_q = mixture_means[l_idx] + mixture_chol_covs[l_idx] @ z_q

            # Measurement residuals: obs - control_contrib - loadings @ theta
            residuals = residual_base - full_loadings @ theta_q

            # Log measurement density: sum of log N(residual_m, 0, sd_m)
            log_meas_density = jnp.sum(
                _log_normal_pdf(residuals, jnp.zeros_like(residuals), meas_sds)
            )

            total = total + mixture_weights[l_idx] * jnp.exp(log_meas_density)

        return total

    # Integrate over quadrature nodes
    contributions = jax.vmap(_node_contribution)(nodes)
    integrated = jnp.dot(weights, contributions)

    return jnp.log(integrated + stability_floor)


def af_loglike_transition(
    free_params: Array,
    *,
    all_params: Array,
    free_mask: Array,
    n_state_factors: int,
    n_measures: int,
    n_controls: int,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    prev_distribution: dict[str, Array],
    state_nodes: Array,
    state_weights: Array,
    shock_nodes: Array,
    shock_weights: Array,
    transition_func: Callable,
    n_transition_params: int,
    stability_floor: float,
) -> Array:
    """Negative log-likelihood for a transition period (Step t).

    Integrate over latent factors at period t and production shocks:

        L_i = sum_q w_q * f(theta_q | prev_data_i)
              * prod_m N(Z_{t,m,i} | c_m + lambda_m' theta_q, sigma_m)
              * [sum_r w_r * prod_m N(Z_{t+1,m,i} | ... f(theta_q) + sd*eta_r ...)]

    For models without future measurements at t+1, the last term is omitted.

    Args:
        free_params: Free parameter values.
        all_params: Full parameter vector with fixed values.
        free_mask: Boolean mask for free parameters.
        n_state_factors: Number of state factors with transition equations.
        n_measures: Number of measurements at period t.
        n_controls: Number of controls.
        measurements: Shape (n_obs, n_measures), measurements at period t.
        controls: Shape (n_obs, n_controls), controls at period t.
        loading_mask: Shape (n_measures, n_state_factors), loading mask.
        prev_distribution: Dict with keys "cond_weights"
            (n_obs, n_components), "means" (n_components, n_factors),
            "chol_covs" (n_components, n_factors, n_factors).
        state_nodes: Shape (n_nodes, n_factors), standard normal nodes.
        state_weights: Shape (n_nodes,), quadrature weights.
        shock_nodes: Shape (n_shock_nodes, n_state_factors), shock nodes.
        shock_weights: Shape (n_shock_nodes,), shock weights.
        transition_func: Vectorized transition function f(states, params) -> states.
        n_transition_params: Number of transition function parameters per factor.
        stability_floor: Numerical stability floor.

    Return:
        Scalar negative log-likelihood.

    """
    params = all_params.at[free_mask].set(free_params)

    parsed = _parse_transition_params(
        params,
        n_state_factors,
        n_measures,
        n_controls,
        n_transition_params,
    )

    log_likes = _transition_loglike_per_obs(
        transition_params=parsed["transition_params"],
        shock_sds=parsed["shock_sds"],
        control_params=parsed["control_params"],
        loadings_flat=parsed["loadings_flat"],
        meas_sds=parsed["meas_sds"],
        measurements=measurements,
        controls=controls,
        loading_mask=loading_mask,
        prev_distribution=prev_distribution,
        state_nodes=state_nodes,
        state_weights=state_weights,
        shock_nodes=shock_nodes,
        shock_weights=shock_weights,
        transition_func=transition_func,
        stability_floor=stability_floor,
    )

    return -jnp.mean(log_likes)


def _parse_transition_params(
    params: Array,
    n_state_factors: int,
    n_measures: int,
    n_controls: int,
    n_transition_params: int,
) -> dict[str, Array]:
    """Parse flat parameter vector for a transition period."""
    idx = 0

    # Transition parameters per factor
    total_trans = n_state_factors * n_transition_params
    transition_params = params[idx : idx + total_trans].reshape(
        n_state_factors,
        n_transition_params,
    )
    idx += total_trans

    # Shock SDs per factor
    shock_sds = params[idx : idx + n_state_factors]
    idx += n_state_factors

    # Control params: (n_measures, n_controls)
    n_ctrl = n_measures * n_controls
    control_params = params[idx : idx + n_ctrl].reshape(n_measures, n_controls)
    idx += n_ctrl

    # Packed loadings
    n_loadings = int(params.shape[0]) - idx - n_measures
    loadings_flat = params[idx : idx + n_loadings]
    idx += n_loadings

    # Measurement SDs
    meas_sds = params[idx : idx + n_measures]

    return {
        "transition_params": transition_params,
        "shock_sds": shock_sds,
        "control_params": control_params,
        "loadings_flat": loadings_flat,
        "meas_sds": meas_sds,
    }


def _transition_loglike_per_obs(
    *,
    transition_params: Array,
    shock_sds: Array,
    control_params: Array,
    loadings_flat: Array,
    meas_sds: Array,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    prev_distribution: dict[str, Array],
    state_nodes: Array,
    state_weights: Array,
    shock_nodes: Array,
    shock_weights: Array,
    transition_func: Callable,
    stability_floor: float,
) -> Array:
    """Compute per-observation log-likelihood for a transition period."""
    n_measures, n_factors = loading_mask.shape
    full_loadings = jnp.zeros((n_measures, n_factors))
    full_loadings = full_loadings.at[loading_mask].set(loadings_flat)

    control_contrib = controls @ control_params.T
    residuals_base = measurements - control_contrib

    cond_weights = prev_distribution["cond_weights"]
    means = prev_distribution["means"]
    chol_covs = prev_distribution["chol_covs"]

    def _single_obs(residual_base: Array, obs_cond_weights: Array) -> Array:
        return _integrate_transition_single_obs(
            residual_base=residual_base,
            full_loadings=full_loadings,
            meas_sds=meas_sds,
            obs_cond_weights=obs_cond_weights,
            means=means,
            chol_covs=chol_covs,
            state_nodes=state_nodes,
            state_weights=state_weights,
            _shock_nodes=shock_nodes,
            _shock_weights=shock_weights,
            _transition_func=transition_func,
            _transition_params=transition_params,
            _shock_sds=shock_sds,
            stability_floor=stability_floor,
        )

    return jax.vmap(_single_obs)(residuals_base, cond_weights)


def _integrate_transition_single_obs(
    *,
    residual_base: Array,
    full_loadings: Array,
    meas_sds: Array,
    obs_cond_weights: Array,
    means: Array,
    chol_covs: Array,
    state_nodes: Array,
    state_weights: Array,
    _shock_nodes: Array,
    _shock_weights: Array,
    _transition_func: Callable,
    _transition_params: Array,
    _shock_sds: Array,
    stability_floor: float,
) -> Array:
    """Quadrature integration for one observation at a transition period.

    Integrate over (theta_t) and production shocks (eta_t).
    The measurement likelihood at period t is evaluated at each quadrature node.
    """
    n_components = obs_cond_weights.shape[0]

    def _node_contribution(z_q: Array) -> Array:
        total = jnp.array(0.0)

        for l_idx in range(n_components):
            theta_q = means[l_idx] + chol_covs[l_idx] @ z_q

            # Measurement density at period t
            residuals = residual_base - full_loadings @ theta_q
            log_meas = jnp.sum(
                _log_normal_pdf(residuals, jnp.zeros_like(residuals), meas_sds)
            )

            total = total + obs_cond_weights[l_idx] * jnp.exp(log_meas)

        return total

    contributions = jax.vmap(_node_contribution)(state_nodes)
    integrated = jnp.dot(state_weights, contributions)

    return jnp.log(integrated + stability_floor)


def _log_normal_pdf(x: Array, mean: Array, sd: Array) -> Array:
    """Log of normal PDF, element-wise."""
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sd) - 0.5 * ((x - mean) / sd) ** 2


def create_loglike_and_gradient(
    loglike_fn: Callable,
    **kwargs: Any,  # noqa: ANN401
) -> Callable:
    """Create a jitted function returning (loglike, gradient).

    Args:
        loglike_fn: The negative log-likelihood function.
        **kwargs: Keyword arguments to partially apply (data, nodes, etc.).

    Return:
        Function mapping free_params -> (neg_loglike, gradient).

    """
    partial_fn = functools.partial(loglike_fn, **kwargs)
    value_and_grad_fn = jax.value_and_grad(partial_fn)
    return jax.jit(value_and_grad_fn)
