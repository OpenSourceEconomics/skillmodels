"""JAX-based likelihood functions for AF estimation.

All functions are JAX-compatible (jittable, differentiable via jax.grad).
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def af_per_obs_loglike_initial(
    params: Array,
    *,
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
    n_latent_factors: int | None = None,
    observed_factor_values: Array | None = None,
    n_obs_per_batch: int | None = None,
) -> Array:
    """Per-observation log-likelihood for the initial period (Step 0).

    Same inputs as `af_loglike_initial`; returns the shape-``(n_obs,)``
    vector of per-observation log-likelihoods instead of the aggregated
    negative mean. Used for score-based inference.
    """
    n_latent = n_factors if n_latent_factors is None else n_latent_factors
    n_obs_factors = n_factors - n_latent

    parsed = _parse_initial_params(
        params,
        n_factors,
        n_mixture_components,
        n_measures,
        n_controls,
    )

    if n_obs_factors == 0:
        return _initial_loglike_per_obs(
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
            n_obs_per_batch=n_obs_per_batch,
        )
    assert observed_factor_values is not None  # noqa: S101
    return _initial_loglike_per_obs_conditional(
        mixture_weights=parsed["mixture_weights"],
        mixture_means=parsed["mixture_means"],
        mixture_chol_covs=parsed["mixture_chol_covs"],
        control_params=parsed["control_params"],
        loadings=parsed["loadings"],
        meas_sds=parsed["meas_sds"],
        measurements=measurements,
        controls=controls,
        observed_factor_values=observed_factor_values,
        loading_mask=loading_mask,
        nodes=nodes,
        weights=weights,
        n_latent=n_latent,
        stability_floor=stability_floor,
        n_obs_per_batch=n_obs_per_batch,
    )


def af_loglike_initial(
    params: Array,
    *,
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
    n_latent_factors: int | None = None,
    observed_factor_values: Array | None = None,
    n_obs_per_batch: int | None = None,
) -> Array:
    """Negative log-likelihood for the initial period (Step 0).

    Integrate over latent factors using Halton quadrature.

    When `n_latent_factors == n_factors` (no observed factors in the joint
    distribution), the likelihood reduces to::

        L_i = sum_q w_q * sum_l pi_l
              * prod_m N(Z_{0,m,i} | c_m + lam_m' theta_q,l, sd_m)

    where theta_q,l = mu_l + L_l @ z_q.

    When `n_latent_factors < n_factors` (joint distribution over
    (latent, observed)), for each individual i::

        L_i = p(Y_i) * sum_q w_q * sum_l pi_{l|Y_i}
              * prod_m N(Z_{0,m,i} | c_m + lam_m' theta_{q,l|Y_i}, sd_m)

    where theta_{q,l|Y_i} is drawn from the conditional N(mu_{theta|Y,l,i},
    Sigma_{theta|Y,l}) via the Schur complement, and pi_{l|Y_i} are the
    posterior component weights given Y_i.

    Args:
        params: Full parameter vector in template order. Fixed entries are
            held constant by optimagic `FixedConstraint`s attached outside.
        n_factors: Number of factors in the joint initial distribution
            (state latents + observed). Reconstructed factors
            (``has_initial_distribution=False``) are excluded from this
            count; their period-0 measurements are estimated in the
            period 0->1 transition step instead.
        n_mixture_components: Number of mixture components.
        n_measures: Number of measurement variables in period 0.
        n_controls: Number of control variables (including constant).
        measurements: Shape (n_obs, n_measures), observed measurements.
        controls: Shape (n_obs, n_controls), control variable values.
        loading_mask: Shape (n_measures, n_state_latent), True where loading
            exists.
        nodes: Shape (n_nodes, n_state_latent), standard normal quadrature
            nodes.
        weights: Shape (n_nodes,), quadrature weights.
        stability_floor: Small constant added for numerical stability.
        n_latent_factors: Number of state latent factors in the mixture.
            Defaults to ``n_factors`` when no observed factors are present.
        observed_factor_values: Shape (n_obs, n_obs_factors), observed factor
            values used for Schur-complement conditioning. Required when
            ``n_latent_factors < n_factors``.
        n_obs_per_batch: Observations per reverse-mode autodiff chunk.
            ``None`` falls back to ``jax.vmap`` (single kernel); a positive
            integer uses ``jax.lax.map`` so the backward-pass tape only
            retains one chunk at a time.

    Return:
        Scalar negative log-likelihood.

    """
    log_likes = af_per_obs_loglike_initial(
        params,
        n_factors=n_factors,
        n_mixture_components=n_mixture_components,
        n_measures=n_measures,
        n_controls=n_controls,
        measurements=measurements,
        controls=controls,
        loading_mask=loading_mask,
        nodes=nodes,
        weights=weights,
        stability_floor=stability_floor,
        n_latent_factors=n_latent_factors,
        observed_factor_values=observed_factor_values,
        n_obs_per_batch=n_obs_per_batch,
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
        "loadings": loadings_flat,
        "meas_sds": meas_sds,
    }


def _map_over_obs(
    f: Callable,
    *xs: Array,
    n_obs_per_batch: int | None,
) -> Array:
    """Map ``f`` over the leading axis of ``xs``, optionally in batches.

    When ``n_obs_per_batch`` is ``None`` or at least as large as the
    leading axis, falls back to ``jax.vmap`` (single kernel). Otherwise
    uses ``jax.lax.map`` so the reverse-mode autodiff tape only needs to
    retain one chunk at a time. Combined with ``jax.checkpoint`` on
    ``f``, this makes reverse-mode memory proportional to
    ``n_obs_per_batch`` rather than to the full ``n_obs``.
    """
    n_obs = xs[0].shape[0]
    if n_obs_per_batch is None or n_obs_per_batch >= n_obs:
        return jax.vmap(f)(*xs)

    def _tupled(args: tuple[Array, ...]) -> Array:
        return f(*args)

    return jax.lax.map(_tupled, xs, batch_size=n_obs_per_batch)


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
    n_obs_per_batch: int | None = None,
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

    @jax.checkpoint
    def _single_obs_loglike(residual_base: Array) -> Array:
        """Log-likelihood for a single observation, integrated over factors.

        `jax.checkpoint` keeps the forward pass small: the per-observation
        quadrature tape is discarded and recomputed during the backward
        pass, so reverse-mode autodiff memory scales with the per-obs
        parameter footprint instead of ``n_obs * n_quadrature_nodes``.
        """
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

    return _map_over_obs(
        _single_obs_loglike, residuals_base, n_obs_per_batch=n_obs_per_batch
    )


def _initial_loglike_per_obs_conditional(
    *,
    mixture_weights: Array,
    mixture_means: Array,
    mixture_chol_covs: Array,
    control_params: Array,
    loadings: Array,
    meas_sds: Array,
    measurements: Array,
    controls: Array,
    observed_factor_values: Array,
    loading_mask: Array,
    nodes: Array,
    weights: Array,
    n_latent: int,
    stability_floor: float,
    n_obs_per_batch: int | None = None,
) -> Array:
    """Per-observation log-likelihood with Schur-complement conditioning.

    For each individual i with observed factors Y_i, the likelihood is::

        L_i = p(Y_i) * integral p(Z_i | theta) p(theta | Y_i) dtheta
            = sum_l pi_l N(Y_i | mu_Y_l, Sigma_YY_l)
              * sum_q w_q prod_m N(residual_m | 0, sd_m)

    where theta is drawn from p(theta | Y_i, component l) using the
    conditional mean and Cholesky factor derived from the joint
    (latent, observed) covariance matrix via the Schur complement.

    Note the identity: combining the log-mixture over components l with
    the measurement density gives an equivalent formulation where each
    component's contribution is weighted by pi_l * N(Y_i | mu_Y_l, Sigma_YY_l).

    """
    n_measures = loading_mask.shape[0]
    full_loadings = jnp.zeros((n_measures, n_latent))
    full_loadings = full_loadings.at[loading_mask].set(loadings)

    control_contrib = controls @ control_params.T
    residuals_base = measurements - control_contrib

    @jax.checkpoint
    def _single_obs_loglike(residual_base: Array, y_i: Array) -> Array:
        return _integrate_initial_single_obs_conditional(
            residual_base=residual_base,
            y_i=y_i,
            full_loadings=full_loadings,
            meas_sds=meas_sds,
            mixture_weights=mixture_weights,
            mixture_means=mixture_means,
            mixture_chol_covs=mixture_chol_covs,
            nodes=nodes,
            weights=weights,
            n_latent=n_latent,
            stability_floor=stability_floor,
        )

    return _map_over_obs(
        _single_obs_loglike,
        residuals_base,
        observed_factor_values,
        n_obs_per_batch=n_obs_per_batch,
    )


def _integrate_initial_single_obs_conditional(
    *,
    residual_base: Array,
    y_i: Array,
    full_loadings: Array,
    meas_sds: Array,
    mixture_weights: Array,
    mixture_means: Array,
    mixture_chol_covs: Array,
    nodes: Array,
    weights: Array,
    n_latent: int,
    stability_floor: float,
) -> Array:
    """Quadrature integration for one individual with observed-factor conditioning.

    Per component l:
    - Split joint (mu, L) into latent and observed blocks.
    - Compute marginal p(Y_i | l) from (mu_Y_l, L_Y_l).
    - Compute conditional mean mu_{theta | Y_i, l} and Cholesky L_{theta | Y, l}
      via Schur complement.
    - Transform nodes: theta_q = mu_{theta|Y,l} + L_{theta|Y,l} @ z_q.
    - Evaluate measurement density at theta_q, sum over quadrature.

    Aggregate with log-sum-exp over components.
    """
    n_components = mixture_weights.shape[0]

    def _component_log_kernel(l_idx: Array) -> Array:
        mu_full = mixture_means[l_idx]
        chol_full = mixture_chol_covs[l_idx]
        cov_full = chol_full @ chol_full.T

        mu_theta = mu_full[:n_latent]
        mu_y = mu_full[n_latent:]
        cov_tt = cov_full[:n_latent, :n_latent]
        cov_ty = cov_full[:n_latent, n_latent:]
        cov_yy = cov_full[n_latent:, n_latent:]

        # Marginal density of Y_i under component l
        chol_yy = jnp.linalg.cholesky(cov_yy)
        log_marg_y = _log_mvn_pdf_chol(y_i, mu_y, chol_yy)

        # Conditional mean and Cholesky of theta | Y_i
        alpha = jax.scipy.linalg.cho_solve((chol_yy, True), (y_i - mu_y))
        cond_mean = mu_theta + cov_ty @ alpha
        # Sigma_{theta|Y} = Sigma_tt - Sigma_ty Sigma_yy^{-1} Sigma_yt
        solve_tt = jax.scipy.linalg.cho_solve((chol_yy, True), cov_ty.T)
        cond_cov = cov_tt - cov_ty @ solve_tt
        # Jitter for numerical stability before Cholesky
        cond_cov = cond_cov + 1e-10 * jnp.eye(n_latent)
        cond_chol = jnp.linalg.cholesky(cond_cov)

        def _log_node(z_q: Array) -> Array:
            theta_q = cond_mean + cond_chol @ z_q
            residuals = residual_base - full_loadings @ theta_q
            return jnp.sum(
                _log_normal_pdf(residuals, jnp.zeros_like(residuals), meas_sds)
            )

        log_meas = jax.vmap(_log_node)(nodes)
        log_integral = jax.scipy.special.logsumexp(log_meas + jnp.log(weights))

        return (
            jnp.log(mixture_weights[l_idx] + stability_floor)
            + log_marg_y
            + log_integral
        )

    comp_log = jax.vmap(_component_log_kernel)(jnp.arange(n_components))
    return jax.scipy.special.logsumexp(comp_log)


def _log_mvn_pdf_chol(x: Array, mean: Array, chol: Array) -> Array:
    """Log pdf of multivariate normal given the lower-triangular Cholesky."""
    diff = x - mean
    sol = jax.scipy.linalg.solve_triangular(chol, diff, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(chol)))
    k = x.shape[0]
    return -0.5 * k * jnp.log(2 * jnp.pi) - log_det - 0.5 * jnp.dot(sol, sol)


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


def af_per_obs_loglike_transition(
    params: Array,
    *,
    n_state_factors: int,
    n_endogenous_factors: int,
    n_measures: int,
    n_controls: int,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    prev_measurements: Array,
    prev_controls: Array,
    prev_loading_mask: Array,
    prev_control_params: Array,
    prev_loadings_flat: Array,
    prev_meas_sds: Array,
    prev_distribution: dict[str, Array],
    joint_nodes: Array,
    joint_weights: Array,
    transition_func: Callable,
    total_n_transition_params: int,
    total_n_inv_params: int,
    n_inv_eq_params_per: int,
    observed_factor_values: Array,
    stability_floor: float,
    state_factor_indices_in_latent: Array | None = None,
    n_shock_factors: int | None = None,
    shock_factor_indices: Array | None = None,
    n_obs_per_batch: int | None = None,
) -> Array:
    """Per-observation log-likelihood for a transition period (Step t).

    Same inputs as `af_loglike_transition`; returns the shape-``(n_obs,)``
    vector of per-observation log-likelihoods instead of the aggregated
    negative mean. Used for score-based inference.
    """
    effective_n_shock = n_state_factors if n_shock_factors is None else n_shock_factors
    if shock_factor_indices is None:
        shock_factor_indices = jnp.arange(effective_n_shock)
    if state_factor_indices_in_latent is None:
        # Default: assume state factors precede endogenous factors in the
        # latent-factor ordering (the existing convention). Callers that
        # don't follow that convention must pass explicit indices.
        state_factor_indices_in_latent = jnp.arange(n_state_factors)

    parsed = _parse_transition_params(
        params,
        n_state_factors,
        n_endogenous_factors,
        n_measures,
        n_controls,
        total_n_transition_params,
        total_n_inv_params,
        n_inv_eq_params_per,
        n_shock_factors=effective_n_shock,
    )

    n_prev_measures = prev_loading_mask.shape[0]
    n_prev_factors = prev_loading_mask.shape[1]
    prev_full_loadings = jnp.zeros((n_prev_measures, n_prev_factors))
    prev_full_loadings = prev_full_loadings.at[prev_loading_mask].set(
        prev_loadings_flat
    )
    prev_control_contrib = prev_controls @ prev_control_params.T
    prev_residuals_base = prev_measurements - prev_control_contrib

    return _transition_loglike_per_obs(
        transition_params=parsed["transition_params"],
        shock_sds=parsed["shock_sds"],
        inv_eq_params=parsed["inv_eq_params"],
        inv_sds=parsed["inv_sds"],
        control_params=parsed["control_params"],
        loadings_flat=parsed["loadings_flat"],
        meas_sds=parsed["meas_sds"],
        measurements=measurements,
        controls=controls,
        loading_mask=loading_mask,
        prev_residuals_base=prev_residuals_base,
        prev_full_loadings=prev_full_loadings,
        prev_meas_sds=prev_meas_sds,
        prev_distribution=prev_distribution,
        joint_nodes=joint_nodes,
        joint_weights=joint_weights,
        transition_func=transition_func,
        n_state_factors=n_state_factors,
        n_endogenous_factors=n_endogenous_factors,
        n_shock_factors=effective_n_shock,
        shock_factor_indices=shock_factor_indices,
        state_factor_indices_in_latent=state_factor_indices_in_latent,
        observed_factor_values=observed_factor_values,
        stability_floor=stability_floor,
        n_obs_per_batch=n_obs_per_batch,
    )


def af_loglike_transition(
    params: Array,
    *,
    n_state_factors: int,
    n_endogenous_factors: int,
    n_measures: int,
    n_controls: int,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    prev_measurements: Array,
    prev_controls: Array,
    prev_loading_mask: Array,
    prev_control_params: Array,
    prev_loadings_flat: Array,
    prev_meas_sds: Array,
    prev_distribution: dict[str, Array],
    joint_nodes: Array,
    joint_weights: Array,
    transition_func: Callable,
    total_n_transition_params: int,
    total_n_inv_params: int,
    n_inv_eq_params_per: int,
    observed_factor_values: Array,
    stability_floor: float,
    state_factor_indices_in_latent: Array | None = None,
    n_shock_factors: int | None = None,
    shock_factor_indices: Array | None = None,
    n_obs_per_batch: int | None = None,
) -> Array:
    """Negative log-likelihood for a transition period (Step t).

    Integrate over latent factors at period t-1 and production shocks.
    The likelihood conditions on individual data via re-evaluation of
    previous-period measurements at each quadrature node::

        L_i = sum_q w_q * sum_l pi_{l,i}
              * [prod_m N(Z_{t-1,m,i} | c~_m + lam~_m' th_{t-1}, sd~_m)]
              * [sum_r w_r * prod_m N(Z_{t,m,i} | c_m + lam_m' th_t, sd_m)]

    where ``th_t = f(th_{t-1}; delta) + sd_shock * eta_r`` and tildes
    denote already-estimated parameters from the previous step.

    Args:
        params: Full parameter vector in template order. Fixed entries are
            held constant by optimagic `FixedConstraint`s attached outside.
        n_state_factors: Number of state factors with transition equations.
        n_endogenous_factors: Number of endogenous (investment) factors.
        n_measures: Number of measurements at period t.
        n_controls: Number of controls at period t.
        measurements: Shape (n_obs, n_measures), measurements at period t.
        controls: Shape (n_obs, n_controls), controls at period t.
        loading_mask: Shape (n_measures, n_state_factors), loading mask.
        prev_measurements: Shape (n_obs, n_prev_measures), measurements t-1.
        prev_controls: Shape (n_obs, n_prev_controls), controls at t-1.
        prev_loading_mask: Shape (n_prev_measures, n_factors), prev loadings.
        prev_control_params: Shape (n_prev_measures, n_prev_controls), fixed.
        prev_loadings_flat: Packed loadings from previous period, fixed.
        prev_meas_sds: Shape (n_prev_measures,), fixed from previous step.
        prev_distribution: Dict with keys "cond_weights", "means", "chol_covs".
        joint_nodes: Shape (n_halton, n_state + n_shock + n_endogenous),
            standard-normal Halton draws partitioned into state, production
            shock, and investment shock components. `n_shock` equals
            `n_shock_factors` (defaults to `n_state_factors`).
        joint_weights: Shape (n_halton,) quadrature weights (uniform
            1/n_halton for Halton integration).
        transition_func: Combined transition f(states, params) -> new_states.
        total_n_transition_params: Total transition params across all factors.
        total_n_inv_params: Total investment equation parameters.
        n_inv_eq_params_per: Investment equation parameters per endogenous factor.
        observed_factor_values: Shape (n_obs, n_obs_factors), observed factor data.
        stability_floor: Numerical stability floor.
        state_factor_indices_in_latent: Shape (n_state_factors,) int array
            mapping each state factor to its column index in the
            previous-period loading mask (which is in `latent_factors` order
            = state + endogenous, possibly interleaved). Used to restrict
            the prev-meas factor to state-factor loadings, mirroring
            MATLAB's `create_nodes_weights_12` (which omits prev-period
            inv measurements from the chained-sample importance weight).
            Defaults to `arange(n_state_factors)` (assuming state factors
            precede endogenous in the latent ordering).
        n_shock_factors: Number of state factors that get a production shock.
            Defaults to `n_state_factors`. Factors without a shock are
            integrated deterministically (their shock dimension is dropped
            from the joint Halton draw).
        shock_factor_indices: Shape (n_shock_factors,) int array mapping each
            shock slot to its position in the state-factor ordering. Required
            when `n_shock_factors < n_state_factors`.
        n_obs_per_batch: Observations per reverse-mode autodiff chunk.
            ``None`` falls back to ``jax.vmap`` (single kernel); a positive
            integer uses ``jax.lax.map`` so the backward-pass tape only
            retains one chunk at a time.

    Return:
        Scalar negative log-likelihood.

    """
    log_likes = af_per_obs_loglike_transition(
        params,
        n_state_factors=n_state_factors,
        n_endogenous_factors=n_endogenous_factors,
        n_measures=n_measures,
        n_controls=n_controls,
        measurements=measurements,
        controls=controls,
        loading_mask=loading_mask,
        prev_measurements=prev_measurements,
        prev_controls=prev_controls,
        prev_loading_mask=prev_loading_mask,
        prev_control_params=prev_control_params,
        prev_loadings_flat=prev_loadings_flat,
        prev_meas_sds=prev_meas_sds,
        prev_distribution=prev_distribution,
        joint_nodes=joint_nodes,
        joint_weights=joint_weights,
        transition_func=transition_func,
        total_n_transition_params=total_n_transition_params,
        total_n_inv_params=total_n_inv_params,
        n_inv_eq_params_per=n_inv_eq_params_per,
        observed_factor_values=observed_factor_values,
        stability_floor=stability_floor,
        state_factor_indices_in_latent=state_factor_indices_in_latent,
        n_shock_factors=n_shock_factors,
        shock_factor_indices=shock_factor_indices,
        n_obs_per_batch=n_obs_per_batch,
    )
    return -jnp.mean(log_likes)


def _parse_transition_params(
    params: Array,
    n_state_factors: int,
    n_endogenous_factors: int,
    n_measures: int,
    n_controls: int,
    total_n_transition_params: int,
    total_n_inv_params: int,
    _n_inv_eq_params_per: int,
    *,
    n_shock_factors: int | None = None,
) -> dict[str, Array]:
    """Parse flat parameter vector for a transition period."""
    effective_n_shock = n_state_factors if n_shock_factors is None else n_shock_factors
    idx = 0

    # Transition parameters (flat, for state factors only)
    transition_params = params[idx : idx + total_n_transition_params]
    idx += total_n_transition_params

    # Shock SDs per shock-bearing state factor (subset of state factors).
    shock_sds = params[idx : idx + effective_n_shock]
    idx += effective_n_shock

    # Investment equation params (if any endogenous factors)
    inv_eq_params = params[idx : idx + total_n_inv_params]
    idx += total_n_inv_params

    # Investment shock SDs
    inv_sds = params[idx : idx + n_endogenous_factors]
    idx += n_endogenous_factors

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
        "inv_eq_params": inv_eq_params,
        "inv_sds": inv_sds,
        "control_params": control_params,
        "loadings_flat": loadings_flat,
        "meas_sds": meas_sds,
    }


def _transition_loglike_per_obs(
    *,
    transition_params: Array,
    shock_sds: Array,
    inv_eq_params: Array,
    inv_sds: Array,
    control_params: Array,
    loadings_flat: Array,
    meas_sds: Array,
    measurements: Array,
    controls: Array,
    loading_mask: Array,
    prev_residuals_base: Array,
    prev_full_loadings: Array,
    prev_meas_sds: Array,
    prev_distribution: dict[str, Array],
    joint_nodes: Array,
    joint_weights: Array,
    transition_func: Callable,
    n_state_factors: int,
    n_endogenous_factors: int,
    n_shock_factors: int,
    shock_factor_indices: Array,
    state_factor_indices_in_latent: Array,
    observed_factor_values: Array,
    stability_floor: float,
    n_obs_per_batch: int | None = None,
) -> Array:
    """Compute per-observation log-likelihood for a transition period."""
    n_measures, n_loading_factors = loading_mask.shape
    full_loadings = jnp.zeros((n_measures, n_loading_factors))
    full_loadings = full_loadings.at[loading_mask].set(loadings_flat)

    control_contrib = controls @ control_params.T
    residuals_base = measurements - control_contrib

    cond_weights = prev_distribution["cond_weights"]
    # samples shape (n_components, n_halton, n_obs, n_state). Re-shape to
    # (n_obs, n_components, n_halton, n_state) so we can map per-obs.
    samples_stacked = prev_distribution["samples_per_component"]
    samples_by_obs = jnp.transpose(samples_stacked, (2, 0, 1, 3))

    @jax.checkpoint
    def _single_obs(
        residual_base: Array,
        prev_residual_base: Array,
        obs_cond_weights: Array,
        obs_factor_values: Array,
        obs_samples: Array,
    ) -> Array:
        return _integrate_transition_single_obs(
            residual_base=residual_base,
            full_loadings=full_loadings,
            meas_sds=meas_sds,
            prev_residual_base=prev_residual_base,
            prev_full_loadings=prev_full_loadings,
            prev_meas_sds=prev_meas_sds,
            obs_cond_weights=obs_cond_weights,
            prev_samples_per_component=obs_samples,
            joint_nodes=joint_nodes,
            joint_weights=joint_weights,
            transition_func=transition_func,
            transition_params=transition_params,
            shock_sds=shock_sds,
            inv_eq_params=inv_eq_params,
            inv_sds=inv_sds,
            n_state_factors=n_state_factors,
            n_endogenous_factors=n_endogenous_factors,
            n_shock_factors=n_shock_factors,
            shock_factor_indices=shock_factor_indices,
            state_factor_indices_in_latent=state_factor_indices_in_latent,
            obs_factor_values=obs_factor_values,
            stability_floor=stability_floor,
        )

    return _map_over_obs(
        _single_obs,
        residuals_base,
        prev_residuals_base,
        cond_weights,
        observed_factor_values,
        samples_by_obs,
        n_obs_per_batch=n_obs_per_batch,
    )


def _compute_investment(
    theta_prev: Array,
    obs_factor_values: Array,
    inv_eq_params: Array,
    inv_sds: Array,
    eps_i: Array,
    n_endogenous_factors: int,
    n_state_factors: int,
) -> Array:
    """Compute investment from the AF investment equation.

    I_j = beta_0 + beta_k @ theta + beta_y @ Y + sigma_I * eps_I

    """
    n_obs_factors = obs_factor_values.shape[0]
    n_per = 1 + n_state_factors + n_obs_factors
    result = jnp.zeros(n_endogenous_factors)
    for j in range(n_endogenous_factors):
        beta = inv_eq_params[j * n_per : (j + 1) * n_per]
        intercept = beta[0]
        state_coeffs = beta[1 : 1 + n_state_factors]
        obs_coeffs = beta[1 + n_state_factors :]
        inv_j = (
            intercept
            + jnp.dot(state_coeffs, theta_prev)
            + jnp.dot(obs_coeffs, obs_factor_values)
            + inv_sds[j] * eps_i[j]
        )
        result = result.at[j].set(inv_j)
    return result


def _integrate_transition_single_obs(
    *,
    residual_base: Array,
    full_loadings: Array,
    meas_sds: Array,
    prev_residual_base: Array,
    prev_full_loadings: Array,
    prev_meas_sds: Array,
    obs_cond_weights: Array,
    prev_samples_per_component: Array,
    joint_nodes: Array,
    joint_weights: Array,
    transition_func: Callable,
    transition_params: Array,
    shock_sds: Array,
    inv_eq_params: Array,
    inv_sds: Array,
    n_state_factors: int,
    n_endogenous_factors: int,
    n_shock_factors: int,
    shock_factor_indices: Array,
    state_factor_indices_in_latent: Array,
    obs_factor_values: Array,
    stability_floor: float,
) -> Array:
    """Importance-sample integration for one observation at a transition period.

    The previous-period skills distribution is supplied as a Halton-driven
    importance sample ``prev_samples_per_component`` of shape
    ``(n_components, n_halton, n_state_factors)``. Each row j is a chained
    realisation of skills_{t-1} for this observation, built deterministically
    from the previous period's Halton design + the previous period's
    estimated parameters. This preserves the non-Gaussian shape of skills_{t-1}
    across periods (vs. the moment-matched Gaussian re-draw, which is the
    bug Mario Rothfelder identified that biased investment-shock SDs
    downward by ~50%).

    The joint Halton design at this period covers the *fresh* period-t
    shocks only:
    ``joint_nodes`` has shape ``(n_halton, n_shock_factors + n_endogenous_factors)``
    (no z_state column — that's absorbed into the importance sample).

    The previous-period measurement density factor is restricted to
    measurements that load on STATE factors only — endogenous-factor
    (investment) measurements at period t-1 are deliberately omitted,
    matching MATLAB's ``create_nodes_weights_12`` (which evaluates only
    skill measurements at the chained sample, not investment measurements;
    investment measurements at period t-1 were already evaluated as
    *current-period* measurements at the (t-2)→(t-1) step).
    ``state_factor_indices_in_latent`` selects the state-factor columns of
    ``prev_full_loadings`` regardless of how state vs. endogenous factors
    are interleaved in the latent-factor ordering.
    """
    n_components = obs_cond_weights.shape[0]

    def _log_draw_contribution(j_idx: Array) -> Array:
        """Per-draw log kernel at Halton index j, LogSumExp over mixture comps."""
        z_at_j = joint_nodes[j_idx]
        z_shock = z_at_j[:n_shock_factors]
        z_inv_shock = z_at_j[n_shock_factors:]

        log_component_vals = []
        for l_idx in range(n_components):
            theta_prev = prev_samples_per_component[l_idx, j_idx]
            inv = _compute_investment(
                theta_prev,
                obs_factor_values,
                inv_eq_params,
                inv_sds,
                z_inv_shock,
                n_endogenous_factors,
                n_state_factors,
            )
            full_prev_with_obs = jnp.concatenate([theta_prev, inv, obs_factor_values])

            # Previous-period measurement density: state-factor (skill)
            # measurements at theta_prev only. Endogenous-factor (inv)
            # measurements at t-1 are NOT re-evaluated here -- they were
            # already used as current-period measurements at the (t-2)->(t-1)
            # step (matches MATLAB's likelihood_12, which omits Z_inv_est_0
            # from the chained-sample importance weight). For rows that load
            # only on endogenous factors, the slice picks zero loadings and
            # the residual reduces to the centered measurement, contributing
            # a per-obs constant that is invariant under the parameters.
            prev_state_loadings = prev_full_loadings[:, state_factor_indices_in_latent]
            prev_residuals = prev_residual_base - prev_state_loadings @ theta_prev
            log_prev_inv_meas = jnp.sum(
                _log_normal_pdf(
                    prev_residuals,
                    jnp.zeros_like(prev_residuals),
                    prev_meas_sds,
                )
            )

            # Current-period measurement density. Shocks only apply to
            # factors with has_production_shock=True; scatter them into the
            # state-factor ordering and leave deterministic factors as is.
            state_shock_contrib = (
                jnp.zeros(n_state_factors)
                .at[shock_factor_indices]
                .set(shock_sds * z_shock)
            )
            theta_t = (
                transition_func(full_prev_with_obs, transition_params)
                + state_shock_contrib
            )
            all_factors_t = jnp.concatenate([theta_t, inv])
            residuals = residual_base - full_loadings @ all_factors_t
            log_meas = jnp.sum(
                _log_normal_pdf(residuals, jnp.zeros_like(residuals), meas_sds)
            )

            log_kernel = (
                jnp.log(obs_cond_weights[l_idx] + stability_floor)
                + log_prev_inv_meas
                + log_meas
            )
            log_component_vals.append(log_kernel)

        return jax.scipy.special.logsumexp(jnp.array(log_component_vals))

    n_halton = joint_nodes.shape[0]
    log_contribs = jax.vmap(_log_draw_contribution)(jnp.arange(n_halton))
    return jax.scipy.special.logsumexp(log_contribs + jnp.log(joint_weights))


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
