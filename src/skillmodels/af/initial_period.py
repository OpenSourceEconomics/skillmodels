"""Step 0 of the AF estimator: initial period estimation.

Estimate the joint distribution of latent factors at period 0 and the
measurement system parameters, using a mixture-of-normals model with
Halton quadrature for numerical integration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
from jax import Array

from skillmodels.af.batching import auto_n_obs_per_batch
from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.likelihood import (
    _log_mvn_pdf_chol,
    af_loglike_initial,
    create_loglike_and_gradient,
)
from skillmodels.af.moment_init import spearman_factor_moments
from skillmodels.af.params import (
    apply_fixed_params,
    apply_start_params,
    build_optimagic_inputs,
    create_af_params_template,
    get_initial_period_params_index,
    get_measurements_per_factor,
    get_normalizations_for_period,
)
from skillmodels.af.types import (
    AFEstimationOptions,
    AFPeriodResult,
    ConditionalDistribution,
    MixtureComponent,
)
from skillmodels.model_spec import ModelSpec
from skillmodels.types import ProcessedModel


def estimate_initial_period(
    model_spec: ModelSpec,
    processed_model: ProcessedModel,
    measurements: Array,
    controls: Array,
    af_options: AFEstimationOptions,
    state_factors: tuple[str, ...] | None = None,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
    observed_factors: tuple[str, ...] = (),
    observed_factor_values: Array | None = None,
) -> tuple[AFPeriodResult, ConditionalDistribution]:
    """Estimate the initial period (Step 0) of the AF procedure.

    Fit a mixture-of-normals distribution for the joint vector of latent
    factors (and, optionally, observed factors) at period 0, together with
    the measurement system parameters, via MLE with Halton quadrature.

    When `observed_factors` is non-empty, the joint distribution is modelled
    over (latent, observed) and per-individual observed values are used to
    condition the Halton draws via the Schur complement. This concentrates
    nodes on the region of latent space consistent with each individual's
    observed data, improving quadrature precision.

    Args:
        model_spec: Model specification.
        processed_model: Processed model from `process_model()`.
        measurements: Shape (n_obs, n_measures), period 0 measurement values.
        controls: Shape (n_obs, n_controls), period 0 control values.
        af_options: AF estimation options.
        state_factors: Subset of latent factors used as state factors for
            AF propagation. If `None`, all latent factors are used.
        start_params: Optional starting values. Matching index entries
            override heuristic defaults.
        fixed_params: Optional DataFrame with a "value" column pinning
            specified parameters (value + bounds both clamped to the value).
        observed_factors: Names of observed factors included in the joint
            initial distribution. Defaults to empty.
        observed_factor_values: Shape (n_obs, n_observed_factors) array of
            observed factor values. Required iff `observed_factors` is
            non-empty.

    Return:
        Tuple of (AFPeriodResult, ConditionalDistribution) where the
        distribution represents the estimated f(theta_0 | data_0), restricted
        to latent (or `state_factors`) coordinates.

    """
    n_components = af_options.n_mixture_components
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    n_obs_factors = len(observed_factors)

    reconstructed_factors = tuple(
        f for f in factors if not model_spec.factors[f].has_initial_distribution
    )
    state_latent_factors = tuple(f for f in factors if f not in reconstructed_factors)
    n_state_latent = len(state_latent_factors)
    n_joint = n_state_latent + n_obs_factors

    if n_obs_factors > 0 and observed_factor_values is None:
        msg = "observed_factor_values required when observed_factors is non-empty."
        raise ValueError(msg)
    obs_values = (
        observed_factor_values
        if observed_factor_values is not None
        else jnp.zeros((measurements.shape[0], 0))
    )

    # Build parameter index and template
    measurements_p0 = get_measurements_per_factor(model_spec.factors, period=0)
    params_index = get_initial_period_params_index(
        n_mixture_components=n_components,
        latent_factors=factors,
        measurements_period_0=measurements_p0,
        controls=controls_names,
        observed_factors=observed_factors,
        reconstructed_factors=reconstructed_factors,
    )
    normalizations = get_normalizations_for_period(model_spec.factors, period=0)
    params_template = create_af_params_template(
        params_index,
        normalizations,
        period=0,
    )

    # Initialize parameters via simple heuristics
    params_template = _initialize_params_heuristic(
        params_template,
        measurements,
        controls,
        n_state_latent,
        n_components,
        observed_factors=observed_factors,
        observed_factor_values=obs_values,
    )

    # Optionally override SDs / loadings / Cholesky diagonals via Spearman
    # moments. This places the optimizer near the strongly-identified MLE
    # neighborhood instead of at the static default 0.5 / obs_sd*0.5; for
    # parameters on weakly-identified ridges (notably sigma_inv vs sigma_meas) the
    # moment-based seed is the difference between converging at truth and
    # drifting to the boundary.
    if af_options.initialization_strategy == "moment_based":
        all_measures_full = _get_ordered_measures(measurements_p0)
        params_template = _apply_moment_based_overrides_initial(
            params_template,
            measurements,
            measurements_per_factor=measurements_p0,
            all_measures=all_measures_full,
            normalizations=normalizations,
            n_components=n_components,
        )

    # Override with user-supplied starting values where available
    if start_params is not None:
        apply_start_params(params_template, start_params)

    # Align template values with user-supplied fixes (bounds are not clamped;
    # pinning happens via FixedConstraintWithValue further below).
    if fixed_params is not None:
        apply_fixed_params(params_template, fixed_params)

    # Period-0 measurements and loading mask cover state-latent factors only.
    # Reconstructed factors' period-0 measurements are handled in the
    # transition step 0->1.
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

    # Halton quadrature nodes: dimension equals the state-latent count
    # (observed factors are conditioned on, not integrated over, via the
    # Schur complement).
    nodes, weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_state_latent,
    )

    # Translate normalization fixes and user-supplied fixes into FixedConstraints
    # so they compose with other constraints (e.g. ProbabilityConstraint).
    full_params_df, fixed_constraints = build_optimagic_inputs(
        params_template, fixed_params
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

    loglike_and_grad = create_loglike_and_gradient(
        af_loglike_initial,
        **loglike_kwargs,
    )

    def fun(params_df: pd.DataFrame) -> float:
        val, _grad = loglike_and_grad(jnp.array(params_df["value"].to_numpy()))
        return float(val)

    def fun_and_jac(params_df: pd.DataFrame) -> tuple[float, np.ndarray]:
        val, grad = loglike_and_grad(jnp.array(params_df["value"].to_numpy()))
        return float(val), np.array(grad)

    opt_res = om.minimize(
        fun=fun,
        params=full_params_df[["value"]],
        algorithm=af_options.optimizer_algorithm,
        bounds=om.Bounds(
            lower=full_params_df["lower_bound"],
            upper=full_params_df["upper_bound"],
        ),
        constraints=list(fixed_constraints) or None,
        fun_and_jac=fun_and_jac,
        **dict(af_options.optimizer_options),
    )

    # Write optimized values back into full template
    result_params = params_template.copy()
    result_params["value"] = opt_res.params["value"].to_numpy()

    # Extract conditional distribution (state factors only for AF propagation),
    # building the per-obs importance sample of skills_0 from the same Halton
    # design used for the optimization.
    sf = state_factors if state_factors is not None else factors
    cond_dist = _extract_conditional_distribution(
        result_params,
        len(sf),
        n_components,
        sf,
        nodes=nodes,
        observed_factor_values=obs_values,
    )

    period_result = AFPeriodResult(
        period=0,
        params=result_params,
        loglikelihood=-float(opt_res.fun),
        success=bool(opt_res.success),
        optimize_result=opt_res,
    )

    return period_result, cond_dist


def _get_ordered_measures(
    measurements_per_factor: dict[str, tuple[str, ...]],
) -> list[str]:
    """Get all measurement variables in a deterministic order."""
    seen: set[str] = set()
    result: list[str] = []
    for measures in measurements_per_factor.values():
        for m in measures:
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result


def _build_loading_mask(
    all_measures: list[str],
    factors: tuple[str, ...],
    measurements_per_factor: dict[str, tuple[str, ...]],
) -> np.ndarray:
    """Build boolean mask for which (measure, factor) pairs have loadings."""
    n_measures = len(all_measures)
    n_factors = len(factors)
    mask = np.zeros((n_measures, n_factors), dtype=bool)
    meas_idx = {m: i for i, m in enumerate(all_measures)}
    fac_idx = {f: i for i, f in enumerate(factors)}
    for factor, measures in measurements_per_factor.items():
        fi = fac_idx[factor]
        for m in measures:
            mi = meas_idx[m]
            mask[mi, fi] = True
    return mask


def _initialize_params_heuristic(
    params_template: pd.DataFrame,
    measurements: Array,
    _controls: Array,
    _n_factors: int,
    n_components: int,
    observed_factors: tuple[str, ...] = (),
    observed_factor_values: Array | None = None,
) -> pd.DataFrame:
    """Initialize parameters using simple heuristics.

    Use measurement means and variances to set reasonable starting values
    for mixture means, variances, loadings, and measurement SDs. When
    observed factors are present, their means come from sample means and
    their Cholesky diagonals from sample SDs.
    """
    params = params_template.copy()
    meas_np = np.array(measurements)

    # Overall mean and SD of first measurement as proxy for latent factor distribution
    meas_mean = float(np.nanmean(meas_np[:, 0]))
    meas_sd = float(np.nanstd(meas_np[:, 0]))
    if meas_sd < 1e-8:
        meas_sd = 1.0

    obs_means, obs_sds = _observed_factor_stats(
        observed_factors, observed_factor_values, n_rows=meas_np.shape[0]
    )

    # Set mixture weights to uniform
    weight_mask = params.index.get_level_values("category") == "mixture_weights"
    params.loc[weight_mask, "value"] = 1.0 / n_components

    _set_initial_mixture_means(
        params, n_components, meas_mean, meas_sd, obs_means, obs_sds
    )
    _set_initial_cholcov_diagonals(params, meas_sd, obs_sds)

    # Set measurement SDs to half the observed SD
    sd_mask = params.index.get_level_values("category") == "meas_sds"
    for i, idx in enumerate(params.index[sd_mask]):
        obs_sd = float(np.nanstd(meas_np[:, i])) if i < meas_np.shape[1] else 1.0
        params.loc[idx, "value"] = max(obs_sd * 0.5, 0.01)

    # Set loadings to 1.0 (where not fixed)
    load_mask = params.index.get_level_values("category") == "loadings"
    for idx in params.index[load_mask]:
        if params.loc[idx, "lower_bound"] != params.loc[idx, "upper_bound"]:
            params.loc[idx, "value"] = 1.0

    # Set control intercepts to measurement means (where not fixed)
    ctrl_mask = params.index.get_level_values("category") == "controls"
    for idx in params.index[ctrl_mask]:
        if (
            idx[3] == "constant"
            and params.loc[idx, "lower_bound"] != params.loc[idx, "upper_bound"]
        ):
            params.loc[idx, "value"] = 0.0

    return params


def _set_initial_mixture_means(
    params: pd.DataFrame,
    n_components: int,
    meas_mean: float,
    meas_sd: float,
    obs_means: dict[str, float],
    obs_sds: dict[str, float],
) -> None:
    """Set initial_states values in place: spread components around sample means."""
    mean_mask = params.index.get_level_values("category") == "initial_states"
    mean_vals = params.loc[mean_mask, "value"].copy()
    for idx in mean_vals.index:
        comp = idx[2]
        factor = idx[3]
        component_offset = (int(comp.split("_")[1]) - (n_components - 1) / 2) * 0.5
        if factor in obs_means:
            mean_vals.loc[idx] = obs_means[factor] + component_offset * obs_sds[factor]
        else:
            mean_vals.loc[idx] = meas_mean + component_offset * meas_sd
    params.loc[mean_mask, "value"] = mean_vals


def _set_initial_cholcov_diagonals(
    params: pd.DataFrame,
    meas_sd: float,
    obs_sds: dict[str, float],
) -> None:
    """Set initial_cholcovs diagonals to factor sample SD, off-diags to 0."""
    chol_mask = params.index.get_level_values("category") == "initial_cholcovs"
    for idx in params.index[chol_mask]:
        parts = idx[3].split("-")
        if len(parts) == 2 and parts[0] == parts[1]:
            params.loc[idx, "value"] = obs_sds.get(parts[0], meas_sd * 0.5)
        else:
            params.loc[idx, "value"] = 0.0


def _observed_factor_stats(
    observed_factors: tuple[str, ...],
    observed_factor_values: Array | None,
    n_rows: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-observed-factor sample means and SDs (SDs clipped to >= 0.01)."""
    obs_vals_np = (
        np.array(observed_factor_values)
        if observed_factor_values is not None
        else np.zeros((n_rows, 0))
    )
    obs_means = {
        factor: float(np.nanmean(obs_vals_np[:, i]))
        for i, factor in enumerate(observed_factors)
    }
    obs_sds = {
        factor: max(float(np.nanstd(obs_vals_np[:, i])), 0.01)
        for i, factor in enumerate(observed_factors)
    }
    return obs_means, obs_sds


def _extract_conditional_distribution(  # noqa: PLR0915
    params: pd.DataFrame,
    _n_factors: int,
    n_components: int,
    factors: tuple[str, ...],
    nodes: Array,
    observed_factor_values: Array,
) -> ConditionalDistribution:
    """Extract the initial distribution and build the period-0 importance sample.

    For each mixture component l, build a per-obs importance sample of
    skills_0 of shape ``(n_halton, n_obs, n_state)``, conditional (where
    applicable) on the observed factor values via the Schur complement.
    Per-obs mixture weights `p(l | Y_i)` are computed by Bayes' rule from
    the marginal density of Y_i under each component.

    These samples are propagated forward across periods (rather than being
    re-collapsed to a Gaussian mixture and re-drawn freshly) so the
    non-Gaussian shape of skills_t survives transitions through the CES
    production function.
    """
    # Mixture weights
    weight_mask = params.index.get_level_values("category") == "mixture_weights"
    weights_raw = jnp.array(params.loc[weight_mask, "value"].to_numpy())
    weights = weights_raw / weights_raw.sum()

    # Determine joint factor ordering from the stored initial_states entries
    joint_factors = _get_joint_factors_in_order(params, n_components)
    n_state = len(factors)
    n_obs = int(observed_factor_values.shape[0])
    n_obs_factors = int(observed_factor_values.shape[1])

    # Indices into joint_factors:
    # - target_idx: positions of `factors` (the state factors we want samples for).
    # - obs_idx:    positions of observed factors at the joint's tail.
    # Joint stores (state_latent_factors, observed_factors) in that order.
    target_idx = jnp.array([joint_factors.index(f) for f in factors], dtype=jnp.int32)
    obs_idx = jnp.array(
        [
            joint_factors.index(joint_factors[len(joint_factors) - n_obs_factors + k])
            for k in range(n_obs_factors)
        ],
        dtype=jnp.int32,
    )

    components: list[MixtureComponent] = []
    samples_per_component: list[Array] = []
    log_unnorm_weights_per_component: list[Array] = []
    cond_means_per_component: list[Array] = []
    cond_chols_per_component: list[Array] = []

    for m in range(n_components):
        joint_mean = jnp.array(
            [
                float(params.loc[("initial_states", 0, f"mixture_{m}", fac), "value"])  # ty: ignore[invalid-argument-type]
                for fac in joint_factors
            ]
        )
        joint_chol = _assemble_joint_chol(params, joint_factors, m)
        joint_cov = joint_chol @ joint_chol.T

        mu_theta = joint_mean[target_idx]
        cov_tt = joint_cov[target_idx[:, None], target_idx[None, :]]

        if n_obs_factors == 0:
            sub_mean = mu_theta
            sub_chol = jnp.linalg.cholesky(cov_tt + 1e-10 * jnp.eye(n_state))
            z_for_state = nodes[:, :n_state]
            per_node = sub_mean[None, :] + z_for_state @ sub_chol.T
            samples = jnp.broadcast_to(
                per_node[:, None, :], (nodes.shape[0], n_obs, n_state)
            )
            log_unnorm = jnp.full((n_obs,), float(jnp.log(weights[m] + 1e-300)))
            # Per-obs cond_means broadcast (n_obs, n_state); shared chol.
            cond_means_obs = jnp.broadcast_to(sub_mean[None, :], (n_obs, n_state))
            cond_chol_comp = sub_chol
        else:
            mu_y = joint_mean[obs_idx]
            cov_ty = joint_cov[target_idx[:, None], obs_idx[None, :]]
            cov_yy = joint_cov[obs_idx[:, None], obs_idx[None, :]]

            chol_yy = jnp.linalg.cholesky(cov_yy)
            solve_tt = jax.scipy.linalg.cho_solve((chol_yy, True), cov_ty.T)
            cond_cov = cov_tt - cov_ty @ solve_tt + 1e-10 * jnp.eye(n_state)
            cond_chol = jnp.linalg.cholesky(cond_cov)

            def _per_obs(
                y_i: Array,
                chol_yy: Array = chol_yy,
                mu_y: Array = mu_y,
                mu_theta: Array = mu_theta,
                cov_ty: Array = cov_ty,
            ) -> tuple[Array, Array]:
                alpha = jax.scipy.linalg.cho_solve((chol_yy, True), y_i - mu_y)
                cond_mean = mu_theta + cov_ty @ alpha
                log_marg_y = _log_mvn_pdf_chol(y_i, mu_y, chol_yy)
                return cond_mean, log_marg_y

            cond_means, log_margs = jax.vmap(_per_obs)(observed_factor_values)
            z_for_state = nodes[:, :n_state]
            samples = cond_means[None, :, :] + (z_for_state @ cond_chol.T)[:, None, :]
            sub_mean = mu_theta
            sub_chol = cond_chol
            log_unnorm = jnp.log(weights[m] + 1e-300) + log_margs
            cond_means_obs = cond_means
            cond_chol_comp = cond_chol

        components.append(MixtureComponent(mean=sub_mean, chol_cov=sub_chol))
        samples_per_component.append(samples)
        log_unnorm_weights_per_component.append(log_unnorm)
        cond_means_per_component.append(cond_means_obs)
        cond_chols_per_component.append(cond_chol_comp)

    if n_obs_factors > 0:
        log_w_stack = jnp.stack(
            log_unnorm_weights_per_component, axis=-1
        )  # (n_obs, n_components)
        cond_weights = jax.nn.softmax(log_w_stack, axis=-1)
    else:
        cond_weights = jnp.broadcast_to(weights[None, :], (n_obs, n_components))

    return ConditionalDistribution(
        mixture_weights=weights,
        components=tuple(components),
        samples_per_component=tuple(samples_per_component),
        conditional_weights=cond_weights,
        cond_means=jnp.stack(cond_means_per_component, axis=0),
        cond_chols=jnp.stack(cond_chols_per_component, axis=0),
    )


def _get_joint_factors_in_order(
    params: pd.DataFrame,
    n_components: int,
) -> tuple[str, ...]:
    """Return the joint factor ordering used in initial_states entries."""
    mask = (params.index.get_level_values("category") == "initial_states") & (
        params.index.get_level_values("name1") == f"mixture_{n_components - 1}"
    )
    del n_components
    return tuple(params.loc[mask].index.get_level_values("name2"))


def _assemble_joint_chol(
    params: pd.DataFrame,
    joint_factors: tuple[str, ...],
    component: int,
) -> Array:
    """Build the lower-triangular joint Cholesky matrix for one component."""
    n = len(joint_factors)
    chol = jnp.zeros((n, n))
    for row, f1 in enumerate(joint_factors):
        for col, f2 in enumerate(joint_factors):
            if col <= row:
                loc = ("initial_cholcovs", 0, f"mixture_{component}", f"{f1}-{f2}")
                val = float(params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
                chol = chol.at[row, col].set(val)  # noqa: PD008
    return chol


def _apply_moment_based_overrides_initial(  # noqa: C901, PLR0912
    params: pd.DataFrame,
    measurements: Array,
    measurements_per_factor: dict[str, tuple[str, ...]],
    all_measures: list[str],
    normalizations: dict[str, dict[tuple[str, str], float]],
    n_components: int,
) -> pd.DataFrame:
    """Override static initialization with Spearman cross-cov moments.

    For each latent factor with at least two period-0 measurements, apply
    `spearman_factor_moments` to the corresponding columns of
    `measurements` and write the recovered loadings, sigma_meas, and per-component
    Cholesky-diagonal sqrt(Var(F)) values into `params`. Skip rows where
    `lower_bound == upper_bound` (i.e. user normalizations or fixed
    constraints).

    The anchor measurement is determined from `normalizations["loadings"]`
    when a loading is pinned for the factor; otherwise the first measurement
    is the anchor.
    """
    out = params.copy()
    meas_np = np.array(measurements)
    n_obs = meas_np.shape[0]
    if n_obs == 0:
        return out
    meas_index = {m: i for i, m in enumerate(all_measures)}
    loading_norms = normalizations.get("loadings", {})

    for factor, factor_meas in measurements_per_factor.items():
        if len(factor_meas) < 2:
            continue
        cols = [meas_index[m] for m in factor_meas if m in meas_index]
        if len(cols) < 2:
            continue
        sub = meas_np[:, cols]

        # Anchor: pick the measurement whose loading is pinned for this
        # factor, falling back to the first measurement.
        anchor_loading = 1.0
        anchor_local = 0
        for local_idx, meas_name in enumerate(factor_meas):
            if (meas_name, factor) in loading_norms:
                anchor_local = local_idx
                anchor_loading = float(loading_norms[(meas_name, factor)])
                break

        result = spearman_factor_moments(
            sub,
            anchor_idx=anchor_local,
            anchor_loading=anchor_loading,
        )
        if not result.valid:
            continue

        # Override loadings (skip pinned rows).
        for local_idx, meas_name in enumerate(factor_meas):
            loc = ("loadings", 0, meas_name, factor)
            if loc not in out.index:
                continue
            if out.loc[loc, "lower_bound"] != out.loc[loc, "upper_bound"]:
                out.loc[loc, "value"] = float(result.loadings[local_idx])

        # Override measurement SDs (skip pinned rows).
        for local_idx, meas_name in enumerate(factor_meas):
            loc = ("meas_sds", 0, meas_name, "-")
            if loc not in out.index:
                continue
            if out.loc[loc, "lower_bound"] != out.loc[loc, "upper_bound"]:
                out.loc[loc, "value"] = float(result.meas_sds[local_idx])

        # Override per-component Cholesky diagonal for this factor with
        # sqrt(Var(F)). Off-diagonals stay at 0 (set by the heuristic).
        sd_factor = float(np.sqrt(max(result.latent_var, 1e-12)))
        for comp in range(n_components):
            loc = (
                "initial_cholcovs",
                0,
                f"mixture_{comp}",
                f"{factor}-{factor}",
            )
            if loc not in out.index:
                continue
            if out.loc[loc, "lower_bound"] != out.loc[loc, "upper_bound"]:
                out.loc[loc, "value"] = sd_factor

    return out
