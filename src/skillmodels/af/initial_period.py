"""Step 0 of the AF estimator: initial period estimation.

Estimate the joint distribution of latent factors at period 0 and the
measurement system parameters, using a mixture-of-normals model with
Halton quadrature for numerical integration.
"""

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
from jax import Array

from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.likelihood import af_loglike_initial, create_loglike_and_gradient
from skillmodels.af.params import (
    apply_fixed_params,
    apply_start_params,
    create_af_params_template,
    get_free_mask,
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
    n_latent = processed_model.dimensions.n_latent_factors
    n_components = af_options.n_mixture_components
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    n_obs_factors = len(observed_factors)
    n_joint = n_latent + n_obs_factors

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
        n_latent,
        n_components,
        observed_factors=observed_factors,
        observed_factor_values=obs_values,
    )

    # Override with user-supplied starting values where available
    if start_params is not None:
        apply_start_params(params_template, start_params)

    # Pin any user-fixed parameters (clamps value + bounds)
    if fixed_params is not None:
        apply_fixed_params(params_template, fixed_params)

    # Build loading mask: (n_measures, n_factors) boolean
    all_measures = _get_ordered_measures(measurements_p0)
    loading_mask = _build_loading_mask(all_measures, factors, measurements_p0)

    # Halton quadrature nodes: dimension equals n_latent (observed factors
    # are conditioned on, not integrated over, via the Schur complement).
    nodes, weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_latent,
    )

    # Set up optimization
    free_mask_np = get_free_mask(params_template)
    free_mask = jnp.array(free_mask_np)
    all_params_init = jnp.array(params_template["value"].to_numpy())

    loglike_kwargs = {
        "all_params": all_params_init,
        "free_mask": free_mask,
        "n_factors": n_joint,
        "n_latent_factors": n_latent,
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

    # Create free params DataFrame for optimagic
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

    # Write optimized values back into full template
    result_params = params_template.copy()
    result_params.loc[free_index, "value"] = opt_res.params["value"].to_numpy()

    # Extract conditional distribution (state factors only for AF propagation)
    sf = state_factors if state_factors is not None else factors
    cond_dist = _extract_conditional_distribution(
        result_params,
        len(sf),
        n_components,
        sf,
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


def _extract_conditional_distribution(
    params: pd.DataFrame,
    _n_factors: int,
    n_components: int,
    factors: tuple[str, ...],
) -> ConditionalDistribution:
    """Extract the estimated initial distribution for the given factors.

    The joint covariance over (latent, observed) may be stored; this
    function extracts the marginal over `factors` by taking the diagonal
    submatrix of the joint covariance, recomputing its Cholesky.
    """
    # Mixture weights
    weight_mask = params.index.get_level_values("category") == "mixture_weights"
    weights_raw = jnp.array(params.loc[weight_mask, "value"].to_numpy())
    weights = weights_raw / weights_raw.sum()

    # Determine joint factor ordering from the stored initial_states entries
    joint_factors = _get_joint_factors_in_order(params, n_components)

    components: list[MixtureComponent] = []
    for m in range(n_components):
        joint_mean = jnp.array(
            [
                float(params.loc[("initial_states", 0, f"mixture_{m}", fac), "value"])  # ty: ignore[invalid-argument-type]
                for fac in joint_factors
            ]
        )
        joint_chol = _assemble_joint_chol(params, joint_factors, m)
        if tuple(factors) == joint_factors:
            sub_chol = joint_chol
            sub_mean = joint_mean
        else:
            fac_idx = jnp.array([joint_factors.index(f) for f in factors])
            joint_cov = joint_chol @ joint_chol.T
            sub_cov = joint_cov[fac_idx[:, None], fac_idx[None, :]]
            sub_chol = jnp.linalg.cholesky(sub_cov)
            sub_mean = joint_mean[fac_idx]
        components.append(MixtureComponent(mean=sub_mean, chol_cov=sub_chol))

    return ConditionalDistribution(
        mixture_weights=weights,
        components=tuple(components),
        conditional_weights=None,
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
