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
) -> tuple[AFPeriodResult, ConditionalDistribution]:
    """Estimate the initial period (Step 0) of the AF procedure.

    Fit a mixture-of-normals distribution for the latent factors at period 0,
    jointly with the measurement system parameters (loadings, intercepts,
    error SDs), using MLE with Halton quadrature.

    Args:
        model_spec: Model specification.
        processed_model: Processed model from `process_model()`.
        measurements: Shape (n_obs, n_measures), period 0 measurement values.
        controls: Shape (n_obs, n_controls), period 0 control values.
        af_options: AF estimation options.
        state_factors: Subset of latent factors used as state factors for
            AF propagation. If `None`, all latent factors are used.

    Return:
        Tuple of (AFPeriodResult, ConditionalDistribution) where the
        distribution represents the estimated f(theta_0 | data_0).

    """
    n_factors = processed_model.dimensions.n_latent_factors
    n_components = af_options.n_mixture_components
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls

    # Build parameter index and template
    measurements_p0 = get_measurements_per_factor(model_spec.factors, period=0)
    params_index = get_initial_period_params_index(
        n_mixture_components=n_components,
        latent_factors=factors,
        measurements_period_0=measurements_p0,
        controls=controls_names,
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
        n_factors,
        n_components,
    )

    # Build loading mask: (n_measures, n_factors) boolean
    all_measures = _get_ordered_measures(measurements_p0)
    loading_mask = _build_loading_mask(all_measures, factors, measurements_p0)

    # Halton quadrature nodes
    nodes, weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_factors,
    )

    # Set up optimization
    free_mask_np = get_free_mask(params_template)
    free_mask = jnp.array(free_mask_np)
    all_params_init = jnp.array(params_template["value"].to_numpy())

    loglike_kwargs = {
        "all_params": all_params_init,
        "free_mask": free_mask,
        "n_factors": n_factors,
        "n_mixture_components": n_components,
        "n_measures": len(all_measures),
        "n_controls": len(controls_names),
        "measurements": measurements,
        "controls": controls,
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
) -> pd.DataFrame:
    """Initialize parameters using simple heuristics.

    Use measurement means and variances to set reasonable starting values
    for mixture means, variances, loadings, and measurement SDs.
    """
    params = params_template.copy()
    meas_np = np.array(measurements)

    # Overall mean and SD of first measurement as proxy for factor distribution
    meas_mean = float(np.nanmean(meas_np[:, 0]))
    meas_sd = float(np.nanstd(meas_np[:, 0]))
    if meas_sd < 1e-8:
        meas_sd = 1.0

    # Set mixture weights to uniform
    weight_mask = params.index.get_level_values("category") == "mixture_weights"
    params.loc[weight_mask, "value"] = 1.0 / n_components

    # Set mixture means: spread around measurement mean
    mean_mask = params.index.get_level_values("category") == "initial_states"
    mean_vals = params.loc[mean_mask, "value"].copy()
    for m in range(n_components):
        offset = (m - (n_components - 1) / 2) * meas_sd * 0.5
        component_mask = mean_vals.index.get_level_values("name1") == f"mixture_{m}"
        mean_vals.loc[component_mask] = meas_mean + offset
    params.loc[mean_mask, "value"] = mean_vals

    # Set Cholesky diagonals to measurement SD, off-diags to 0
    chol_mask = params.index.get_level_values("category") == "initial_cholcovs"
    for idx in params.index[chol_mask]:
        pair = idx[3]
        parts = pair.split("-")
        if len(parts) == 2 and parts[0] == parts[1]:
            params.loc[idx, "value"] = meas_sd * 0.5
        else:
            params.loc[idx, "value"] = 0.0

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


def _extract_conditional_distribution(
    params: pd.DataFrame,
    n_factors: int,
    n_components: int,
    factors: tuple[str, ...],
) -> ConditionalDistribution:
    """Extract the estimated initial distribution for the given factors."""
    # Mixture weights
    weight_mask = params.index.get_level_values("category") == "mixture_weights"
    weights_raw = jnp.array(params.loc[weight_mask, "value"].to_numpy())
    weights = weights_raw / weights_raw.sum()

    # Components
    components: list[MixtureComponent] = []
    for m in range(n_components):
        # Mean: select only the requested factors
        mean_vals = []
        for fac in factors:
            loc = ("initial_states", 0, f"mixture_{m}", fac)
            if loc in params.index:
                mean_vals.append(float(params.loc[loc, "value"]))  # ty: ignore[invalid-argument-type]
        mean = jnp.array(mean_vals)

        # Cholesky: extract submatrix for requested factors
        chol_vals = []
        for row_fac in factors:
            for col_fac in factors:
                if factors.index(col_fac) <= factors.index(row_fac):
                    loc = (
                        "initial_cholcovs",
                        0,
                        f"mixture_{m}",
                        f"{row_fac}-{col_fac}",
                    )
                    if loc in params.index:
                        chol_vals.append(float(params.loc[loc, "value"]))  # ty: ignore[invalid-argument-type]
        chol_flat = jnp.array(chol_vals)
        chol = jnp.zeros((n_factors, n_factors))
        chol = chol.at[jnp.tril_indices(n_factors)].set(chol_flat)  # noqa: PD008

        components.append(MixtureComponent(mean=mean, chol_cov=chol))

    return ConditionalDistribution(
        mixture_weights=weights,
        components=tuple(components),
        conditional_weights=None,
    )
