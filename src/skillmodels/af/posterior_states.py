"""Compute posterior state estimates from AF estimation results.

For each individual and period, compute E[theta_t | Z_{0:t,i}] using
Halton quadrature and the estimated conditional distributions.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.initial_period import _build_loading_mask, _get_ordered_measures
from skillmodels.af.likelihood import _log_normal_pdf
from skillmodels.af.params import get_measurements_per_factor
from skillmodels.af.types import AFEstimationResult, ConditionalDistribution
from skillmodels.chs.process_debug_data import create_state_ranges
from skillmodels.model_spec import ModelSpec


def get_af_posterior_states(
    af_result: AFEstimationResult,
    model_spec: ModelSpec,
    data: pd.DataFrame,
    n_halton_points: int = 100,
) -> dict[str, dict[str, Any]]:
    """Compute posterior state means from AF estimation results.

    For each individual i and period t, compute::

        E[theta_t | Z_t,i] = sum_q w_q theta_q p(Z_t,i | theta_q)
                              / sum_q w_q p(Z_t,i | theta_q)

    where theta_q are quadrature nodes from the estimated conditional
    distribution at period t, and p(Z_t,i | theta_q) is the measurement
    density.

    Args:
        af_result: Result from `estimate_af()`.
        model_spec: Model specification.
        data: Dataset in long format with MultiIndex (id, period).
        n_halton_points: Quadrature points for posterior computation.

    Return:
        Dict with "unanchored_states" containing "states" DataFrame
        (columns: id, period, factor1, ...) and "state_ranges".

    """
    jax.config.update("jax_enable_x64", val=True)

    idx_names = data.index.names
    id_col = str(idx_names[0])
    period_col = str(idx_names[1])

    # Identify state factors from the conditional distribution dimension
    n_state = af_result.conditional_distributions[0].components[0].mean.shape[0]
    state_factors = tuple(
        f for f in model_spec.factors if not model_spec.factors[f].is_endogenous
    )[:n_state]

    rows: list[dict[str, float | int]] = []

    for t, (period_result, cond_dist) in enumerate(
        zip(
            af_result.period_results,
            af_result.conditional_distributions,
            strict=True,
        )
    ):
        measurements_pt = get_measurements_per_factor(model_spec.factors, period=t)
        if not measurements_pt:
            continue

        meas_info = _extract_period_measurement_info(
            period_result.params,
            model_spec,
            state_factors,
            t,
        )

        period_mask = data.index.get_level_values(period_col) == t
        period_df = data.loc[period_mask]
        ids = period_df.index.get_level_values(id_col)

        all_measures = _get_ordered_measures(measurements_pt)
        meas_cols = [c for c in all_measures if c in period_df.columns]
        measurements = jnp.array(
            period_df[meas_cols].to_numpy(dtype=np.float64, na_value=np.nan),
        )

        # Build per-observation control contribution
        ctrl_arrays = []
        for ctrl in meas_info["control_names"]:
            if ctrl == "constant":
                ctrl_arrays.append(np.ones(len(period_df)))
            elif ctrl in period_df.columns:
                ctrl_arrays.append(period_df[ctrl].to_numpy(dtype=np.float64))
            else:
                ctrl_arrays.append(np.zeros(len(period_df)))
        controls = jnp.array(np.column_stack(ctrl_arrays))
        control_contrib = controls @ meas_info["control_params"].T

        nodes, weights = create_halton_nodes_and_weights(n_halton_points, n_state)

        posterior_means = _compute_posterior_means(
            cond_dist=cond_dist,
            measurements=measurements,
            control_contrib=control_contrib,
            full_loadings=meas_info["full_loadings"],
            meas_sds=meas_info["meas_sds"],
            nodes=nodes,
            weights=weights,
        )

        for idx_i, obs_id in enumerate(ids):
            row: dict[str, float | int] = {id_col: obs_id, "period": t}
            for f_idx, factor in enumerate(state_factors):
                row[factor] = float(posterior_means[idx_i, f_idx])
            rows.append(row)

    states_df = pd.DataFrame(rows)
    state_ranges = create_state_ranges(
        filtered_states=states_df,
        factors=state_factors,
    )

    return {
        "unanchored_states": {
            "states": states_df,
            "state_ranges": state_ranges,
        },
    }


def _extract_period_measurement_info(
    period_params: pd.DataFrame,
    model_spec: ModelSpec,
    factors: tuple[str, ...],
    period: int,
) -> dict[str, Any]:
    """Extract measurement loadings, control contribution, and SDs."""
    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    all_measures = _get_ordered_measures(measurements_pt)
    loading_mask = _build_loading_mask(all_measures, factors, measurements_pt)

    loadings_list = []
    for mi, meas in enumerate(all_measures):
        for fi, factor in enumerate(factors):
            if loading_mask[mi, fi]:
                loc = ("loadings", period, meas, factor)
                if loc in period_params.index:
                    loadings_list.append(
                        float(period_params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
                    )

    full_loadings = jnp.zeros((len(all_measures), len(factors)))
    full_loadings = full_loadings.at[jnp.array(loading_mask)].set(  # noqa: PD008
        jnp.array(loadings_list)
    )

    # Extract ALL control coefficients (not just "constant")
    ctrl_entries = period_params.loc[
        period_params.index.get_level_values("category") == "controls"
    ]
    ctrl_names = (
        sorted(set(ctrl_entries.index.get_level_values("name2")))
        if len(ctrl_entries) > 0
        else ["constant"]
    )
    ctrl_params_list = []
    for meas in all_measures:
        for ctrl in ctrl_names:
            loc = ("controls", period, meas, ctrl)
            if loc in period_params.index:
                ctrl_params_list.append(float(period_params.loc[loc, "value"]))
            else:
                ctrl_params_list.append(0.0)
    control_params = jnp.array(ctrl_params_list).reshape(
        len(all_measures), len(ctrl_names)
    )

    sd_list = [
        float(period_params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
        if (loc := ("meas_sds", period, meas, "-")) in period_params.index
        else 0.5
        for meas in all_measures
    ]

    return {
        "full_loadings": full_loadings,
        "control_params": control_params,
        "control_names": ctrl_names,
        "meas_sds": jnp.array(sd_list),
    }


def _compute_posterior_means(
    *,
    cond_dist: ConditionalDistribution,
    measurements: Array,
    full_loadings: Array,
    control_contrib: Array,
    meas_sds: Array,
    nodes: Array,
    weights: Array,
) -> Array:
    """Compute posterior means for all individuals at one period.

    Return shape (n_obs, n_factors).
    """
    n_components = len(cond_dist.components)
    means = jnp.stack([c.mean for c in cond_dist.components])
    chol_covs = jnp.stack([c.chol_cov for c in cond_dist.components])
    mix_weights = cond_dist.mixture_weights

    residuals_base = measurements - control_contrib

    def _single_obs(residual_base: Array) -> Array:
        """Posterior mean for one individual."""

        def _node_kernel(z_q: Array) -> tuple[Array, Array]:
            """Return (log_weight, weighted_theta) for one quadrature node."""
            log_component_vals = []
            theta_components = []
            for l_idx in range(n_components):
                theta = means[l_idx] + chol_covs[l_idx] @ z_q
                residuals = residual_base - full_loadings @ theta
                log_lik = jnp.sum(
                    _log_normal_pdf(
                        residuals,
                        jnp.zeros_like(residuals),
                        meas_sds,
                    )
                )
                log_component_vals.append(
                    jnp.log(mix_weights[l_idx] + 1e-300) + log_lik
                )
                theta_components.append(theta)

            log_w = jax.scipy.special.logsumexp(jnp.array(log_component_vals))
            # Weighted theta across mixture components
            comp_weights = jax.nn.softmax(jnp.array(log_component_vals))
            avg_theta = jnp.zeros_like(theta_components[0])
            for cw, tv in zip(comp_weights, theta_components, strict=True):
                avg_theta = avg_theta + cw * tv
            return log_w, avg_theta

        log_ws, thetas = jax.vmap(_node_kernel)(nodes)

        # Posterior weights: softmax of log_ws + log(quadrature_weights)
        log_posterior = log_ws + jnp.log(weights)
        posterior_weights = jax.nn.softmax(log_posterior)

        return jnp.sum(posterior_weights[:, None] * thetas, axis=0)

    return jax.vmap(_single_obs)(residuals_base)
