"""Compute posterior state estimates from AF estimation results.

For each individual and period, compute E[theta_t | Z_{0:t,i}] using
Halton quadrature and the estimated conditional distributions.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from beartype import beartype
from jax import Array

from skillmodels._beartype_conf import ESTIMATION_CONF
from skillmodels.af.initial_period import _build_loading_mask, _get_ordered_measures
from skillmodels.af.likelihood import _log_normal_pdf
from skillmodels.af.params import get_measurements_per_factor
from skillmodels.af.types import AFEstimationResult, ConditionalDistribution
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.state_ranges import create_state_ranges


@beartype(conf=ESTIMATION_CONF)
def get_af_posterior_states(
    af_result: AFEstimationResult,
    model_spec: ModelSpec,
    data: pd.DataFrame,
    n_halton_points: int = 100,  # noqa: ARG001
) -> dict[str, dict[str, Any]]:
    """Compute posterior state means from AF estimation results.

    For each individual i and period t, compute the posterior mean
    E[theta_t | Z_{0:t,i}] against the per-observation, income-conditioned
    chained importance sample carried by the estimator
    (`ConditionalDistribution.samples_per_component`). Each stored sample is
    a prior draw already encoding the period-0 income conditioning; it is
    weighted by the per-observation prior mixture weights
    (`conditional_weights`) and reweighted by the current-period measurement
    likelihood, then the posterior mean is the weighted average of the
    samples.

    Args:
        af_result: Result from `estimate_af()`. Must NOT have gone through
            `to_numpy()`, which drops the per-obs chained sample this
            computation needs.
        model_spec: Model specification.
        data: Dataset in long format with MultiIndex (id, period).
        n_halton_points: Retained for API/backward compatibility only; no
            longer used (posterior means are computed from the carried
            chained sample, not from freshly drawn Halton nodes).

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

        posterior_means = _compute_posterior_means(
            cond_dist=cond_dist,
            measurements=measurements,
            control_contrib=control_contrib,
            full_loadings=meas_info["full_loadings"],
            meas_sds=meas_info["meas_sds"],
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
) -> Array:
    """Compute posterior means for all individuals at one period.

    Use the per-observation, income-conditioned chained importance sample
    (`samples_per_component`) carried by the estimator, weighted by the
    per-observation prior mixture weights (`conditional_weights`), and
    reweight each sample by the current-period measurement likelihood.

    Return shape (n_obs, n_factors).
    """
    if not cond_dist.samples_per_component:
        msg = (
            "get_af_posterior_states needs the per-observation chained "
            "importance sample (`samples_per_component`). It is dropped by "
            "AFEstimationResult.to_numpy(); call get_af_posterior_states on "
            "the estimation result BEFORE to_numpy()."
        )
        raise ValueError(msg)

    # Stack to shape n_components by n_summary by n_obs by n_state.
    samples = jnp.stack([jnp.asarray(s) for s in cond_dist.samples_per_component])
    n_components, n_summary, n_obs = (
        samples.shape[0],
        samples.shape[1],
        samples.shape[2],
    )

    if cond_dist.conditional_weights is not None:
        cond_weights = jnp.asarray(cond_dist.conditional_weights)
    else:
        cond_weights = jnp.broadcast_to(
            jnp.asarray(cond_dist.mixture_weights)[None, :], (n_obs, n_components)
        )

    residuals_base = measurements - control_contrib  # (n_obs, n_meas)
    # obs-major samples for vmap: (n_obs, n_components, n_summary, n_state)
    samples_obs_major = jnp.transpose(samples, (2, 0, 1, 3))

    def _single_obs(
        residual_base: Array,  # (n_meas,)
        obs_samples: Array,  # (n_components, n_summary, n_state)
        obs_weights: Array,  # (n_components,)
    ) -> Array:
        def _per_sample(theta: Array) -> Array:  # (n_state,) -> scalar
            residuals = residual_base - full_loadings @ theta
            return jnp.sum(
                _log_normal_pdf(residuals, jnp.zeros_like(residuals), meas_sds)
            )

        log_lik = jax.vmap(jax.vmap(_per_sample))(obs_samples)  # (n_comp, n_summary)
        log_prior = jnp.log(obs_weights + 1e-300)[:, None] - jnp.log(n_summary)
        log_post = (log_prior + log_lik).reshape(-1)
        post_weights = jax.nn.softmax(log_post)
        flat_theta = obs_samples.reshape(-1, obs_samples.shape[-1])
        return jnp.sum(post_weights[:, None] * flat_theta, axis=0)

    return jax.vmap(_single_obs)(residuals_base, samples_obs_major, cond_weights)
