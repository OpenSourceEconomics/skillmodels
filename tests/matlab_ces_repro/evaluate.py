"""Evaluate skillmodels' AF log-likelihood at a given parameter vector.

This mirrors the setup in ``skillmodels.af.initial_period`` and
``skillmodels.af.transition_period`` up to building the jitted likelihood
but stops short of running the optimizer. It lets tests score MATLAB-
translated parameters under skillmodels' own likelihood so we can ask
"does MATLAB's optimum give a higher likelihood than ours?" without
having to run a second optimisation pass.

Public entry points:

- ``evaluate_af_initial_loglike(model_spec, period_0_data, params_df,
  af_options, observed_factor_values)`` → scalar ``log L``.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.af.batching import auto_n_obs_per_batch
from skillmodels.af.halton import create_halton_nodes_and_weights
from skillmodels.af.initial_period import (
    _build_loading_mask,
    _get_ordered_measures,
)
from skillmodels.af.likelihood import af_loglike_initial, create_loglike_and_gradient
from skillmodels.af.params import (
    get_initial_period_params_index,
    get_measurements_per_factor,
    get_normalizations_for_period,
)
from skillmodels.af.types import AFEstimationOptions
from skillmodels.model_spec import ModelSpec
from skillmodels.process_model import process_model


def evaluate_af_initial_loglike(
    *,
    model_spec: ModelSpec,
    measurements: Array,
    controls: Array,
    params_df: pd.DataFrame,
    af_options: AFEstimationOptions,
    observed_factors: tuple[str, ...] = (),
    observed_factor_values: Array | None = None,
) -> float:
    """Return ``-neg_log_likelihood`` i.e. the log-likelihood per observation.

    Args:
        model_spec: The AF model spec.
        measurements: Shape ``(n_obs, n_measures)`` period-0 measurement
            values.
        controls: Shape ``(n_obs, n_controls)`` period-0 control values.
        params_df: Full parameter DataFrame with the initial-period
            MultiIndex produced by ``get_initial_period_params_index``.
            Must have a ``"value"`` column.
        af_options: AF options (uses the same Halton count as the
            estimator would).
        observed_factors: Names of observed factors in the initial joint.
        observed_factor_values: Shape ``(n_obs, n_observed_factors)`` of
            observed factor values.

    Return:
        Average log-likelihood per observation (matches what the estimator
        reports as ``AFPeriodResult.loglikelihood``).
    """
    processed_model = process_model(model_spec)
    n_latent = processed_model.dimensions.n_latent_factors
    n_components = af_options.n_mixture_components
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    n_obs_factors = len(observed_factors)
    n_joint = n_latent + n_obs_factors

    obs_values = (
        observed_factor_values
        if observed_factor_values is not None
        else jnp.zeros((measurements.shape[0], 0))
    )

    measurements_p0 = get_measurements_per_factor(model_spec.factors, period=0)
    params_index = get_initial_period_params_index(
        n_mixture_components=n_components,
        latent_factors=factors,
        measurements_period_0=measurements_p0,
        controls=controls_names,
        observed_factors=observed_factors,
    )
    # Sanity check that the caller-supplied params_df matches the AF index.
    if not params_df.index.equals(params_index):
        msg = (
            "params_df has a different MultiIndex than the AF initial-period "
            "index. Build it via get_initial_period_params_index."
        )
        raise ValueError(msg)
    # Unused but kept as a lookup in case future calls need it.
    _ = get_normalizations_for_period(model_spec.factors, period=0)

    all_measures = _get_ordered_measures(measurements_p0)
    loading_mask = _build_loading_mask(all_measures, factors, measurements_p0)
    nodes, weights = create_halton_nodes_and_weights(
        af_options.n_halton_points,
        n_latent,
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
        "n_obs_per_batch": n_obs_per_batch,
    }

    loglike_and_grad = create_loglike_and_gradient(af_loglike_initial, **loglike_kwargs)

    params_array = jnp.array(params_df["value"].to_numpy(dtype=np.float64))
    neg_ll, _grad = loglike_and_grad(params_array)
    return -float(neg_ll)
