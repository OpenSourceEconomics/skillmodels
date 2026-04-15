"""Main driver for the AF estimation procedure."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.af.initial_period import estimate_initial_period
from skillmodels.af.params import get_measurements_per_factor
from skillmodels.af.transition_period import estimate_transition_period
from skillmodels.af.types import (
    AFEstimationOptions,
    AFEstimationResult,
    AFPeriodResult,
    ConditionalDistribution,
)
from skillmodels.af.validate import validate_af_model
from skillmodels.model_spec import ModelSpec
from skillmodels.process_model import process_model


def estimate_af(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
    _start_params: pd.DataFrame | None = None,
) -> AFEstimationResult:
    """Estimate a latent factor model using the Antweiler-Freyberger method.

    Sequential period-by-period MLE with Halton quadrature for numerical
    integration, following Antweiler and Freyberger (2025).

    The procedure estimates one period at a time:
    - Step 0: Fit initial distribution and measurement params for period 0
    - Step t (t >= 1): Estimate transition and measurement params using the
      estimated distribution from previous periods

    Args:
        model_spec: Model specification (same as for CHS estimation).
        data: Dataset in long format with MultiIndex (id, period).
        af_options: AF-specific estimation options. If None, uses defaults.
        _start_params: Optional starting parameter values (not yet implemented).

    Return:
        AFEstimationResult with per-period results and combined parameters.

    """
    jax.config.update("jax_enable_x64", val=True)

    if af_options is None:
        af_options = AFEstimationOptions()

    validate_af_model(model_spec)
    processed_model = process_model(model_spec)

    # Extract data arrays per period
    n_periods = processed_model.dimensions.n_periods
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls

    period_data = _extract_period_data(
        data,
        n_periods,
        factors,
        controls_names,
        model_spec,
    )

    # Step 0: Initial period
    period_0_result, cond_dist = estimate_initial_period(
        model_spec=model_spec,
        processed_model=processed_model,
        measurements=period_data[0]["measurements"],
        controls=period_data[0]["controls"],
        af_options=af_options,
    )

    period_results: list[AFPeriodResult] = [period_0_result]
    conditional_dists: list[ConditionalDistribution] = [cond_dist]

    # Steps 1..T-1: Transition periods
    for t in range(1, n_periods):
        measurements_pt = get_measurements_per_factor(model_spec.factors, period=t)
        if not measurements_pt:
            break

        period_t_result, cond_dist = estimate_transition_period(
            period=t,
            model_spec=model_spec,
            processed_model=processed_model,
            measurements=period_data[t]["measurements"],
            controls=period_data[t]["controls"],
            prev_distribution=cond_dist,
            af_options=af_options,
        )
        period_results.append(period_t_result)
        conditional_dists.append(cond_dist)

    # Combine parameters from all periods
    all_params = pd.concat([r.params for r in period_results])

    return AFEstimationResult(
        period_results=tuple(period_results),
        all_params=all_params,
        model_spec=model_spec,
        conditional_distributions=tuple(conditional_dists),
    )


def _extract_period_data(
    data: pd.DataFrame,
    n_periods: int,
    _factors: tuple[str, ...],
    controls_names: tuple[str, ...],
    model_spec: ModelSpec,
) -> dict[int, dict[str, Array]]:
    """Extract measurement and control arrays for each period.

    Args:
        data: Long-format DataFrame with MultiIndex (id, period).
        n_periods: Number of periods in the model.
        _factors: Latent factor names (unused, reserved for future use).
        controls_names: Control variable names (includes "constant").
        model_spec: Model specification for measurement variable names.

    Return:
        Dict mapping period -> {"measurements": Array, "controls": Array}.

    """
    period_data: dict[int, dict[str, Array]] = {}

    # Get all individuals and periods
    idx_names = data.index.names
    period_col = str(idx_names[1])

    for t in range(n_periods):
        measurements_pt = get_measurements_per_factor(model_spec.factors, period=t)
        if not measurements_pt:
            continue

        # Get all unique measurement variable names for this period
        all_measures: list[str] = []
        seen: set[str] = set()
        for measures in measurements_pt.values():
            for m in measures:
                if m not in seen:
                    seen.add(m)
                    all_measures.append(m)

        # Select data for this period
        period_mask = data.index.get_level_values(period_col) == t
        period_df = data.loc[period_mask]

        # Measurements array
        meas_cols = [c for c in all_measures if c in period_df.columns]
        meas_array = jnp.array(
            period_df[meas_cols].to_numpy(dtype=np.float64, na_value=np.nan),
        )

        # Controls array (constant + control variables)
        ctrl_arrays = []
        for ctrl in controls_names:
            if ctrl == "constant":
                ctrl_arrays.append(np.ones(len(period_df)))
            elif ctrl in period_df.columns:
                ctrl_arrays.append(period_df[ctrl].to_numpy(dtype=np.float64))
            else:
                ctrl_arrays.append(np.zeros(len(period_df)))

        ctrl_array = jnp.array(np.column_stack(ctrl_arrays))

        period_data[t] = {
            "measurements": meas_array,
            "controls": ctrl_array,
        }

    return period_data
