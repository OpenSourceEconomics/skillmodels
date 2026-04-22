"""Parameter index construction and parsing for AF estimation."""

from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from skillmodels.types import Normalizations, TransitionInfo


def get_initial_period_params_index(
    *,
    n_mixture_components: int,
    latent_factors: tuple[str, ...],
    measurements_period_0: dict[str, tuple[str, ...]],
    controls: tuple[str, ...],
    observed_factors: tuple[str, ...] = (),
) -> pd.MultiIndex:
    """Build parameter index for the initial period (Step 0).

    Parameters estimated in Step 0:
    - Mixture weights, means, Cholesky covariances for the joint distribution
      of latent and observed factors at period 0
    - Measurement loadings, intercepts, SDs for period 0

    When `observed_factors` is non-empty, the initial distribution is modelled
    over the joint vector (latent, observed). Per-individual observed values
    let the likelihood condition on them via the Schur complement, which
    concentrates Halton draws and improves estimation precision.

    Args:
        n_mixture_components: Number of Gaussian mixture components.
        latent_factors: Names of latent factors.
        measurements_period_0: Factor name -> tuple of measurement variable names.
        controls: Control variable names (includes "constant").
        observed_factors: Names of observed factors included in the joint
            initial distribution.

    Return:
        MultiIndex with levels (category, period, name1, name2).

    """
    ind_tups: list[tuple[str, int, str, str]] = []
    joint_factors = (*latent_factors, *observed_factors)

    # Mixture weights
    for m in range(n_mixture_components):
        ind_tups.append(("mixture_weights", 0, f"mixture_{m}", "-"))

    # Initial means per component per joint factor
    for m in range(n_mixture_components):
        for factor in joint_factors:
            ind_tups.append(("initial_states", 0, f"mixture_{m}", factor))

    # Initial Cholesky covariances per component (lower triangular) over joint factors
    for m in range(n_mixture_components):
        for row, f1 in enumerate(joint_factors):
            for col, f2 in enumerate(joint_factors):
                if col <= row:
                    ind_tups.append(
                        (
                            "initial_cholcovs",
                            0,
                            f"mixture_{m}",
                            f"{f1}-{f2}",
                        )
                    )

    # Measurement params for period 0
    ind_tups.extend(
        _measurement_index_tuples(
            period=0,
            latent_factors=latent_factors,
            measurements=measurements_period_0,
            controls=controls,
        )
    )

    return pd.MultiIndex.from_tuples(
        ind_tups,
        names=["category", "period", "name1", "name2"],
    )


def get_transition_period_params_index(
    *,
    period: int,
    latent_factors: tuple[str, ...],
    transition_info: TransitionInfo,
    measurements_at_period: dict[str, tuple[str, ...]],
    controls: tuple[str, ...],
    endogenous_factors: tuple[str, ...] = (),
    observed_factors: tuple[str, ...] = (),
) -> pd.MultiIndex:
    """Build parameter index for a transition period (Step t, t >= 1).

    Parameters estimated in Step t:
    - Transition parameters and shock SDs for period t-1 -> t
    - Measurement loadings, intercepts, SDs for period t
    - Investment equation params for each endogenous factor (if any)

    Args:
        period: Calendar period (t >= 1).
        latent_factors: Names of latent (non-endogenous) state factors.
        transition_info: Transition function info from ProcessedModel.
        measurements_at_period: Factor name -> measurement variables at period t.
        controls: Control variable names.
        endogenous_factors: Names of endogenous (investment) factors.
        observed_factors: Names of observed factors.

    Return:
        MultiIndex with levels (category, period, name1, name2).

    """
    ind_tups: list[tuple[str, int, str, str]] = []

    # Transition parameters (for t-1 -> t)
    for factor in latent_factors:
        if factor in transition_info.param_names:
            for name in transition_info.param_names[factor]:
                ind_tups.append(("transition", period - 1, factor, name))

    # Shock SDs (for t-1 -> t)
    for factor in latent_factors:
        ind_tups.append(("shock_sds", period - 1, factor, "-"))

    # Investment equation parameters (for t-1)
    for endog_factor in endogenous_factors:
        # Intercept
        ind_tups.append(("investment_eq", period - 1, endog_factor, "constant"))
        # Coefficients on each state factor
        for factor in latent_factors:
            ind_tups.append(("investment_eq", period - 1, endog_factor, factor))
        # Coefficients on observed factors
        for obs_factor in observed_factors:
            ind_tups.append(("investment_eq", period - 1, endog_factor, obs_factor))
        # Investment shock SD
        ind_tups.append(("investment_sds", period - 1, endog_factor, "-"))

    # Measurement params for period t (loadings for ALL factors, not just state)
    all_factor_measurements = dict(measurements_at_period)
    all_latent = (*latent_factors, *endogenous_factors)
    ind_tups.extend(
        _measurement_index_tuples(
            period=period,
            latent_factors=all_latent,
            measurements=all_factor_measurements,
            controls=controls,
        )
    )

    return pd.MultiIndex.from_tuples(
        ind_tups,
        names=["category", "period", "name1", "name2"],
    )


def _measurement_index_tuples(
    *,
    period: int,
    latent_factors: tuple[str, ...],
    measurements: dict[str, tuple[str, ...]],
    controls: tuple[str, ...],
) -> list[tuple[str, int, str, str]]:
    """Generate index tuples for measurement system parameters.

    Includes controls (intercept/control coefficients), loadings, and
    measurement error SDs for all measurements in the given period.

    """
    ind_tups: list[tuple[str, int, str, str]] = []

    # Collect all unique measurement variables for this period, preserving order
    all_measures: list[str] = []
    measure_to_factors: dict[str, list[str]] = {}
    for factor, measures in measurements.items():
        for m in measures:
            if m not in measure_to_factors:
                all_measures.append(m)
                measure_to_factors[m] = []
            measure_to_factors[m].append(factor)

    # Controls (intercept + control variables) per measurement
    for meas in all_measures:
        for ctrl in controls:
            ind_tups.append(("controls", period, meas, ctrl))

    # Loadings: one per (measurement, factor) pair
    for meas in all_measures:
        for factor in latent_factors:
            if factor in measure_to_factors.get(meas, []):
                ind_tups.append(("loadings", period, meas, factor))

    # Measurement error SDs
    for meas in all_measures:
        ind_tups.append(("meas_sds", period, meas, "-"))

    return ind_tups


def get_measurements_per_factor(
    factors: MappingProxyType[str, Any],
    period: int,
) -> dict[str, tuple[str, ...]]:
    """Extract measurement variable names per factor for a given period.

    Args:
        factors: ModelSpec.factors mapping.
        period: Calendar period index.

    Return:
        Dict mapping factor name to tuple of measurement variable names.

    """
    result: dict[str, tuple[str, ...]] = {}
    for name, spec in factors.items():
        if period < len(spec.measurements) and len(spec.measurements[period]) > 0:
            result[name] = spec.measurements[period]
    return result


def get_normalizations_for_period(
    factors: MappingProxyType[str, Any],
    period: int,
) -> dict[str, dict[tuple[str, str], float]]:
    """Extract normalization constraints for a given period.

    Return:
        Dict of category ("loadings" or "intercepts") to dict of
        (measurement, factor_or_control) -> fixed value.

    """
    loading_fixes: dict[tuple[str, str], float] = {}
    intercept_fixes: dict[tuple[str, str], float] = {}

    for factor_name, spec in factors.items():
        norms: Normalizations | None = spec.normalizations
        if norms is None:
            continue

        if norms.loadings is not None and period < len(norms.loadings):
            for meas, value in norms.loadings[period].items():
                loading_fixes[(meas, factor_name)] = value

        if norms.intercepts is not None and period < len(norms.intercepts):
            for meas, value in norms.intercepts[period].items():
                # intercept normalizations fix the constant control
                intercept_fixes[(meas, "constant")] = value

    return {"loadings": loading_fixes, "intercepts": intercept_fixes}


def create_af_params_template(
    params_index: pd.MultiIndex,
    normalizations: dict[str, dict[tuple[str, str], float]],
    period: int,
    *,
    bounds_distance: float = 0.001,
) -> pd.DataFrame:
    """Create parameter template DataFrame with bounds and fixed values.

    Args:
        params_index: Parameter MultiIndex for this period.
        normalizations: Loading and intercept normalizations.
        period: Calendar period.
        bounds_distance: Minimum distance from zero for SD parameters.

    Return:
        DataFrame with columns: value, lower_bound, upper_bound.

    """
    params = pd.DataFrame(
        index=params_index,
        data={
            "value": np.nan,
            "lower_bound": -np.inf,
            "upper_bound": np.inf,
        },
    )

    # Set bounds for SD parameters
    sd_categories = ("meas_sds", "shock_sds", "investment_sds")
    for cat in sd_categories:
        mask = params.index.get_level_values("category") == cat
        params.loc[mask, "lower_bound"] = bounds_distance
        params.loc[mask, "value"] = 0.5

    # Set bounds for mixture weights
    weight_mask = params.index.get_level_values("category") == "mixture_weights"
    params.loc[weight_mask, "lower_bound"] = 0.001
    params.loc[weight_mask, "upper_bound"] = 0.999

    # Set bounds for Cholesky diagonals (must be positive)
    chol_mask = params.index.get_level_values("category") == "initial_cholcovs"
    for idx in params.index[chol_mask]:
        # Diagonal entries have matching factor names (e.g., "fac1-fac1")
        pair = idx[3]  # name2 level
        parts = pair.split("-")
        if len(parts) == 2 and parts[0] == parts[1]:
            params.loc[idx, "lower_bound"] = bounds_distance

    # Apply normalization fixes
    loading_fixes = normalizations.get("loadings", {})
    for (meas, factor), val in loading_fixes.items():
        loc = ("loadings", period, meas, factor)
        if loc in params.index:
            params.loc[loc, "value"] = val
            params.loc[loc, "lower_bound"] = val
            params.loc[loc, "upper_bound"] = val

    intercept_fixes = normalizations.get("intercepts", {})
    for (meas, ctrl), val in intercept_fixes.items():
        loc = ("controls", period, meas, ctrl)
        if loc in params.index:
            params.loc[loc, "value"] = val
            params.loc[loc, "lower_bound"] = val
            params.loc[loc, "upper_bound"] = val

    # Default values for parameters still NaN
    still_nan = params["value"].isna()
    params.loc[still_nan, "value"] = 0.5

    return params


def is_fixed(row: pd.Series) -> bool:
    """Check if a parameter row is fixed (lower == upper == value)."""
    return row["lower_bound"] == row["upper_bound"]


def get_free_mask(params_template: pd.DataFrame) -> np.ndarray:
    """Return boolean mask for free (non-fixed) parameters."""
    return (params_template["lower_bound"] != params_template["upper_bound"]).to_numpy()


def apply_start_params(
    params_template: pd.DataFrame,
    start_params: pd.DataFrame,
) -> None:
    """Override heuristic defaults with user-supplied starting values.

    Match on the 4-level MultiIndex. Only free (non-fixed) parameters whose
    index appears in `start_params` are updated. Fixed parameters and
    parameters not in `start_params` are left unchanged. Modifies
    `params_template` in place.
    """
    common = params_template.index.intersection(start_params.index)
    if common.empty:
        return
    free = (
        params_template.loc[common, "lower_bound"]
        != params_template.loc[common, "upper_bound"]
    )
    to_update = common[free]
    if not to_update.empty:
        params_template.loc[to_update, "value"] = start_params.loc[to_update, "value"]
