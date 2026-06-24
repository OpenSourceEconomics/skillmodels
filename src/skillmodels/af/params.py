"""Parameter index construction and parsing for AF estimation."""

from types import MappingProxyType
from typing import Any

import numpy as np
import optimagic as om
import pandas as pd

from skillmodels.common.constraints import FixedConstraintWithValue
from skillmodels.common.types import Normalizations, TransitionInfo


def get_initial_period_params_index(
    *,
    n_mixture_components: int,
    latent_factors: tuple[str, ...],
    measurements_period_0: dict[str, tuple[str, ...]],
    controls: tuple[str, ...],
    observed_factors: tuple[str, ...] = (),
    reconstructed_factors: tuple[str, ...] = (),
) -> pd.MultiIndex:
    """Build parameter index for the initial period (Step 0).

    Parameters estimated in Step 0:
    - Mixture weights, means, Cholesky covariances for the joint distribution
      of the *state* latent factors (those with
      ``has_initial_distribution=True``) and observed factors at period 0.
    - Investment equation parameters (one block per ``reconstructed_factor``)
      and an investment shock SD per reconstructed factor. These pin the
      period-0 value of each reconstructed factor as a deterministic
      function of the state latents plus a shock.
    - Measurement loadings, intercepts, SDs for period 0.

    When ``observed_factors`` is non-empty, the initial distribution is
    modelled over the joint vector (state_latent, observed). Per-individual
    observed values let the likelihood condition on them via the Schur
    complement, which concentrates Halton draws and improves estimation
    precision.

    Args:
        n_mixture_components: Number of Gaussian mixture components.
        latent_factors: Names of *all* latent factors (including reconstructed
            ones). Used for loading entries in the measurement block so
            reconstructed factors can still load on period-0 measurements.
        measurements_period_0: Factor name -> tuple of measurement variable names.
        controls: Control variable names (includes "constant").
        observed_factors: Names of observed factors included in the joint
            initial distribution.
        reconstructed_factors: Latent factors with
            ``has_initial_distribution=False``. These are excluded from the
            mixture and receive their own investment-equation block at
            period 0 instead.

    Return:
        MultiIndex with levels (category, period, name1, name2).

    """
    ind_tups: list[tuple[str, int, str, str]] = []
    state_latent_factors = tuple(
        f for f in latent_factors if f not in reconstructed_factors
    )
    joint_factors = (*state_latent_factors, *observed_factors)

    # Measurements for the initial step exclude those that only load on
    # reconstructed factors; their period-0 measurement params are
    # estimated in the transition step 0->1 instead (matching MATLAB's
    # transition_01 block convention).
    measurements_period_0_filtered = {
        f: m for f, m in measurements_period_0.items() if f in state_latent_factors
    }

    # Mixture weights
    for m in range(n_mixture_components):
        ind_tups.append(("mixture_weights", 0, f"mixture_{m}", "-"))

    # Initial means per component per joint factor (state latent + observed)
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

    # Measurement params for period 0 over state-latent factors only.
    # Reconstructed factors' period-0 measurement params live in the
    # transition step 0->1 params index.
    ind_tups.extend(
        _measurement_index_tuples(
            period=0,
            latent_factors=state_latent_factors,
            measurements=measurements_period_0_filtered,
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
    shock_factors: tuple[str, ...] | None = None,
    measurement_index_tuples: list[tuple[str, int, str, str]] | None = None,
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
        shock_factors: Subset of `latent_factors` for which a production shock
            SD is estimated. Factors omitted here get no shock SD parameter
            and are integrated deterministically (dropping their shock
            dimension from the Halton draw). Defaults to `latent_factors`.
        measurement_index_tuples: Pre-compiled mixed-calendar measurement index
            rows (source/destination calendar adapter). When given, used verbatim
            for the measurement block instead of single-period emission.

    Return:
        MultiIndex with levels (category, period, name1, name2).

    """
    if shock_factors is None:
        shock_factors = latent_factors
    ind_tups: list[tuple[str, int, str, str]] = []

    # Transition parameters (for t-1 -> t)
    for factor in latent_factors:
        if factor in transition_info.param_names:
            for name in transition_info.param_names[factor]:
                ind_tups.append(("transition", period - 1, factor, name))

    # Shock SDs (for t-1 -> t): only factors that have a production shock
    for factor in shock_factors:
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

    # Measurement params. By default emit period-t rows for all factors; when the
    # source/destination calendar adapter supplies a pre-compiled mixed-calendar
    # measurement index (destination skills at d, source investment at s, in global
    # category order), use it verbatim instead.
    if measurement_index_tuples is not None:
        ind_tups.extend(measurement_index_tuples)
    else:
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

    # Bound the log_ces substitution parameter phi from above. Without
    # an upper bound the optimizer can drift phi to large positive
    # values where exp(states * phi) overflows and the gradient turns
    # to NaN. The lower side is well-behaved (phi -> -inf collapses to
    # a finite minimum via logsumexp), so leave it unbounded to match
    # MATLAB's (-inf, 1 - c) convention.
    phi_mask = (params.index.get_level_values("category") == "transition") & (
        params.index.get_level_values("name2") == "phi"
    )
    params.loc[phi_mask, "upper_bound"] = 1.0 - bounds_distance

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


def apply_fixed_params(
    params_template: pd.DataFrame,
    fixed_params: pd.DataFrame,
) -> None:
    """Set template values to match user-provided fixed values.

    Used to pin parameters that would otherwise be free -- e.g., identity
    transitions and zero shock SDs for time-invariant latent factors. The
    pinning itself is enforced through `FixedConstraintWithValue` objects
    emitted by `build_optimagic_inputs`; this helper only aligns the
    template's starting values with the fixes so early likelihood evaluations
    use the correct values. Modifies `params_template` in place.
    """
    common = params_template.index.intersection(fixed_params.index)
    if common.empty:
        return
    params_template.loc[common, "value"] = fixed_params.loc[common, "value"]


def build_optimagic_inputs(
    params_template: pd.DataFrame,
    fixed_params: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[om.constraints.Constraint]]:
    """Prepare the params DataFrame and fixed-constraint list for `om.minimize`.

    The AF template encodes normalization fixes by clamping
    ``lower_bound == upper_bound`` on affected rows. User-provided
    `fixed_params` add further pinned rows. Both are translated into
    `FixedConstraintWithValue` objects so optimagic can treat them uniformly
    -- in particular so fixes that overlap a `ProbabilityConstraint` selector
    get folded correctly. The returned DataFrame has infinite bounds on every
    row that is pinned by a constraint, since optimagic rejects finite bounds
    on probability selectors.

    Args:
        params_template: AF parameter template with value/lower_bound/upper_bound.
        fixed_params: Optional user-provided fixes (DataFrame with a "value"
            column and the same 4-level MultiIndex as the template).

    Return:
        Tuple of (full_params_df, fixed_constraints) where full_params_df
        carries the template values plus any user fixes on all rows, and
        fixed_constraints is a list of `FixedConstraintWithValue` objects
        covering every pinned row (normalisation and user fixes alike).

    """
    params = params_template.copy()

    if fixed_params is not None:
        common = params.index.intersection(fixed_params.index)
        if not common.empty:
            params.loc[common, "value"] = fixed_params.loc[common, "value"]

    fixed_from_bounds = (
        params["lower_bound"].to_numpy() == params["upper_bound"].to_numpy()
    )
    fixed_from_user: np.ndarray
    if fixed_params is not None:
        common = params.index.intersection(fixed_params.index)
        fixed_from_user = np.asarray(params.index.isin(common))
    else:
        fixed_from_user = np.zeros(len(params), dtype=bool)

    pinned = fixed_from_bounds | fixed_from_user

    constraints: list[om.constraints.Constraint] = []
    for idx in params.index[pinned]:
        constraints.append(
            FixedConstraintWithValue(
                loc=idx,
                value=float(params.loc[idx, "value"]),
            )
        )

    # Relax bounds on pinned rows: optimagic rejects finite bounds that
    # overlap a probability selector, and the FixedConstraint now does the
    # pinning.
    pinned_idx = params.index[pinned]
    params.loc[pinned_idx, "lower_bound"] = -np.inf
    params.loc[pinned_idx, "upper_bound"] = np.inf

    return params, constraints
