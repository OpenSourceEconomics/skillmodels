"""Constraint objects for a model specification."""

import functools
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import optimagic as om
import pandas as pd

import skillmodels.common.transition_functions as t_f_module
from skillmodels.common.fixed_constraint import FixedConstraintWithValue
from skillmodels.common.selector import align_index_names, select_by_loc
from skillmodels.common.types import (
    Anchoring,
    Dimensions,
    EndogenousFactorsInfo,
    Labels,
    MeasurementType,
    Normalizations,
)

__all__ = [
    "FixedConstraintWithValue",
    "add_bounds",
    "align_index_names",
    "collect_fixed_locs",
    "enforce_fixed_constraints",
    "filter_within_step_constraints",
    "get_constraints",
    "project_to_probability_constraints",
    "reconcile_start_to_equality",
    "select_by_loc",
]


def _equality_constraint_loc(c: om.constraints.Constraint) -> pd.MultiIndex | None:
    """Return the `loc` MultiIndex of a `select_by_loc`-style EqualityConstraint.

    Returns `None` for any other constraint type or selector shape, so
    callers can `continue` past unrecognised entries without nested
    guard clauses.
    """
    if not isinstance(c, om.EqualityConstraint):
        return None
    keywords = getattr(c.selector, "keywords", None)
    if not keywords:
        return None
    loc = keywords.get("loc")
    return loc if isinstance(loc, pd.MultiIndex) else None


def filter_within_step_constraints(
    user_constraints: list[om.constraints.Constraint] | None,
    params_index: pd.Index,
) -> list[om.constraints.Constraint]:
    """Return user equality constraints fully contained in `params_index`.

    Used by AF's per-step optimizers to forward only those user-supplied
    `om.EqualityConstraint` objects whose `select_by_loc` `loc` MultiIndex
    is a subset of the current step's params index. Cross-period
    equalities (whose members straddle multiple steps) are handled
    separately by `_propagate_equality_groups` in
    `skillmodels.af.estimate`.
    """
    if not user_constraints:
        return []
    idx_set = set(params_index)
    out: list[om.constraints.Constraint] = []
    for c in user_constraints:
        loc = _equality_constraint_loc(c)
        if loc is not None and all(tup in idx_set for tup in loc):
            out.append(c)
    return out


def reconcile_start_to_equality(
    params: pd.DataFrame,
    equality_constraints: list[om.constraints.Constraint],
) -> pd.DataFrame:
    """Average each equality group's `value` so the start point satisfies it.

    `om.minimize` raises `InvalidParamsError` when an equality
    constraint is violated at the starting point. For each constraint
    in `equality_constraints` whose selector is
    `functools.partial(select_by_loc, loc=...)`, set every member's
    `value` to the mean of the group's current values. Returns a copy;
    `params` is not modified.
    """
    if not equality_constraints:
        return params
    out = params.copy()
    for c in equality_constraints:
        loc = _equality_constraint_loc(c)
        if loc is None or not all(tup in out.index for tup in loc):
            continue
        out.loc[loc, "value"] = float(out.loc[loc, "value"].mean())
    return out


def collect_fixed_locs(
    constraints: Iterable[om.constraints.Constraint],
) -> set[tuple[Any, ...]]:
    """Flatten every `FixedConstraintWithValue.loc` into a single set of tuples.

    Used by `project_to_probability_constraints` to decide which
    entries of a `ProbabilityConstraint` group are already pinned by
    an overlapping `FixedConstraintWithValue` and therefore must not
    be touched by the rescaling step.

    Handles every shape that `FixedConstraintWithValue.loc` permits
    per its type annotation: a single 4-tuple (`("loadings", 0, ...)`),
    a `tuple` / `list` of 4-tuples (used by the anchoring
    constraints), and a `pd.MultiIndex` (the type annotation allows
    it; the runtime needs to follow). String `loc`s (like
    `"mixture_weights"`) are deliberately skipped: they refer to
    a category prefix in the params index, not to a single
    parameter, and never belong to a probability fold.
    """
    fixed_locs: set[tuple[Any, ...]] = set()
    for c in constraints:
        if not isinstance(c, FixedConstraintWithValue):
            continue
        loc = c.loc
        if isinstance(loc, pd.MultiIndex):
            fixed_locs.update(tuple(t) for t in loc)
        elif isinstance(loc, tuple) and loc and not isinstance(loc[0], tuple):
            fixed_locs.add(loc)
        elif isinstance(loc, (list, tuple)):
            fixed_locs.update(sub for sub in loc if isinstance(sub, tuple))
    return fixed_locs


def project_to_probability_constraints(
    params_template: pd.DataFrame,
    constraints: Iterable[om.constraints.Constraint],
) -> pd.DataFrame:
    """Project starting values onto each `ProbabilityConstraint`'s simplex.

    Spearman / AMN seeding does not know about probability folds: the
    seeded entries don't sum to one. Walk every `ProbabilityConstraint`
    whose selector is the `select_by_loc(loc=list_of_tuples)` form and
    rescale its free members so they sum to `1 - sum(fixed_values)`.
    Entries also bound by a `FixedConstraintWithValue` keep their
    pinned value; only the remaining (free) entries are rescaled.
    Groups where the free entries sum to zero are left untouched --
    the user is on the hook for supplying a feasible start in that
    degenerate case.
    """
    fixed_locs = collect_fixed_locs(constraints)

    out = params_template
    for c in constraints:
        if not isinstance(c, om.ProbabilityConstraint):
            continue
        keywords = getattr(c.selector, "keywords", None)
        loc = keywords.get("loc") if keywords else None
        if not isinstance(loc, list):
            continue

        free_loc = [tup for tup in loc if tup not in fixed_locs]
        pinned_loc = [tup for tup in loc if tup in fixed_locs]
        if not free_loc:
            continue
        try:
            free_values = out.loc[free_loc, "value"]
        except KeyError:
            continue
        free_total = float(free_values.sum())
        if free_total <= 0 or not np.isfinite(free_total):
            continue

        pinned_total = float(out.loc[pinned_loc, "value"].sum()) if pinned_loc else 0.0
        target = max(0.0, 1.0 - pinned_total)
        if abs(free_total - target) < 1e-12:
            continue

        if out is params_template:
            out = params_template.copy()
        out.loc[free_loc, "value"] = free_values * (target / free_total)
    return out


def get_constraints(
    dimensions: Dimensions,
    labels: Labels,
    anchoring_info: Anchoring,
    update_info: pd.DataFrame,
    normalizations: Mapping[str, Normalizations],
    endogenous_factors_info: EndogenousFactorsInfo,
    bounds_distance: float,
) -> list[om.constraints.Constraint]:
    """Generate constraints implied by the model specification.

    Args:
        dimensions: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        anchoring_info: Information about anchoring. See :ref:`anchoring`
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        normalizations: Nested dictionary with information on normalized factor
            loadings and intercepts for each factor. See :ref:`normalizations`.
        endogenous_factors_info: Information about endogenous factors in the model.
        bounds_distance: Distance from zero/one used for soft-pinning shock
            standard deviations in carry-forward augmented periods.

    Returns:
        List of optimagic constraint objects.

    """
    constraints: list[om.constraints.Constraint] = []

    constraints += _get_normalization_constraints(
        normalizations=normalizations, factors=labels.latent_factors
    )
    constraints += _get_mixture_weights_constraints(dimensions.n_mixtures)
    constraints += _get_stage_constraints(
        stagemap=labels.aug_stagemap,
        stages=labels.aug_stages,
    )
    constraints += _get_constant_factors_constraints(labels=labels)
    constraints += _get_initial_states_constraints(
        n_mixtures=dimensions.n_mixtures,
        factors=labels.latent_factors,
    )
    constraints += _get_transition_constraints(labels=labels)
    constraints += _get_anchoring_constraints(
        update_info=update_info,
        controls=labels.controls,
        anchoring_info=anchoring_info,
        periods=labels.aug_periods,
    )
    if endogenous_factors_info.has_endogenous_factors:
        constraints += _get_constraints_for_augmented_periods(
            labels=labels,
            endogenous_factors_info=endogenous_factors_info,
            bounds_distance=bounds_distance,
        )

    return constraints


def add_bounds(params: pd.DataFrame, bounds_distance: float) -> pd.DataFrame:
    """Add bounds for standard deviations to params.

    Lower and upper bounds are set to (minus) infinity; lower bounds for standard
    deviation-like parameters are set to *bounds_distance*. Note that the latter will be
    overridden for parameters where fixed constraints are imposed.

    Args:
        params: see :ref:`params`.
        bounds_distance: set standard deviation-like to this amount.

    Returns:
        Modified copy of params

    """
    df = params.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing past lexsort depth may impact performance.",
        )
        if "lower_bound" not in df.columns:
            df["lower_bound"] = -np.inf
        if "upper_bound" not in df.columns:
            df["upper_bound"] = np.inf

        df.loc["meas_sds", "lower_bound"] = bounds_distance
        df.loc["shock_sds", "lower_bound"] = bounds_distance

        cholcov_index = df.query("category == 'initial_cholcovs'").index.tolist()
        ind_tups = [tup for tup in cholcov_index if _is_diagonal_entry(tup)]
        df.loc[ind_tups, "lower_bound"] = bounds_distance

    return df


def _is_diagonal_entry(ind_tup: tuple[Any, ...]) -> bool:
    name2 = ind_tup[-1]
    middle_pos = int(len(name2) // 2)
    if (
        len(name2) % 2 == 0
        or name2[middle_pos] != "-"
        or name2[:middle_pos] != name2[middle_pos + 1 :]
    ):
        is_diag = False
    else:
        is_diag = True
    return is_diag


def _get_normalization_constraints(
    normalizations: Mapping[str, Normalizations],
    factors: tuple[str, ...],
) -> list[om.constraints.Constraint]:
    """List of constraints to enforce normalizations.

    Args:
        normalizations: Mapping from factor name to Normalizations instance.
        factors: Tuple of factor names to process.

    Returns:
        List of constraint objects.

    """
    periods = range(len(normalizations[factors[0]].loadings))

    constraints: list[om.constraints.Constraint] = []
    for factor in factors:
        for period in periods:
            for meas, normval in normalizations[factor].loadings[period].items():
                loc = ("loadings", period, meas, factor)
                constraints.append(FixedConstraintWithValue(loc=loc, value=normval))
            for meas, normval in normalizations[factor].intercepts[period].items():
                loc = ("controls", period, meas, "constant")
                constraints.append(FixedConstraintWithValue(loc=loc, value=normval))

    return constraints


def _get_mixture_weights_constraints(
    n_mixtures: int,
) -> list[om.constraints.Constraint]:
    """Constrain mixture weights to be between 0 and 1 and sum to 1."""
    loc = "mixture_weights"
    if n_mixtures == 1:
        return [
            FixedConstraintWithValue(loc=loc, value=1.0),
        ]
    return [
        om.ProbabilityConstraint(selector=functools.partial(select_by_loc, loc=loc))
    ]


def _get_stage_constraints(
    stagemap: tuple[int, ...],
    stages: tuple[int, ...],
) -> list[om.constraints.Constraint]:
    """Equality constraints for transition and shock parameters within stages.

    Args:
        stagemap: map aug_periods to aug_stages
        stages: aug_stages

    Returns:
        List of constraint objects.

    """
    constraints: list[om.constraints.Constraint] = []

    stages_to_periods: dict[int, list[int]] = {stage: [] for stage in stages}
    for aug_period, stage in enumerate(stagemap):
        stages_to_periods[stage].append(aug_period)

    for stage_periods in stages_to_periods.values():
        if len(stage_periods) > 1:
            loc_trans = [("transition", p) for p in stage_periods]
            loc_q = [("shock_sds", p) for p in stage_periods]
            constraints.append(
                om.PairwiseEqualityConstraint(
                    selectors=[
                        functools.partial(select_by_loc, loc=loc) for loc in loc_trans
                    ],
                ),
            )
            constraints.append(
                om.PairwiseEqualityConstraint(
                    selectors=[
                        functools.partial(select_by_loc, loc=loc) for loc in loc_q
                    ],
                ),
            )

    return constraints


def _get_constant_factors_constraints(
    labels: Labels,
) -> list[om.constraints.Constraint]:
    """Fix shock variances of constant factors to zero.

    Args:
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        List of constraint objects.

    """
    constraints: list[om.constraints.Constraint] = []
    for f, factor in enumerate(labels.latent_factors):
        if labels.transition_names[f] == "constant":
            for aug_period in labels.aug_periods[:-1]:
                loc = ("shock_sds", aug_period, factor, "-")
                constraints.append(
                    FixedConstraintWithValue(loc=loc, value=0.0),
                )
    return constraints


def _get_initial_states_constraints(
    n_mixtures: int,
    factors: tuple[str, ...],
) -> list[om.constraints.Constraint]:
    """Enforce that the x values of the first factor are increasing.

    Otherwise the model would only be identified up to the order of the start factors.

    Args:
        n_mixtures: number of elements in the mixture of normal of the factors.
        factors: the latent factors of the model

    Returns:
        List of constraint objects.

    """
    if n_mixtures > 1:
        locs = [
            ("initial_states", 0, f"mixture_{emf}", factors[0])
            for emf in range(n_mixtures)
        ]
        return [
            om.IncreasingConstraint(selector=functools.partial(select_by_loc, loc=locs))
        ]
    return []


def _get_transition_constraints(
    labels: Labels,
) -> list[om.constraints.Constraint]:
    """Collect possible constraints on transition parameters.

    Args:
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        List of constraint objects.

    """
    constraints: list[om.constraints.Constraint] = []
    for f, factor in enumerate(labels.latent_factors):
        tname = labels.transition_names[f]
        for aug_period in labels.aug_periods[:-1]:
            funcname = f"constraints_{tname}"
            if func := getattr(t_f_module, funcname, False):
                constraints.append(
                    func(  # ty: ignore[call-non-callable]
                        factor=factor,
                        factors=labels.all_factors,
                        aug_period=aug_period,
                    )
                )
    return constraints


def _get_anchoring_constraints(  # noqa: C901
    update_info: pd.DataFrame,
    controls: tuple[str, ...],
    anchoring_info: Anchoring,
    periods: tuple[int, ...],
) -> list[om.constraints.Constraint]:
    """Constraints on anchoring parameters.

    Args:
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        controls: List of control variables
        anchoring_info: Information about anchoring. See :ref:`anchoring`
        periods: Period of the model

    Returns:
        List of constraint objects.

    """
    anchoring_updates = update_info[update_info["purpose"] == "anchoring"].index

    constraints: list[om.constraints.Constraint] = []
    if not anchoring_info.free_constant:
        locs = []
        for period, meas in anchoring_updates:
            locs.append(("controls", period, meas, "constant"))
        if locs:
            loc = tuple(locs)
            constraints.append(
                FixedConstraintWithValue(loc=loc, value=0),
            )

    if not anchoring_info.free_controls:
        ind_tups = []
        for period, meas in anchoring_updates:
            for cont in [c for c in controls if c != "constant"]:
                ind_tups.append(("controls", period, meas, cont))
        if ind_tups:
            loc = tuple(ind_tups)
            constraints.append(
                FixedConstraintWithValue(loc=loc, value=0),
            )

    if not anchoring_info.free_loadings:
        ind_tups = []
        for period in periods:
            for factor in anchoring_info.factors:
                outcome = anchoring_info.outcomes[factor]
                meas = f"{outcome}_{factor}"
                ind_tups.append(("loadings", period, meas, factor))

        if ind_tups:
            loc = tuple(ind_tups)
            constraints.append(
                FixedConstraintWithValue(loc=loc, value=1),
            )

    return constraints


def _get_constraints_for_augmented_periods(
    labels: Labels,
    endogenous_factors_info: EndogenousFactorsInfo,
    bounds_distance: float,
) -> list[om.constraints.Constraint]:
    """Constraints for augmented periods.

    - Carry forward states from uneven periods to even periods
    - Carry forward endogenous factors even periods to uneven periods
    - Set shock_sds to 0 when carrying anything forward

    Both depend on the transition function.

    Args:
        labels: Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        endogenous_factors_info: Information about endogenous factors and their
            relationship to augmented periods.
        bounds_distance: Value to pin shock standard deviations to in
            carry-forward augmented periods.

    Returns:
        List of constraint objects.

    """
    constraints: list[om.constraints.Constraint] = []
    for f, factor in enumerate(labels.latent_factors):
        tname = labels.transition_names[f]
        if tname == "constant":
            continue
        # We are restricting transitions and shocks, not measurements. So this might
        # look counterintuitive...
        aug_period_meas_type_to_constrain = (
            MeasurementType.STATES
            if endogenous_factors_info.factor_info[factor].is_state
            else MeasurementType.ENDOGENOUS_FACTORS
        )
        aug_period_meas_types = (
            endogenous_factors_info.aug_periods_to_aug_period_meas_types
        )
        aug_periods_to_constrain = [
            k
            for k, v in aug_period_meas_types.items()
            if v == aug_period_meas_type_to_constrain
        ]
        # The last entry of `aug_periods_to_constrain` is the aug-period
        # half of the last calendar period for this factor's meas-type.
        # `get_transition_index_tuples` stops at `aug_periods[:-2]` when
        # endogenous factors are present (or `[:-1]` otherwise), so the
        # params index has no transition entries at that final aug-period
        # for any factor. Emitting identity constraints there would target
        # locs that don't exist and trip the optimagic selector. The
        # shock-sds loop below already uses `[:-1]` for the same reason
        # — keep them symmetric.
        for aug_period in aug_periods_to_constrain[:-1]:
            if func := getattr(t_f_module, f"identity_constraints_{tname}", False):
                constraints += func(  # ty: ignore[call-non-callable]
                    factor=factor,
                    aug_period=aug_period,
                    all_factors=labels.all_factors,
                )
        for aug_period in aug_periods_to_constrain[:-1]:
            loc = ("shock_sds", aug_period, factor, "-")
            constraints.append(
                FixedConstraintWithValue(
                    loc=loc,
                    value=bounds_distance,
                )
            )

    return constraints


def enforce_fixed_constraints(
    params_template: pd.DataFrame,
    constraints: list[om.constraints.Constraint],
) -> pd.DataFrame:
    """Enforce fixed constraints on params_template.

    For fixed constraints, we also set the lower and upper bounds to the fixed value.
    This means that any robust bounds will be overridden for fixed parameters.

    Args:
        params_template: see :ref:`params_df`.
        constraints: list of optimagic constraint objects.

    Returns:
        pd.DataFrame: modified copy of params_template
    """
    params = params_template.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing past lexsort depth may impact performance.",
        )
        for constraint in constraints:
            if isinstance(constraint, FixedConstraintWithValue):
                params.loc[constraint.loc, "value"] = constraint.value

    # Setting via loc may expand the index, so reduce to the original index
    return params.loc[params_template.index].astype(float)
