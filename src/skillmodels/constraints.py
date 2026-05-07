"""Constraint objects for a model specification."""

import functools
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import optimagic as om
import pandas as pd

import skillmodels.transition_functions as t_f_module
from skillmodels.types import (
    Anchoring,
    Dimensions,
    EndogenousFactorsInfo,
    Labels,
    MeasurementType,
    Normalizations,
)


def select_by_loc(params: pd.DataFrame, loc: Any) -> pd.DataFrame:  # noqa: ANN401
    """Select parameters by location."""
    return params.loc[loc]


@dataclass(frozen=True)
class FixedConstraintWithValue(om.FixedConstraint):
    """Fixed constraint that carries the target value and parameter location.

    `om.FixedConstraint` fixes parameters at their start values but does not carry a
    target value. This wrapper adds `loc` (the parameter location in the params
    DataFrame) and `value` (the value to set before optimization).
    """

    loc: pd.MultiIndex | tuple | str | None = None
    """Parameter location in the params DataFrame."""
    value: float | None = None
    """Value to enforce on the parameter."""

    def __post_init__(self) -> None:
        """Validate that `loc` and `value` are not None and derive `selector`."""
        if self.loc is None:
            msg = "loc must not be None"
            raise TypeError(msg)
        if self.value is None:
            msg = "value must not be None"
            raise TypeError(msg)
        object.__setattr__(
            self,
            "selector",
            functools.partial(select_by_loc, loc=self.loc),
        )


def get_constraints(
    dimensions: Dimensions,
    labels: Labels,
    anchoring_info: Anchoring,
    update_info: pd.DataFrame,
    normalizations: Mapping[str, Normalizations],
    endogenous_factors_info: EndogenousFactorsInfo,
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


def _is_diagonal_entry(ind_tup: tuple[str, ...]) -> bool:
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
                    value=endogenous_factors_info.bounds_distance,
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
