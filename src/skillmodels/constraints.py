"""List of constraints for a model, which can be converted to optimagic constraints."""

import functools
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optimagic as om
import pandas as pd

import skillmodels.transition_functions as t_f_module


def get_constraints_dicts(
    dimensions,
    labels,
    anchoring_info,
    update_info,
    normalizations,
    endogenous_factors_info,
) -> list[dict]:
    """Generate constraints implied by the model specification.

    The result can easily be converted to optimagic-style constraints.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        anchoring (dict): Information about anchoring. See :ref:`anchoring`
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        normalizations (dict): Nested dictionary with information on normalized factor
            loadings and intercepts for each factor. See :ref:`normalizations`.

    Returns:
        A list of constraints dictionaries with entries:
        - "type": str, one of "fixed", "equality", "probability", "increasing",
            "pairwise_equality". Must map to an optimagic constraint, see
            :func:`constraints_dicts_to_om`.
        - "loc": The location of the affected row(s) in the params DataFrame
        - "value": float, only present if type is "fixed"
        - "description": str, optional description of the constraint

    """
    constraints_dicts = []

    constraints_dicts += _get_normalization_constraints(
        normalizations, labels["latent_factors"]
    )
    constraints_dicts += _get_mixture_weights_constraints(dimensions["n_mixtures"])
    constraints_dicts += _get_stage_constraints(
        stagemap=labels["aug_stagemap"],
        stages=labels["aug_stages"],
    )
    constraints_dicts += _get_constant_factors_constraints(labels=labels)
    constraints_dicts += _get_initial_states_constraints(
        n_mixtures=dimensions["n_mixtures"],
        factors=labels["latent_factors"],
    )
    constraints_dicts += _get_transition_constraints(labels=labels)
    constraints_dicts += _get_anchoring_constraints(
        update_info=update_info,
        controls=labels["controls"],
        anchoring_info=anchoring_info,
        periods=labels["aug_periods"],
    )
    if endogenous_factors_info["has_endogenous_factors"]:
        constraints_dicts += _get_constraints_for_augmented_periods(
            labels=labels,
            endogenous_factors_info=endogenous_factors_info,
        )

    for i, c in enumerate(constraints_dicts):
        c["id"] = i

    return constraints_dicts


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


def _is_diagonal_entry(ind_tup):
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


def _get_normalization_constraints(normalizations, factors) -> list[dict]:
    """List of constraints to enforce normalizations.

    Args:
        normalizations (dict): Nested dictionary with information on normalized factor
        loadings and intercepts for each factor. See :ref:`normalizations`.

    Returns:
        constraints_dicts

    """
    msg = "This constraint was generated because of an explicit normalization."
    periods = range(len(normalizations[factors[0]]["loadings"]))

    constraints_dicts = []
    for factor in factors:
        if "variances" in normalizations[factor]:
            raise ValueError("normalization for variances cannot be provided")
        for period in periods:
            for meas, normval in normalizations[factor]["loadings"][period].items():
                constraints_dicts.append(
                    {
                        "loc": ("loadings", period, meas, factor),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )
            for meas, normval in normalizations[factor]["intercepts"][period].items():
                constraints_dicts.append(
                    {
                        "loc": ("controls", period, meas, "constant"),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )

    return constraints_dicts


def _get_mixture_weights_constraints(n_mixtures) -> list[dict]:
    """Constrain mixture weights to be between 0 and 1 and sum to 1."""
    if n_mixtures == 1:
        msg = "Set the mixture weight to 1 if there is only one mixture element."
        constraints_dicts = [
            {
                "loc": "mixture_weights",
                "type": "fixed",
                "value": 1.0,
                "description": msg,
            },
        ]
    else:
        msg = "Ensure that weights are between 0 and 1 and sum to 1."
        constraints_dicts = [
            {"loc": "mixture_weights", "type": "probability", "description": msg}
        ]
    return constraints_dicts


def _get_stage_constraints(stagemap, stages) -> list[dict]:
    """Equality constraints for transition and shock parameters within stages.

    Args:
        stagemap (list): map aug_periods to aug_stages
        stages (list): aug_stages
    Returns:
        constraints_dicts

    """
    msg = (
        "This constraint was generated because all involved periods belong to stage {}."
    )
    constraints_dicts = []

    stages_to_periods = {stage: [] for stage in stages}
    for aug_period, stage in enumerate(stagemap):
        stages_to_periods[stage].append(aug_period)

    for stage, stage_periods in stages_to_periods.items():
        if len(stage_periods) > 1:
            loc_trans = [("transition", p) for p in stage_periods]
            loc_q = [("shock_sds", p) for p in stage_periods]
            constraints_dicts.append(
                {
                    "loc": loc_trans,
                    "type": "pairwise_equality",
                    "description": msg.format(stage),
                },
            )
            constraints_dicts.append(
                {
                    "loc": loc_q,
                    "type": "pairwise_equality",
                    "description": msg.format(stage),
                },
            )

    return constraints_dicts


def _get_constant_factors_constraints(labels) -> list[dict]:
    """Fix shock variances of constant factors to `bounds_distance`.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints_dicts

    """
    constraints_dicts = []
    for f, factor in enumerate(labels["latent_factors"]):
        if labels["transition_names"][f] == "constant":
            msg = f"This constraint was generated because {factor} is constant."
            for aug_period in labels["aug_periods"][:-1]:
                constraints_dicts.append(
                    {
                        "loc": ("shock_sds", aug_period, factor, "-"),
                        "type": "fixed",
                        "value": 0.0,
                        "description": msg,
                    },
                )
    return constraints_dicts


def _get_initial_states_constraints(n_mixtures, factors) -> list[dict]:
    """Enforce that the x values of the first factor are increasing.

    Otherwise the model would only be identified up to the order of the start factors.

    Args:
        n_mixtures (int): number of elements in the mixture of normal of the factors.
        factors (list): the latent factors of the model

    Returns:
        constraints_dicts

    """
    msg = (
        "This constraint enforces an ordering on the initial means of the states "
        "across the components of the factor distribution. This is necessary to ensure "
        "uniqueness of the maximum likelihood estimator."
    )

    if n_mixtures > 1:
        locs = [
            ("initial_states", 0, f"mixture_{emf}", factors[0])
            for emf in range(n_mixtures)
        ]
        constraints_dicts = [{"loc": locs, "type": "increasing", "description": msg}]
    else:
        constraints_dicts = []

    return constraints_dicts


def _get_transition_constraints(labels) -> list[dict]:
    """Collect possible constraints on transition parameters.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints_dicts

    """
    constraints_dicts = []
    for f, factor in enumerate(labels["latent_factors"]):
        tname = labels["transition_names"][f]
        msg = f"This constraint is inherent to the {tname} production function."
        for aug_period in labels["aug_periods"][:-1]:
            funcname = f"constraints_{tname}"
            if func := getattr(t_f_module, funcname, False):
                c = func(
                    factor=factor, factors=labels["all_factors"], aug_period=aug_period
                )
                if "description" not in c:
                    c["description"] = msg
                constraints_dicts.append(c)
    return constraints_dicts


def _get_anchoring_constraints(
    update_info, controls, anchoring_info, periods
) -> list[dict]:
    """Constraints on anchoring parameters.

    Args:
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        controls (list): List of control variables
        anchoring_info (dict): Information about anchoring. See :ref:`anchoring`
        periods (list): Period of the model

    Returns:
        constraints_dicts

    """
    anchoring_updates = update_info[update_info["purpose"] == "anchoring"].index

    constraints_dicts = []
    if not anchoring_info["free_constant"]:
        msg = (
            "This constraint was generated because free_constant in the anchoring "
            "section of the model specification is set to False."
        )
        locs = []
        for period, meas in anchoring_updates:
            locs.append(("controls", period, meas, "constant"))
        constraints_dicts.append(
            {"loc": locs, "type": "fixed", "value": 0, "description": msg},
        )

    if not anchoring_info["free_controls"]:
        msg = (
            "This constraint was generated because free_controls in the anchoring "
            "section of the model specification is set to False."
        )
        ind_tups = []
        for period, meas in anchoring_updates:
            for cont in [c for c in controls if c != "constant"]:
                ind_tups.append(("controls", period, meas, cont))
        constraints_dicts.append(
            {"loc": ind_tups, "type": "fixed", "value": 0, "description": msg},
        )

    if not anchoring_info["free_loadings"]:
        msg = (
            "This constraint was generated because free_loadings in the anchoring "
            "section of the model specification is set to False."
        )
        ind_tups = []
        for period in periods:
            for factor in anchoring_info["factors"]:
                outcome = anchoring_info["outcomes"][factor]
                meas = f"{outcome}_{factor}"
                ind_tups.append(("loadings", period, meas, factor))

        constraints_dicts.append(
            {"loc": ind_tups, "type": "fixed", "value": 1, "description": msg},
        )

    constraints_dicts = [c for c in constraints_dicts if c["loc"] != []]

    return constraints_dicts


def _get_constraints_for_augmented_periods(
    labels, endogenous_factors_info
) -> list[dict]:
    """Constraints for augmented periods.

    - Carry forward states from uneven periods to even periods
    - Carry forward endogenous factors even periods to uneven periods
    - Set shock_sds to 0 when carrying anything forward

    Both depend on the transition function.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints_dicts

    """
    constraints_dicts = []
    for f, factor in enumerate(labels["latent_factors"]):
        tname = labels["transition_names"][f]
        if tname == "constant":
            continue
        # We are restricting transitions and shocks, not measurements. So this might
        # look counterintuitive...
        aug_period_meas_type_to_constrain = (
            "states"
            if endogenous_factors_info[factor]["is_state"]
            else "endogenous_factors"
        )
        aug_periods_to_constrain = [
            k
            for k, v in endogenous_factors_info[
                "aug_periods_to_aug_period_meas_types"
            ].items()
            if v == aug_period_meas_type_to_constrain
        ]
        for aug_period in aug_periods_to_constrain:
            if func := getattr(t_f_module, f"identity_constraints_{tname}", False):
                constraints_dicts += func(
                    factor=factor,
                    aug_period=aug_period,
                    all_factors=labels["all_factors"],
                )
        for aug_period in aug_periods_to_constrain[:-1]:
            constraints_dicts.append(
                {
                    "loc": ("shock_sds", aug_period, factor, "-"),
                    "type": "fixed",
                    "value": endogenous_factors_info["bounds_distance"],
                    "description": "Identity constraint.",
                }
            )

    return constraints_dicts


def _sel(params, loc):
    return params.loc[loc]


@dataclass(frozen=True)
class SkillmodelsPairwiseEqualityConstraint(om.PairwiseEqualityConstraint):
    """Thin wrapper around om.PairwiseEqualityConstraint.

    Adds fields to preserve information from the internal constraints dictionary.
    """

    loc: pd.MultiIndex | tuple | str | None = None
    description: str | None = None
    type: str = "Just to be able to use **constraints_dict"
    id: int | None = None


@dataclass(frozen=True)
class SkillmodelsFixedConstraint(om.FixedConstraint):
    """Thin wrapper around om.FixedConstraint.

    Adds fields to preserve information from the internal constraints dictionary.
    """

    loc: pd.MultiIndex | tuple | str | None = None
    description: str | None = None
    type: str = "Just to be able to use **constraints_dict"
    id: int | None = None
    value: float | None = None


@dataclass(frozen=True)
class SkillmodelsEqualityConstraint(om.EqualityConstraint):
    """Thin wrapper around om.EqualityConstraint.

    Adds fields to preserve information from the internal constraints dictionary.
    """

    loc: pd.MultiIndex | tuple | str | None = None
    description: str | None = None
    type: str = "Just to be able to use **constraints_dict"
    id: int | None = None


@dataclass(frozen=True)
class SkillmodelsProbabilityConstraint(om.ProbabilityConstraint):
    """Thin wrapper around om.ProbabilityConstraint.

    Adds fields to preserve information from the internal constraints dictionary.
    """

    loc: pd.MultiIndex | tuple | str | None = None
    description: str | None = None
    type: str = "Just to be able to use **constraints_dict"
    id: int | None = None


@dataclass(frozen=True)
class SkillmodelsIncreasingConstraint(om.IncreasingConstraint):
    """Thin wrapper around om.IncreasingConstraint.

    Adds fields to preserve information from the internal constraints dictionary.
    """

    loc: pd.MultiIndex | tuple | str | None = None
    description: str | None = None
    type: str = "Just to be able to use **constraints_dict"
    id: int | None = None


def constraints_dicts_to_om(
    constraints_dicts: list[dict],
) -> list[om.constraints.Constraint]:
    """Convert constraints provided in dictionary form to optimagic constraints.

    Args:
        constraints_dicts (list): see :ref:`get_constraints_dicts`.

    Returns:
        List of optimagic constraints.
    """
    om_style = []
    for c_d in constraints_dicts:
        if c_d["type"] == "pairwise_equality":
            om_style.append(
                SkillmodelsPairwiseEqualityConstraint(
                    selectors=[functools.partial(_sel, loc=loc) for loc in c_d["loc"]],
                    **c_d,
                )
            )
        else:
            sel = functools.partial(_sel, loc=c_d["loc"])
            if c_d["type"] == "fixed":
                om_style.append(SkillmodelsFixedConstraint(selector=sel, **c_d))
            elif c_d["type"] == "equality":
                om_style.append(SkillmodelsEqualityConstraint(selector=sel, **c_d))
            elif c_d["type"] == "probability":
                om_style.append(SkillmodelsProbabilityConstraint(selector=sel, **c_d))
            elif c_d["type"] == "increasing":
                om_style.append(SkillmodelsIncreasingConstraint(selector=sel, **c_d))
            else:
                raise TypeError(c_d["type"])
    return om_style


def enforce_fixed_constraints(
    params_template: pd.DataFrame,
    constraints_dicts: list[dict[str, Any]],
) -> pd.DataFrame:
    """Enforce fixed constraints on params_template.

    For fixed constraints, we also set the lower and upper bounds to the fixed value.
    This means that any robust bounds will be overridden for fixed parameters.

    Args:
        params_template (pd.DataFrame): see :ref:`params_df`.
        constraints_dicts (list): see :ref:`get_constraints_dicts`.

    Returns:
        pd.DataFrame: modified copy of params_template
    """
    params = params_template.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing past lexsort depth may impact performance.",
        )
        for constraint in constraints_dicts:
            if constraint["type"] == "fixed":
                params.loc[constraint["loc"], "value"] = constraint["value"]

    # Setting via loc may expand the index, so reduce to the original index
    return params.loc[params_template.index].astype(float)
