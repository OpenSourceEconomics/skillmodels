"""Construct an estimagic constraints list for a model."""
import numpy as np

import skillmodels.transition_functions_jax as tf
from skillmodels.params_index_jax import loading_index_tuples


def constraints(dimensions, labels, anchoring_info, update_info, normalizations):
    """Generate the estimagic constraints implied by the model specification.

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
        list: List of estimagic compatible constraints.

    """
    constr = []

    constr += _normalization_constraints(normalizations)
    constr += _not_measured_constraints(update_info, anchoring_info, labels)
    constr += _mixture_weight_constraints(dimensions["n_mixtures"])
    constr += _stage_constraints(labels["stagemap"], labels["stages"])
    constr += _constant_factors_constraints(labels)
    constr += _initial_mean_constraints(dimensions["n_mixtures"], labels["factors"])
    constr += _trans_coeff_constraints(labels)
    constr += _anchoring_constraints(
        update_info, labels["controls"], anchoring_info, labels["periods"]
    )

    for i, c in enumerate(constr):
        c["id"] = i

    return constr


def add_bounds(params_df, bounds_distance=0.0):
    """Add the bounds to params_df that are not implied by other constraints.

    Args:
        params_df (DataFrame): see :ref:`params_df`.
        bounds_distance (float): sets bounds stricter by this amount. Default 0.0.

    Returns:
        df (DataFrame): modified copy of params_df

    """
    df = params_df.copy()
    if "lower" not in df.columns:
        df["lower"] = -np.inf
    df.loc["meas_sds", "lower"] = bounds_distance
    df.loc["shock_sds", "lower"] = bounds_distance
    return df


def _normalization_constraints(normalizations):
    """List of constraints to enforce normalizations.

    Args:
        normalizations (dict): Nested dictionary with information on normalized factor
        loadings and intercepts for each factor. See :ref:`normalizations`.

    Returns:
        constraints (list)

    """
    msg = "This constraint was generated because of an explicit normalization."
    constraints = []
    factors = sorted(normalizations.keys())
    periods = range(len(normalizations[factors[0]]["loadings"]))
    for factor in factors:
        if "variances" in normalizations[factor].keys():
            raise ValueError("normalization for variances cannot be provided")
        for period in periods:
            loading_norminfo = normalizations[factor]["loadings"][period]
            for meas, normval in loading_norminfo.items():
                constraints.append(
                    {
                        "loc": ("loadings", period, meas, factor),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )
            intercept_norminfo = normalizations[factor]["intercepts"][period]
            for meas, normval in intercept_norminfo.items():
                constraints.append(
                    {
                        "loc": ("controls", period, meas, "constant"),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )

    return constraints


def _not_measured_constraints(update_info, labels):
    """Fix all loadings for non-measured factors to 0.

    Args:
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`


    Returns:
        constraints (list)

    """

    msg = (
        "This constraint sets the loadings of those factors that are not measured by "
        "a measurement to 0."
    )

    factors = labels["factors"]
    all_loading_indices = loading_index_tuples(factors, update_info)
    to_fix = ~update_info[factors].to_numpy().flatten().astype(bool)
    locs = [tup for i, tup in enumerate(all_loading_indices) if to_fix[i]]

    return [{"loc": locs, "type": "fixed", "value": 0, "description": msg}]


def _mixture_weight_constraints(n_mixtures):
    """Constrain mixture weights to be between 0 and 1 and sum to 1."""
    if n_mixtures == 1:
        msg = "Set the mixture weight to 1 if there is only one mixture element."
        return [
            {
                "loc": "mixture_weights",
                "type": "fixed",
                "value": 1.0,
                "description": msg,
            }
        ]
    else:
        msg = "Ensure that weights are between 0 and 1 and sum to 1."
        return [{"loc": "mixture_weights", "type": "probability", "description": msg}]


def _stage_constraints(stagemap, stages):
    """Equality constraints for transition and shock parameters within stages.

    Args:
        stagemap (list): map periods to stages
        stages (list): stages
    Returns:
        constrainst (list)

    """
    msg = (
        "This constraint was generated because you have a 'stagemap' in your model "
        "specification."
    )
    constraints = []

    stages_to_periods = {stage: [] for stage in stages}
    for stage in stages:
        for period, stage in enumerate(stagemap):
            stages_to_periods[stage].append(period)

    for stage in stages:
        locs_trans = [("transition", p) for p in stages_to_periods[stage]]
        locs_q = [("shock_sds", p) for p in stages_to_periods[stage]]
        constraints.append(
            {"locs": locs_trans, "type": "pairwise_equality", "description": msg}
        )
        constraints.append(
            {"locs": locs_q, "type": "pairwise_equality", "description": msg}
        )

    return constraints


def _constant_factors_constraints(labels):
    """Fix shock variances of constant factors to 0.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints (list)

    """
    constraints = []
    for f, factor in enumerate(labels["factors"]):
        if labels["transition_names"][f] == "constant":
            msg = f"This constraint was generated because {factor} is constant."
            for period in labels["periods"][:-1]:
                constraints.append(
                    {
                        "loc": ("shock_sds", period, factor, "-"),
                        "type": "fixed",
                        "value": 0.0,
                        "description": msg,
                    }
                )
    return constraints


def _initial_mean_constraints(n_mixtures, factors):
    """Enforce that the x values of the first factor are increasing.

    Otherwise the model would only be identified up to the order of the start factors.

    Args:
        n_mixtures (int): number of elements in the mixture of normal of the factors.
        factors (list): the latent factors of the model

    Returns:
        constraints (list)

    """
    msg = (
        "This constraint enforces an ordering on the initial means of the states "
        "across the components of the factor distribution. This is necessary to ensure "
        "uniqueness of the maximum likelihood estimator."
    )

    ind_tups = [
        ("initial_states", 0, f"mixture_{emf}", factors[0]) for emf in range(n_mixtures)
    ]
    constr = [{"loc": ind_tups, "type": "increasing", "description": msg}]

    return constr


def _trans_coeff_constraints(labels):
    """Collect possible constraints on transition parameters.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints (list)

    """
    constraints = []
    for f, factor in enumerate(labels["factors"]):
        tname = labels["transition_names"][f]
        msg = f"This constraint is inherent to the {tname} production function."
        for period in labels["periods"][:-1]:
            funcname = f"constraints_{tname}"
            if hasattr(tf, funcname):
                func = getattr(tf, funcname)
                constr = func(factor, labels["factors"], period)
                if "description" not in constr:
                    constr["description"] = msg
                constraints.append(constr)
    return constraints


def _anchoring_constraints(update_info, controls, anchoring_info, periods):
    """Constraints on anchoring parameters.

    Args:
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        controls (list): List of control variables
        anchoring_info (dict): Information about anchoring. See :ref:`anchoring`
        periods (list): Period of the model

    Returns:
        constraints (list)

    """
    anch_outcome = anchoring_info["outcome"]
    anchoring_updates = update_info[update_info["purpose"] == "anchoring"].index

    constraints = []
    if not anchoring_info["use_constant"]:
        for period, meas in anchoring_updates:
            ind_tup = ("controls", period, meas, "constant")
            constraints.append({"loc": ind_tup, "type": "fixed", "value": 0})

    if not anchoring_info["use_controls"]:
        for period, meas in anchoring_updates:
            ind_tups = []
            for cont in controls:
                ind_tups.append(("controls", period, meas, cont))
            constraints.append({"loc": ind_tups, "type": "fixed", "value": 0})

    if not anchoring_info["free_loadings"]:
        for period in periods:
            ind_tups = []
            for factor in anchoring_info["factors"]:
                meas = f"{anch_outcome}_{factor}"
                ind_tups.append(("loadings", period, meas, factor))

            constraints.append({"loc": ind_tups, "type": "fixed", "value": 1})

    return constraints
