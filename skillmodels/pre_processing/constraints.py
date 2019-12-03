"""Construct an estimagic constraints list for a model."""
import numpy as np

import skillmodels.model_functions.transition_functions as tf


def constraints(
    update_info,
    controls,
    factors,
    normalizations,
    measurements,
    nmixtures,
    stagemap,
    transition_names,
    included_factors,
    invariant_meas_system,
    anchored_factors,
    anch_outcome,
    bounds_distance,
):

    periods = list(range(len(stagemap)))
    constr = []

    if invariant_meas_system:
        constr += _invariant_meas_system_constraints(update_info, controls, factors)
    constr += _normalization_constraints(normalizations)
    constr += _not_measured_constraints(
        update_info, measurements, anchored_factors, anch_outcome
    )
    constr += _mixture_weight_constraints(nmixtures)
    constr += _initial_cov_constraints(nmixtures, bounds_distance)
    constr += _stage_constraints(stagemap, factors, transition_names, included_factors)
    constr += _constant_factors_constraints(factors, transition_names, periods)
    constr += _initial_mean_constraints(nmixtures, factors)
    constr += _trans_coeff_constraints(
        factors, transition_names, included_factors, periods
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
    df.loc["meas_sd", "lower"] = bounds_distance
    df.loc["shock_variance", "lower"] = bounds_distance
    return df


def _invariant_meas_system_constraints(update_info, controls, factors):
    """List of constraints that enforce a time invariant measurement system.

    Args:
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.
            The column 'is_repeated' indicates the if the update is repeated. The
            column 'first_occurrence' indicates in which period it already occurred
            or equals np.nan.
        controls (list): List of lists. There is one sublist per period which contains
            the names of the control variables in that period. Constant not included.
        factors (list): Names of the latent factors.

    Returns:
        constraints (list)

    """
    msg = (
        "This constraint was generated because you set 'time_invariante_measurement_"
        "system' to True in the general section of your model specification."
    )

    locs = []
    for period, meas in update_info.index:
        if update_info.loc[(period, meas), "is_repeated"]:
            first = update_info.loc[(period, meas), "first_occurence"]
            for cont in ["constant"] + controls[period]:
                locs.append(
                    [("delta", period, meas, cont), ("delta", int(first), meas, cont)]
                )
            for factor in factors:
                locs.append(
                    [
                        ("loading", period, meas, factor),
                        ("loading", int(first), meas, factor),
                    ]
                )
            locs.append(
                [("meas_sd", period, meas, "-"), ("meas_sd", int(first), meas, "-")]
            )

    constraints = [{"type": "equality", "loc": loc, "description": msg} for loc in locs]
    return constraints


def _normalization_constraints(normalizations):
    """List of constraints to enforce normalizations.

    Args:
        normalizations (dict): Nested dictionary with the normalization info.
            normalizations[factor][norm_type][period] is a dictionary where the keys
            are measurements and the values are the normalized value. norm_type can
            take the values ['loadings', 'intercepts', 'variances']

    Returns:
        constraints (list)

    """
    msg = "This constraint was generated because of an explicit normalization."
    constraints = []
    factors = sorted(normalizations.keys())
    periods = range(len(normalizations[factors[0]]["loadings"]))
    for factor in factors:
        for period in periods:
            loading_norminfo = normalizations[factor]["loadings"][period]
            for meas, normval in loading_norminfo.items():
                constraints.append(
                    {
                        "loc": ("loading", period, meas, factor),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )
            intercept_norminfo = normalizations[factor]["intercepts"][period]
            for meas, normval in intercept_norminfo.items():
                constraints.append(
                    {
                        "loc": ("delta", period, meas, "constant"),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )
            variance_norminfo = normalizations[factor]["variances"][period]
            for meas, normval in variance_norminfo.items():
                constraints.append(
                    {
                        "loc": ("meas_sd", period, meas, "-"),
                        "type": "fixed",
                        "value": normval,
                        "description": msg,
                    }
                )
    return constraints


def _not_measured_constraints(
    update_info, measurements, anchored_factors, anch_outcome
):
    """Fix all loadings for non-measured factors to 0.

    Args:
        update_info (DataFrame): DataFrame with one row per update.
            It has aMultiIndex that indicates the period and name of the measurement
            for that update.
        measurements (dict): measurements[factor][period] is a list of all measurements
            of *factor* in *period*.

    Returns:
        constraints (list)

    """
    msg = (
        "This constraint sets the loadings of those factors that are not measured by "
        "a measurement to 0."
    )

    factors = sorted(measurements.keys())
    periods = range(len(measurements[factors[0]]))

    locs = []
    for period in periods:
        all_measurements = update_info.loc[period].index
        for factor in factors:
            used_measurements = measurements[factor][period]
            if period == periods[-1] and factor in anchored_factors:
                used_measurements = used_measurements + [anch_outcome]
            for meas in all_measurements:
                if meas not in used_measurements:
                    locs.append(("loading", period, meas, factor))

    constraints = [{"loc": locs, "type": "fixed", "value": 0, "description": msg}]

    return constraints


def _mixture_weight_constraints(nmixtures):
    """Constrain mixture weights to be between 0 and 1 and sum to 1."""
    if nmixtures == 1:
        msg = "Set the mixture weight to 1 if there is only one mixture element."
        return [
            {"loc": "mixture_weight", "type": "fixed", "value": 1.0, "description": msg}
        ]
    else:
        msg = "Ensure that weights are between 0 and 1 and sum to 1."
        return [{"loc": "mixture_weight", "type": "probability", "description": msg}]


def _initial_cov_constraints(nmixtures, bounds_distance):
    """Constraint initial covariance matrices to be positive semi-definite.

    Args:
        nmixtures (int): number of elements in the mixture of normal of the factors.

    Returns:
        constraints (list)

    """
    msg = "Make sure that the covariance matrix is valid."
    constraints = []
    for emf in range(nmixtures):
        constraints.append(
            {
                "loc": ("initial_cov", 0, f"mixture_{emf}"),
                "type": "covariance",
                "bounds_distance": bounds_distance,
                "description": msg,
            }
        )
    return constraints


def _stage_constraints(stagemap, factors, transition_names, included_factors):
    """Equality constraints for transition and shock parameters within stages.

    Args:
        stagemap (list): map periods to stages
        factors (list): the latent factors of the model
        transition_names (list): name of the transition equation of each factor
        included_factors (list): the factors that appear on the right hand side of
            the transition equations of the latent factors.

    Returns:
        constrainst (list)

    """
    msg = (
        "This constraint was generated because you have a 'stagemap' in your model "
        "specification."
    )
    constraints = []

    stages = sorted(np.unique(stagemap))
    stages_to_periods = {stage: [] for stage in stages}
    for stage in stages:
        for period, stage in enumerate(stagemap[:-1]):
            stages_to_periods[stage].append(period)

    for stage in stages:
        locs_trans = [("trans", p) for p in stages_to_periods[stage]]
        locs_q = [("shock_variance", p) for p in stages_to_periods[stage]]
        constraints.append(
            {"locs": locs_trans, "type": "pairwise_equality", "description": msg}
        )
        constraints.append(
            {"locs": locs_q, "type": "pairwise_equality", "description": msg}
        )

    return constraints


def _constant_factors_constraints(factors, transition_names, periods):
    """Fix shock variances of constant factors to 0.

    Args:
        factors (list): the latent factors of the model
        transition_names (list): name of the transition equation of each factor
        periods (list): the periods of the model.

    Returns:
        constraints (list)

    """
    constraints = []
    for f, factor in enumerate(factors):
        if transition_names[f] == "constant":
            msg = f"This constraint was generated because {factor} is constant."
            for period in periods[:-1]:
                constraints.append(
                    {
                        "loc": ("shock_variance", period, factor, "-"),
                        "type": "fixed",
                        "value": 0.0,
                        "description": msg,
                    }
                )
    return constraints


def _initial_mean_constraints(nmixtures, factors):
    """Enforce that the x values of the first factor are increasing.

    Otherwise the model would only be identified up to the order of the start factors.

    Args:
        nmixtures (int): number of elements in the mixture of normal of the factors.
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
        ("initial_mean", 0, f"mixture_{emf}", factors[0]) for emf in range(nmixtures)
    ]
    constr = [{"loc": ind_tups, "type": "increasing", "description": msg}]

    return constr


def _trans_coeff_constraints(factors, transition_names, included_factors, periods):
    """Collect possible constraints on transition parameters.

    Args:
        factors (list): the latent factors of the model
        transition_names (list): name of the transition equation of each factor
        included_factors (list): the factors that appear on the right hand side of
            the transition equations of the latent factors.
        periods (list): the periods of the model.

    Returns:
        constraints (list)

    """
    constraints = []
    for f, factor in enumerate(factors):
        tname = transition_names[f]
        msg = f"This constraint is inherent to the {tname} production function."
        for period in periods[:-1]:
            funcname = f"constraints_{tname}"
            if hasattr(tf, funcname):
                func = getattr(tf, funcname)
                constr = func(factor, included_factors[f], period)
                if "description" not in constr:
                    constr["description"] = msg
                constraints.append(constr)
    return constraints
