"""Construct an estimagic constraints list for a model."""
import skillmodels.model_functions.transition_functions as tf
import numpy as np


def constraints(
    update_info,
    controls,
    factors,
    normalizations,
    measurements,
    nemf,
    stagemap,
    transition_names,
    included_factors,
    invariant_meas_system,
):

    periods = list(range(len(stagemap)))
    constr = []

    if invariant_meas_system:
        constr += _invariant_meas_system_constraints(update_info, controls, factors)
    constr += _normalization_constraints(normalizations)
    constr += _not_measured_constraints(update_info, measurements)
    constr += _w_constraints()
    constr += _p_constraints(nemf)
    constr += _stage_constraints(stagemap, factors, transition_names, included_factors)
    constr += _constant_factors_constraints(factors, transition_names, periods)
    constr += _ar1_contraints(factors, transition_names, included_factors, periods)
    constr += _x_constraints(nemf, factors)
    constr += _trans_coeff_constraints(
        factors, transition_names, included_factors, periods
    )

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
    df.loc["r", "lower"] = bounds_distance
    df.loc["q", "lower"] = bounds_distance
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
    locs = []
    for period, meas in update_info.index:
        if update_info.loc[(period, meas), "is_repeated"] == True:
            first = update_info.loc[(period, meas), "first_occurence"]
            for cont in ["constant"] + controls[period]:
                locs.append(
                    [("delta", period, meas, cont), ("delta", int(first), meas, cont)]
                )
            for factor in factors:
                locs.append(
                    [("h", period, meas, factor), ("h", int(first), meas, factor)]
                )
            locs.append([("r", period, meas, ""), ("r", int(first), meas, "")])

    constraints = [{"type": "equality", "loc": loc} for loc in locs]
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
    constraints = []
    factors = sorted(list(normalizations.keys()))
    periods = range(len(normalizations[factors[0]]["loadings"]))
    for factor in factors:
        for period in periods:
            loading_norminfo = normalizations[factor]["loadings"][period]
            for meas, normval in loading_norminfo.items():
                constraints.append(
                    {
                        "loc": ("h", period, meas, factor),
                        "type": "fixed",
                        "value": normval,
                    }
                )
            intercept_norminfo = normalizations[factor]["intercepts"][period]
            for meas, normval in intercept_norminfo.items():
                constraints.append(
                    {
                        "loc": ("delta", period, meas, "constant"),
                        "type": "fixed",
                        "value": normval,
                    }
                )
            variance_norminfo = normalizations[factor]["variances"][period]
            for meas, normval in variance_norminfo.items():
                constraints.append(
                    {"loc": ("r", period, meas, ""), "type": "fixed", "value": normval}
                )
    return constraints


def _not_measured_constraints(update_info, measurements):
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
    factors = sorted(list(measurements.keys()))
    periods = range(len(measurements[factors[0]]))

    locs = []
    for period in periods:
        all_measurements = update_info.loc[period].index
        for factor in factors:
            used_measurements = measurements[factor][period]
            for meas in all_measurements:
                if meas not in used_measurements:
                    locs.append(("h", period, meas, factor))

    constraints = [{"loc": loc, "type": "fixed", "value": 0} for loc in locs]

    return constraints


def _w_constraints():
    """Constrain mixture weights to be between 0 and 1 and sum to 1."""
    return [{"loc": "w", "type": "probability"}]


def _p_constraints(nemf):
    """Constraint initial covariance matrices to be positive semi-definite.

    Args:
        nemf (int): number of elements in the mixture of normal of the factors.

    Returns:
        constraints (list)

    """
    constraints = []
    for emf in range(nemf):
        constraints.append({"loc": ("p", 0, emf), "type": "covariance"})
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
    constraints = []
    periods = range(len(stagemap))
    for period in periods[1:-1]:
        stage = stagemap[period]
        need_equality = stage == stagemap[period - 1]
        for f, factor in enumerate(factors):
            if need_equality:
                func = getattr(tf, "index_tuples_{}".format(transition_names[f]))
                ind1 = func(factor, included_factors[f], period - 1)
                ind2 = func(factor, included_factors[f], period)
                constraints += _pairwise_equality_constraint(ind1, ind2)

                constraints.append(
                    {
                        "loc": [
                            ("q", period - 1, factor, ""),
                            ("q", period, factor, ""),
                        ],
                        "type": "equality",
                    }
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
            for period in periods[:-1]:
                constraints.append(
                    {"loc": ("q", period, factor, ""), "type": "fixed", "value": 0.0}
                )
    return constraints


def _ar1_contraints(factors, transition_names, included_factors, periods):
    """Equality constraints on transition and shock parameters for ar1 factors.

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
        if transition_names[f] == "ar1":
            for period in periods[1:-1]:
                constraints.append(
                    {
                        "loc": [
                            ("q", period - 1, factor, ""),
                            ("q", period, factor, ""),
                        ],
                        "type": "equality",
                    }
                )
                func = getattr(tf, "index_tuples_ar1")
                ind1 = func(factor, included_factors[f], period - 1)
                ind2 = func(factor, included_factors[f], period)
                constraints += _pairwise_equality_constraint(ind1, ind2)
    return constraints


def _x_constraints(nemf, factors):
    """Enforce that the x values of the first factor are increasing.

    Otherwise the model would only be identified up to the order of the start factors.

    Args:
        nemf (int): number of elements in the mixture of normal of the factors.
        factors (list): the latent factors of the model

    Returns:
        constraints (list)

    """
    ind_tups = [("x", 0, emf, factors[0]) for emf in range(nemf)]
    return [{"loc": ind_tups, "type": "increasing"}]


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
        for period in periods[:-1]:
            funcname = "constraints_{}".format(transition_names[f])
            if hasattr(tf, funcname):
                func = getattr(tf, funcname)
                constraints.append(func(factor, included_factors[f], period))
    return constraints


# in the long run a sophisticated version of this might move to estimagic
def _pairwise_equality_constraint(index1, index2):
    assert len(index1) == len(index2), "index1 and index2 must have the same length."

    constraints = []
    for i1, i2 in zip(index1, index2):
        constraints.append({"loc": [i1, i2], "type": "equality"})
    return constraints
