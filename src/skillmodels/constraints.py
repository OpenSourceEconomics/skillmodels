"""List of constraints for a model, which can be converted to optimagic constraints."""

import warnings

import numpy as np

import skillmodels.transition_functions as tf


def get_constraints(dimensions, labels, anchoring_info, update_info, normalizations):
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
        list[dict]: List of constraints.

    """
    constr = []

    constr += _get_normalization_constraints(normalizations, labels["latent_factors"])
    constr += _get_mixture_weights_constraints(dimensions["n_mixtures"])
    constr += _get_stage_constraints(labels["stagemap"], labels["stages"])
    constr += _get_constant_factors_constraints(labels)
    constr += _get_initial_states_constraints(
        dimensions["n_mixtures"],
        labels["latent_factors"],
    )
    constr += _get_transition_constraints(labels)
    constr += _get_anchoring_constraints(
        update_info,
        labels["controls"],
        anchoring_info,
        labels["periods"],
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing past lexsort depth may impact performance.",
        )
        if "lower_bound" not in df.columns:
            df["lower_bound"] = -np.inf
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


def _get_normalization_constraints(normalizations, factors):
    """List of constraints to enforce normalizations.

    Args:
        normalizations (dict): Nested dictionary with information on normalized factor
        loadings and intercepts for each factor. See :ref:`normalizations`.

    Returns:
        constraints (list)

    """
    msg = "This constraint was generated because of an explicit normalization."
    periods = range(len(normalizations[factors[0]]["loadings"]))

    index_tuples = []
    fixed_values = []

    for factor in factors:
        if "variances" in normalizations[factor]:
            raise ValueError("normalization for variances cannot be provided")
        for period in periods:
            loading_norminfo = normalizations[factor]["loadings"][period]
            for meas, normval in loading_norminfo.items():
                index_tuples.append(("loadings", period, meas, factor))
                fixed_values.append(normval)

            intercept_norminfo = normalizations[factor]["intercepts"][period]
            for meas, normval in intercept_norminfo.items():
                index_tuples.append(("controls", period, meas, "constant"))
                fixed_values.append(normval)

    if index_tuples:
        constraints = [
            {
                "loc": index_tuples,
                "type": "fixed",
                "value": fixed_values,
                "description": msg,
            },
        ]
    else:
        constraints = []

    return constraints


def _get_mixture_weights_constraints(n_mixtures):
    """Constrain mixture weights to be between 0 and 1 and sum to 1."""
    if n_mixtures == 1:
        msg = "Set the mixture weight to 1 if there is only one mixture element."
        return [
            {
                "loc": "mixture_weights",
                "type": "fixed",
                "value": 1.0,
                "description": msg,
            },
        ]
    msg = "Ensure that weights are between 0 and 1 and sum to 1."
    return [{"loc": "mixture_weights", "type": "probability", "description": msg}]


def _get_stage_constraints(stagemap, stages):
    """Equality constraints for transition and shock parameters within stages.

    Args:
        stagemap (list): map periods to stages
        stages (list): stages
    Returns:
        constrainst (list)

    """
    msg = (
        "This constraint was generated because all involved periods belong to stage {}."
    )
    constraints = []

    stages_to_periods = {stage: [] for stage in stages}
    for period, stage in enumerate(stagemap):
        stages_to_periods[stage].append(period)

    for stage, stage_periods in stages_to_periods.items():
        if len(stage_periods) > 1:
            locs_trans = [("transition", p) for p in stage_periods]
            locs_q = [("shock_sds", p) for p in stage_periods]
            constraints.append(
                {
                    "locs": locs_trans,
                    "type": "pairwise_equality",
                    "description": msg.format(stage),
                },
            )
            constraints.append(
                {
                    "locs": locs_q,
                    "type": "pairwise_equality",
                    "description": msg.format(stage),
                },
            )

    return constraints


def _get_constant_factors_constraints(labels):
    """Fix shock variances of constant factors to 0.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints (list)

    """
    constraints = []
    for f, factor in enumerate(labels["latent_factors"]):
        if labels["transition_names"][f] == "constant":
            msg = f"This constraint was generated because {factor} is constant."
            for period in labels["periods"][:-1]:
                constraints.append(
                    {
                        "loc": ("shock_sds", period, factor, "-"),
                        "type": "fixed",
                        "value": 0.0,
                        "description": msg,
                    },
                )
    return constraints


def _get_initial_states_constraints(n_mixtures, factors):
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

    if n_mixtures > 1:
        ind_tups = [
            ("initial_states", 0, f"mixture_{emf}", factors[0])
            for emf in range(n_mixtures)
        ]
        constr = [{"loc": ind_tups, "type": "increasing", "description": msg}]
    else:
        constr = []

    return constr


def _get_transition_constraints(labels):
    """Collect possible constraints on transition parameters.

    Args:
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        constraints (list)

    """
    constraints = []
    for f, factor in enumerate(labels["latent_factors"]):
        tname = labels["transition_names"][f]
        msg = f"This constraint is inherent to the {tname} production function."
        for period in labels["periods"][:-1]:
            funcname = f"constraints_{tname}"
            if hasattr(tf, funcname):
                func = getattr(tf, funcname)
                constr = func(factor, labels["all_factors"], period)
                if "description" not in constr:
                    constr["description"] = msg
                constraints.append(constr)
    return constraints


def _get_anchoring_constraints(update_info, controls, anchoring_info, periods):
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
    anchoring_updates = update_info[update_info["purpose"] == "anchoring"].index

    constraints = []
    if not anchoring_info["free_constant"]:
        msg = (
            "This constraint was generated because free_constant in the anchoring "
            "section of the model specification is set to False."
        )
        locs = []
        for period, meas in anchoring_updates:
            locs.append(("controls", period, meas, "constant"))
        constraints.append(
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
        constraints.append(
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

        constraints.append(
            {"loc": ind_tups, "type": "fixed", "value": 1, "description": msg},
        )

    constraints = [c for c in constraints if c["loc"] != []]

    return constraints
