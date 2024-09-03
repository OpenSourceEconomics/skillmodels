import numpy as np


def check_model(model_dict, labels, dimensions, anchoring):
    """Check consistency and validity of the model specification.

    labels, dimensions and anchoring information are done before the model checking
    because processing them will not raise any errors except for easy to understand
    KeyErrors.

    Other specifications are checked in the model dict before processing to make sure
    that the assumptions we make during the processing are fulfilled.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.

        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

        anchoring (dict): Dictionary with information about anchoring.
            See :ref:`anchoring`

    Raises:
        ValueError

    """
    report = _check_stagemap(
        labels["stagemap"],
        labels["stages"],
        dimensions["n_periods"],
    )
    report += _check_anchoring(anchoring)
    report += _check_measurements(model_dict, labels["latent_factors"])
    report += _check_normalizations(model_dict, labels["latent_factors"])

    report = "\n".join(report)
    if report != "":
        raise ValueError(f"Invalid model specification:\n{report}")


def _check_stagemap(stagemap, stages, n_periods):
    report = []
    if len(stagemap) != n_periods - 1:
        report.append(
            "The stagemap needs to be of length n_periods - 1. n_periods is "
            f"{n_periods}, the stagemap has length {len(stagemap)}.",
        )

    if stages != list(range(len(stages))):
        report.append("Stages need to be integers, start at zero and increase by 1.")

    if not np.isin(np.array(stagemap[1:]) - np.array(stagemap[:-1]), (0, 1)).all():
        report.append("Consecutive entries in stagemap must be equal or increase by 1.")
    return report


def _check_anchoring(anchoring):
    report = []
    if not isinstance(anchoring["anchoring"], bool):
        report.append("anchoring['anchoring'] must be a bool.")
    if not isinstance(anchoring["outcomes"], dict):
        report.append("anchoring['outcomes'] must be a dict")
    else:
        variables = list(anchoring["outcomes"].values())
        for var in variables:
            if not isinstance(var, (str, int, tuple)):
                report.append("Outcomes variables have to be valid variable names.")

    if not isinstance(anchoring["free_controls"], bool):
        report.append("anchoring['use_controls'] must be a bool")
    if not isinstance(anchoring["free_constant"], bool):
        report.append("anchoring['use_constant'] must be a bool.")
    if not isinstance(anchoring["free_loadings"], bool):
        report.append("anchoring['free_loadings'] must be a bool.")
    return report


def _check_measurements(model_dict, factors):
    report = []
    for factor in factors:
        candidate = model_dict["factors"][factor]["measurements"]
        if not _is_list_of(candidate, list):
            report.append(
                f"measurements must lists of lists. Check measurements of {factor}.",
            )
        else:
            for period, meas_list in enumerate(candidate):
                for meas in meas_list:
                    if not isinstance(meas, (int, str, tuple)):
                        report.append(
                            "Measurements need to be valid pandas column names. Check "
                            f"{meas} for {factor} in period {period}.",
                        )
    return report


def _check_normalizations(model_dict, factors):
    report = []
    for factor in factors:
        norminfo = model_dict["factors"][factor].get("normalizations", {})
        for norm_type in ["loadings", "intercepts"]:
            candidate = norminfo.get(norm_type, [])
            if not _is_list_of(candidate, dict):
                report.append(
                    f"normalizations must be lists of dicts. Check {norm_type} "
                    f"normalizations for {factor}.",
                )
            else:
                report += _check_normalized_variables_are_present(
                    candidate,
                    model_dict,
                    factor,
                )

                if norm_type == "loadings":
                    report += _check_loadings_are_not_normalized_to_zero(
                        candidate,
                        factor,
                    )
    return report


def _check_normalized_variables_are_present(list_of_normdicts, model_dict, factor):
    report = []
    for period, norm_dict in enumerate(list_of_normdicts):
        for var in norm_dict:
            if var not in model_dict["factors"][factor]["measurements"][period]:
                report.append(
                    "You can only normalize variables that are specified as "
                    f"measurements. Check {var} for {factor} in period "
                    f"{period}.",
                )

    return report


def _check_loadings_are_not_normalized_to_zero(list_of_normdicts, factor):
    report = []
    for period, norm_dict in enumerate(list_of_normdicts):
        for var, val in norm_dict.items():
            if val == 0:
                report.append(
                    f"loadings cannot be normalized to 0. Check measurement {var} "
                    f"of {factor} in period {period}.",
                )
    return report


def _is_list_of(candidate, type_):
    """Check if candidate is a list that only contains elements of type.

    Note that this is always falls if candidate is not a list and always true if
    it is an empty list.

    Examples:
    >>> _is_list_of([["a"], ["b"]], list)
    True
    >>> _is_list_of([{}], list)
    False
    >>> _is_list_of([], dict)
    True

    """
    return isinstance(candidate, list) and all(isinstance(i, type_) for i in candidate)
