import numpy as np
import pandas as pd
from pandas import DataFrame

import skillmodels.transition_functions as tf
from skillmodels.check_model import check_model


def process_model(model_dict):
    """Check, clean, extend and transform the model specs.

    Check the completeness, consistency and validity of the model specifications.

    Set default values and extend the model specification where necessary.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`

    Returns:
        dict: nested dictionary of model specs. It has the following entries:
        - dimensions (dict): Dimensional information like n_states, n_periods,
        n_controls, n_mixtures. See :ref:`dimensions`.
        - labels (dict): Dict of lists with labels for the model quantities like
        factors, periods, controls, stagemap and stages. See :ref:`labels`
        - anchoring (dict): Information about anchoring. See :ref:`anchoring`
        - transition_functions (tuple): Tuple of tuples of length n_periods. Each inner
        tuple has the following two entries: (name_of_transition_function, callable).
        - update_info (pandas.DataFrame): DataFrame with one row per Kalman update
        needed in the likelihood function. See :ref:`update_info`.
        - normalizations (dict): Nested dictionary with information on normalized factor
        loadings and intercepts for each factor. See :ref:`normalizations`.

    """
    dims = _get_dimensions(model_dict)
    labels = _get_labels(model_dict, dims)
    anchoring = _process_anchoring(model_dict)
    check_model(model_dict, labels, dims, anchoring)

    processed = {
        "dimensions": dims,
        "labels": labels,
        "anchoring": anchoring,
        "estimation_options": _process_estimation_options(model_dict),
        "transition_functions": _get_transition_functions(labels["transition_names"]),
        "update_info": _get_update_info(model_dict, dims, labels, anchoring),
        "normalizations": _process_normalizations(model_dict, dims, labels),
    }
    return processed


def _get_dimensions(model_dict):
    """Extract the dimensions of the model.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`

    Returns:
        dict: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.

    """
    all_n_periods = [len(d["measurements"]) for d in model_dict["factors"].values()]
    dims = {
        "n_states": len(model_dict["factors"]),
        "n_periods": max(all_n_periods),
        # plus 1 for the constant
        "n_controls": len(model_dict.get("controls", [])) + 1,
        "n_mixtures": model_dict["estimation_options"].get("n_mixtures", 1),
    }
    return dims


def _get_labels(model_dict, dimensions):
    """Extract labels of the model quantities.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.

    Returns:
        dict: Dict of lists with labels for the model quantities like
        factors, periods, controls, stagemap and stages. See :ref:`labels`

    """
    stagemap = model_dict.get("stagemap", list(range(dimensions["n_periods"] - 1)))

    labels = {
        "factors": sorted(model_dict["factors"]),
        "controls": ["constant"] + sorted(model_dict.get("controls", [])),
        "periods": list(range(dimensions["n_periods"])),
        "stagemap": stagemap,
        "stages": sorted(np.unique(stagemap)),
    }

    trans_names = []
    for factor in labels["factors"]:
        trans_names.append(model_dict["factors"][factor]["transition_function"])
    labels["transition_names"] = trans_names

    return labels


def _process_estimation_options(model_dict):
    """Process options.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`

    Returns:
        dict: Tuning parameters for the estimation. See :ref:`options`.

    """
    default_options = {
        "sigma_points_scale": 2,
        "robust_bounds": True,
        "bounds_distance": 1e-3,
        "clipping_lower_bound": -1e250,
        "clipping_upper_bound": None,
        "clipping_lower_hardness": 1,
        "clipping_upper_hardness": 1,
    }
    default_options.update(model_dict.get("estimation_options", {}))

    if not default_options["robust_bounds"]:
        default_options["bounds_distance"] = 0

    return default_options


def _process_anchoring(model_dict):
    """Process the specification that governs how latent factors are anchored.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`

    Returns:
        dict: Dictionary with information about anchoring. See :ref:`anchoring`

    """
    anchinfo = {
        "anchoring": False,
        "outcomes": {},
        "factors": [],
        "free_controls": False,
        "free_constant": False,
        "free_loadings": False,
        "ignore_constant_when_anchoring": False,
    }

    if "anchoring" in model_dict:
        anchinfo.update(model_dict["anchoring"])
        anchinfo["anchoring"] = True
        anchinfo["factors"] = sorted(anchinfo["outcomes"].keys())

    return anchinfo


def _get_transition_functions(transition_names):
    """Collect the transition functions in a nested tuple.

    Args:
        transition_names (list): Names of transition functions for each factor.

    Returns:
        tuple: Tuple of tuples of length n_periods. Each inner tuple
            has the following two entries: (name_of_transition_function, callable).

    """
    return tuple((name, getattr(tf, name)) for name in transition_names)


def _get_update_info(model_dict, dimensions, labels, anchoring_info):
    """Construct a DataFrame with information on each Kalman update.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        anchoring_info (dict): Information about anchoring. See :ref:`anchoring`

    Returns:
        pandas.DataFrame: DataFrame with one row per Kalman update needed in
            the likelihood function. See :ref:`update_info`.

    """
    index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["period", "variable"])
    uinfo = DataFrame(index=index, columns=labels["factors"] + ["purpose"])

    measurements = {}
    for factor in labels["factors"]:
        measurements[factor] = _fill_list(
            model_dict["factors"][factor]["measurements"], [], dimensions["n_periods"]
        )

    for period in labels["periods"]:
        for factor in labels["factors"]:
            for meas in measurements[factor][period]:
                uinfo.loc[(period, meas), factor] = True
                uinfo.loc[(period, meas), "purpose"] = "measurement"
        for factor in anchoring_info["factors"]:
            outcome = anchoring_info["outcomes"][factor]
            name = f"{outcome}_{factor}"
            uinfo.loc[(period, name), factor] = True
            uinfo.loc[(period, name), "purpose"] = "anchoring"

    uinfo.fillna(False, inplace=True)
    return uinfo


def _process_normalizations(model_dict, dimensions, labels):
    """Process the normalizations of intercepts and factor loadings.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        normalizations (dict): Nested dictionary with information on normalized factor
            loadings and intercepts for each factor. See :ref:`normalizations`.

    """
    normalizations = {}
    for factor in labels["factors"]:
        normalizations[factor] = {}
        norminfo = model_dict["factors"][factor].get("normalizations", {})
        for norm_type in ["loadings", "intercepts"]:
            candidate = norminfo.get(norm_type, [])
            candidate = _fill_list(candidate, {}, dimensions["n_periods"])
            normalizations[factor][norm_type] = candidate

    return normalizations


def _fill_list(short_list, fill_value, length):
    """Extend a list to specified length by filling it with the fill_value.

    Examples:
    >>> _fill_list(["a"], "b", 3)
    ['a', 'b', 'b']

    """
    res = list(short_list)
    diff = length - len(short_list)
    assert diff >= 0, "short_list has to be shorter than length."
    if diff >= 1:
        res += [fill_value] * diff
    return res
