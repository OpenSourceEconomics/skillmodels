import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from skillmodels.params_index import get_params_index
from skillmodels.process_model import get_dimensions
from skillmodels.process_model import process_model


def extract_factors(factors, model_dict, params=None):
    """Reduce a specification to a model with fewer latent factors.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        factors (str or list): Name(s) of the factor(s) to extract.
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    if isinstance(factors, str):
        factors = [factors]

    to_remove = set(model_dict["factors"]).difference(factors)
    out = remove_factors(to_remove, model_dict, params)
    return out


def update_parameter_values(params, others):
    """Update the "value" column of params with values from other.

    Args:
        params (pandas.DataFrame or None): The params DataFrame for the full model.
        other (pandas.DataFrame or list): Another DataFrame with parameters or list
            of thereof. The values from other are used to update the value column
            of ``params``. If other is a list, the updates will be in order, i.e.
            later elements overwrite earlier ones.

    Returns:
        pandas.DataFrame: Updated copy of params.

    """
    if isinstance(others, pd.DataFrame):
        others = [others]

    out = params.copy(deep=True)
    for other in others:
        out["value"].update(other["value"])

    return out


def remove_factors(factors, model_dict, params=None):
    """Remove factors from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    It is possible that the reduced model has fewer periods than the original one.
    This happens if the remaining factors do not have measurements in later periods.

    Args:
        factors (str or list): Name(s) of the factor(s) to remove.
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)

    out["factors"] = _remove_from_dict(out["factors"], factors)

    # adjust anchoring
    if "anchoring" in model_dict:
        out["anchoring"]["outcomes"] = _remove_from_dict(
            out["anchoring"]["outcomes"], factors
        )
        if out["anchoring"]["outcomes"] == {}:
            out = _remove_from_dict(out, "anchoring")

    # Remove periods if necessary
    new_n_periods = get_dimensions(out)["n_periods"]
    out = reduce_n_periods(out, new_n_periods)

    if params is not None:
        out_params = _reduce_params(params, out)
        out = (out, out_params)

    return out


def remove_measurements(measurements, model_dict, params=None):
    """Remove measurements from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        measurements (str or list): Name(s) of the measurement(s) to remove.
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)

    for factor in model_dict["factors"]:
        full = model_dict["factors"][factor]["measurements"]
        reduced = [_remove_from_list(meas_list, measurements) for meas_list in full]
        out["factors"][factor]["measurements"] = reduced

        norminfo = model_dict["factors"][factor].get("normalizations", {})
        if "loadings" in norminfo:
            out["factors"][factor]["normalizations"][
                "loadings"
            ] = _remove_measurements_from_normalizations(
                measurements, norminfo["loadings"]
            )

        if "intercepts" in norminfo:
            out["factors"][factor]["normalizations"][
                "intercepts"
            ] = _remove_measurements_from_normalizations(
                measurements, norminfo["intercepts"]
            )

    if params is not None:
        out_params = _reduce_params(params, out)
        out = (out, out_params)

    return out


def remove_controls(controls, model_dict, params=None):
    """Remove control variables from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        controls (str or list): Name(s) of the contral variable(s) to remove.
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    out["controls"] = _remove_from_list(out["controls"], controls)
    if out["controls"] == []:
        out = _remove_from_dict(out, "controls")

    if params is not None:
        out_params = _reduce_params(params, out)
        out = (out, out_params)

    return out


def switch_translog_to_linear(model_dict, params=None):
    """Switch all translog production functions to linear.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    for factor in model_dict["factors"]:
        if model_dict["factors"][factor]["transition_function"] == "translog":
            out["factors"][factor]["transition_function"] = "linear"

    if params is not None:
        out_params = _reduce_params(params, out)
        out = (out, out_params)

    return out


def switch_linear_to_translog(model_dict, params=None):
    """Switch all linear production functions to translog.

    If provided, a params DataFrame is also extended correspondingly. The fill value
    for the additional terms is 0.05 because experience showed that estimating a
    translog model with start parameters obtained from a linear model is faster when
    the additional parameters are not initialized at zero.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    for factor in model_dict["factors"]:
        if model_dict["factors"][factor]["transition_function"] == "linear":
            out["factors"][factor]["transition_function"] = "translog"

    if params is not None:
        out_params = _extend_params(params, out, 0.05)
        out = (out, out_params)
    return out


def reduce_n_periods(model_dict, new_n_periods, params=None):
    """Remove all periods after n_periods.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`.
        new_n_periods (int): The new number of periods.
        params (pandas.DataFrame or None): The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    for factor in model_dict["factors"]:
        out["factors"][factor]["measurements"] = _shorten_if_necessary(
            out["factors"][factor]["measurements"], new_n_periods
        )

        norminfo = model_dict["factors"][factor].get("normalizations", {})
        if "loadings" in norminfo:
            out["factors"][factor]["normalizations"][
                "loadings"
            ] = _shorten_if_necessary(norminfo["loadings"], new_n_periods)

        if "intercepts" in norminfo:
            out["factors"][factor]["normalizations"][
                "intercepts"
            ] = _shorten_if_necessary(norminfo["intercepts"], new_n_periods)

    if "stagemap" in out:
        out["stagemap"] = _shorten_if_necessary(out["stagemap"], new_n_periods - 1)

    if params is not None:
        out_params = _extend_params(params, out, 0.05)
        out = (out, out_params)

    return out


def _remove_from_list(list_, to_remove):
    if isinstance(to_remove, str):
        to_remove = [to_remove]
    return [element for element in list_ if element not in to_remove]


def _remove_from_dict(dict_, to_remove):
    if isinstance(to_remove, str):
        to_remove = [to_remove]

    return {key: val for key, val in dict_.items() if key not in to_remove}


def _reduce_params(params, model_dict):
    """Reduce a parameter DataFrame from a larger model to a reduced model.

    The reduced model must be nested in the original model for which the params
    DataFrame was constructed.

    Args:
        params (pandas.DataFrame or None): The params DataFrame for the full model.
        model_dict (dict): The model specification. See: :ref:`model_specs`.

    Returns
        pandas.DataFrame: The reduced parameters DataFrame.

    """
    index = _get_params_index_from_model_dict(model_dict)
    out = params.loc[index]
    return out


def _extend_params(params, model_dict, fill_value):
    index = _get_params_index_from_model_dict(model_dict)
    out = params.reindex(index)
    out["value"] = out["value"].fillna(fill_value)
    if "lower_bound" in out:
        out["lower_bound"] = out["lower_bound"].fillna(-np.inf)

    if "upper_bound" in out:
        out["upper_bound"] = out["upper_bound"].fillna(np.inf)

    return out


def _get_params_index_from_model_dict(model_dict):
    mod = process_model(model_dict)
    index = get_params_index(
        update_info=mod["update_info"],
        labels=mod["labels"],
        dimensions=mod["dimensions"],
    )
    return index


def _remove_measurements_from_normalizations(measurements, normalizations):
    reduced = [_remove_from_dict(norm, measurements) for norm in normalizations]
    if reduced != normalizations:
        warnings.warn(
            "Your removed a normalized measurement from a model. Make sure there are "
            "enough normalizations left to ensure identification."
        )
    return reduced


def _shorten_if_necessary(list_, length):
    if len(list_) > length:
        list_ = list_[:length]
    return list_
