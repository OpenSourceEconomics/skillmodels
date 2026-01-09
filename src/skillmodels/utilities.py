import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from skillmodels.params_index import get_params_index
from skillmodels.process_model import (
    get_dimensions,
    get_has_endogenous_factors,
    process_model,
)


def extract_factors(
    factors: str | list[str],
    model_dict: dict[str, Any],
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Reduce a specification to a model with fewer latent factors.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        factors: Name(s) of the factor(s) to extract.
        model_dict: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    if isinstance(factors, str):
        factors = [factors]

    to_remove = list(set(model_dict["factors"]).difference(factors))
    out = remove_factors(to_remove, model_dict, params)
    return out


def update_parameter_values(
    params: pd.DataFrame,
    others: pd.DataFrame | list[pd.DataFrame],
) -> pd.DataFrame:
    """Update the "value" column of params with values from other.

    Args:
        params: The params DataFrame for the full model.
        others: Another DataFrame with parameters or list
            of thereof. The values from other are used to update the value column
            of ``params``. If other is a list, the updates will be in order, i.e.
            later elements overwrite earlier ones.

    Returns:
        pandas.DataFrame: Updated copy of params.

    """
    if isinstance(others, pd.DataFrame):
        others = [others]

    out = params.copy(deep=True)

    # Create a temporary Series to hold the updated values
    temp_series = out["value"].copy()

    # Update the temporary Series
    for other in others:
        temp_series.update(other["value"])

    # Assign the updated Series back to the DataFrame
    out["value"] = temp_series

    return out


def remove_factors(
    factors: str | list[str],
    model_dict: dict[str, Any],
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Remove factors from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    It is possible that the reduced model has fewer periods than the original one.
    This happens if the remaining factors do not have measurements in later periods.

    Args:
        factors: Name(s) of the factor(s) to remove.
        model_dict: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    # We need this for the full model when endogenous factors are present.
    has_endogenous_factors = get_has_endogenous_factors(model_dict["factors"])

    out = deepcopy(model_dict)

    out["factors"] = _remove_from_dict(out["factors"], factors)

    # adjust anchoring
    if "anchoring" in model_dict:
        out["anchoring"]["outcomes"] = _remove_from_dict(
            out["anchoring"]["outcomes"],
            factors,
        )
        if out["anchoring"]["outcomes"] == {}:
            out = _remove_from_dict(out, "anchoring")

    # Remove periods if necessary, but only if no endogenous factors are present.
    # (else we would mess up the mapping between raw periods model periods)
    if not has_endogenous_factors:
        new_n_periods = get_dimensions(out, has_endogenous_factors).n_periods
        out = reduce_n_periods(out, new_n_periods)

    if params is not None:
        out_params = _reduce_params(params, out, has_endogenous_factors)  # ty: ignore[invalid-argument-type]
        out = (out, out_params)

    return out  # ty: ignore[invalid-return-type]


def remove_measurements(
    measurements: str | list[str],
    model_dict: dict[str, Any],
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Remove measurements from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        measurements: Name(s) of the measurement(s) to remove.
        model_dict: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

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
            out["factors"][factor]["normalizations"]["loadings"] = (
                _remove_measurements_from_normalizations(
                    measurements,
                    norminfo["loadings"],
                )
            )

        if "intercepts" in norminfo:
            out["factors"][factor]["normalizations"]["intercepts"] = (
                _remove_measurements_from_normalizations(
                    measurements,
                    norminfo["intercepts"],
                )
            )

    if params is not None:
        # This likely won't work if we have endogenous factors.
        out_params = _reduce_params(params, out, has_endogenous_factors=False)
        out = (out, out_params)

    return out


def remove_controls(
    controls: str | list[str],
    model_dict: dict[str, Any],
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Remove control variables from a model specification.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        controls: Name(s) of the contral variable(s) to remove.
        model_dict: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    out["controls"] = _remove_from_list(out["controls"], controls)
    if out["controls"] == []:
        out = _remove_from_dict(out, "controls")

    if params is not None:
        # This likely won't work if we have endogenous factors.
        out_params = _reduce_params(params, out, has_endogenous_factors=False)
        out = (out, out_params)

    return out


def switch_translog_to_linear(
    model_dict: dict[str, Any],
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Switch all translog production functions to linear.

    If provided, a params DataFrame is also reduced correspondingly.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    for factor in model_dict["factors"]:
        if model_dict["factors"][factor]["transition_function"] == "translog":
            out["factors"][factor]["transition_function"] = "linear"

    if params is not None:
        # This likely won't work if we have endogenous factors.
        out_params = _reduce_params(params, out, has_endogenous_factors=False)
        out = (out, out_params)

    return out


def switch_linear_to_translog(
    model_dict: dict[str, Any],
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Switch all linear production functions to translog.

    If provided, a params DataFrame is also extended correspondingly. The fill value
    for the additional terms is 0.05 because experience showed that estimating a
    translog model with start parameters obtained from a linear model is faster when
    the additional parameters are not initialized at zero.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`.
        params: The params DataFrame for the full model.

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


def reduce_n_periods(
    model_dict: dict[str, Any],
    new_n_periods: int,
    params: pd.DataFrame | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Remove all periods after n_periods.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`.
        new_n_periods: The new number of periods.
        params: The params DataFrame for the full model.

    Returns:
        dict: The reduced model dictionary
        pandas.DataFrame: The reduced parameter DataFrame (only if params is not None)

    """
    out = deepcopy(model_dict)
    for factor in model_dict["factors"]:
        out["factors"][factor]["measurements"] = _shorten_if_necessary(
            out["factors"][factor]["measurements"],
            new_n_periods,
        )

        norminfo = model_dict["factors"][factor].get("normalizations", {})
        if "loadings" in norminfo:
            out["factors"][factor]["normalizations"]["loadings"] = (
                _shorten_if_necessary(norminfo["loadings"], new_n_periods)
            )

        if "intercepts" in norminfo:
            out["factors"][factor]["normalizations"]["intercepts"] = (
                _shorten_if_necessary(norminfo["intercepts"], new_n_periods)
            )

    if "stagemap" in out:
        out["stagemap"] = _shorten_if_necessary(out["stagemap"], new_n_periods - 1)

    if params is not None:
        out_params = _extend_params(params, out, 0.05)
        out = (out, out_params)

    return out


def _remove_from_list(
    list_: list[Any],
    to_remove: str | list[str],
) -> list[Any]:
    if isinstance(to_remove, str):
        to_remove = [to_remove]
    return [element for element in list_ if element not in to_remove]


def _remove_from_dict(
    dict_: dict[str, Any],
    to_remove: str | list[str],
) -> dict[str, Any]:
    if isinstance(to_remove, str):
        to_remove = [to_remove]

    return {key: val for key, val in dict_.items() if key not in to_remove}


def _reduce_params(
    params: pd.DataFrame,
    model_dict: dict[str, Any],
    has_endogenous_factors: bool,
) -> pd.DataFrame:
    """Reduce a parameter DataFrame from a larger model to a reduced model.

    The reduced model must be nested in the original model for which the params
    DataFrame was constructed.

    Args:
        params: The params DataFrame for the full model.
        model_dict: The model specification. See: :ref:`model_specs`.
        has_endogenous_factors: Whether the model has endogenous factors.

    Returns:
        pandas.DataFrame: The reduced parameters DataFrame.

    """
    index = _get_params_index_from_model_dict(model_dict)
    # If we have endogenous factors, we need to keep the periods from params.
    if has_endogenous_factors:
        df = pd.merge(
            left=params.reset_index(),
            right=index.to_frame(index=False)[
                ["category", "name1", "name2"]
            ].drop_duplicates(),
            on=["category", "name1", "name2"],
            how="right",
        )
        index = pd.MultiIndex.from_frame(df[params.index.names])
    return params.loc[index]


def _extend_params(
    params: pd.DataFrame,
    model_dict: dict[str, Any],
    fill_value: float,
) -> pd.DataFrame:
    index = _get_params_index_from_model_dict(model_dict)
    out = params.reindex(index)
    out["value"] = out["value"].fillna(fill_value)
    if "lower_bound" in out:
        out["lower_bound"] = out["lower_bound"].fillna(-np.inf)

    if "upper_bound" in out:
        out["upper_bound"] = out["upper_bound"].fillna(np.inf)

    return out


def _get_params_index_from_model_dict(
    model_dict: dict[str, Any],
) -> pd.MultiIndex:
    mod = process_model(model_dict)
    index = get_params_index(
        update_info=mod.update_info,
        labels=mod.labels,
        dimensions=mod.dimensions,
        transition_info=mod.transition_info,
        endogenous_factors_info=mod.endogenous_factors_info,
    )
    return index


def _remove_measurements_from_normalizations(
    measurements: str | list[str],
    normalizations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reduced = [_remove_from_dict(norm, measurements) for norm in normalizations]
    if reduced != normalizations:
        warnings.warn(
            "Your removed a normalized measurement from a model. Make sure there are "
            "enough normalizations left to ensure identification.",
        )
    return reduced


def _shorten_if_necessary(
    list_: list[Any],
    length: int,
) -> list[Any]:
    if len(list_) > length:
        list_ = list_[:length]
    return list_
