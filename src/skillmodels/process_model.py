from copy import deepcopy
from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd
from dags import concatenate_functions
from dags.signature import rename_arguments
from jax import vmap
from pandas import DataFrame

import skillmodels.transition_functions as tf
from skillmodels.check_model import check_model
from skillmodels.decorators import extract_params, jax_array_output

pd.set_option("future.no_silent_downcasting", True)  # noqa:  FBT003


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
        - transition_info (dict): Everything related to transition functions.
        - update_info (pandas.DataFrame): DataFrame with one row per Kalman update
        needed in the likelihood function. See :ref:`update_info`.
        - normalizations (dict): Nested dictionary with information on normalized factor
        loadings and intercepts for each factor. See :ref:`normalizations`.

    """
    has_investments = get_has_investments(model_dict["factors"])
    dims = get_dimensions(model_dict, has_investments)
    labels = _get_labels(
        model_dict=model_dict, has_investments=has_investments, dimensions=dims
    )
    anchoring = _process_anchoring(model_dict)
    check_model(model_dict, labels, dims, anchoring)
    if has_investments:
        model_dict_aug = _augment_periods_for_investments(model_dict, labels)
    else:
        model_dict_aug = model_dict
    transition_info = _get_transition_info(model_dict_aug, labels)
    labels["transition_names"] = list(transition_info["function_names"].values())

    processed = {
        "has_investments": has_investments,
        "dimensions": dims,
        "labels": labels,
        "anchoring": anchoring,
        "estimation_options": _process_estimation_options(model_dict),
        "transition_info": transition_info,
        "update_info": _get_update_info(model_dict, dims, labels, anchoring),
        "normalizations": _process_normalizations(model_dict, dims, labels),
    }
    return processed


def get_has_investments(factors: dict[str, Any]) -> bool:
    """Return True if any investment factors are present."""
    investments = pd.DataFrame(
        [
            {
                "factor": f,
                "is_investment": v.get("is_investment", False),
                "is_correction": v.get("is_correction", False),
            }
            for f, v in factors.items()
        ]
    ).set_index("factor")
    if (investments.dtypes != bool).any():  # noqa: E721
        raise ValueError(
            "If specified, 'is_investment' and 'is_correction' both need to be of type"
            f"'bool', got:\n{investments}"
        )
    if (~investments["is_investment"] & investments["is_correction"]).any():
        raise ValueError(
            "A factor cannot be a correction and not an investment, got:\n"
            f"{investments}"
        )
    return investments["is_investment"].any()


def get_dimensions(model_dict, has_investments):
    """Extract the dimensions of the model.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        has_investments (bool): Whether investment factors are present.

    Returns:
        dict: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.

    """
    all_n_periods = [len(d["measurements"]) for d in model_dict["factors"].values()]
    n_periods_raw = max(all_n_periods)
    n_periods = 2 * n_periods_raw if has_investments else n_periods_raw

    dims = {
        "n_latent_factors": len(model_dict["factors"]),
        "n_observed_factors": len(model_dict.get("observed_factors", [])),
        "n_periods": n_periods,
        "n_periods_raw": n_periods_raw,
        "n_controls": len(model_dict.get("controls", [])) + 1,  # plus 1: constant
        "n_mixtures": model_dict["estimation_options"].get("n_mixtures", 1),
    }
    dims["n_all_factors"] = dims["n_latent_factors"] + dims["n_observed_factors"]
    return dims


def _get_periods_to_periods_raw(
    dimensions: dict[str, int], has_investments: bool
) -> dict[int, int]:
    """Return mapper of periods potentially augmented for investments to raw periods."""
    periods = list(range(dimensions["n_periods"]))
    return {p: p // 2 for p in periods} if has_investments else {p: p for p in periods}


def _get_periods_to_period_types(
    periods: list[int], has_investments: bool
) -> dict[int, Literal["states", "investments"]]:
    return {
        p: ("states" if p % 2 == 0 else "investments")
        if has_investments
        else {p: "states"}
        for p in periods
    }


def _get_labels(model_dict, has_investments, dimensions):
    """Extract labels of the model quantities.

    Args:
        model_dict (dict): The model specification. See: :ref:`model_specs`
        has_investments (bool): Whether investment factors are present.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.

    Returns:
        dict: Dict of lists with labels for the model quantities like
        factors, periods, controls, stagemap and stages. See :ref:`labels`

    """
    if has_investments and "stagemap" in model_dict:
        raise ValueError("Stages currently not supported when investments are present.")

    stagemap = model_dict.get("stagemap", list(range(dimensions["n_periods"] - 1)))

    periods_to_periods_raw = _get_periods_to_periods_raw(dimensions, has_investments)
    periods_to_period_types = _get_periods_to_period_types(
        list(periods_to_periods_raw.keys()), has_investments
    )

    labels = {
        "latent_factors": list(model_dict["factors"]),
        "observed_factors": list(model_dict.get("observed_factors", [])),
        "controls": ["constant", *list(model_dict.get("controls", []))],
        "periods": list(periods_to_periods_raw.keys()),
        "periods_raw": sorted(set(periods_to_periods_raw.values())),
        "periods_to_periods_raw": periods_to_periods_raw,
        "periods_to_period_types": periods_to_period_types,
        "stagemap": stagemap,
        "stages": sorted(np.unique(stagemap)),
    }

    labels["all_factors"] = labels["latent_factors"] + labels["observed_factors"]

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
        anchinfo["factors"] = list(anchinfo["outcomes"])

    return anchinfo


def _insert_empty_elements_into_list(old, insert_at_modulo, to_insert, p_to_p_raw):
    return [
        to_insert if p % 2 == insert_at_modulo else old[p_raw]
        for p, p_raw in p_to_p_raw.items()
    ]


def _augment_periods_for_investments(
    model_dict: dict[str, Any], labels: dict[str, Any]
) -> dict[str, Any]:
    """Insert periods without measurements / normalisations if investments are present.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        labels: Dictionary with information about periods etc.

    Returns:
        Model dictionary with twice the amount of periods

    """
    aug = deepcopy(model_dict)
    for fac, v in model_dict["factors"].items():
        insert_at_modulo = 0 if v.get("is_investment", False) else 1
        aug["factors"][fac]["measurements"] = _insert_empty_elements_into_list(
            old=v["measurements"],
            insert_at_modulo=insert_at_modulo,
            to_insert=[],
            p_to_p_raw=labels["periods_to_periods_raw"],
        )


def _get_transition_info(model_dict, labels):
    """Collect information about transition functions."""
    func_list, param_names = [], []
    latent_factors = labels["latent_factors"]
    all_factors = labels["all_factors"]

    for factor in latent_factors:
        spec = model_dict["factors"][factor]["transition_function"]
        if isinstance(spec, str):
            func = getattr(tf, spec)
            if spec == "constant":
                func = rename_arguments(func, mapper={"state": factor})
            func_list.append(extract_params(func, key=factor))
            param_names.append(getattr(tf, f"params_{spec}")(all_factors))
        elif callable(spec):
            if not hasattr(spec, "__name__"):
                raise AttributeError(
                    "Custom transition functions must have a __name__ attribute.",
                )
            if hasattr(spec, "__registered_params__"):
                names = spec.__registered_params__
                param_names.append(names)
            else:
                raise AttributeError(
                    "Custom transition_functions must have a __registered_params__ "
                    "attribute. You can set it via the register_params decorator.",
                )
            func_list.append(extract_params(spec, key=factor, names=names))

    function_names = [f.__name__ for f in func_list]

    functions = {
        f"__next_{fac}__": func
        for fac, func in zip(latent_factors, func_list, strict=False)
    }

    # add functions to produce the individual factors out of the 1d states vector.
    # The dag will automatically sort out what we don't need.
    def _extract_factor(states, pos):
        return states[pos]

    for i, factor in enumerate(labels["all_factors"]):
        functions[factor] = partial(_extract_factor, pos=i)

    transition_function = concatenate_functions(
        functions=functions,
        targets=[f"__next_{fac}__" for fac in latent_factors],
    )

    transition_function = jax_array_output(transition_function)

    transition_function = vmap(transition_function, in_axes=(None, 0))

    individual_functions = {}
    for factor in latent_factors:
        func = concatenate_functions(functions=functions, targets=f"__next_{factor}__")
        func = vmap(func, in_axes=(None, 0))
        individual_functions[factor] = func

    out = {
        "func": transition_function,
        "param_names": dict(zip(latent_factors, param_names, strict=False)),
        "individual_functions": individual_functions,
        "function_names": dict(zip(latent_factors, function_names, strict=False)),
    }
    return out


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
    uinfo = DataFrame(index=index, columns=labels["latent_factors"] + ["purpose"])

    measurements = {}
    for factor in labels["latent_factors"]:
        measurements[factor] = fill_list(
            model_dict["factors"][factor]["measurements"],
            [],
            dimensions["n_periods"],
        )

    for period in labels["periods"]:
        for factor in labels["latent_factors"]:
            for meas in measurements[factor][period]:
                uinfo.loc[(period, meas), factor] = True
                uinfo.loc[(period, meas), "purpose"] = "measurement"
        for factor in anchoring_info["factors"]:
            outcome = anchoring_info["outcomes"][factor]
            name = f"{outcome}_{factor}"
            uinfo.loc[(period, name), factor] = True
            uinfo.loc[(period, name), "purpose"] = "anchoring"

    for col in [c for c in uinfo.columns if c != "purpose"]:
        uinfo[col] = uinfo[col].fillna(value=False).astype(bool)
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
    for factor in labels["latent_factors"]:
        normalizations[factor] = {}
        norminfo = model_dict["factors"][factor].get("normalizations", {})
        for norm_type in ["loadings", "intercepts"]:
            candidate = norminfo.get(norm_type, [])
            candidate = fill_list(candidate, {}, dimensions["n_periods"])
            normalizations[factor][norm_type] = candidate

    return normalizations


def fill_list(short_list, fill_value, length):
    """Extend a list to specified length by filling it with the fill_value.

    Examples:
    >>> fill_list(["a"], "b", 3)
    ['a', 'b', 'b']

    """
    res = list(short_list)
    diff = length - len(short_list)
    assert diff >= 0, "short_list has to be shorter than length."
    if diff >= 1:
        res += [fill_value] * diff
    return res


def get_period_measurements(update_info, period):
    if period in update_info.index:
        measurements = list(update_info.loc[period].index)
    else:
        measurements = []
    return measurements
