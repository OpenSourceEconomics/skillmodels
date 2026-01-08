from copy import deepcopy
from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd
from dags import concatenate_functions
from dags.signature import rename_arguments
from jax import vmap
from pandas import DataFrame

import skillmodels.transition_functions as t_f_module
from skillmodels.check_model import check_model, check_stagemap
from skillmodels.decorators import extract_params, jax_array_output
from skillmodels.types import (
    Anchoring,
    Dimensions,
    EndogenousFactorsInfo,
    EstimationOptions,
    FactorEndogenousInfo,
    Labels,
    TransitionInfo,
)

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
    has_endogenous_factors = get_has_endogenous_factors(model_dict["factors"])
    dims = get_dimensions(
        model_dict=model_dict, has_endogenous_factors=has_endogenous_factors
    )
    labels = _get_labels(
        model_dict=model_dict,
        has_endogenous_factors=has_endogenous_factors,
        dimensions=dims,
    )
    anchoring = _process_anchoring(model_dict)
    if has_endogenous_factors:
        _model_dict_aug = _augment_periods_for_endogenous_factors(
            model_dict=model_dict,
            dimensions=dims,
            labels=labels,
        )
        endogenous_factors_info = _get_endogenous_factors_info(
            has_endogenous_factors=has_endogenous_factors,
            model_dict=_model_dict_aug,
            labels=labels,
            bounds_distance=model_dict["estimation_options"]["bounds_distance"],
        )
    else:
        _model_dict_aug = model_dict
        endogenous_factors_info = EndogenousFactorsInfo(
            has_endogenous_factors=has_endogenous_factors,
            aug_periods_to_aug_period_meas_types=_get_aug_periods_to_aug_period_meas_types(
                aug_periods=labels.aug_periods_to_periods.keys(),
                has_endogenous_factors=has_endogenous_factors,
            ),
            bounds_distance=model_dict["estimation_options"].get(
                "bounds_distance", 1e-3
            ),
            aug_periods_from_period=partial(
                _aug_periods_from_period,
                aug_periods_to_periods=labels.aug_periods_to_periods,
            ),
            factor_info={
                fac: FactorEndogenousInfo(
                    is_state=True, is_endogenous=False, is_correction=False
                )
                for fac in labels.latent_factors
            },
        )
    check_model(
        model_dict=_model_dict_aug,
        labels=labels,
        dimensions=dims,
        anchoring=anchoring,
        has_endogenous_factors=has_endogenous_factors,
    )
    transition_info = _get_transition_info(_model_dict_aug, labels)
    # Create new Labels with transition_names (frozen dataclass requires replacement)
    labels = Labels(
        latent_factors=labels.latent_factors,
        observed_factors=labels.observed_factors,
        controls=labels.controls,
        periods=labels.periods,
        stagemap=labels.stagemap,
        stages=labels.stages,
        aug_periods=labels.aug_periods,
        aug_periods_to_periods=labels.aug_periods_to_periods,
        aug_stagemap=labels.aug_stagemap,
        aug_stages=labels.aug_stages,
        aug_stages_to_stages=labels.aug_stages_to_stages,
        transition_names=tuple(transition_info.function_names.values()),
    )

    processed = {
        "dimensions": dims,
        "labels": labels,
        "anchoring": anchoring,
        "estimation_options": _process_estimation_options(_model_dict_aug),
        "transition_info": transition_info,
        "update_info": _get_update_info(_model_dict_aug, dims, labels, anchoring),
        "normalizations": _process_normalizations(_model_dict_aug, dims, labels),
        "endogenous_factors_info": endogenous_factors_info,
    }
    return processed


def get_has_endogenous_factors(factors: dict[str, Any]) -> bool:
    """Return True if any endogenous factors are present."""
    endogenous_factors = pd.DataFrame(
        [
            {
                "factor": f,
                "is_endogenous": v.get("is_endogenous", False),
                "is_correction": v.get("is_correction", False),
            }
            for f, v in factors.items()
        ]
    ).set_index("factor")
    if (endogenous_factors.dtypes != bool).any():  # noqa: E721
        raise ValueError(
            "If specified, 'is_endogenous' and 'is_correction' both need to be of type"
            f"'bool', got:\n{endogenous_factors}"
        )
    if (
        ~endogenous_factors["is_endogenous"] & endogenous_factors["is_correction"]
    ).any():
        raise ValueError(
            "A factor cannot be a correction and not endogenous, got:\n"
            f"{endogenous_factors}"
        )
    return endogenous_factors["is_endogenous"].any()  # ty: ignore[invalid-return-type]


def get_dimensions(model_dict: dict, has_endogenous_factors: bool) -> Dimensions:
    """Extract the dimensions of the model.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        has_endogenous_factors: Whether endogenous factors are present.

    Returns:
        Dimensions dataclass with all dimensional information.

    """
    all_n_periods = [len(d["measurements"]) for d in model_dict["factors"].values()]
    n_periods = max(all_n_periods)
    n_aug_periods = 2 * n_periods if has_endogenous_factors else n_periods

    return Dimensions(
        n_latent_factors=len(model_dict["factors"]),
        n_observed_factors=len(model_dict.get("observed_factors", [])),
        n_controls=len(model_dict.get("controls", [])) + 1,  # plus 1: constant
        n_mixtures=model_dict["estimation_options"].get("n_mixtures", 1),
        n_aug_periods=n_aug_periods,
        n_periods=n_periods,
    )


def _get_aug_periods_to_periods(
    n_aug_periods: int, has_endogenous_factors: bool
) -> dict[int, int]:
    """Return mapper of (potentially) augmented periods to user-provided periods."""
    aug_periods = list(range(n_aug_periods))
    return (
        {p: p // 2 for p in aug_periods}
        if has_endogenous_factors
        else {p: p for p in aug_periods}
    )


def _aug_periods_from_period(
    period: int, aug_periods_to_periods: dict[int, int]
) -> list[int]:
    """The inverse of the the aug_periods_to_periods mapper."""
    return [ap for ap, p in aug_periods_to_periods.items() if p == period]


def _get_labels(
    model_dict: dict, has_endogenous_factors: bool, dimensions: Dimensions
) -> Labels:
    """Extract labels of the model quantities.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        has_endogenous_factors: Whether endogenous factors are present.
        dimensions: Dimensional information.

    Returns:
        Labels dataclass with all label information.

    """
    aug_periods_to_periods = _get_aug_periods_to_periods(
        n_aug_periods=dimensions.n_aug_periods,
        has_endogenous_factors=has_endogenous_factors,
    )

    stagemap = model_dict.get("stagemap", list(range(dimensions.n_periods - 1)))
    stages = sorted(int(v) for v in np.unique(stagemap))

    report = check_stagemap(
        stagemap=stagemap,
        stages=stages,
        n_periods=dimensions.n_periods,
        is_augmented=False,
    )
    if report:
        raise ValueError(f"Invalid stage map: {report}")
    if has_endogenous_factors:
        aug_stagemap: list[int] = []
        aug_stages_to_stages: dict[int, int] = {}
        relevant_aug_periods = sorted(aug_periods_to_periods.keys())[:-2]
        for aug_p in relevant_aug_periods:
            p = aug_periods_to_periods[aug_p]
            s = stagemap[p]
            aug_s = 2 * s + aug_p % 2
            aug_stagemap.append(aug_s)
            aug_stages_to_stages[aug_s] = s
    else:
        aug_stagemap = list(stagemap)
        aug_stages_to_stages = {s: s for s in stages}

    return Labels(
        latent_factors=tuple(model_dict["factors"]),
        observed_factors=tuple(model_dict.get("observed_factors", [])),
        controls=("constant", *model_dict.get("controls", [])),
        periods=tuple(sorted(set(aug_periods_to_periods.values()))),
        stagemap=tuple(stagemap),
        stages=tuple(stages),
        aug_periods=tuple(aug_periods_to_periods.keys()),
        aug_periods_to_periods=aug_periods_to_periods,
        aug_stagemap=tuple(aug_stagemap),
        aug_stages=tuple(sorted(int(v) for v in np.unique(aug_stagemap))),
        aug_stages_to_stages=aug_stages_to_stages,
    )


def _process_estimation_options(model_dict: dict) -> EstimationOptions:
    """Process options.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`

    Returns:
        EstimationOptions dataclass with tuning parameters for the estimation.

    """
    user_opts = model_dict.get("estimation_options", {})

    sigma_points_scale = user_opts.get("sigma_points_scale", 2)
    robust_bounds = user_opts.get("robust_bounds", True)
    bounds_distance = user_opts.get("bounds_distance", 1e-3)
    clipping_lower_bound = user_opts.get("clipping_lower_bound", -1e30)
    clipping_upper_bound = user_opts.get("clipping_upper_bound", None)
    clipping_lower_hardness = user_opts.get("clipping_lower_hardness", 1)
    clipping_upper_hardness = user_opts.get("clipping_upper_hardness", 1)

    if not robust_bounds:
        bounds_distance = 0

    return EstimationOptions(
        sigma_points_scale=sigma_points_scale,
        robust_bounds=robust_bounds,
        bounds_distance=bounds_distance,
        clipping_lower_bound=clipping_lower_bound,
        clipping_upper_bound=clipping_upper_bound,
        clipping_lower_hardness=clipping_lower_hardness,
        clipping_upper_hardness=clipping_upper_hardness,
    )


def _process_anchoring(model_dict: dict) -> Anchoring:
    """Process the specification that governs how latent factors are anchored.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`

    Returns:
        Anchoring dataclass with information about anchoring.

    """
    if "anchoring" in model_dict:
        anch = model_dict["anchoring"]
        return Anchoring(
            anchoring=True,
            outcomes=anch.get("outcomes", {}),
            factors=tuple(anch.get("outcomes", {}).keys()),
            free_controls=anch.get("free_controls", False),
            free_constant=anch.get("free_constant", False),
            free_loadings=anch.get("free_loadings", False),
            ignore_constant_when_anchoring=anch.get(
                "ignore_constant_when_anchoring", False
            ),
        )

    return Anchoring(
        anchoring=False,
        outcomes={},
        factors=(),
        free_controls=False,
        free_constant=False,
        free_loadings=False,
        ignore_constant_when_anchoring=False,
    )


def _insert_empty_elements_into_list(old, insert_at_modulo, to_insert, aug_p_to_p):
    return [
        to_insert if aug_p % 2 == insert_at_modulo else old[p]
        for aug_p, p in aug_p_to_p.items()
    ]


def _augment_periods_for_endogenous_factors(
    model_dict: dict[str, Any], dimensions: Dimensions, labels: Labels
) -> dict[str, Any]:
    """Augment periods if endogenous factors are present.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        dimensions: Dimensional information.
        labels: Labels for model quantities.

    Returns:
        Model dictionary with twice the amount of periods

    """
    aug = deepcopy(model_dict)
    for fac, v in model_dict["factors"].items():
        insert_at_modulo = 0 if v.get("is_endogenous", False) else 1

        # Insert empty elements into measurements when we do not have those.
        if len(v["measurements"]) != dimensions.n_periods:
            raise ValueError(
                "Measurements must be of length `n_periods`, "
                f"got {v['measurements']} for {fac}"
            )
        aug["factors"][fac]["measurements"] = _insert_empty_elements_into_list(
            old=v["measurements"],
            insert_at_modulo=insert_at_modulo,
            to_insert=[],
            aug_p_to_p=labels.aug_periods_to_periods,
        )

        # Insert empty elements into normalizations when we do not have those.
        for norm_type, normalizations in v.get("normalizations", {}).items():
            if not len(normalizations) == dimensions.n_periods:
                raise ValueError(
                    "Normalizations must be lists of length `n_periods`, "
                    f"got {normalizations} for {fac}['normalizations']['{norm_type}']"
                )
            aug["factors"][fac]["normalizations"][norm_type] = (
                _insert_empty_elements_into_list(
                    old=normalizations,
                    insert_at_modulo=insert_at_modulo,
                    to_insert={},
                    aug_p_to_p=labels.aug_periods_to_periods,
                )
            )
    return aug


def _get_transition_info(model_dict: dict, labels: Labels) -> TransitionInfo:
    """Collect information about transition functions."""
    func_list, param_names = [], []
    latent_factors = labels.latent_factors
    all_factors = labels.all_factors

    for factor in latent_factors:
        spec = model_dict["factors"][factor]["transition_function"]
        if isinstance(spec, str):
            func = getattr(t_f_module, spec)
            if spec == "constant":
                func = rename_arguments(func, mapper={"state": factor})
            func_list.append(extract_params(func, key=factor))
            param_names.append(getattr(t_f_module, f"params_{spec}")(all_factors))
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

    for i, factor in enumerate(labels.all_factors):
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

    return TransitionInfo(
        func=transition_function,
        param_names=dict(zip(latent_factors, param_names, strict=False)),
        individual_functions=individual_functions,
        function_names=dict(zip(latent_factors, function_names, strict=False)),
    )


def _get_endogenous_factors_info(
    has_endogenous_factors: bool,
    model_dict: dict[str, Any],
    labels: Labels,
    bounds_distance: float,
) -> EndogenousFactorsInfo:
    """Collect information about endogenous factors."""
    factor_info = {}
    for fac, v in model_dict["factors"].items():
        factor_info[fac] = FactorEndogenousInfo(
            is_state=(
                not v.get("is_endogenous", False) and not v.get("is_correction", False)
            ),
            is_endogenous=v.get("is_endogenous", False),
            is_correction=v.get("is_correction", False),
        )

    return EndogenousFactorsInfo(
        has_endogenous_factors=has_endogenous_factors,
        aug_periods_to_aug_period_meas_types=_get_aug_periods_to_aug_period_meas_types(
            aug_periods=labels.aug_periods_to_periods.keys(),
            has_endogenous_factors=has_endogenous_factors,
        ),
        bounds_distance=bounds_distance,
        aug_periods_from_period=partial(
            _aug_periods_from_period,
            aug_periods_to_periods=labels.aug_periods_to_periods,
        ),
        factor_info=factor_info,
    )


def _get_aug_periods_to_aug_period_meas_types(
    aug_periods, has_endogenous_factors: bool
) -> dict[int, Literal["states", "endogenous_factors"]]:
    if has_endogenous_factors:
        return {
            aug_p: ("states" if aug_p % 2 == 0 else "endogenous_factors")
            for aug_p in aug_periods
        }
    return dict.fromkeys(aug_periods, "states")


def _get_update_info(
    model_dict: dict, dimensions: Dimensions, labels: Labels, anchoring_info: Anchoring
) -> DataFrame:
    """Construct a DataFrame with information on each Kalman update.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        dimensions: Dimensional information.
        labels: Labels for model quantities.
        anchoring_info: Information about anchoring. See :ref:`anchoring`

    Returns:
        DataFrame with one row per Kalman update needed in the likelihood function.

    """
    index = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["aug_period", "variable"]
    )
    uinfo = DataFrame(index=index, columns=[*labels.latent_factors, "purpose"])

    measurements = {}
    for factor in labels.latent_factors:
        measurements[factor] = model_dict["factors"][factor]["measurements"]
        if len(measurements[factor]) != dimensions.n_aug_periods:
            raise ValueError(
                "Measurements must be of length `n_aug_periods`, "
                f"got {measurements[factor]} for {factor}"
            )

    for aug_period in labels.aug_periods:
        for factor in labels.latent_factors:
            for meas in measurements[factor][aug_period]:
                uinfo.loc[(aug_period, meas), factor] = True
                uinfo.loc[(aug_period, meas), "purpose"] = "measurement"
        for factor in anchoring_info.factors:
            outcome = anchoring_info.outcomes[factor]
            name = f"{outcome}_{factor}"
            uinfo.loc[(aug_period, name), factor] = True
            uinfo.loc[(aug_period, name), "purpose"] = "anchoring"

    for col in [c for c in uinfo.columns if c != "purpose"]:
        uinfo[col] = uinfo[col].fillna(value=False).astype(bool)
    return uinfo


def _process_normalizations(
    model_dict: dict, dimensions: Dimensions, labels: Labels
) -> dict[str, dict[str, list]]:
    """Process the normalizations of intercepts and factor loadings.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        dimensions: Dimensional information.
        labels: Labels for model quantities.

    Returns:
        Nested dictionary with information on normalized factor loadings and
        intercepts for each factor.

    """
    normalizations = {}
    for factor in labels.latent_factors:
        normalizations[factor] = {}
        norminfo = model_dict["factors"][factor].get("normalizations", {})
        for norm_type in ["loadings", "intercepts"]:
            candidate = norminfo.get(
                norm_type, [{} for _ in range(dimensions.n_aug_periods)]
            )
            if not len(candidate) == dimensions.n_aug_periods:
                raise ValueError(
                    "Normalizations must be of length `n_aug_periods`, "
                    f"got {norminfo} for {factor}['{norm_type}']"
                )
            normalizations[factor][norm_type] = candidate

    return normalizations
