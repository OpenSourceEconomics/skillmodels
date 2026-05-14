"""Functions to process model specifications from user-friendly to internal form."""

from collections.abc import KeysView, Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType

import numpy as np
import pandas as pd
from dags import concatenate_functions
from dags.signature import rename_arguments
from jax import Array, vmap
from pandas import DataFrame

import skillmodels.common.transition_functions as t_f_module
from skillmodels.common.check_model import check_model, check_stagemap
from skillmodels.common.decorators import extract_params, jax_array_output
from skillmodels.common.model_spec import FactorSpec, ModelSpec
from skillmodels.common.types import (
    Anchoring,
    Dimensions,
    EndogenousFactorsInfo,
    FactorInfo,
    Labels,
    MeasurementType,
    Normalizations,
    ProcessedModel,
    TransitionInfo,
)


def process_model(model_spec: ModelSpec) -> ProcessedModel:
    """Check, clean, extend and transform the model specs.

    Check the completeness, consistency and validity of the model specifications.

    Set default values and extend the model specification where necessary.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`

    Returns:
        ProcessedModel with the following entries:
        - dimensions: Dimensional information like n_states, n_periods,
          n_controls, n_mixtures. See :ref:`dimensions`.
        - labels: Dict of lists with labels for the model quantities like
          factors, periods, controls, stagemap and stages. See :ref:`labels`
        - anchoring: Information about anchoring. See :ref:`anchoring`
        - transition_info: Everything related to transition functions.
        - update_info: DataFrame with one row per Kalman update
          needed in the likelihood function. See :ref:`update_info`.
        - normalizations: Nested dictionary with information on normalized factor
          loadings and intercepts for each factor. See :ref:`normalizations`.

    """
    has_endogenous_factors = get_has_endogenous_factors(model_spec.factors)
    dims = get_dimensions(
        model_spec=model_spec, has_endogenous_factors=has_endogenous_factors
    )
    labels = _get_labels(
        model_spec=model_spec,
        has_endogenous_factors=has_endogenous_factors,
        dimensions=dims,
    )
    anchoring = _process_anchoring(model_spec)
    if has_endogenous_factors:
        _model_spec_aug = _augment_periods_for_endogenous_factors(
            model_spec=model_spec,
            dimensions=dims,
            labels=labels,
        )
    else:
        _model_spec_aug = model_spec
    endogenous_factors_info = _get_endogenous_factors_info(
        has_endogenous_factors=has_endogenous_factors,
        model_spec=_model_spec_aug,
        labels=labels,
    )
    check_model(
        model_spec=_model_spec_aug,
        labels=labels,
        dimensions=dims,
        anchoring=anchoring,
        has_endogenous_factors=has_endogenous_factors,
    )
    transition_info = _get_transition_info(model_spec=_model_spec_aug, labels=labels)
    labels = replace(
        labels, transition_names=tuple(transition_info.function_names.values())
    )

    return ProcessedModel(
        dimensions=dims,
        labels=labels,
        anchoring=anchoring,
        transition_info=transition_info,
        update_info=_get_update_info(
            model_spec=_model_spec_aug,
            dimensions=dims,
            labels=labels,
            anchoring_info=anchoring,
        ),
        normalizations=_process_normalizations(
            model_spec=_model_spec_aug, dimensions=dims, labels=labels
        ),
        endogenous_factors_info=endogenous_factors_info,
    )


def get_has_endogenous_factors(factors: Mapping[str, FactorSpec]) -> bool:
    """Return True if any endogenous factors are present."""
    endogenous_factors = pd.DataFrame(
        [
            {
                "factor": f,
                "is_endogenous": v.is_endogenous,
                "is_correction": v.is_correction,
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


def get_dimensions(
    model_spec: ModelSpec, *, has_endogenous_factors: bool
) -> Dimensions:
    """Extract the dimensions of the model.

    Args:
        model_spec: The model specification.
        has_endogenous_factors: Whether endogenous factors are present.

    Returns:
        Dimensions dataclass with all dimensional information.

    """
    all_n_periods = [len(fspec.measurements) for fspec in model_spec.factors.values()]
    n_periods = max(all_n_periods)
    n_aug_periods = 2 * n_periods if has_endogenous_factors else n_periods

    return Dimensions(
        n_latent_factors=len(model_spec.factors),
        n_observed_factors=len(model_spec.observed_factors),
        n_controls=len(model_spec.controls) + 1,  # plus 1: constant
        n_mixtures=model_spec.n_mixtures,
        n_aug_periods=n_aug_periods,
        n_periods=n_periods,
    )


def _get_aug_periods_to_periods(
    n_aug_periods: int, *, has_endogenous_factors: bool
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
    model_spec: ModelSpec, *, has_endogenous_factors: bool, dimensions: Dimensions
) -> Labels:
    """Extract labels of the model quantities.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
        has_endogenous_factors: Whether endogenous factors are present.
        dimensions: Dimensional information.

    Returns:
        Labels dataclass with all label information.

    """
    aug_periods_to_periods = _get_aug_periods_to_periods(
        n_aug_periods=dimensions.n_aug_periods,
        has_endogenous_factors=has_endogenous_factors,
    )

    stagemap: list[int] = (
        list(model_spec.stagemap)
        if model_spec.stagemap is not None
        else list(range(dimensions.n_periods - 1))
    )
    stages = sorted(int(v) for v in np.unique(stagemap))

    report = check_stagemap(
        stagemap=tuple(stagemap),
        stages=tuple(stages),
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
        latent_factors=tuple(model_spec.factors),
        observed_factors=tuple(model_spec.observed_factors),
        controls=("constant", *model_spec.controls),
        periods=tuple(sorted(set(aug_periods_to_periods.values()))),
        stagemap=tuple(stagemap),
        stages=tuple(stages),
        aug_periods=tuple(aug_periods_to_periods.keys()),
        aug_periods_to_periods=MappingProxyType(aug_periods_to_periods),
        aug_stagemap=tuple(aug_stagemap),
        aug_stages=tuple(sorted(int(v) for v in np.unique(aug_stagemap))),
        aug_stages_to_stages=MappingProxyType(aug_stages_to_stages),
    )


def _process_anchoring(model_spec: ModelSpec) -> Anchoring:
    """Process the specification that governs how latent factors are anchored.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`

    Returns:
        Anchoring dataclass with information about anchoring.

    """
    anch = model_spec.anchoring
    if anch is not None:
        return Anchoring.from_config(
            outcomes=dict(anch.outcomes),
            free_controls=anch.free_controls,
            free_constant=anch.free_constant,
            free_loadings=anch.free_loadings,
            ignore_constant_when_anchoring=anch.ignore_constant_when_anchoring,
        )

    return Anchoring.disabled()


def _augment_periods_for_endogenous_factors(
    model_spec: ModelSpec, dimensions: Dimensions, labels: Labels
) -> ModelSpec:
    """Augment periods if endogenous factors are present.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
        dimensions: Dimensional information.
        labels: Labels for model quantities.

    Returns:
        ModelSpec with twice the amount of periods.

    """
    new_factors: dict[str, FactorSpec] = {}
    for fac, fspec in model_spec.factors.items():
        insert_at_modulo = 0 if fspec.is_endogenous else 1

        # Insert empty elements into measurements when we do not have those.
        if len(fspec.measurements) != dimensions.n_periods:
            raise ValueError(
                "Measurements must be of length `n_periods`, "
                f"got {fspec.measurements} for {fac}"
            )
        aug_measurements = tuple(
            () if aug_p % 2 == insert_at_modulo else fspec.measurements[p]
            for aug_p, p in labels.aug_periods_to_periods.items()
        )

        # Insert empty elements into normalizations when we do not have those.
        aug_normalizations = None
        if fspec.normalizations is not None:
            aug_norm_parts: dict[str, tuple[Mapping[str, float], ...]] = {}
            for norm_type in ("loadings", "intercepts"):
                norms = getattr(fspec.normalizations, norm_type)
                if len(norms) != dimensions.n_periods:
                    raise ValueError(
                        "Normalizations must be lists of length `n_periods`, "
                        f"got {norms} for {fac}['normalizations']['{norm_type}']"
                    )
                aug_norm_parts[norm_type] = tuple(
                    {} if aug_p % 2 == insert_at_modulo else norms[p]
                    for aug_p, p in labels.aug_periods_to_periods.items()
                )
            aug_normalizations = Normalizations(
                loadings=aug_norm_parts["loadings"],
                intercepts=aug_norm_parts["intercepts"],
            )

        new_factors[fac] = FactorSpec(
            measurements=aug_measurements,
            normalizations=aug_normalizations,
            is_endogenous=fspec.is_endogenous,
            is_correction=fspec.is_correction,
            transition_function=fspec.transition_function,
            has_production_shock=fspec.has_production_shock,
            has_initial_distribution=fspec.has_initial_distribution,
        )

    return model_spec._replace(factors=new_factors)


def _get_transition_info(model_spec: ModelSpec, labels: Labels) -> TransitionInfo:
    """Collect information about transition functions."""
    func_list, param_names = [], []
    latent_factors = labels.latent_factors
    all_factors = labels.all_factors

    for factor in latent_factors:
        spec = model_spec.factors[factor].transition_function
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
                names: list[str] = spec.__registered_params__  # ty: ignore[invalid-assignment]
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
    def _extract_factor(states: Array, pos: int) -> Array:
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
        param_names=MappingProxyType(
            dict(zip(latent_factors, param_names, strict=False))
        ),
        individual_functions=MappingProxyType(individual_functions),
        function_names=MappingProxyType(
            dict(zip(latent_factors, function_names, strict=False))
        ),
    )


def _get_endogenous_factors_info(
    *,
    has_endogenous_factors: bool,
    model_spec: ModelSpec,
    labels: Labels,
) -> EndogenousFactorsInfo:
    """Collect information about endogenous factors."""
    factor_info = {}
    for fac, fspec in model_spec.factors.items():
        factor_info[fac] = FactorInfo.from_flags(
            is_endogenous=fspec.is_endogenous,
            is_correction=fspec.is_correction,
        )

    return EndogenousFactorsInfo(
        has_endogenous_factors=has_endogenous_factors,
        aug_periods_to_aug_period_meas_types=_get_aug_periods_to_aug_period_meas_types(
            aug_periods=labels.aug_periods_to_periods.keys(),
            has_endogenous_factors=has_endogenous_factors,
        ),
        aug_periods_from_period=partial(
            _aug_periods_from_period,
            aug_periods_to_periods=labels.aug_periods_to_periods,
        ),
        factor_info=MappingProxyType(factor_info),
    )


def _get_aug_periods_to_aug_period_meas_types(
    aug_periods: tuple[int, ...] | KeysView[int],
    *,
    has_endogenous_factors: bool,
) -> MappingProxyType[int, MeasurementType]:
    if has_endogenous_factors:
        return MappingProxyType(
            {
                aug_p: (
                    MeasurementType.STATES
                    if aug_p % 2 == 0
                    else MeasurementType.ENDOGENOUS_FACTORS
                )
                for aug_p in aug_periods
            }
        )
    return MappingProxyType(dict.fromkeys(aug_periods, MeasurementType.STATES))


def _get_update_info(
    model_spec: ModelSpec,
    dimensions: Dimensions,
    labels: Labels,
    anchoring_info: Anchoring,
) -> DataFrame:
    """Construct a DataFrame with information on each Kalman update.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
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
        measurements[factor] = model_spec.factors[factor].measurements
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
    uinfo["purpose"] = uinfo["purpose"].astype(pd.StringDtype(na_value=np.nan))
    return uinfo


def _process_normalizations(
    model_spec: ModelSpec, dimensions: Dimensions, labels: Labels
) -> MappingProxyType[str, Normalizations]:
    """Process the normalizations of intercepts and factor loadings.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
        dimensions: Dimensional information.
        labels: Labels for model quantities.

    Returns:
        Mapping from factor name to Normalizations instance.

    """
    result: dict[str, Normalizations] = {}
    for factor in labels.latent_factors:
        fspec = model_spec.factors[factor]
        parts: dict[str, tuple[Mapping[str, float], ...]] = {}
        for norm_type in ("loadings", "intercepts"):
            if fspec.normalizations is not None:
                candidate = list(getattr(fspec.normalizations, norm_type))
            else:
                candidate = [{} for _ in range(dimensions.n_aug_periods)]
            if len(candidate) != dimensions.n_aug_periods:
                raise ValueError(
                    "Normalizations must be of length `n_aug_periods`, "
                    f"got {candidate} for {factor}['{norm_type}']"
                )
            parts[norm_type] = tuple(candidate)
        result[factor] = Normalizations(**parts)

    return MappingProxyType(result)
