"""Functions to process model specifications from user-friendly to internal form."""

import inspect
from collections.abc import Callable, KeysView, Mapping
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
from skillmodels.common.control_function import (
    build_cf_node,
    build_kappa_addition_node,
    build_kappa_term_evaluators,
    build_prediction_node,
    generate_kappa_terms,
)
from skillmodels.common.decorators import extract_params, jax_array_output
from skillmodels.common.model_spec import FactorSpec, ModelSpec
from skillmodels.common.types import (
    Anchoring,
    ControlFunctionInfo,
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
    transition_info = _get_transition_info(
        model_spec=_model_spec_aug,
        labels=labels,
        control_function=endogenous_factors_info.control_function,
    )
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
    for factor, fspec in factors.items():
        if not isinstance(fspec.is_endogenous, bool):
            msg = (
                f"'is_endogenous' must be a bool, got {fspec.is_endogenous!r} "
                f"for {factor}."
            )
            raise TypeError(msg)
    return any(fspec.is_endogenous for fspec in factors.values())


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
    period: int, aug_periods_to_periods: Mapping[int, int]
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
        # insert_at_modulo decides measurement parity: endogenous (investment)
        # factors are measured at ODD aug_periods (the endogenous half), ordinary
        # state factors at EVEN aug_periods (the states half). This co-placement
        # produces the investment level and the first-class control-function
        # prediction at the same odd aug_period, so the residual cf = level -
        # prediction is a genuine same-period residual.
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
            correction=fspec.correction,
            transition_function=fspec.transition_function,
            has_production_shock=fspec.has_production_shock,
            has_initial_distribution=fspec.has_initial_distribution,
        )

    return model_spec._replace(factors=new_factors)


def _inject_control_function_nodes(
    *,
    functions: dict,
    control_function: ControlFunctionInfo,
    all_factors: tuple[str, ...],
) -> None:
    """Graft the control-function nodes into the transition DAG in place.

    Adds a contemporaneous first-stage prediction node and a residual `cf`
    node, and rewrites each target factor's `__next_<target>__` node to add
    `kappa * cf` on top of the (untouched) base production node. The base node
    is preserved under `__base_next_<target>__`, so its arguments, positional
    parameter layout, and DAG dependencies are unchanged.

    Args:
        functions: The DAG node mapping built by `_get_transition_info`.
        control_function: The resolved control-function configuration.
        all_factors: Factor order of the `states` vector (latent then observed).

    """
    inv = control_function.investment_factor
    inv_pos = all_factors.index(inv)
    predictor_positions = tuple(
        all_factors.index(factor)
        for factor in (
            *control_function.state_predictors,
            *control_function.instruments,
        )
    )
    factor_positions = {factor: i for i, factor in enumerate(all_factors)}

    prediction_node = f"__prediction_{inv}__"
    functions[prediction_node] = build_prediction_node(
        beta_key=f"__first_stage_{inv}__",
        predictor_positions=predictor_positions,
    )
    functions["cf"] = rename_arguments(
        build_cf_node(inv_pos=inv_pos),
        mapper={"prediction": prediction_node},
    )

    for target in control_function.targets:
        base_node = f"__base_next_{target}__"
        functions[base_node] = functions.pop(f"__next_{target}__")
        evaluators = build_kappa_term_evaluators(
            kappa_terms=control_function.kappa_terms[target],
            factor_positions=factor_positions,
        )
        functions[f"__next_{target}__"] = rename_arguments(
            build_kappa_addition_node(
                kappa_key=f"__kappa_{target}__",
                kappa_evaluators=evaluators,
            ),
            mapper={"base_value": base_node},
        )


def _build_base_transition_funcs(
    model_spec: ModelSpec,
    latent_factors: tuple[str, ...],
    all_factors: tuple[str, ...],
) -> tuple[list[Callable], list[list[str]]]:
    """Build the per-factor base transition callables and their parameter names."""
    func_list: list[Callable] = []
    param_names: list[list[str]] = []
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
            if not hasattr(spec, "__registered_params__"):
                raise AttributeError(
                    "Custom transition_functions must have a __registered_params__ "
                    "attribute. You can set it via the register_params decorator.",
                )
            names: list[str] = spec.__registered_params__  # ty: ignore[invalid-assignment]
            param_names.append(names)
            func_list.append(extract_params(spec, key=factor, names=names))
    return func_list, param_names


def _get_transition_info(
    model_spec: ModelSpec,
    labels: Labels,
    control_function: ControlFunctionInfo | None = None,
) -> TransitionInfo:
    """Collect information about transition functions."""
    latent_factors = labels.latent_factors
    all_factors = labels.all_factors

    func_list, param_names = _build_base_transition_funcs(
        model_spec=model_spec,
        latent_factors=latent_factors,
        all_factors=all_factors,
    )

    # `extract_params` preserves `__name__` via `functools.wraps`, but `ty` only
    # sees the `Callable` return type, which does not expose it.
    function_names = [f.__name__ for f in func_list]  # ty: ignore[unresolved-attribute]

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

    if control_function is not None:
        _inject_control_function_nodes(
            functions=functions,
            control_function=control_function,
            all_factors=all_factors,
        )

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
            dict(zip(latent_factors, (tuple(p) for p in param_names), strict=False))
        ),
        individual_functions=MappingProxyType(individual_functions),
        function_names=MappingProxyType(
            dict(zip(latent_factors, function_names, strict=False))
        ),
    )


def _resolve_control_function(model_spec: ModelSpec) -> ControlFunctionInfo | None:
    """Resolve the `CorrectionSpec` declared on the investment factor.

    Expand the user-facing `CorrectionSpec` (which may leave fields empty as
    "all state factors" / "the default `cf` regressor") into a fully resolved
    `ControlFunctionInfo` read by both estimators. Return `None` when no factor
    declares a `correction`.

    Args:
        model_spec: The model specification.

    Returns:
        Resolved `ControlFunctionInfo`, or `None` if there is no correction.

    Raises:
        ValueError: If a non-endogenous factor declares a `correction`.
        NotImplementedError: If more than one factor declares a `correction`.

    """
    with_correction = [
        (fac, fspec.correction)
        for fac, fspec in model_spec.factors.items()
        if fspec.correction is not None
    ]
    if not with_correction:
        return None
    if len(with_correction) > 1:
        names = ", ".join(fac for fac, _ in with_correction)
        msg = (
            "The control function supports exactly one investment factor, but "
            f"a correction was declared on multiple factors: {names}."
        )
        raise NotImplementedError(msg)

    investment_factor, spec = with_correction[0]
    if not model_spec.factors[investment_factor].is_endogenous:
        msg = (
            f"Factor {investment_factor!r} declares a correction but is not "
            "endogenous. A control function requires an endogenous investment "
            "factor."
        )
        raise ValueError(msg)

    not_observed = tuple(
        i for i in spec.instruments if i not in model_spec.observed_factors
    )
    if not_observed:
        msg = (
            f"The correction on {investment_factor!r} lists instruments "
            f"{not_observed} that are not declared observed factors. Control-"
            "function instruments must be observed factors of the model."
        )
        raise ValueError(msg)

    state_factors = tuple(
        fac for fac, fspec in model_spec.factors.items() if not fspec.is_endogenous
    )
    state_predictors = spec.state_predictors or state_factors
    targets = spec.targets or state_factors
    if not targets:
        msg = (
            f"Factor {investment_factor!r} declares a correction but the model "
            "has no state factors to apply it to. A control function needs at "
            "least one non-endogenous state factor as a target."
        )
        raise ValueError(msg)

    # Built-in production transitions enumerate (and the constraint machinery pins
    # to 0) an instrument coefficient, but a custom transition could consume an
    # instrument as an input with no pinnable coefficient. Instruments are
    # first-stage-only, so reject that loudly.
    instruments = set(spec.instruments)
    for fac, fspec in model_spec.factors.items():
        tfunc = fspec.transition_function
        if fac == investment_factor or not callable(tfunc):
            continue
        consumed = set(inspect.signature(tfunc).parameters) - {"params"}
        if leaked := sorted(consumed & instruments):
            msg = (
                f"The custom transition for {fac!r} consumes control-function "
                f"instrument(s) {leaked}. Instruments are first-stage-only and must "
                "not enter a production transition."
            )
            raise ValueError(msg)

    if spec.kappa_terms is not None:
        kappa_terms = {t: spec.kappa_terms.get(t, ("cf",)) for t in targets}
    else:
        degree = spec.kappa_degree if spec.kappa_degree is not None else 1
        basis = generate_kappa_terms(state_factors, max_degree=degree)
        kappa_terms = dict.fromkeys(targets, basis)

    return ControlFunctionInfo(
        investment_factor=investment_factor,
        state_predictors=tuple(state_predictors),
        instruments=tuple(spec.instruments),
        targets=tuple(targets),
        kappa_terms=MappingProxyType(kappa_terms),
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
        factor_info[fac] = FactorInfo.from_flags(is_endogenous=fspec.is_endogenous)

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
        control_function=_resolve_control_function(model_spec),
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
