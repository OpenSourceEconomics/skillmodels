"""Functions to validate model specifications."""

from collections.abc import Mapping

import numpy as np

from skillmodels.model_spec import ModelSpec
from skillmodels.types import Anchoring, Dimensions, Labels


def check_model(
    model_spec: ModelSpec,
    labels: Labels,
    dimensions: Dimensions,
    anchoring: Anchoring,
    *,
    has_endogenous_factors: bool,
) -> None:
    """Check consistency and validity of the model specification.

    labels, dimensions and anchoring information are done before the model checking
    because processing them will not raise any errors except for easy to understand
    KeyErrors.

    Other specifications are checked in the model spec before processing to make sure
    that the assumptions we make during the processing are fulfilled.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
        dimensions: Dimensional information.
        labels: Labels for model quantities.
        anchoring: Information about anchoring.
        has_endogenous_factors: Whether the model has any endogenous factors

    Raises:
        ValueError

    """
    report = check_stagemap(
        stagemap=labels.aug_stagemap,
        stages=labels.aug_stages,
        n_periods=dimensions.n_aug_periods,
        is_augmented=has_endogenous_factors,
    )
    report += _check_anchoring(anchoring)
    invalid_measurements = _check_measurements(
        model_spec=model_spec, factors=labels.latent_factors
    )
    if invalid_measurements:
        report += invalid_measurements
    elif has_endogenous_factors:
        # Make this conditional because the check only works for valid meas.
        report += _check_no_overlap_in_measurements_of_states_and_inv(
            model_spec=model_spec, labels=labels
        )
    report += _check_normalizations(
        model_spec=model_spec, factors=labels.latent_factors
    )

    report = "\n".join(report)
    if report != "":
        raise ValueError(f"Invalid model specification: {report}")


def check_stagemap(
    stagemap: tuple[int, ...],
    stages: tuple[int, ...],
    n_periods: int,
    *,
    is_augmented: bool,
) -> list[str]:
    """Validate the stagemap configuration against model dimensions."""
    report: list[str] = []
    step_size = 2 if is_augmented else 1
    if len(stagemap) != n_periods - step_size:
        report.append(
            f"The stagemap needs to be of length n_periods - {step_size}. "
            f" n_periods is {n_periods}, the stagemap has length {len(stagemap)}.",
        )
    if stages != tuple(range(len(stages))):
        report.append("Stages need to be integers, start at zero and increase by 1.")

    # Hijacking the stagemap for endogenous factors leads to interleaved elements.
    to_consider = [stagemap] if not is_augmented else [stagemap[0::2], stagemap[1::2]]
    for sm in to_consider:
        if not np.isin(np.array(sm[1:]) - np.array(sm[:-1]), (0, step_size)).all():
            report.append(
                "Consecutive entries in stagemap must be equal or increase by 1."
            )
    return report


def _check_anchoring(anchoring: Anchoring) -> list[str]:
    report = []
    if not isinstance(anchoring.anchoring, bool):
        report.append("anchoring.anchoring must be a bool.")

    if not isinstance(anchoring.outcomes, Mapping):
        report.append("anchoring.outcomes must be a Mapping")
    else:
        variables = list(anchoring.outcomes.values())
        for var in variables:
            if not isinstance(var, str | int | tuple):
                report.append("Outcomes variables have to be valid variable names.")

    if not isinstance(anchoring.free_controls, bool):
        report.append("anchoring.free_controls must be a bool")
    if not isinstance(anchoring.free_constant, bool):
        report.append("anchoring.free_constant must be a bool.")
    if not isinstance(anchoring.free_loadings, bool):
        report.append("anchoring.free_loadings must be a bool.")
    return report


def _check_measurements(
    model_spec: ModelSpec,
    factors: tuple[str, ...],
) -> list[str]:
    report: list[str] = []
    for factor in factors:
        candidate = model_spec.factors[factor].measurements
        if not _is_sequence_of(candidate=candidate, type_=tuple):
            report.append(
                "measurements must be tuples of tuples. "
                f"Check measurements of {factor}.",
            )
        else:
            for period, meas_list in enumerate(candidate):
                for meas in meas_list:
                    if not isinstance(meas, int | str | tuple):
                        report.append(
                            "Measurements need to be valid pandas column names. Check "
                            f"{meas} for {factor} in period {period}.",
                        )
    return report


def _check_no_overlap_in_measurements_of_states_and_inv(
    model_spec: ModelSpec, labels: Labels
) -> list[str]:
    report = []
    for period in labels.periods:
        meas: dict[str, set] = {}
        for factor in labels.latent_factors:
            fspec = model_spec.factors[factor]
            if fspec.is_endogenous:
                meas["endogenous_factors"] = set(fspec.measurements[period])
            else:
                meas["states"] = set(fspec.measurements[period])
        if overlap := meas["states"].intersection(meas["endogenous_factors"]):
            report.append(
                "Measurements for exogenous and endogenous latent factors must not "
                f"overlap.\n\nCheck measurements {overlap} in period {period}.",
            )
    return report


def _check_normalizations(
    model_spec: ModelSpec,
    factors: tuple[str, ...],
) -> list[str]:
    report: list[str] = []
    for factor in factors:
        fspec = model_spec.factors[factor]
        if fspec.normalizations is None:
            continue
        for norm_type in ["loadings", "intercepts"]:
            norms = getattr(fspec.normalizations, norm_type)
            candidate = [dict(m) for m in norms]
            if not _is_sequence_of(candidate=candidate, type_=dict):
                report.append(
                    f"normalizations must be sequences of dicts. Check {norm_type} "
                    f"normalizations for {factor}.",
                )
            else:
                report += _check_normalized_variables_are_present(
                    list_of_normdicts=candidate,
                    model_spec=model_spec,
                    factor=factor,
                )

                if norm_type == "loadings":
                    report += _check_loadings_are_not_normalized_to_zero(
                        list_of_normdicts=candidate,
                        factor=factor,
                    )
    return report


def _check_normalized_variables_are_present(
    list_of_normdicts: list[dict],
    model_spec: ModelSpec,
    factor: str,
) -> list[str]:
    report: list[str] = []
    for period, norm_dict in enumerate(list_of_normdicts):
        for var in norm_dict:
            if var not in model_spec.factors[factor].measurements[period]:
                report.append(
                    "You can only normalize variables that are specified as "
                    f"measurements. Check {var} for {factor} in period "
                    f"{period}.",
                )

    return report


def _check_loadings_are_not_normalized_to_zero(
    list_of_normdicts: list[dict],
    factor: str,
) -> list[str]:
    report: list[str] = []
    for period, norm_dict in enumerate(list_of_normdicts):
        for var, val in norm_dict.items():
            if val == 0:
                report.append(
                    f"loadings cannot be normalized to 0. Check measurement {var} "
                    f"of {factor} in period {period}.",
                )
    return report


def _is_sequence_of(candidate: object, type_: type) -> bool:
    """Check if candidate is a sequence that only contains elements of type.

    Works with both lists and tuples.

    Examples:
    >>> _is_sequence_of([["a"], ["b"]], list)
    True
    >>> _is_sequence_of((("a",), ("b",)), tuple)
    True
    >>> _is_sequence_of([{}], list)
    False
    >>> _is_sequence_of([], dict)
    True

    """
    return isinstance(candidate, list | tuple) and all(
        isinstance(i, type_) for i in candidate
    )
