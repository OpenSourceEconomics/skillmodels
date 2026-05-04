"""AF-specific ModelSpec validation."""

from skillmodels.model_spec import FactorSpec, ModelSpec

# Transition functions compatible with AF estimation (parametric, differentiable).
_AF_COMPATIBLE_TRANSITIONS = frozenset(
    {
        "linear",
        "translog",
        "robust_translog",
        "log_ces",
        "log_ces_with_constant",
        "log_ces_general",
        "linear_and_squares",
    }
)

_MIN_MEASURES_PER_FACTOR = 3


def validate_af_model(model_spec: ModelSpec) -> None:
    """Validate that a ModelSpec is compatible with AF estimation.

    Check:
    - At least 3 measurements per factor in each period where the factor is measured
    - Transition functions are parametric (built-in or registered)
    - Normalizations are present for each factor

    Raise:
        ValueError: If validation fails, with a detailed error message.

    """
    errors: list[str] = []
    for factor_name, factor_spec in model_spec.factors.items():
        errors.extend(_validate_factor(factor_name, factor_spec))

    if errors:
        msg = "ModelSpec is not compatible with AF estimation:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)


def _validate_factor(factor_name: str, factor_spec: FactorSpec) -> list[str]:
    """Return a list of error messages for a single factor."""
    errors: list[str] = []

    # Check measurements: need >= 3 per factor in each active period
    for period, measures in enumerate(factor_spec.measurements):
        if len(measures) == 0:
            continue
        if len(measures) < _MIN_MEASURES_PER_FACTOR:
            errors.append(
                f"Factor '{factor_name}' period {period}: AF requires at least "
                f"{_MIN_MEASURES_PER_FACTOR} measurements, got {len(measures)}."
            )

    # Check transition function is parametric
    tf = factor_spec.transition_function
    if tf is not None and isinstance(tf, str) and tf not in _AF_COMPATIBLE_TRANSITIONS:
        errors.append(
            f"Factor '{factor_name}': transition function '{tf}' is not in the "
            f"set of AF-compatible functions: {sorted(_AF_COMPATIBLE_TRANSITIONS)}."
        )
    # Custom callables are accepted if they have __registered_params__
    if callable(tf) and not hasattr(tf, "__registered_params__"):
        errors.append(
            f"Factor '{factor_name}': custom transition function must be decorated "
            f"with @register_params to be used with AF estimation."
        )

    # Check normalizations exist
    if factor_spec.normalizations is None:
        errors.append(
            f"Factor '{factor_name}': AF requires explicit normalizations "
            f"(loading=1, intercept=0 for at least one measurement per period)."
        )

    # has_initial_distribution=False requires is_endogenous=True so the
    # factor can be reconstructed via the investment equation at period 0.
    if not factor_spec.has_initial_distribution and not factor_spec.is_endogenous:
        errors.append(
            f"Factor '{factor_name}': has_initial_distribution=False is only "
            f"supported for endogenous factors (set is_endogenous=True)."
        )

    # Factors without an initial distribution must also not be measured at
    # period 0: their value at period 0 is not drawn from any mixture, so a
    # measurement density there would have no latent value to hit.
    if (
        not factor_spec.has_initial_distribution
        and len(factor_spec.measurements) > 0
        and len(factor_spec.measurements[0]) > 0
    ):
        errors.append(
            f"Factor '{factor_name}': has_initial_distribution=False requires "
            f"empty measurements at period 0 (got "
            f"{factor_spec.measurements[0]!r}). Drop them from the FactorSpec; "
            f"their contribution would typically be absorbed into the "
            f"transition step 0->1 in a MATLAB-style reproduction."
        )

    return errors
