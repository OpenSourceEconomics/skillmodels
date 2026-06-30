"""Per-exception `BeartypeConf` instances used at the skillmodels perimeter.

Decorators at user-facing entry points configure beartype to raise the
existing project exception class on parameter-type violations,
preserving the documented exception hierarchy in
`skillmodels.exceptions`.

The constructors and call sites decorated through this module are the
"perimeter": ModelSpec / FactorSpec / AnchoringSpec / Normalizations,
the three estimation-options dataclasses, and every public function
exposed from the top-level package or the subpackage `__init__`s. The
internal helpers below the perimeter are unannotated for beartype and
trust the perimeter to have already validated parameter types.
"""

from collections.abc import Callable

from beartype import BeartypeConf, BeartypeStrategy, beartype

from skillmodels.exceptions import (
    DiagnosticsCallError,
    EstimationCallError,
    InferenceCallError,
    ModelSpecInitializationError,
    OptionsInitializationError,
    SimulationCallError,
)


def _conf(exc: type[Exception]) -> BeartypeConf:
    """Build a `BeartypeConf` that raises `exc` on parameter-type violations.

    `On` strategy: full O(n) container validation so every bad entry in
    a mapping/sequence is reported, not just one sampled element. The
    decorated entry points are called rarely (construction, estimate,
    simulate, plot), so per-call cost is invisible compared to the
    JIT-compiled hot path each one kicks off.

    `is_pep484_tower=True`: respect the PEP-484 numeric tower so `int`
    satisfies `float`-typed parameters (matches the implicit numeric
    conversion that Python and ruff's PYI041 both assume).
    """
    return BeartypeConf(
        violation_param_type=exc,
        strategy=BeartypeStrategy.On,
        is_pep484_tower=True,
    )


def beartype_init[T](conf: BeartypeConf) -> Callable[[type[T]], type[T]]:
    """Class decorator that wraps only `__init__` with `@beartype(conf=conf)`.

    Bare `@beartype` on a class wraps every method, which surfaces
    non-public annotation drift on instance methods that has nothing
    to do with parameter validation at construction time (e.g. a
    helper method that takes a JAX array typed loosely as `Any`). The
    only annotations we actively curate at the perimeter are the
    public-facing `__init__` parameters; restrict to those.

    The decorator is generic in the decorated class so the type checker keeps
    seeing `ModelSpec` (etc.) as the class itself, not as an opaque `type`
    value -- otherwise every `model_spec: ModelSpec` annotation downstream
    reads as an illegal `type`-valued annotation.
    """

    def wrap(cls: type[T]) -> type[T]:
        cls.__init__ = beartype(conf=conf)(cls.__init__)  # ty: ignore[invalid-assignment]
        return cls

    return wrap


# Construction of the four user-facing model-spec dataclasses.
MODEL_SPEC_CONF = _conf(ModelSpecInitializationError)

# Construction of CHSEstimationOptions, AFEstimationOptions,
# AMNEstimationOptions.
OPTIONS_CONF = _conf(OptionsInitializationError)

# `get_maximization_inputs`, `get_individual_states`, `estimate_af`,
# `estimate_amn`, `get_af_posterior_states`,
# `get_amn_posterior_states`.
ESTIMATION_CONF = _conf(EstimationCallError)

# `compute_af_standard_errors`, `compute_amn_standard_errors`.
INFERENCE_CONF = _conf(InferenceCallError)

# `simulate_dataset`, `simulate_policy_effect`.
SIMULATION_CONF = _conf(SimulationCallError)

# Diagnostics + visualisation entry points.
DIAGNOSTICS_CONF = _conf(DiagnosticsCallError)
