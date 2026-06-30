"""Project-specific exception types raised by skillmodels' user-facing API.

The beartype decorators applied at the public entry points
(`skillmodels._beartype_conf`) route parameter-type violations through
one of the classes defined here, so callers can write narrowly-scoped
`except` clauses against a stable skillmodels-specific hierarchy
instead of catching the framework-supplied `BeartypeCallHintParamViolation`.

All classes inherit from `TypeError` so existing `except TypeError`
handlers continue to fire; the subclasses are additive.
"""


class SkillmodelsInputError(TypeError):
    """Base class for all skillmodels parameter-validation errors."""


class ModelSpecInitializationError(SkillmodelsInputError):
    """Bad argument to a model-spec dataclass.

    Raised on construction of `ModelSpec`, `FactorSpec`,
    `AnchoringSpec`, or `Normalizations`.
    """


class OptionsInitializationError(SkillmodelsInputError):
    """Bad argument to an estimation-options dataclass.

    Raised on construction of `CHSEstimationOptions`,
    `AFEstimationOptions`, or `AMNEstimationOptions`.
    """


class EstimationCallError(SkillmodelsInputError):
    """Bad argument to an estimation entry point.

    Raised by `get_maximization_inputs`, `get_individual_states`,
    `estimate_af`, `estimate_amn`, `get_af_posterior_states`, or
    `get_amn_posterior_states` when arguments don't match the
    declared types.
    """


class InferenceCallError(SkillmodelsInputError):
    """Bad argument to a standard-error / bootstrap helper.

    Raised by `compute_af_standard_errors` and
    `compute_amn_standard_errors`.
    """


class SimulationCallError(SkillmodelsInputError):
    """Bad argument to a simulation helper.

    Raised by `simulate_dataset` and `simulate_policy_effect`.
    """


class DiagnosticsCallError(SkillmodelsInputError):
    """Bad argument to a diagnostics / visualisation helper.

    Raised by `decompose_measurement_variance`,
    `summarize_measurement_reliability`, `plot_residual_boxplots`,
    `plot_likelihood_contributions`, `create_state_ranges`, and the
    factor-distribution / transition-equation plotting helpers.
    """
