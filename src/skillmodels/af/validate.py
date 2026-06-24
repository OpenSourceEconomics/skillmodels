"""AF-specific ModelSpec validation."""

import warnings

import optimagic as om
import pandas as pd

from skillmodels.common.identification import (
    check_identification,
    warn_if_overrestricted,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec

# Transition functions compatible with AF estimation (parametric, differentiable).
_AF_COMPATIBLE_TRANSITIONS = frozenset(
    {
        "linear",
        "translog",
        "translog_af",
        "robust_translog",
        "log_ces",
        "log_ces_af",
        "log_ces_with_constant",
        "log_ces_general",
        "linear_and_squares",
    }
)

# Built-in production transition functions that enumerate parameters over
# `all_factors` (latent + observed). When such a function is used for a
# (non-endogenous) production factor and observed factors are present, the
# observed factors (e.g. income) receive free linear / square / interaction /
# CES-weight coefficients and silently enter the production function — which
# changes the AF estimand (income must affect skills only through the
# investment equation). `translog_af` / `log_ces_af` are deliberately NOT in
# this set because they are documented to receive production factors only.
_LEAKY_BUILTIN_PRODUCTION = frozenset(
    {
        "linear",
        "linear_and_squares",
        "translog",
        "robust_translog",
        "log_ces",
        "log_ces_with_constant",
        "log_ces_general",
    }
)

# Hard minimum: 2 measurements + a loading normalization just-identify the
# per-period measurement system (3 moments — Var(Z1), Var(Z2), Cov(Z1,Z2) —
# vs 1 free loading + 2 sigma_meas) given Var(F) pinned by the chain.
_MIN_MEASURES_PER_FACTOR = 2
# Recommended minimum: the AF paper's identification arguments assume 3
# indicators per factor per period (over-identified Spearman moments).
# Below this, Stage-B Spearman is noisy and cross-period equality
# constraints on loadings / sigma_meas become load-bearing for ID.
_RECOMMENDED_MEASURES_PER_FACTOR = 3


def validate_af_model(
    model_spec: ModelSpec,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> None:
    """Validate that a ModelSpec is compatible with AF estimation.

    Check:
    - At least 3 measurements per factor in each period where the factor is measured
    - Transition functions are parametric (built-in or registered)
    - Normalizations are present for each factor
    - Each factor's period-0 (initial-distribution) affine orbit is anchored:
      the initial distribution is not produced by any transition, so its scale
      and location must be pinned directly by a loading/intercept normalization,
      a `fixed_params` pin, or an equality constraint. A `Normalizations` object
      with empty period-0 maps would otherwise leave the trans-log model
      under-identified. Periods t>0 are not checked here -- the transition can
      legitimately propagate the anchor, so verifying their identification needs
      a transition-aware diagnostic (tracked separately).

    The optional `fixed_params` and `constraints` (the same objects passed to
    `estimate_af`) supply the alternative anchors. They default to None so the
    measurement-system and transition checks can be run on a bare ModelSpec.

    Also emit a loud `UserWarning` (not an error) when a built-in production
    transition function would silently absorb observed factors (income).
    Built-in transitions enumerate parameters over ALL factors (latent +
    observed), so an observed factor enters the production function with free
    coefficients. The AF model assumes observed factors (e.g. income) affect
    skills ONLY through the investment equation; using a built-in transition
    on a production factor while observed factors are present violates that
    assumption. The warning (rather than a hard error) keeps existing models
    runnable while surfacing the wrong-estimand risk so it is an explicit
    choice; use `translog_af` / `log_ces_af` (production-factors only) to
    avoid the leakage entirely.

    Raise:
        ValueError: If validation fails, with a detailed error message.
        NotImplementedError: If the model declares a control-function correction
            (`FactorSpec.correction`), which AF does not implement.

    """
    corrected = [
        name for name, spec in model_spec.factors.items() if spec.correction is not None
    ]
    if corrected:
        msg = (
            "AF estimation does not implement the control-function correction "
            f"(kappa != 0) declared by FactorSpec.correction on {corrected}. AF "
            "covers only the kappa=0 (exogenous-investment) special case. Use "
            "estimate_chs for the correction, or strip it with "
            "ModelSpec.without_correction()."
        )
        raise NotImplementedError(msg)

    errors: list[str] = []
    for factor_name, factor_spec in model_spec.factors.items():
        errors.extend(_validate_factor(factor_name, factor_spec))

    errors.extend(check_identification(model_spec, fixed_params, constraints))

    warn_if_overrestricted(model_spec, fixed_params, constraints)
    _warn_on_observed_factor_leakage(model_spec)

    if errors:
        msg = "ModelSpec is not compatible with AF estimation:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)


def _validate_factor(factor_name: str, factor_spec: FactorSpec) -> list[str]:
    """Return a list of error messages for a single factor."""
    errors: list[str] = []

    # Check measurements: need >= 2 per factor in each active period; warn
    # below 3 (the recommended count from the AF paper).
    for period, measures in enumerate(factor_spec.measurements):
        if len(measures) == 0:
            continue
        if len(measures) < _MIN_MEASURES_PER_FACTOR:
            errors.append(
                f"Factor '{factor_name}' period {period}: AF requires at least "
                f"{_MIN_MEASURES_PER_FACTOR} measurements, got {len(measures)}."
            )
        elif len(measures) < _RECOMMENDED_MEASURES_PER_FACTOR:
            warnings.warn(
                f"Factor '{factor_name}' period {period}: only {len(measures)} "
                f"measurements (AF paper assumes at least "
                f"{_RECOMMENDED_MEASURES_PER_FACTOR}). Identification of "
                f"loadings + sigma_meas at this period relies on "
                f"cross-period equality constraints across the AF MLE chain; "
                f"supply explicit `fixed_params` for the loading if needed.",
                stacklevel=3,
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

    # Check normalizations exist. The required pattern is transition-specific
    # (audit F7a): trans-log needs a per-factor affine anchor (loading=1 and
    # intercept=0) at the initial period; restricted CES (psi=1) needs only a
    # SINGLE scale anchor (lambda_theta,0,1=1) -- CES identifies the remaining
    # loadings -- with location supplied by the simplex (plain log_ces) or a
    # pinned intercept (log_ces_with_constant). So do not prescribe the trans-log
    # "one per period" pattern for every factor.
    if factor_spec.normalizations is None:
        errors.append(
            f"Factor '{factor_name}': AF requires explicit normalizations. "
            f"Provide a scale anchor (a finite nonzero loading=1) for one "
            f"measurement, plus a finite period-0 measurement-intercept (or "
            f"equivalent latent-location pin) as the absolute initial location "
            f"anchor mu_theta,0,1=0. The CES simplex and the free constant of "
            f"log_ces_with_constant do NOT supply that initial location anchor. "
            f"Trans-log needs the affine anchor at the initial period; restricted "
            f"CES needs only the single scale anchor lambda_theta,0,1=1 (the "
            f"later loadings are then identified)."
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


def _warn_on_observed_factor_leakage(model_spec: ModelSpec) -> None:
    """Warn when a built-in production function silently absorbs observed factors.

    Built-in transition functions enumerate parameters over `all_factors`
    (latent + observed), so observed factors (e.g. income) get FREE linear /
    square / interaction / CES-weight coefficients in the production function.
    The AF model assumes income affects skills ONLY through the investment
    equation. Emit a loud `UserWarning` so the wrong-estimand risk is an
    explicit, visible choice rather than a silent default. Use a
    production-factors-only transition (`translog_af` / `log_ces_af`, or a
    custom `@register_params` callable consuming only production factors) to
    remove the leakage. The warning is deliberately NOT an error: it must not
    break existing, intentionally-leaky models, and `validate_af_model` has no
    access to `fixed_params` to detect an explicit opt-out (pinned-zero
    observed-factor coefficients).
    """
    observed = tuple(model_spec.observed_factors)
    if not observed:
        return
    for fac, fspec in model_spec.factors.items():
        if fspec.is_endogenous:
            # The investment equation legitimately uses observed factors.
            continue
        tf = fspec.transition_function
        if isinstance(tf, str) and tf in _LEAKY_BUILTIN_PRODUCTION:
            warnings.warn(
                f"Factor '{fac}': built-in transition '{tf}' enumerates "
                f"parameters over ALL factors including observed factors "
                f"{observed}, so they enter the production function with free "
                f"coefficients. The AF model assumes observed factors affect "
                f"skills only through the investment equation. Use a "
                f"production-factors-only transition ('translog_af' or "
                f"'log_ces_af'), or pin every observed-factor transition "
                f"coefficient to 0.0 via `fixed_params`, to avoid changing "
                f"the production estimand.",
                stacklevel=3,
            )


def fail_if_unsupported_kappa_params(
    start_params: pd.DataFrame | None,
    fixed_params: pd.DataFrame | None,
    constraints: list | None,  # noqa: ARG001
) -> None:
    """Raise if any supplied params reference an unimplemented kappa category.

    AF integrates production and investment shocks as independent draws
    (kappa_t = 0). A nonzero kappa_t control-function term is not
    implemented; fail loudly rather than silently dropping such entries.
    Unknown parameter categories never enter any period index and would
    otherwise vanish without error, letting a caller believe endogenous
    coupling is being estimated.
    """
    bad = {"kappa", "kappa_t"}

    def _has_kappa(df: pd.DataFrame | None) -> bool:
        if df is None or df.index.nlevels < 1:
            return False
        cats = df.index.get_level_values(0)
        return any(c in bad for c in cats)

    if _has_kappa(start_params) or _has_kappa(fixed_params):
        msg = (
            "AF estimation does not implement endogenous investment "
            "(kappa_t != 0): production and investment shocks are assumed "
            "independent (kappa_t = 0). Remove 'kappa'/'kappa_t' parameters."
        )
        raise NotImplementedError(msg)
