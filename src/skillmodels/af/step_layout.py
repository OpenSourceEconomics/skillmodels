"""Source/destination calendar layout for the AF estimator.

The AF estimator is sequential: step `s -> d = s+1` jointly estimates the `s -> d`
transition, the period-`s` investment equation, and a measurement block. The public
`ModelSpec` is contemporaneous -- an investment indicator declared at calendar period
`c` measures `I_c` -- so a calendar-to-step adapter is needed to feed AF's blocks. This
module compiles that adapter as row-level `AFMeasurementTerm`s grouped into one
`AFStepLayout` per transition.

For step `s -> d` the compiled blocks (validated against MATLAB `likelihood_01`/
`likelihood_12` + `create_nodes_weights_12`) are:

- FREE **target**: destination dynamic-state (skill) indicators scored on `theta_d`,
  and source endogenous (investment) indicators scored on `I_s`.
- FIXED **importance**: source dynamic-state indicators scored on `theta_s`, plus every
  static-persistent factor's period-0 indicators scored on its time-invariant value
  (re-applied at every step -- the dropped-MC/MN importance fix).

Carrying `family`/`lower`/`upper` on each term lets limited-dependent (probit/Tobit)
measurements compose with the shared measurement kernel without a second calendar pass.
"""

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from skillmodels.common.measurement_models import MeasurementFamily

# A term's role: a free, this-step target density vs a fixed, historical importance
# reweighting density.
AFTermRole = Literal["target", "importance"]


class AFFactorRole(enum.Enum):
    """How a latent factor participates in AF's sequential calendar."""

    DYNAMIC = "dynamic"
    """Real transition, measured across periods (e.g. skills)."""
    STATIC_PERSISTENT = "static_persistent"
    """Time-invariant; its period-0 measurement density is re-applied as an importance
    factor at every transition step (e.g. MC, MN)."""
    ENDOGENOUS = "endogenous"
    """Reconstructed from the investment equation (`has_initial_distribution=False`)."""


class AFEval(enum.Enum):
    """Which assembled latent vector a measurement term is scored against."""

    THETA_DEST = "theta_dest"
    """Destination dynamic state `theta_d`."""
    THETA_SRC = "theta_src"
    """Source dynamic state `theta_s`."""
    INV_SRC = "inv_src"
    """Source endogenous reconstruction `I_s = g(theta_s, Y_s)`."""
    STATIC = "static"
    """Time-invariant static-persistent latent coordinate."""


@dataclass(frozen=True)
class AFFactorInfo:
    """A factor's AF role and its per-calendar-period measurement declarations."""

    name: str
    """Factor name."""
    role: AFFactorRole
    """The factor's AF calendar role."""
    measurements_by_period: tuple[tuple[str, ...], ...]
    """Per-period tuple of measurement variable names (empty where inactive)."""


@dataclass(frozen=True)
class AFMeasurementTerm:
    """One observed measurement equation instance within an AF step."""

    measurement: str
    """Observed measurement variable name."""
    factor_loadings: tuple[str, ...]
    """Latent factors this row loads on (a single factor for non-cross-loaded rows)."""
    data_period: int
    """Calendar period the observed values are read from."""
    param_period: int
    """Calendar period the measurement params are indexed under."""
    control_period: int
    """Calendar period the controls are read from."""
    eval_node: AFEval
    """Which assembled latent vector the loadings are scored against."""
    family: MeasurementFamily
    """Measurement family (Gaussian/probit/Tobit)."""
    lower: float
    """Tobit lower censoring bound (`-inf` if not applicable)."""
    upper: float
    """Tobit upper censoring bound (`+inf` if not applicable)."""
    role: AFTermRole
    """`"target"` (free, this step) or `"importance"` (fixed, historical)."""
    free: bool
    """Whether this row's params are estimated at this step vs fixed from history."""


@dataclass(frozen=True)
class AFStepLayout:
    """The compiled measurement layout for one AF transition step `s -> d`."""

    source_period: int
    """The source calendar period `s`."""
    destination_period: int
    """The destination calendar period `d = s + 1`."""
    terms: tuple[AFMeasurementTerm, ...]
    """All target and importance measurement terms for the step."""
    equation_period: int
    """Calendar period indexing the transition, investment-eq, and shock params."""
    observed_factor_period: int
    """Calendar period the observed factors (income) entering `g` are read from."""

    def target_terms(self) -> tuple[AFMeasurementTerm, ...]:
        """Return the free, this-step target terms."""
        return tuple(t for t in self.terms if t.role == "target")

    def importance_terms(self) -> tuple[AFMeasurementTerm, ...]:
        """Return the fixed, historical importance terms."""
        return tuple(t for t in self.terms if t.role == "importance")


def compile_af_step_layouts(
    factor_infos: Sequence[AFFactorInfo],
    n_periods: int,
    families: Mapping[str, tuple[MeasurementFamily, float, float]] | None = None,
) -> tuple[AFStepLayout, ...]:
    """Compile one `AFStepLayout` per transition step from contemporaneous declarations.

    Args:
        factor_infos: Per-factor AF role + per-period measurement declarations.
        n_periods: Number of calendar periods (transitions are `0..n_periods-2`).
        families: Optional measurement -> `(family, lower, upper)`; absent measurements
            default to Gaussian.

    Return:
        One `AFStepLayout` per transition step `s -> s+1`.

    """
    fam_map = dict(families) if families is not None else {}
    layouts = []
    for source in range(n_periods - 1):
        destination = source + 1
        terms: list[AFMeasurementTerm] = []
        for info in factor_infos:
            terms.extend(_terms_for_factor(info, source, destination, fam_map))
        layouts.append(
            AFStepLayout(
                source_period=source,
                destination_period=destination,
                terms=tuple(terms),
                equation_period=source,
                observed_factor_period=source,
            )
        )
    return tuple(layouts)


def _terms_for_factor(
    info: AFFactorInfo,
    source: int,
    destination: int,
    fam_map: Mapping[str, tuple[MeasurementFamily, float, float]],
) -> list[AFMeasurementTerm]:
    """Build the step's measurement terms contributed by one factor."""
    if info.role == AFFactorRole.DYNAMIC:
        return [
            *_terms_at(
                info, destination, AFEval.THETA_DEST, "target", free=True, fam=fam_map
            ),
            *_terms_at(
                info, source, AFEval.THETA_SRC, "importance", free=False, fam=fam_map
            ),
        ]
    if info.role == AFFactorRole.ENDOGENOUS:
        # Source-period indicators measure I_s (the inv that drives this transition).
        return _terms_at(info, source, AFEval.INV_SRC, "target", free=True, fam=fam_map)
    # STATIC_PERSISTENT: each declared period's indicators re-applied as importance.
    terms: list[AFMeasurementTerm] = []
    for period in range(len(info.measurements_by_period)):
        terms.extend(
            _terms_at(
                info, period, AFEval.STATIC, "importance", free=False, fam=fam_map
            )
        )
    return terms


def _terms_at(
    info: AFFactorInfo,
    period: int,
    eval_node: AFEval,
    role: AFTermRole,
    *,
    free: bool,
    fam: Mapping[str, tuple[MeasurementFamily, float, float]],
) -> list[AFMeasurementTerm]:
    """Build terms for one factor's measurements at a single calendar period."""
    measures = (
        info.measurements_by_period[period]
        if period < len(info.measurements_by_period)
        else ()
    )
    terms = []
    for measure in measures:
        family, lower, upper = fam.get(
            measure, (MeasurementFamily.GAUSSIAN, -math.inf, math.inf)
        )
        terms.append(
            AFMeasurementTerm(
                measurement=measure,
                factor_loadings=(info.name,),
                data_period=period,
                param_period=period,
                control_period=period,
                eval_node=eval_node,
                family=family,
                lower=lower,
                upper=upper,
                role=role,
                free=free,
            )
        )
    return terms
