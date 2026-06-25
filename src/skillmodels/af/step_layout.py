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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, SupportsFloat, cast

import pandas as pd

from skillmodels.common.measurement_models import MeasurementFamily
from skillmodels.common.model_spec import ModelSpec

# A term's role: a free, this-step target density vs a fixed, historical importance
# reweighting density.
AFTermRole = Literal["target", "importance"]


def model_uses_calendar_adapter(model_spec: ModelSpec) -> bool:
    """Return whether a model takes the AF source/destination calendar-adapter path.

    The adapter activates for a reconstructed endogenous factor (`is_endogenous` with
    `has_initial_distribution=False`) or any static-persistent factor. Plain AF models
    keep the single-period path.
    """
    specs = model_spec.factors.values()
    reconstructed_endog = any(
        spec.is_endogenous and not spec.has_initial_distribution for spec in specs
    )
    static_persistent = any(spec.af_state_role == "static_persistent" for spec in specs)
    return reconstructed_endog or static_persistent


def fail_if_calendar_adapter_unsupported(model_spec: ModelSpec, feature: str) -> None:
    """Raise if `feature` is requested for a calendar-adapter model.

    AF standard-error inference and posterior-state extraction still reconstruct the
    single-period measurement layout, which is wrong for the mixed-calendar adapter:
    source-investment params live in the next step's result and the per-step parser
    counts differ. Reject adapter models on those paths -- rather than return standard
    errors that differentiate the wrong objective or posterior rows attached to the
    wrong latent -- until they consume the compiled `AFStepLayout`.
    """
    if model_uses_calendar_adapter(model_spec):
        msg = (
            f"{feature} is not yet supported for AF models that use the "
            "source/destination calendar adapter (a reconstructed-endogenous or "
            "static-persistent factor). That path must be migrated to the compiled "
            "AFStepLayout before it returns valid results."
        )
        raise NotImplementedError(msg)


def fail_if_spearman_unsupported_on_adapter(
    model_spec: ModelSpec, strategy: str
) -> None:
    """Raise if a Spearman moment start is requested for a calendar-adapter model.

    The Spearman/OLS start routine discovers measurements at the destination period and
    writes loading/SD rows there, so it does not seed the mixed-calendar target
    (destination skills at `d` plus source investment at `s`). Rather than silently fall
    back to the constant start -- which contradicts the requested option -- reject it on
    the adapter path until a layout-aware moment initializer exists.
    """
    if strategy == "spearman" and model_uses_calendar_adapter(model_spec):
        msg = (
            "start_params_strategy='spearman' is not supported for AF models that use "
            "the source/destination calendar adapter: the moment start discovers "
            "measurements at the destination period and would mis-seed the "
            "source-investment block. Use start_params_strategy='constant' (or 'amn') "
            "until a calendar-aware moment initializer is implemented."
        )
        raise NotImplementedError(msg)


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


@dataclass(frozen=True)
class HistoricalParams:
    """Cumulative AF parameter registry keyed by the full param MultiIndex.

    Importance terms re-apply fixed period-0 static-factor (MC/MN) densities, whose
    params live in the initial-step result, not the immediately-previous step result.
    A single concatenated registry makes every earlier estimate reachable.
    """

    table: pd.DataFrame
    """Concatenated per-step params with a unique MultiIndex and a `value` column."""

    @classmethod
    def from_param_frames(cls, frames: Iterable[pd.DataFrame]) -> HistoricalParams:
        """Concatenate per-step param frames, rejecting duplicate index entries."""
        combined = pd.concat(list(frames))
        if combined.index.has_duplicates:
            dups = sorted(set(combined.index[combined.index.duplicated()].tolist()))
            msg = f"Duplicate parameter index entries across AF steps: {dups}"
            raise ValueError(msg)
        return cls(table=combined)

    def value(self, category: str, period: int, name1: str, name2: str) -> float:
        """Return the estimated value at one full-index coordinate."""
        # The index is unique (validated), so this `.loc` is a scalar; pandas-stubs
        # still type it as a broad union, hence the cast to a float-coercible value.
        cell = self.table.loc[(category, period, name1, name2), "value"]
        return float(cast("SupportsFloat", cell))


def compile_target_measurement_index(
    layout: AFStepLayout,
    controls: tuple[str, ...],
) -> list[tuple[str, int, str, str]]:
    """Build the free target measurement param index in the flat parser's global order.

    Emits all control rows, then all loading rows, then all measurement-SD rows (the
    order `_parse_transition_params` expects), each tagged with its term's true
    `param_period` so a mixed-calendar target (destination skills at `d`, source
    investment at `s`) parses correctly.
    """
    targets = layout.target_terms()
    ind_tups: list[tuple[str, int, str, str]] = []
    for term in targets:
        for ctrl in controls:
            ind_tups.append(("controls", term.param_period, term.measurement, ctrl))
    for term in targets:
        for factor in term.factor_loadings:
            ind_tups.append(("loadings", term.param_period, term.measurement, factor))
    for term in targets:
        ind_tups.append(("meas_sds", term.param_period, term.measurement, "-"))
    return ind_tups


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
        # @pro: THE calendar fix. The endogenous (investment) indicators are sourced
        # from the SOURCE period s and scored on I_s -- the investment that drives the
        # s->d transition -- not from destination d (the prior I_{d-1} mispairing).
        # Confirm this matches MATLAB likelihood_01/12, where the period-s investment
        # block enters transition_{s->d}.
        return _terms_at(info, source, AFEval.INV_SRC, "target", free=True, fam=fam_map)
    # STATIC_PERSISTENT: only the period-0 indicators are re-applied as importance.
    # @pro: a static factor's period-0 measurement block is re-emitted as a FIXED
    # importance term at every step s->d (period-0 rows only, re-applied each step), so
    # its time-invariant density carries over in the importance weight without any later
    # period's declaration leaking into an earlier step. Is re-anchoring the static
    # block at its declared period-0 the correct carry-over weight at every later step?
    return _terms_at(info, 0, AFEval.STATIC, "importance", free=False, fam=fam_map)


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
