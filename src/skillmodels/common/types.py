"""Dataclass definitions for skillmodels internal data structures."""

import copyreg
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, NewType, cast

import pandas as pd
from jax import Array

from skillmodels._beartype_conf import MODEL_SPEC_CONF, beartype_init


def _make_immutable(value: Any) -> Any:  # noqa: ANN401
    """Recursively convert a value to its immutable equivalent."""
    if isinstance(value, (MappingProxyType, tuple, frozenset)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({k: _make_immutable(v) for k, v in value.items()})
    if isinstance(value, set):
        return frozenset(_make_immutable(v) for v in value)
    if isinstance(value, list):
        return tuple(_make_immutable(v) for v in value)
    return value


def ensure_containers_are_immutable[K, V](
    value: Mapping[K, V],
) -> MappingProxyType[K, V]:
    """Recursively convert mutable containers to immutable equivalents.

    Conversions:
        - dict/Mapping -> MappingProxyType
        - list -> tuple
        - set -> frozenset

    Values that are already immutable (MappingProxyType, tuple, frozenset) are
    returned as-is.

    Args:
        value: Any Mapping to convert.

    Returns:
        A MappingProxyType with all nested containers converted to their
        immutable equivalents.

    """
    return cast("MappingProxyType[K, V]", _make_immutable(value))


def _to_plain(value: Any) -> Any:  # noqa: ANN401
    """Inverse of `_make_immutable`: recursively unwrap to mutable Python.

    Used at boundaries where a downstream library (e.g. optimagic's
    `om.minimize(algo_options=...)`) does a strict `isinstance(..., dict)`
    check that rejects `MappingProxyType`.
    """
    if isinstance(value, MappingProxyType | Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    if isinstance(value, frozenset | set):
        return {_to_plain(v) for v in value}
    return value


def to_plain_dict[K, V](
    mp: Mapping[K, V],
) -> dict[K, V]:
    """Recursively unwrap a `MappingProxyType` tree into plain `dict`."""
    return cast("dict[K, V]", _to_plain(mp))


def _reduce_mapping_proxy(mp: MappingProxyType) -> tuple:
    return ensure_containers_are_immutable, (dict(mp),)


copyreg.pickle(MappingProxyType, _reduce_mapping_proxy)


# NewType definitions for domain safety
# These prevent accidentally mixing up semantically different int values
Period = NewType("Period", int)
AugPeriod = NewType("AugPeriod", int)
Stage = NewType("Stage", int)
AugStage = NewType("AugStage", int)


class FactorType(Enum):
    """Type of a latent factor in the model."""

    STATE = auto()  # Regular state factor
    ENDOGENOUS = auto()  # Endogenous factor (not a correction)
    CORRECTION = auto()  # Correction factor (is_endogenous=True, is_correction=True)


class MeasurementType(Enum):
    """Type of measurement in an augmented period."""

    STATES = auto()
    ENDOGENOUS_FACTORS = auto()


@dataclass(frozen=True)
class Dimensions:
    """Dimensional information for a skill formation model."""

    n_latent_factors: int
    """Number of latent (unobserved) factors."""
    n_observed_factors: int
    """Number of observed factors."""
    n_controls: int
    """Number of control variables (always >= 1 due to constant)."""
    n_mixtures: int
    """Number of mixture components in the distribution."""
    n_aug_periods: int
    """Number of augmented periods (doubled when endogenous factors are present)."""
    n_periods: int
    """Number of original model periods."""

    @property
    def n_all_factors(self) -> int:
        """Total number of factors (latent + observed)."""
        return self.n_latent_factors + self.n_observed_factors


@dataclass(frozen=True)
class Labels:
    """Labels for model quantities."""

    latent_factors: tuple[str, ...]
    """Names of latent (unobserved) factors."""
    observed_factors: tuple[str, ...]
    """Names of observed factors."""
    controls: tuple[str, ...]
    """Names of control variables (first is always `"constant"`)."""
    periods: tuple[int, ...]
    """Original period indices `(0, 1, 2, ...)`."""
    stagemap: tuple[int, ...]
    """Map each transition to a development stage."""
    stages: tuple[int, ...]
    """Unique stage indices."""
    aug_periods: tuple[int, ...]
    """Augmented period indices (doubled when endogenous factors are present)."""
    aug_periods_to_periods: MappingProxyType[int, int]
    """Map each augmented period to its original period."""
    aug_stagemap: tuple[int, ...]
    """Stage mapping for augmented periods."""
    aug_stages: tuple[int, ...]
    """Unique augmented stage indices."""
    aug_stages_to_stages: MappingProxyType[int, int]
    """Map each augmented stage to its original stage."""
    transition_names: tuple[str, ...] = ()
    """Names of the transition functions per factor."""

    @property
    def all_factors(self) -> tuple[str, ...]:
        """All factor names (latent + observed)."""
        return self.latent_factors + self.observed_factors


@dataclass(frozen=True)
class Anchoring:
    """Information about how latent factors are anchored to observed outcomes."""

    anchoring: bool = False
    """Whether anchoring is enabled."""
    outcomes: MappingProxyType[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping from factor names to outcome variable names."""
    factors: tuple[str, ...] = ()
    """Names of anchored factors (derived from `outcomes` keys)."""
    free_controls: bool = False
    """Whether control coefficients are free in anchoring equations."""
    free_constant: bool = False
    """Whether the constant is free in anchoring equations."""
    free_loadings: bool = False
    """Whether loadings are free in anchoring equations."""
    ignore_constant_when_anchoring: bool = False
    """Whether to ignore the constant when anchoring."""

    @classmethod
    def disabled(cls) -> Anchoring:
        """Create an Anchoring config with anchoring disabled."""
        return cls()

    @classmethod
    def from_config(
        cls,
        outcomes: dict[str, str],
        *,
        free_controls: bool = False,
        free_constant: bool = False,
        free_loadings: bool = False,
        ignore_constant_when_anchoring: bool = False,
    ) -> Anchoring:
        """Create an Anchoring config from a configuration dictionary.

        Args:
            outcomes: Mapping from factor names to outcome variable names.
            free_controls: Whether control parameters are free in anchoring equations.
            free_constant: Whether constant is free in anchoring equations.
            free_loadings: Whether loadings are free in anchoring equations.
            ignore_constant_when_anchoring: Whether to ignore constant when anchoring.

        Returns:
            Configured Anchoring instance with anchoring enabled.

        """
        return cls(
            anchoring=True,
            outcomes=MappingProxyType(outcomes),
            factors=tuple(outcomes.keys()),
            free_controls=free_controls,
            free_constant=free_constant,
            free_loadings=free_loadings,
            ignore_constant_when_anchoring=ignore_constant_when_anchoring,
        )


@dataclass(frozen=True)
class TransitionInfo:
    """Information about transition functions."""

    func: Callable
    """Combined transition function for all factors."""
    param_names: MappingProxyType[str, tuple[str, ...]]
    """Mapping from factor name to its transition parameter names."""
    individual_functions: MappingProxyType[str, Callable]
    """Mapping from factor name to its transition function."""
    function_names: MappingProxyType[str, str]
    """Mapping from factor name to its transition function name."""


@dataclass(frozen=True)
class FactorInfo:
    """Information for a single factor."""

    factor_type: FactorType
    """Whether this factor is a state, endogenous, or correction factor."""

    @property
    def is_state(self) -> bool:
        """Whether the factor is a regular state factor."""
        return self.factor_type == FactorType.STATE

    @property
    def is_endogenous(self) -> bool:
        """Whether the factor is endogenous (ENDOGENOUS or CORRECTION)."""
        return self.factor_type in (FactorType.ENDOGENOUS, FactorType.CORRECTION)

    @property
    def is_correction(self) -> bool:
        """Whether the factor is a correction factor."""
        return self.factor_type == FactorType.CORRECTION

    @classmethod
    def from_flags(
        cls, *, is_endogenous: bool = False, is_correction: bool = False
    ) -> FactorInfo:
        """Create FactorInfo from boolean flags.

        Args:
            is_endogenous: Whether the factor is endogenous.
            is_correction: Whether the factor is a correction (must be endogenous).

        Returns:
            FactorInfo with the appropriate FactorType.

        Raises:
            ValueError: If is_correction is True but is_endogenous is False.

        """
        if is_correction and not is_endogenous:
            msg = "A correction factor must also be endogenous"
            raise ValueError(msg)
        if is_correction:
            return cls(factor_type=FactorType.CORRECTION)
        if is_endogenous:
            return cls(factor_type=FactorType.ENDOGENOUS)
        return cls(factor_type=FactorType.STATE)


@dataclass(frozen=True)
class EndogenousFactorsInfo:
    """Information about endogenous factors in the model."""

    has_endogenous_factors: bool
    """Whether the model contains any endogenous factors."""
    aug_periods_to_aug_period_meas_types: MappingProxyType[int, MeasurementType]
    """Map each augmented period to whether it measures states or endogenous
    factors."""
    aug_periods_from_period: Callable[[int], list[int]]
    """Return the augmented period indices for a given original period."""
    factor_info: MappingProxyType[str, FactorInfo]
    """Mapping from factor name to its `FactorInfo`."""


@beartype_init(MODEL_SPEC_CONF)
@dataclass(frozen=True)
class Normalizations:
    """Normalizations for factor identification."""

    loadings: tuple[Mapping[str, float], ...]
    """Per-period loading normalizations. Each element maps variable name to
    fixed value."""
    intercepts: tuple[Mapping[str, float], ...]
    """Per-period intercept normalizations. Each element maps variable name to
    fixed value."""

    def __post_init__(self) -> None:  # noqa: D105
        object.__setattr__(
            self,
            "loadings",
            tuple(ensure_containers_are_immutable(m) for m in self.loadings),
        )
        object.__setattr__(
            self,
            "intercepts",
            tuple(ensure_containers_are_immutable(m) for m in self.intercepts),
        )


@dataclass(frozen=True)
class ProcessedModel:
    """Complete processed model specification.

    Main output of `process_model()` containing all information needed for estimation.
    """

    dimensions: Dimensions
    """Counts of model components."""
    labels: Labels
    """String identifiers for factors, periods, controls, and stages."""
    anchoring: Anchoring
    """Anchoring configuration."""
    transition_info: TransitionInfo
    """Transition function details."""
    update_info: pd.DataFrame
    """DataFrame describing each Kalman update step."""
    normalizations: MappingProxyType[str, Normalizations]
    """Per-factor identification normalizations."""
    endogenous_factors_info: EndogenousFactorsInfo
    """Endogenous factor configuration."""


@dataclass(frozen=True)
class LoadingsParsingInfo:
    """Information for parsing factor loadings from parameter vector."""

    slice: Array | slice
    """Slice or index array into the flat parameter vector."""
    flat_indices: Array
    """Flat indices for reshaping the loadings array."""
    shape: tuple[int, ...]
    """Target shape of the loadings array."""
    size: int
    """Total number of loading parameters."""


@dataclass(frozen=True)
class ParsingInfo:
    """Information for parsing the parameter vector.

    Maps model quantities to positions or slices of the parameter vector.
    """

    initial_states: Array | slice
    """Slice for initial state means."""
    initial_cholcovs: Array | slice
    """Slice for initial Cholesky covariance factors."""
    mixture_weights: Array | slice
    """Slice for mixture weight parameters."""
    controls: Array | slice
    """Slice for control variable coefficients."""
    meas_sds: Array | slice
    """Slice for measurement standard deviations."""
    shock_sds: Array | slice
    """Slice for shock standard deviations."""
    loadings: LoadingsParsingInfo
    """Parsing info for factor loadings."""
    transition: MappingProxyType[str, Array | slice]
    """Mapping from transition parameter name to its slice."""
    is_anchoring_loading: Array
    """Boolean array flagging anchoring loadings."""
    is_anchored_factor: Array
    """Boolean array flagging anchored factors."""
    is_anchoring_update: Array
    """Boolean array flagging anchoring updates."""
    ignore_constant_when_anchoring: bool
    """Whether to ignore constant when anchoring."""
    has_endogenous_factors: bool
    """Whether the model has endogenous factors."""


@dataclass(frozen=True)
class ParsedParams:
    """Parsed parameters from the flat parameter vector."""

    controls: Array
    """Control variable coefficients."""
    loadings: Array
    """Factor loading parameters."""
    meas_sds: Array
    """Measurement standard deviations."""
    shock_sds: Array
    """Shock standard deviations."""
    transition: MappingProxyType[str, Array]
    """Mapping from transition parameter name to its array."""
    anchoring_scaling_factors: Array
    """Scaling factors for anchoring equations."""
    anchoring_constants: Array
    """Constants for anchoring equations."""


@dataclass(frozen=True)
class ProcessedData:
    """Processed data arrays for estimation.

    All arrays are JAX arrays ready for use in the likelihood function.
    """

    measurements: Array
    """Array of shape `(n_updates, n_obs)` with measurement data."""
    controls: Array
    """Array of shape `(n_periods, n_obs, n_controls)` with controls."""
    observed_factors: Array
    """Array of shape `(n_periods, n_obs, n_observed_factors)` with observed
    factor data."""


@dataclass(frozen=True)
class KalmanState:
    """State carried through Kalman filter iterations.

    Used as the carry state in `jax.lax.scan`.
    """

    states: Array
    """State means per mixture component."""
    upper_chols: Array
    """Upper Cholesky factors of state covariance matrices."""
    log_mixture_weights: Array
    """Log weights for each mixture component."""
