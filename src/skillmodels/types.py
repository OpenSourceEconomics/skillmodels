"""Dataclass definitions for skillmodels internal data structures."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import NewType

import pandas as pd
from jax import Array

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
    """Dimensional information for a skill formation model.

    All fields represent counts of model components.
    """

    n_latent_factors: int
    n_observed_factors: int
    n_controls: int
    n_mixtures: int
    n_aug_periods: int
    n_periods: int

    @property
    def n_all_factors(self) -> int:
        """Total number of factors (latent + observed)."""
        return self.n_latent_factors + self.n_observed_factors


@dataclass(frozen=True)
class Labels:
    """Labels for model quantities.

    Contains string identifiers for factors, periods, controls, and stages.
    """

    latent_factors: tuple[str, ...]
    observed_factors: tuple[str, ...]
    controls: tuple[str, ...]
    periods: tuple[int, ...]
    stagemap: tuple[int, ...]
    stages: tuple[int, ...]
    aug_periods: tuple[int, ...]
    aug_periods_to_periods: MappingProxyType[int, int]
    aug_stagemap: tuple[int, ...]
    aug_stages: tuple[int, ...]
    aug_stages_to_stages: MappingProxyType[int, int]
    transition_names: tuple[str, ...] = ()

    @property
    def all_factors(self) -> tuple[str, ...]:
        """All factor names (latent + observed)."""
        return self.latent_factors + self.observed_factors


@dataclass(frozen=True)
class Anchoring:
    """Information about how latent factors are anchored to observed outcomes."""

    anchoring: bool
    outcomes: MappingProxyType[str, str]
    factors: tuple[str, ...]
    free_controls: bool
    free_constant: bool
    free_loadings: bool
    ignore_constant_when_anchoring: bool

    @classmethod
    def disabled(cls) -> Anchoring:
        """Create an Anchoring config with anchoring disabled."""
        return cls(
            anchoring=False,
            outcomes=MappingProxyType({}),
            factors=(),
            free_controls=False,
            free_constant=False,
            free_loadings=False,
            ignore_constant_when_anchoring=False,
        )

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
class EstimationOptions:
    """Tuning parameters for the estimation."""

    sigma_points_scale: float
    robust_bounds: bool
    bounds_distance: float
    clipping_lower_bound: float | None
    clipping_upper_bound: float | None
    clipping_lower_hardness: float
    clipping_upper_hardness: float


@dataclass(frozen=True)
class TransitionInfo:
    """Information about transition functions."""

    func: Callable
    param_names: MappingProxyType[str, list[str]]
    individual_functions: MappingProxyType[str, Callable]
    function_names: MappingProxyType[str, str]


@dataclass(frozen=True)
class FactorInfo:
    """Information for a single factor."""

    factor_type: FactorType

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
    aug_periods_to_aug_period_meas_types: MappingProxyType[int, MeasurementType]
    bounds_distance: float
    aug_periods_from_period: Callable[[int], list[int]]
    factor_info: MappingProxyType[str, FactorInfo]


@dataclass(frozen=True)
class ProcessedModel:
    """Complete processed model specification.

    This is the main output of process_model() containing all information
    needed for estimation.
    """

    dimensions: Dimensions
    labels: Labels
    anchoring: Anchoring
    estimation_options: EstimationOptions
    transition_info: TransitionInfo
    update_info: pd.DataFrame
    normalizations: dict[str, dict[str, list]]
    endogenous_factors_info: EndogenousFactorsInfo


@dataclass(frozen=True)
class LoadingsParsingInfo:
    """Information for parsing factor loadings from parameter vector."""

    slice: Array | slice
    flat_indices: Array
    shape: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class ParsingInfo:
    """Information for parsing the parameter vector.

    Maps model quantities to positions or slices of the parameter vector.
    """

    initial_states: Array | slice
    initial_cholcovs: Array | slice
    mixture_weights: Array | slice
    controls: Array | slice
    meas_sds: Array | slice
    shock_sds: Array | slice
    loadings: LoadingsParsingInfo
    transition: dict[str, Array | slice]
    is_anchoring_loading: Array
    is_anchored_factor: Array
    is_anchoring_update: Array
    ignore_constant_when_anchoring: bool
    has_endogenous_factors: bool


@dataclass(frozen=True)
class ParsedParams:
    """Parsed parameters from the flat parameter vector.

    Contains all model parameters in structured arrays.
    """

    controls: Array
    loadings: Array
    meas_sds: Array
    shock_sds: Array
    transition: dict[str, Array]
    anchoring_scaling_factors: Array
    anchoring_constants: Array


@dataclass(frozen=True)
class ProcessedData:
    """Processed data arrays for estimation.

    All arrays are JAX arrays ready for use in the likelihood function.
    """

    measurements: Array
    controls: Array
    observed_factors: Array


@dataclass(frozen=True)
class KalmanState:
    """State carried through Kalman filter iterations.

    Used as the carry state in jax.lax.scan.
    """

    states: Array
    upper_chols: Array
    log_mixture_weights: Array
