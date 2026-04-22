"""Frozen dataclass definitions for the AF estimator."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandas as pd
from jax import Array

from skillmodels.types import ensure_containers_are_immutable

if TYPE_CHECKING:
    from skillmodels.model_spec import ModelSpec


@dataclass(frozen=True, init=False)
class AFEstimationOptions:
    """Configuration options for the AF estimator."""

    n_halton_points: int
    """Halton quadrature nodes per dimension."""

    n_halton_points_shock: int
    """Quadrature nodes for production shock integration."""

    n_mixture_components: int
    """Gaussian mixture components for initial distribution."""

    optimizer_algorithm: str
    """Optimization algorithm for each period's MLE."""

    optimizer_options: MappingProxyType[str, Any]
    """Additional options passed to optimagic."""

    two_stage: bool
    """Whether to use coarse-then-fine grid strategy."""

    coarse_fraction: float
    """Fraction of quadrature points for coarse stage (if two_stage is True)."""

    stability_floor: float
    """Floor added to likelihood for numerical stability."""

    n_obs_per_batch: int | None
    """Observations per reverse-mode autodiff chunk.

    When `None` (default), an auto-detected value is derived from the
    available GPU/CPU memory in `estimate_af`. Setting this to a small
    integer trades compile time and throughput for lower peak VRAM; the
    likelihood value is unchanged.
    """

    def __init__(  # noqa: D107
        self,
        n_halton_points: int = 50,
        n_halton_points_shock: int = 30,
        n_mixture_components: int = 2,
        optimizer_algorithm: str = "fides",
        optimizer_options: Mapping[str, Any] | None = None,
        *,
        two_stage: bool = False,
        coarse_fraction: float = 0.5,
        stability_floor: float = 1e-217,
        n_obs_per_batch: int | None = None,
    ) -> None:
        object.__setattr__(self, "n_halton_points", n_halton_points)
        object.__setattr__(self, "n_halton_points_shock", n_halton_points_shock)
        object.__setattr__(self, "n_mixture_components", n_mixture_components)
        object.__setattr__(self, "optimizer_algorithm", optimizer_algorithm)
        object.__setattr__(
            self,
            "optimizer_options",
            ensure_containers_are_immutable(optimizer_options or {}),
        )
        object.__setattr__(self, "two_stage", two_stage)
        object.__setattr__(self, "coarse_fraction", coarse_fraction)
        object.__setattr__(self, "stability_floor", stability_floor)
        object.__setattr__(self, "n_obs_per_batch", n_obs_per_batch)


@dataclass(frozen=True)
class MixtureComponent:
    """Single component of a Gaussian mixture distribution."""

    mean: Array
    """Mean vector, shape (n_factors,)."""

    chol_cov: Array
    """Lower-triangular Cholesky factor of covariance, shape (n_factors, n_factors)."""


@dataclass(frozen=True)
class ConditionalDistribution:
    """Estimated conditional distribution of latent factors at a given period.

    Represents f(ln theta_t | data_{0:t}) as a mixture of Gaussians, where the
    mixture parameters may depend on individual-level data from previous periods.
    """

    mixture_weights: Array
    """Mixture weights, shape (n_components,)."""

    components: tuple[MixtureComponent, ...]
    """Per-component distribution parameters."""

    conditional_weights: Array | None = None
    """Individual-specific conditional mixture weights, shape (n_obs, n_components).

    When not None, these override `mixture_weights` for each observation (computed
    from Bayes' rule using data from previous periods).
    """


@dataclass(frozen=True)
class AFPeriodResult:
    """Result from estimating a single period."""

    period: int
    """Calendar period index."""

    params: pd.DataFrame
    """Estimated parameters with 4-level MultiIndex (category, period, name1, name2)."""

    loglikelihood: float
    """Log-likelihood value at the optimum."""

    success: bool
    """Whether optimization converged."""

    optimize_result: Any
    """Raw optimagic result object."""


@dataclass(frozen=True)
class AFEstimationResult:
    """Complete result from AF estimation across all periods."""

    period_results: tuple[AFPeriodResult, ...]
    """Per-period estimation results, ordered by period."""

    all_params: pd.DataFrame
    """Combined parameters from all periods with standard 4-level MultiIndex."""

    model_spec: ModelSpec
    """The ModelSpec used for estimation."""

    conditional_distributions: tuple[ConditionalDistribution, ...]
    """Estimated conditional distributions per period (for filtered states)."""
