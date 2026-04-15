"""Frozen dataclass definitions for the AF estimator."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pandas as pd
from jax import Array


@dataclass(frozen=True)
class AFEstimationOptions:
    """Configuration options for the AF estimator."""

    n_halton_points: int = 50
    """Halton quadrature nodes per dimension."""

    n_halton_points_shock: int = 30
    """Quadrature nodes for production shock integration."""

    n_mixture_components: int = 2
    """Gaussian mixture components for initial distribution."""

    optimizer_algorithm: str = "fides"
    """Optimization algorithm for each period's MLE."""

    optimizer_options: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Additional options passed to optimagic."""

    two_stage: bool = False
    """Whether to use coarse-then-fine grid strategy."""

    coarse_fraction: float = 0.5
    """Fraction of quadrature points for coarse stage (if two_stage is True)."""

    stability_floor: float = 1e-217
    """Floor added to likelihood for numerical stability (exp(-500) ~ 7e-218)."""


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

    model_spec: Any
    """The ModelSpec used for estimation."""

    conditional_distributions: tuple[ConditionalDistribution, ...]
    """Estimated conditional distributions per period (for filtered states)."""
