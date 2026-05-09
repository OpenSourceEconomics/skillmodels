"""Frozen dataclass definitions for the AF estimator."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import jax
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

    initialization_strategy: Literal["constant", "moment_based"]
    """Strategy for seeding optimizer start values.

    `"moment_based"` (default) uses Spearman cross-covariance moments
    (factor-analysis identification) to seed loadings, sigma_meas,
    sigma_shock, and sigma_inv from the data. `"constant"` reproduces
    the legacy 0.5 / 0.5*obs_sd defaults; provided for regression
    testing and pre-fix reproducibility.
    """

    two_stage_measurement: bool
    """Estimate the measurement system in a Stage-1 pre-step.

    When True, run `estimate_measurement_system` (Spearman /
    multi-indicator factor-analysis identification) before AF Stage-2
    optimization, and hold the recovered loadings and sigma_meas fixed
    in Stage 2. This eliminates the sigma_inv / sigma_meas
    constant-Var(I_meas) ridge that causes ~30-50% sigma_inv_0 boundary
    collapse on translog-style DGPs.

    Standard-error caveat: when True, the score bootstrap currently
    holds Stage-1 outputs fixed across replicates and therefore
    underestimates variance for Stage-2 parameters that covary with
    sigma_meas. Users wanting fully-correct SEs should run a parametric
    bootstrap (resample data, redo `estimate_af`) until the
    per-replicate-Spearman bootstrap extension lands.

    No default: users must make an explicit choice given this trade-off
    between point-estimate robustness (favors True) and SE correctness
    within the existing bootstrap (favors False). When False, sigma_meas
    enters the AF MLE chain and the score bootstrap captures Spearman-
    free SEs correctly; when True, point estimates are far more
    reliable but SEs miss the Stage-1 contribution.
    """

    def __init__(  # noqa: D107
        self,
        n_halton_points: int = 50,
        n_halton_points_shock: int = 30,
        n_mixture_components: int = 2,
        optimizer_algorithm: str = "fides",
        optimizer_options: Mapping[str, Any] | None = None,
        *,
        two_stage_measurement: bool,
        two_stage: bool = False,
        coarse_fraction: float = 0.5,
        stability_floor: float = 1e-217,
        n_obs_per_batch: int | None = None,
        initialization_strategy: Literal["constant", "moment_based"] = "moment_based",
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
        object.__setattr__(self, "initialization_strategy", initialization_strategy)
        object.__setattr__(self, "two_stage_measurement", two_stage_measurement)


@dataclass(frozen=True)
class MixtureComponent:
    """Single component of a Gaussian mixture distribution."""

    mean: Array
    """Mean vector, shape (n_factors,)."""

    chol_cov: Array
    """Lower-triangular Cholesky factor of covariance, shape (n_factors, n_factors)."""


@dataclass(frozen=True)
class ChainLink:
    """Frozen-period parameters for one prior step in the θ_0→θ_{t-1} chain.

    Used by the AF transition likelihood to rebuild the chained importance
    sample on-demand from a single joint Halton design at every transition
    step (mirroring MATLAB's ``create_nodes_weights_01/12``). Each
    `ChainLink` carries the just-fitted parameters of one prior transition
    so the chain can be replayed inside the next step's likelihood call.
    """

    period: int
    """Calendar period at which this link applies (1-indexed; the link
    transforms θ_{period-1} → θ_period)."""

    transition_func: Callable
    """Combined per-factor transition function f(full_states, params)."""

    transition_params: Array
    """Flat transition parameter vector for this period, shape
    ``(total_n_transition_params,)``."""

    shock_sds: Array
    """Production shock SDs for shock-bearing state factors, shape
    ``(n_shock_factors,)``."""

    shock_factor_indices: Array
    """Mapping each shock slot to its position in the state-factor
    ordering, shape ``(n_shock_factors,)`` int."""

    inv_eq_params: Array
    """Flat investment-equation parameters, shape
    ``(n_endogenous * n_inv_eq_params_per,)``."""

    inv_sds: Array
    """Investment shock SDs, shape ``(n_endogenous,)``."""

    n_inv_eq_params_per: int
    """Investment equation parameters per endogenous factor (1 + n_state +
    n_observed_factors when n_endogenous > 0; 0 otherwise)."""

    obs_factor_values: Array
    """Observed factor values at this link's source period (i.e. period -
    1), shape ``(n_obs, n_observed_factors)``. Used in the chain rebuild
    for the inv equation and the transition function."""


# Register ChainLink as a JAX pytree so tuples of ChainLinks can be passed
# through `jax.jit` in the AF transition likelihood. Array fields are
# leaves; the period index, transition function, and per-link int counts
# are static metadata baked into the trace.
jax.tree_util.register_dataclass(
    ChainLink,
    data_fields=[
        "transition_params",
        "shock_sds",
        "shock_factor_indices",
        "inv_eq_params",
        "inv_sds",
        "obs_factor_values",
    ],
    meta_fields=["period", "transition_func", "n_inv_eq_params_per"],
)


@dataclass(frozen=True)
class ConditionalDistribution:
    """Estimated conditional distribution of latent factors at a given period.

    Holds two things that downstream code consumes:

    * Per-component summary statistics (`mean`, `chol_cov`) of the chained
      sample at this period — used by `posterior_states.py` and the
      inference sandwich code.
    * The chain history (`chain_links`) needed to rebuild the chained
      sample on-demand inside the next transition step's likelihood (joint
      Halton design — see `_rebuild_chain_at_period` in
      `af.likelihood`).

    For the period-0 distribution: per-obs `cond_means` / `cond_chols`
    encode the Schur conditional of latent factors given observed factors
    (`Y_0`); `conditional_weights` are the Bayes posterior mixture weights
    given `Y_0`. For later periods these are unused (chain replays from
    period 0).

    Note: `samples_per_component` is retained for backward compatibility
    and posterior-state-summary computation, but is no longer load-bearing
    inside the transition likelihood (which rebuilds the chain on-demand).
    """

    mixture_weights: Array
    """Mixture weights, shape (n_components,)."""

    components: tuple[MixtureComponent, ...]
    """Per-component summary statistics (mean, chol_cov) derived from the
    importance sample. Used by `posterior_states` and `inference`; not used
    in the transition likelihood itself."""

    samples_per_component: tuple[Array, ...]
    """One importance-sample array per mixture component, each shape
    ``(n_halton, n_obs, n_state)``. Retained for posterior-state summary
    statistics; not consumed by the transition likelihood (which rebuilds
    the chain on-demand from a joint Halton). May use a smaller Halton
    count than the likelihood's `n_halton_points`."""

    conditional_weights: Array | None = None
    """Individual-specific conditional mixture weights, shape (n_obs, n_components).

    When not None, these override `mixture_weights` for each observation (computed
    from Bayes' rule using data from previous periods).
    """

    cond_means: Array | None = None
    """Per-obs Schur-conditional means of the latent state given observed
    factors at period 0, shape ``(n_components, n_obs, n_state)``. Built
    by the initial period only. None for transition-period distributions.
    """

    cond_chols: Array | None = None
    """Per-component Schur-conditional Cholesky factors at period 0, shape
    ``(n_components, n_state, n_state)``. Shared across observations
    because the conditional covariance does not depend on Y_i (it's the
    prior cov_yy minus a Schur term). None for transition-period
    distributions."""

    chain_links: tuple[ChainLink, ...] = field(default_factory=tuple)
    """Sequence of frozen prior-period parameter packages, one per
    transition already estimated. Empty before period 1; one entry after
    period 1 estimation; two entries after period 2; etc. Used by the
    transition likelihood to rebuild the chained sample on-demand from a
    single joint Halton."""


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
