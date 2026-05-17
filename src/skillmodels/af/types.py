"""Frozen dataclass definitions for the AF estimator."""

import dataclasses
import gc
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import jax
import numpy as np
import pandas as pd
from jax import Array

from skillmodels._beartype_conf import OPTIONS_CONF, beartype_init
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.types import ensure_containers_are_immutable


@beartype_init(OPTIONS_CONF)
@dataclass(frozen=True, init=False)
class AFEstimationOptions:
    """Configuration options for the AF estimator."""

    n_halton_points: int
    """Halton quadrature nodes per dimension."""

    n_halton_points_shock: int
    """Quadrature nodes for production shock integration."""

    n_mixture_components: int
    """Gaussian mixture components for initial distribution."""

    optimizer_backend: Literal["auto", "optimagic", "jaxopt"]
    """Optimizer backend.

    `"auto"` (default) picks `"jaxopt"` when a JAX GPU is visible and
    the model is jaxopt-compatible (no `log_ces` transitions that
    would trigger `ProbabilityConstraint`s, no cross-section
    `EqualityConstraint`s passed via `estimate_af(constraints=...)`).
    Otherwise falls back to `"optimagic"`. The decision is taken
    once at the start of `estimate_af` and the resolved value is
    available on `AFEstimationResult.af_options.optimizer_backend`.

    `"optimagic"` explicit: each period's MLE runs via
    `optimagic.minimize` with the algorithm in `optimizer_algorithm`.
    Supports the full set of optimagic constraint kinds
    (`FixedConstraintWithValue`, `ProbabilityConstraint`,
    `EqualityConstraint`).

    `"jaxopt"` explicit: keeps the parameter vector on device through
    the L-BFGS-B iterations via `jaxopt.LBFGSB`, eliminating the
    host<->device transfer that occurs once per likelihood call when
    optimagic is used. Supports only `FixedConstraintWithValue`
    plus bounds; raises on probability or equality constraints
    (i.e. models with log_ces transitions or cross-section
    equalities must use `"optimagic"`).
    """

    optimizer_algorithm: str
    """Optimization algorithm for each period's MLE.

    Only consulted by the `"optimagic"` backend; ignored when
    `optimizer_backend="jaxopt"` (jaxopt always uses L-BFGS-B).
    """

    optimizer_options: MappingProxyType[str, Any]
    """Additional options passed to the optimizer.

    Forwarded to `optimagic.minimize(**optimizer_options)` for the
    `"optimagic"` backend or to `jaxopt.LBFGSB(**optimizer_options)`
    for the `"jaxopt"` backend.
    """

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

    initialization_strategy: Literal["constant", "spearman", "amn"]
    """Strategy for seeding optimizer start values.

    `"amn"` (default) runs the full AMN 2020 three-stage estimator
    upfront and uses its parameter estimates as start values for the
    per-period MLE. `"spearman"` uses Spearman cross-covariance
    moments per period (factor-analysis identification) to seed
    loadings, sigma_meas, sigma_shock, and sigma_inv. `"constant"`
    reproduces the legacy 0.5 / 0.5*obs_sd defaults; provided for
    regression testing and pre-fix reproducibility.
    """

    keep_conditional_distributions: bool
    """If True (default), the result's `conditional_distributions` field
    holds the per-period filtered state distributions, materialised on
    host as numpy. Set to False to skip the device→host transfer of
    these arrays entirely and return an empty tuple; useful on small
    GPUs (e.g. P100 12 GB) where the final materialisation OOMs even
    after the per-period optimiser has finished.
    """

    n_halton_points_posterior_summary: int
    """Halton draws kept per period for `samples_per_component`, the
    posterior-state summary tensor.

    `samples_per_component` is an `(n, n_obs, n_state)` array per mixture
    component used only by `posterior_states.py` and the inference
    sandwich to compute summary statistics; it is NOT consumed by the
    transition likelihood (which rebuilds the chain on-demand from a
    joint Halton via `_rebuild_chain_at_period`). The likelihood always
    uses `n_halton_points`; this knob only controls the persistent
    summary tensor's size.

    Defaults to 256, which keeps the per-period summary tensor under
    a few MB even at `n_obs = 50_000`. Bump higher (e.g. 2_000) if
    posterior-state summary precision matters for downstream analysis.
    """

    def __init__(  # noqa: D107
        self,
        n_halton_points: int = 50,
        n_halton_points_shock: int = 30,
        n_mixture_components: int = 2,
        optimizer_backend: Literal["auto", "optimagic", "jaxopt"] = "auto",
        optimizer_algorithm: str = "fides",
        optimizer_options: Mapping[str, Any] | None = None,
        *,
        two_stage: bool = False,
        coarse_fraction: float = 0.5,
        stability_floor: float = 1e-217,
        n_obs_per_batch: int | None = None,
        initialization_strategy: Literal["constant", "spearman", "amn"] = "amn",
        keep_conditional_distributions: bool = True,
        n_halton_points_posterior_summary: int = 256,
    ) -> None:
        if n_halton_points_posterior_summary < 1:
            msg = (
                "n_halton_points_posterior_summary must be >= 1, "
                f"got {n_halton_points_posterior_summary}."
            )
            raise ValueError(msg)
        if optimizer_backend not in ("auto", "optimagic", "jaxopt"):
            msg = (
                'optimizer_backend must be "auto", "optimagic", or "jaxopt", '
                f"got {optimizer_backend!r}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "n_halton_points", n_halton_points)
        object.__setattr__(self, "n_halton_points_shock", n_halton_points_shock)
        object.__setattr__(self, "n_mixture_components", n_mixture_components)
        object.__setattr__(self, "optimizer_backend", optimizer_backend)
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
        object.__setattr__(
            self, "keep_conditional_distributions", keep_conditional_distributions
        )
        object.__setattr__(
            self,
            "n_halton_points_posterior_summary",
            n_halton_points_posterior_summary,
        )


@dataclass(frozen=True)
class MixtureComponent:
    """Single component of a Gaussian mixture distribution."""

    mean: Array | np.ndarray
    """Mean vector, shape (n_factors,)."""

    chol_cov: Array | np.ndarray
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

    transition_params: Array | np.ndarray
    """Flat transition parameter vector for this period, shape
    ``(total_n_transition_params,)``."""

    shock_sds: Array | np.ndarray
    """Production shock SDs for shock-bearing state factors, shape
    ``(n_shock_factors,)``."""

    shock_factor_indices: Array | np.ndarray
    """Mapping each shock slot to its position in the state-factor
    ordering, shape ``(n_shock_factors,)`` int."""

    inv_eq_params: Array | np.ndarray
    """Flat investment-equation parameters, shape
    ``(n_endogenous * n_inv_eq_params_per,)``."""

    inv_sds: Array | np.ndarray
    """Investment shock SDs, shape ``(n_endogenous,)``."""

    n_inv_eq_params_per: int
    """Investment equation parameters per endogenous factor (1 + n_state +
    n_observed_factors when n_endogenous > 0; 0 otherwise)."""

    obs_factor_values: Array | np.ndarray
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

    mixture_weights: Array | np.ndarray
    """Mixture weights, shape (n_components,)."""

    components: tuple[MixtureComponent, ...]
    """Per-component summary statistics (mean, chol_cov) derived from the
    importance sample. Used by `posterior_states` and `inference`; not used
    in the transition likelihood itself."""

    samples_per_component: tuple[Array | np.ndarray, ...]
    """One importance-sample array per mixture component, each shape
    ``(n_halton, n_obs, n_state)``. Retained for posterior-state summary
    statistics; not consumed by the transition likelihood (which rebuilds
    the chain on-demand from a joint Halton). May use a smaller Halton
    count than the likelihood's `n_halton_points`."""

    conditional_weights: Array | np.ndarray | None = None
    """Individual-specific conditional mixture weights, shape (n_obs, n_components).

    When not None, these override `mixture_weights` for each observation (computed
    from Bayes' rule using data from previous periods).
    """

    cond_means: Array | np.ndarray | None = None
    """Per-obs Schur-conditional means of the latent state given observed
    factors at period 0, shape ``(n_components, n_obs, n_state)``. Built
    by the initial period only. None for transition-period distributions.
    """

    cond_chols: Array | np.ndarray | None = None
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

    def to_numpy(self) -> AFEstimationResult:
        """Return a copy with all device arrays materialised as numpy.

        Drops `samples_per_component` (per-period
        `(n_halton, n_obs, n_state)` importance buffers, typically
        multi-GB) and replaces every `jax.Array` inside the conditional
        distributions with a host-side `np.ndarray`.

        Call this before pickling the result or when device memory needs
        to be released. `estimate_af` itself returns arrays on-device so
        repeated calls can reuse the JAX/XLA compilation cache; that
        cache is freed here (the side effect is necessary because the
        host-staging buffer for the GPU→host copy must fit, and on a
        device loaded with compiled per-period likelihoods + gradients
        it routinely OOMs without this).
        """
        # Free compiled executables + unreferenced device buffers so the
        # host staging copy below has room.
        jax.clear_caches()
        gc.collect()
        new_cds = tuple(
            _conditional_distribution_to_numpy(cd)
            for cd in self.conditional_distributions
        )
        return dataclasses.replace(self, conditional_distributions=new_cds)


def _array_to_numpy(value: Array | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(jax.device_get(value))


def _chain_link_to_numpy(link: ChainLink) -> ChainLink:
    return dataclasses.replace(
        link,
        transition_params=_array_to_numpy(link.transition_params),
        shock_sds=_array_to_numpy(link.shock_sds),
        shock_factor_indices=_array_to_numpy(link.shock_factor_indices),
        inv_eq_params=_array_to_numpy(link.inv_eq_params),
        inv_sds=_array_to_numpy(link.inv_sds),
        obs_factor_values=_array_to_numpy(link.obs_factor_values),
    )


def _conditional_distribution_to_numpy(
    cond_dist: ConditionalDistribution,
) -> ConditionalDistribution:
    new_components = tuple(
        MixtureComponent(
            mean=_array_to_numpy(c.mean),  # ty: ignore[invalid-argument-type]
            chol_cov=_array_to_numpy(c.chol_cov),  # ty: ignore[invalid-argument-type]
        )
        for c in cond_dist.components
    )
    new_chain_links = tuple(_chain_link_to_numpy(cl) for cl in cond_dist.chain_links)
    return dataclasses.replace(
        cond_dist,
        mixture_weights=_array_to_numpy(cond_dist.mixture_weights),
        components=new_components,
        samples_per_component=(),
        conditional_weights=_array_to_numpy(cond_dist.conditional_weights),
        cond_means=_array_to_numpy(cond_dist.cond_means),
        cond_chols=_array_to_numpy(cond_dist.cond_chols),
        chain_links=new_chain_links,
    )
