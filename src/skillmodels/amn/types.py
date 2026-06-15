"""Frozen dataclass definitions for the AMN estimator.

Mirrors the structure of `skillmodels.af.types` for consistency. The
three-stage Attanasio-Meghir-Nix (2020) procedure produces a stack of
intermediate results (reduced-form mixture, structural recovery,
production-function regression); each stage's output is held in
`AMNStageResults`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from skillmodels._beartype_conf import OPTIONS_CONF, beartype_init
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.types import ensure_containers_are_immutable


@beartype_init(OPTIONS_CONF)
@dataclass(frozen=True, init=False)
class AMNEstimationOptions:
    """Configuration options for the AMN estimator."""

    em_max_iter: int
    """Maximum EM iterations in Stage 1."""

    em_tol: float
    """Log-likelihood tolerance for EM convergence."""

    em_n_init: int
    """Number of EM restarts; keep the highest-likelihood fit."""

    em_reg_covar: float
    """Diagonal ridge added to each EM covariance for numerical stability."""

    n_simulation_draws: int
    """Synthetic latent-factor panel size for Stage 3."""

    minimum_distance_weighting: Literal["identity", "optimal"]
    """Stage 2 minimum-distance weighting. `"identity"` (the paper's default
    and currently the only implemented option) is an unweighted identity metric
    over per-component means and full covariance matrices. `"optimal"` is
    reserved for a future 2-step Avar-weighted criterion and currently raises
    `NotImplementedError`."""

    allow_ces_overnormalization: bool
    """Opt out of the CES minimal-normalization guard. When True, extra
    normalized CES loadings are treated as a deliberate fixed-loadings
    analysis rather than a (Freyberger-adaptation-defeating) error."""

    optimizer_algorithm: str
    """optimagic algorithm name for Stage 2 minimum-distance optimization."""

    optimizer_options: MappingProxyType[str, Any]
    """Additional kwargs forwarded to optimagic in Stage 2."""

    keep_synthetic_panel: bool
    """Retain the Stage-3 simulated panel on the result for diagnostics. Off
    by default to keep result objects compact."""

    seed: int
    """RNG seed used for Stage 3 simulation and bootstrap inference."""

    def __init__(  # noqa: D107
        self,
        em_max_iter: int = 500,
        em_tol: float = 1e-6,
        em_n_init: int = 5,
        em_reg_covar: float = 1e-6,
        n_simulation_draws: int = 100_000,
        minimum_distance_weighting: Literal["identity", "optimal"] = "identity",
        optimizer_algorithm: str = "scipy_lbfgsb",
        optimizer_options: Mapping[str, Any] | None = None,
        *,
        allow_ces_overnormalization: bool = False,
        keep_synthetic_panel: bool = False,
        seed: int = 0,
    ) -> None:
        object.__setattr__(self, "em_max_iter", em_max_iter)
        object.__setattr__(self, "em_tol", em_tol)
        object.__setattr__(self, "em_n_init", em_n_init)
        object.__setattr__(self, "em_reg_covar", em_reg_covar)
        object.__setattr__(self, "n_simulation_draws", n_simulation_draws)
        object.__setattr__(
            self, "minimum_distance_weighting", minimum_distance_weighting
        )
        object.__setattr__(
            self, "allow_ces_overnormalization", allow_ces_overnormalization
        )
        object.__setattr__(self, "optimizer_algorithm", optimizer_algorithm)
        object.__setattr__(
            self,
            "optimizer_options",
            ensure_containers_are_immutable(optimizer_options or {}),
        )
        object.__setattr__(self, "keep_synthetic_panel", keep_synthetic_panel)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True)
class AugmentedMeasureLayout:
    """Index bookkeeping for the augmented measure vector.

    AMN Stage 1 fits a Gaussian mixture on the joint vector of:
    1. Factor measurements at each period (have measurement error),
    2. Observed factor values at each period (no measurement error,
       loading fixed at 1, intercept free),
    3. Controls (time-invariant, no measurement error).

    The layout records which slot in the stacked vector corresponds to
    which conceptual quantity, so Stage 2 can map the fitted Pi/Psi back
    onto the structural Lambda/A/Sigma/mu/Omega.
    """

    columns: tuple[str, ...]
    """Human-readable label per augmented-vector column."""

    measurement_slots: tuple[int, ...]
    """Indices of slots that correspond to factor measurements (with
    measurement error). One per (period, measurement) update."""

    observed_factor_slots: tuple[int, ...]
    """Indices of slots that correspond to observed factor values (no
    measurement error). One per (period, observed factor)."""

    control_slots: tuple[int, ...]
    """Indices of slots that correspond to controls (no measurement
    error)."""

    measurement_meta: tuple[tuple[int, str, str], ...]
    """For each measurement slot: (period, factor_name, measurement_name)."""

    observed_factor_meta: tuple[tuple[int, str], ...]
    """For each observed-factor slot: (period, observed_factor_name)."""

    control_meta: tuple[str, ...]
    """Control name for each control slot."""


@dataclass(frozen=True)
class MixtureFitResult:
    """Output of Stage 1: reduced-form mixture parameters.

    The fitted distribution is
    ``sum_k weights[k] * Normal(means[k], covariances[k])`` on the
    augmented measure vector. Matches AMN eq. (11)-(14).
    """

    weights: np.ndarray
    """Mixture weights, shape ``(n_components,)``."""

    means: np.ndarray
    """Per-component mean vectors, shape ``(n_components, n_aug)``."""

    covariances: np.ndarray
    """Per-component covariance matrices, shape
    ``(n_components, n_aug, n_aug)``."""

    loglikelihood: float
    """Final EM log-likelihood (summed across observations)."""

    n_iter: int
    """EM iterations run by the best restart."""

    converged: bool
    """Whether the best restart converged within `em_tol`."""

    layout: AugmentedMeasureLayout
    """Slot bookkeeping for the augmented measure vector this mixture was
    fit on."""


@dataclass(frozen=True)
class MinimumDistanceResult:
    """Output of Stage 2: structural parameters from the reduced-form mixture.

    All arrays are in the standard skillmodels ordering established by
    `process_model.process_model`.
    """

    loadings: pd.DataFrame
    """Recovered factor loadings, MultiIndexed by (period, measurement,
    factor)."""

    measurement_intercepts: pd.DataFrame
    """Recovered measurement intercepts, MultiIndexed by (period,
    measurement, control)."""

    measurement_sds: pd.DataFrame
    """Recovered measurement-error SDs, MultiIndexed by (period,
    measurement)."""

    factor_mixture_means: np.ndarray
    """Per-component means of the latent factors stacked across periods,
    shape ``(n_components, n_factor_period_slots)``."""

    factor_mixture_covariances: np.ndarray
    """Per-component covariances of the same stacked factor vector, shape
    ``(n_components, n_factor_period_slots, n_factor_period_slots)``."""

    factor_period_slots: tuple[tuple[int, str], ...]
    """Ordered ``(period, factor_name)`` for the
    ``factor_mixture_*`` arrays."""

    objective_value: float
    """Minimum-distance criterion at the optimum."""

    success: bool
    """Whether the Stage-2 optimization converged."""


@dataclass(frozen=True)
class ProductionFitResult:
    """Output of Stage 3: production-function and investment-equation params.

    Fitted by regression on a simulated latent-factor panel; see AMN 2020
    eqs. 4-5, 7-8.
    """

    production_params: pd.DataFrame
    """Production-function parameters, in the standard skillmodels
    params-DataFrame format (4-level MultiIndex)."""

    investment_params: pd.DataFrame
    """Investment-equation parameters (eq. 7), 4-level MultiIndex. Populated
    under the control-function correction (a `CorrectionSpec` on the endogenous
    investment factor) with the first-stage `investment_eq` coefficients and
    `investment_sds` residual SD per investment factor and period; empty
    otherwise. When the correction runs, each state factor's production shock
    SD (`shock_sds`) is the corrected SD(eps_C) and the production block gains
    a `cf` row carrying kappa_t."""

    n_draws: int
    """Number of simulated latent-factor trajectories used."""

    seed: int
    """RNG seed used for the simulation."""


@dataclass(frozen=True)
class AMNStageResults:
    """Container for the three stages' intermediate outputs."""

    mixture: MixtureFitResult
    """Stage 1 reduced-form mixture fit."""

    structural: MinimumDistanceResult
    """Stage 2 structural recovery."""

    production: ProductionFitResult
    """Stage 3 production-function regression."""


@dataclass(frozen=True)
class AMNEstimationResult:
    """Complete result from AMN estimation."""

    model_spec: ModelSpec
    """The ModelSpec used for estimation."""

    stages: AMNStageResults
    """Per-stage intermediate outputs."""

    params: pd.DataFrame
    """Combined parameters across stages, in the standard 4-level
    MultiIndex (category, period, name1, name2) format consumed by every
    other skillmodels entry point."""

    success: bool
    """AND across stage convergence flags."""

    md_criterion: float
    """Stage-2 minimum-distance criterion at the optimum (AMN's objective).
    Conforms to `skillmodels.common.estimation.CommonEstimationResult`."""

    loglikelihood: float | None = None
    """Always `None` for AMN (minimum-distance, not likelihood); present to
    satisfy the common result Protocol."""

    synthetic_panel: pd.DataFrame | None = None
    """Stage-3 simulated factor panel, kept iff
    `AMNEstimationOptions.keep_synthetic_panel` is True."""


@dataclass(frozen=True)
class AMNInferenceResult:
    """Cluster-bootstrap standard errors and covariance for AMN params."""

    standard_errors: pd.Series
    """std across replicate_params, indexed by the params MultiIndex."""

    vcov: pd.DataFrame
    """cov(replicate_params), MultiIndexed on both axes."""

    replicate_params: pd.DataFrame
    """One row per bootstrap replicate, columns = params MultiIndex."""

    n_clusters: int
    """Caseids resampled per replicate."""

    n_boot: int
    """Number of bootstrap replicates."""
