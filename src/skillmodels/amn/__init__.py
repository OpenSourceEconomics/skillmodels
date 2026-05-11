"""AMN: Attanasio-Meghir-Nix (2020) latent factor estimator (and start values).

This package exposes two distinct surfaces:

1. **Start-value helpers** -- the Spearman cross-covariance moments
   (`spearman_factor_moments`) and Bartlett-score OLS
   (`seed_beta_from_ols`) that seed every estimator's starting values
   (`get_moment_based_start_params`, used by CHS and AF).

2. **Full AMN estimator** -- a three-stage mixture-EM /
   minimum-distance / simulate-and-regress procedure mirroring AMN 2020,
   plus bootstrap inference and a per-observation posterior-state helper
   for diagnostic plots.

Public API:

* Start-value helpers: `spearman_factor_moments`, `derive_unexplained_sd`,
  `seed_beta_from_ols`, `SpearmanResult`, `get_moment_based_start_params`,
  `pool_equality_groups`.
* AMN estimator: `estimate_amn`, `compute_amn_standard_errors`,
  `get_amn_posterior_states`, `AMNEstimationOptions`,
  `AMNEstimationResult`, `AMNInferenceResult`, `AMNStageResults`.
* Stage 1 building blocks (for testing / advanced use):
  `fit_mixture_em`, `build_augmented_measure_layout`,
  `build_augmented_measure_matrix`, `MixtureFitResult`,
  `AugmentedMeasureLayout`.
"""

from skillmodels.amn.estimate import estimate_amn
from skillmodels.amn.inference import compute_amn_standard_errors
from skillmodels.amn.mixture_em import (
    build_augmented_measure_layout,
    build_augmented_measure_matrix,
    fit_mixture_em,
)
from skillmodels.amn.moments import (
    SpearmanResult,
    derive_unexplained_sd,
    seed_beta_from_ols,
    spearman_factor_moments,
)
from skillmodels.amn.posterior_states import get_amn_posterior_states
from skillmodels.amn.start_values import (
    get_moment_based_start_params,
    pool_equality_groups,
)
from skillmodels.amn.types import (
    AMNEstimationOptions,
    AMNEstimationResult,
    AMNInferenceResult,
    AMNStageResults,
    AugmentedMeasureLayout,
    MinimumDistanceResult,
    MixtureFitResult,
    ProductionFitResult,
)

__all__ = [
    "AMNEstimationOptions",
    "AMNEstimationResult",
    "AMNInferenceResult",
    "AMNStageResults",
    "AugmentedMeasureLayout",
    "MinimumDistanceResult",
    "MixtureFitResult",
    "ProductionFitResult",
    "SpearmanResult",
    "build_augmented_measure_layout",
    "build_augmented_measure_matrix",
    "compute_amn_standard_errors",
    "derive_unexplained_sd",
    "estimate_amn",
    "fit_mixture_em",
    "get_amn_posterior_states",
    "get_moment_based_start_params",
    "pool_equality_groups",
    "seed_beta_from_ols",
    "spearman_factor_moments",
]
