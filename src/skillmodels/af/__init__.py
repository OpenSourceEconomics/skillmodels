"""Antweiler-Freyberger estimator for latent factor models.

Iterative period-by-period MLE with Halton quadrature for numerical
integration, following Antweiler and Freyberger (2025).
"""

from skillmodels.af.estimate import estimate_af
from skillmodels.af.inference import (
    AFInferenceResult,
    compute_af_standard_errors,
)
from skillmodels.af.posterior_states import get_af_posterior_states
from skillmodels.af.types import AFEstimationOptions, AFEstimationResult, AFPeriodResult

__all__ = [
    "AFEstimationOptions",
    "AFEstimationResult",
    "AFInferenceResult",
    "AFPeriodResult",
    "compute_af_standard_errors",
    "estimate_af",
    "get_af_posterior_states",
]
