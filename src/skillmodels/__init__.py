"""Skillmodels: A Python package for estimating latent factor models."""

import contextlib

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

from skillmodels.af import (
    AFEstimationOptions,
    AFEstimationResult,
    AFInferenceResult,
    compute_af_standard_errors,
    estimate_af,
)
from skillmodels.diagnostic_plots import (
    plot_likelihood_contributions,
    plot_residual_boxplots,
)
from skillmodels.filtered_states import get_filtered_states
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.model_spec import (
    AnchoringSpec,
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.simulate_data import simulate_dataset, simulate_policy_effect
from skillmodels.variance_decomposition import (
    decompose_measurement_variance,
    summarize_measurement_reliability,
)

__all__ = [
    "AFEstimationOptions",
    "AFEstimationResult",
    "AFInferenceResult",
    "AnchoringSpec",
    "EstimationOptions",
    "FactorSpec",
    "ModelSpec",
    "Normalizations",
    "compute_af_standard_errors",
    "create_state_ranges",
    "decompose_measurement_variance",
    "estimate_af",
    "get_filtered_states",
    "get_maximization_inputs",
    "plot_likelihood_contributions",
    "plot_residual_boxplots",
    "simulate_dataset",
    "simulate_policy_effect",
    "summarize_measurement_reliability",
]
