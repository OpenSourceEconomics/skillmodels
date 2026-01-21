"""Skillmodels: A Python package for estimating latent factor models."""

import contextlib

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

from skillmodels.filtered_states import get_filtered_states
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.model_spec import (
    AnchoringSpec,
    EstimationOptionsSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.simulate_data import simulate_dataset

__all__ = [
    "AnchoringSpec",
    "EstimationOptionsSpec",
    "FactorSpec",
    "ModelSpec",
    "Normalizations",
    "get_filtered_states",
    "get_maximization_inputs",
    "simulate_dataset",
]
