"""Skillmodels: A Python package for estimating latent factor models."""

# Enable 64-bit JAX before any skillmodels submodule. Every CHS / AF / AMN
# entry point already sets this inside its function body; centralising it
# here makes the package behave consistently for direct callers.
import contextlib
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

with contextlib.suppress(ImportError):
    import pdbp  # noqa: F401

from skillmodels.common.model_spec import (  # noqa: E402
    AnchoringSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)

__all__ = [
    "AnchoringSpec",
    "FactorSpec",
    "ModelSpec",
    "Normalizations",
]
