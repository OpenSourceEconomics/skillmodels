"""Skillmodels: A Python package for estimating latent factor models."""

# Enable 64-bit JAX before any skillmodels submodule -- and crucially before
# any transitive `import jaxopt` -- so jaxopt's module-level jit/sort
# kernels see int64 as the default integer type. Without this, jaxopt's
# `argsort` inside `LBFGSB.update` emits an `s32` accumulator into an
# `s64` scatter operand and XLA's permutation_sort_simplifier verifier
# rejects it on JAX >= 0.10 / cuda13. The package has always assumed
# x64 (every CHS / AF / AMN entry point sets it inside the function);
# centralising it at import time fixes the jaxopt path too and is a
# no-op for callers who already enable it.
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import contextlib

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
