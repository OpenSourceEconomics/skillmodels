"""Skillmodels: A Python package for estimating latent factor models."""

# Enable 64-bit JAX before any skillmodels submodule. Every CHS / AF / AMN
# entry point already sets this inside its function body; centralising it
# here makes the package behave consistently for direct callers.
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

# Workaround for a JAX 0.10 XLA bug surfaced by jaxopt's `LBFGSB.update`.
# The `permutation_sort_simplifier` HLO pass mis-lowers the `argsort`
# inside `update`: it emits an s32 reduction accumulator into the s64
# scatter operand built by the rest of the optimizer, and the HLO
# verifier rejects the resulting mismatch with `INVALID_ARGUMENT:
# Reduction function's accumulator shape at index 0 differs from the
# init_value shape: s32[] vs s64[]`. Disabling just that one pass via
# `XLA_FLAGS` keeps every other XLA optimisation intact and is a no-op
# on JAX < 0.10 (pre-0.10 lacks the pass). Must be set *before* `import
# jax` because XLA reads `XLA_FLAGS` once at backend init.
_xla_pass_disable = "--xla_disable_hlo_passes=permutation_sort_simplifier"  # noqa: S105
_existing_xla_flags = os.environ.get("XLA_FLAGS", "")
if _xla_pass_disable not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = f"{_existing_xla_flags} {_xla_pass_disable}".strip()

import contextlib  # noqa: E402

import jax  # noqa: E402

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
