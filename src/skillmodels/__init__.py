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

from skillmodels.af.estimate import estimate_af  # noqa: E402
from skillmodels.af.types import AFEstimationOptions, AFEstimationResult  # noqa: E402
from skillmodels.amn.estimate import estimate_amn  # noqa: E402
from skillmodels.amn.types import (  # noqa: E402
    AMNEstimationOptions,
    AMNEstimationResult,
)
from skillmodels.chs.estimate import estimate_chs  # noqa: E402
from skillmodels.chs.maximization_inputs import get_maximization_inputs  # noqa: E402
from skillmodels.chs.options import CHSEstimationOptions  # noqa: E402
from skillmodels.chs.types import CHSEstimationResult  # noqa: E402
from skillmodels.common.control_function import generate_kappa_terms  # noqa: E402
from skillmodels.common.estimation import (  # noqa: E402
    CommonEstimationOptions,
    CommonEstimationResult,
)
from skillmodels.common.individual_states import (  # noqa: E402
    get_individual_states,
    get_individual_states_from_params,
)
from skillmodels.common.model_spec import (  # noqa: E402
    AnchoringSpec,
    CorrectionSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)

__all__ = [
    "AFEstimationOptions",
    "AFEstimationResult",
    "AMNEstimationOptions",
    "AMNEstimationResult",
    "AnchoringSpec",
    "CHSEstimationOptions",
    "CHSEstimationResult",
    "CommonEstimationOptions",
    "CommonEstimationResult",
    "CorrectionSpec",
    "FactorSpec",
    "ModelSpec",
    "Normalizations",
    "estimate_af",
    "estimate_amn",
    "estimate_chs",
    "generate_kappa_terms",
    "get_individual_states",
    "get_individual_states_from_params",
    "get_maximization_inputs",
]
