"""CHS (Cunha-Heckman-Schennach 2010) Kalman-filter MLE estimator.

This subpackage holds the state-space machinery that powers the
default skillmodels estimator:

* `kalman_filters` — square-root unscented and extended Kalman filter
  predict/update steps.
* `likelihood` (`+ `_debug`) — Kalman-filter log-likelihood.
* `estimate_chs` — one-call driver wrapping `get_maximization_inputs`
  and `optimagic.maximize`, returning a `CHSEstimationResult`.
* `maximization_inputs` — `get_maximization_inputs()`, the canonical
  power-user entry point that bundles likelihood / gradients /
  constraints / params template for `optimagic.maximize`.
* `process_debug_data` — Kalman-debug-output post-processing.
* `qr`, `clipping` — numerical helpers (square-root QR, soft clipping
  for UKF stability).

The public top-level package re-exports the user-facing entry points
(`estimate_chs`, `get_maximization_inputs`) so most callers don't need
to touch the `chs.` prefix. Cross-estimator state extraction
(`get_individual_states`) and the estimator-agnostic
`create_state_ranges` live under `skillmodels.common`.
"""

from skillmodels.chs.estimate import estimate_chs
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.process_debug_data import process_debug_data
from skillmodels.chs.types import CHSEstimationResult

__all__ = [
    "CHSEstimationOptions",
    "CHSEstimationResult",
    "estimate_chs",
    "get_maximization_inputs",
    "process_debug_data",
]
