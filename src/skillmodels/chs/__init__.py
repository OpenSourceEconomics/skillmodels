"""CHS (Cunha-Heckman-Schennach 2010) Kalman-filter MLE estimator.

This subpackage holds the state-space machinery that powers the
default skillmodels estimator:

* `kalman_filters` — square-root unscented and extended Kalman filter
  predict/update steps.
* `likelihood` (`+ `_debug`) — Kalman-filter log-likelihood.
* `maximization_inputs` — `get_maximization_inputs()`, the canonical
  entry point that bundles likelihood / gradients / constraints /
  params template for `optimagic.maximize`.
* `filtered_states` — `get_filtered_states()` post-estimation helper.
* `process_debug_data` — Kalman-debug-output post-processing.
* `qr`, `clipping` — numerical helpers (square-root QR, soft clipping
  for UKF stability).

The public top-level package re-exports the user-facing entry points
(`get_maximization_inputs`, `get_filtered_states`) so most callers
don't need to touch the `chs.` prefix. The estimator-agnostic
`create_state_ranges` lives under `skillmodels.common.state_ranges`.
"""

from skillmodels.chs.filtered_states import get_filtered_states
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.process_debug_data import process_debug_data
from skillmodels.common.types import CHSEstimationOptions

__all__ = [
    "CHSEstimationOptions",
    "get_filtered_states",
    "get_maximization_inputs",
    "process_debug_data",
]
