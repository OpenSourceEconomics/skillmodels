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
(`get_maximization_inputs`, `get_filtered_states`, `create_state_ranges`)
so most callers don't need to touch the `chs.` prefix.
"""

from skillmodels.chs.filtered_states import get_filtered_states
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.process_debug_data import (
    create_state_ranges,
    process_debug_data,
)

__all__ = [
    "create_state_ranges",
    "get_filtered_states",
    "get_maximization_inputs",
    "process_debug_data",
]
