"""Auto-sizing helpers for the AF likelihood's memory-aware batching.

The AF likelihood replaces the outermost ``jax.vmap`` over observations
with ``jax.lax.map`` when ``n_obs_per_batch`` is smaller than ``n_obs``.
This module provides a simple heuristic that picks an ``n_obs_per_batch``
from a target-bytes budget, mirroring pylcm's approach (see
``pylcm/src/lcm/simulation/initial_conditions.py:547-560``).

The heuristic is intentionally crude: it multiplies the per-observation
Halton grid footprint by a safety factor and divides a budget (256 MB by
default, overridable via the ``SKILLMODELS_AF_TARGET_BATCH_BYTES``
environment variable) by that product. No GPU-specific probing is done;
users who need tighter control can set ``n_obs_per_batch`` explicitly on
``AFEstimationOptions``.
"""

import logging
import os

_DEFAULT_TARGET_BATCH_BYTES = 2**28  # 256 MB
_ENV_VAR_TARGET = "SKILLMODELS_AF_TARGET_BATCH_BYTES"
_BYTES_PER_FLOAT64 = 8

# Empirical multiplier reflecting that a single observation's forward +
# backward tape at full state/shock/inv_shock resolution retains several
# copies of the integrand footprint. This sized conservatively high: a
# smaller batch is always safe, a larger batch can OOM.
_SAFETY_FACTOR = 16

logger = logging.getLogger(__name__)


def target_batch_bytes() -> int:
    """Return the bytes budget per observation batch.

    Honours ``SKILLMODELS_AF_TARGET_BATCH_BYTES`` when set to a positive
    integer, otherwise returns the default 256 MB budget.
    """
    override = os.environ.get(_ENV_VAR_TARGET)
    if override is None:
        return _DEFAULT_TARGET_BATCH_BYTES
    try:
        parsed = int(override)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: not a valid integer.",
            _ENV_VAR_TARGET,
            override,
        )
        return _DEFAULT_TARGET_BATCH_BYTES
    if parsed <= 0:
        logger.warning("Ignoring %s=%r: must be positive.", _ENV_VAR_TARGET, override)
        return _DEFAULT_TARGET_BATCH_BYTES
    return parsed


def auto_n_obs_per_batch(
    *,
    n_obs: int,
    n_halton_points: int,
    n_halton_points_shock: int,  # noqa: ARG001
    n_latent: int,
    n_endogenous: int,
    target_bytes: int | None = None,
) -> int:
    """Pick ``n_obs_per_batch`` from a target-bytes budget.

    The AF transition-period likelihood forms a joint Halton draw of
    size ``(n_halton_points, 2 * n_latent + n_endogenous)`` rather than
    an outer product of per-axis grids, so per-observation memory is
    linear in ``n_halton_points``. The per-observation footprint is
    estimated as

    ``n_halton_points * (n_latent + n_endogenous + 1) * 8 * SAFETY_FACTOR``.

    ``n_halton_points_shock`` is retained in the signature for API
    compatibility with the earlier per-axis layout but is unused now
    that draws are joint.

    Args:
        n_obs: Total number of observations.
        n_halton_points: Halton grid size (joint dimension count unused here).
        n_halton_points_shock: Legacy shock Halton count, ignored.
        n_latent: Latent factor count.
        n_endogenous: Endogenous (investment) factor count.
        target_bytes: Budget per batch. Defaults to `target_batch_bytes()`.

    Return:
        A positive integer no larger than ``n_obs``.
    """
    budget = target_bytes if target_bytes is not None else target_batch_bytes()
    per_obs_bytes = (
        n_halton_points
        * (n_latent + n_endogenous + 1)
        * _BYTES_PER_FLOAT64
        * _SAFETY_FACTOR
    )
    per_obs_bytes = max(per_obs_bytes, 1)
    batch = max(1, budget // per_obs_bytes)
    return min(batch, n_obs)
