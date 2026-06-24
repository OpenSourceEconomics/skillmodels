"""Shared measurement-family log-likelihood kernel.

One JAX-compatible kernel evaluates the log density/probability contribution of a
single measurement given its linear predictor ``eta = c + x'beta + lambda'theta``,
its scale, and a `MeasurementFamily` code. AF estimation, CHS, simulation and
posterior-state reweighting all route their measurement contributions through this
kernel, so the measurement system is defined in exactly one place.

The three families:

- ``GAUSSIAN`` -- ``log N(y; eta, sigma)``.
- ``PROBIT`` -- ``log Phi((2y-1) eta)`` for ``y in {0, 1}``; the latent error SD is
  fixed to 1, so ``sigma`` is ignored. Evaluated through `jax.scipy.special.log_ndtr`
  (never ``log(1 - ndtr)``) so the tails stay finite and stable.
- ``TOBIT`` -- censored normal with known bounds: the interior normal log density on
  ``L < y < U``, the left tail mass ``Phi((L-eta)/sigma)`` at ``y = L``, and the
  right tail mass ``Phi((eta-U)/sigma)`` at ``y = U``. A non-finite bound disables
  that side (no censoring there).
"""

import enum

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm


class MeasurementFamily(enum.IntEnum):
    """Observation model attached to a measurement variable.

    Integer-valued so the code can be stored in a JAX array aligned with the
    measurement system and compared inside traced/jitted code. A distinct name from
    `MeasurementType` (which marks state vs endogenous-factor augmented periods).
    """

    GAUSSIAN = 0
    PROBIT = 1
    TOBIT = 2


def measurement_loglik(
    y: Array,
    eta: Array,
    sigma: Array,
    family: Array,
    lower: Array,
    upper: Array,
) -> Array:
    """Return the log contribution of one measurement under its family.

    Args:
        y: Observed measurement value (0/1 for probit; a censored value for Tobit).
        eta: Linear predictor `c + x'beta + lambda'theta`.
        sigma: Measurement scale (ignored for probit, which fixes it to 1).
        family: A `MeasurementFamily` integer code.
        lower: Tobit lower censoring bound (`-inf` to disable).
        upper: Tobit upper censoring bound (`+inf` to disable).

    Return:
        The scalar log density (Gaussian/Tobit interior) or log probability
        (probit, Tobit tail) of the measurement.

    """
    gaussian = norm.logpdf(y, loc=eta, scale=sigma)
    probit = log_ndtr((2.0 * y - 1.0) * eta)
    tobit = _tobit_loglik(y, eta, sigma, lower, upper)
    return jnp.where(
        family == int(MeasurementFamily.PROBIT),
        probit,
        jnp.where(family == int(MeasurementFamily.TOBIT), tobit, gaussian),
    )


def _tobit_loglik(
    y: Array, eta: Array, sigma: Array, lower: Array, upper: Array
) -> Array:
    """Censored-normal log contribution with safe-gradient handling of inf bounds.

    A non-finite bound is replaced by a finite sentinel before any arithmetic so the
    masked-out tail term carries no `inf`/`NaN` into `jnp.where` (whose gradient
    would otherwise be poisoned), then masked out by the finiteness check.
    """
    interior = norm.logpdf(y, loc=eta, scale=sigma)

    lower_finite = jnp.isfinite(lower)
    upper_finite = jnp.isfinite(upper)
    safe_lower = jnp.where(lower_finite, lower, 0.0)
    safe_upper = jnp.where(upper_finite, upper, 0.0)

    left = log_ndtr((safe_lower - eta) / sigma)
    right = log_ndtr((eta - safe_upper) / sigma)

    at_lower = lower_finite & (y <= lower)
    at_upper = upper_finite & (y >= upper)
    return jnp.where(at_lower, left, jnp.where(at_upper, right, interior))


# Pre-jitted, vmapped batch form over a 1-D measurement vector. Callers that already
# operate inside their own jit/vmap can use `measurement_loglik` directly.
measurement_loglik_batch = jax.jit(
    jax.vmap(measurement_loglik, in_axes=(0, 0, 0, 0, 0, 0))
)
