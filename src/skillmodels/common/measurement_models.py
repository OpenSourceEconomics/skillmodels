"""Shared measurement-family log-likelihood kernel.

One JAX-compatible kernel evaluates the log density/probability contribution of a
single measurement given its linear predictor ``eta = c + x'beta + lambda'theta``,
its scale, and a `MeasurementFamily` code. AF estimation, simulation and
posterior-state reweighting route their measurement contributions through this
kernel. CHS conditions a Gaussian predicted state on each observation instead
(`skillmodels.chs.halfline`), using the same families and bounds; AMN rejects
non-Gaussian measurements.

The families:

- ``GAUSSIAN`` -- ``log N(y; eta, sigma)``.
- ``PROBIT`` -- ``log Phi((2y-1) eta)`` for ``y in {0, 1}``; the latent error SD is
  fixed to 1, so ``sigma`` is ignored. Evaluated through `jax.scipy.special.log_ndtr`
  (never ``log(1 - ndtr)``) so the tails stay finite and stable.
- ``TOBIT`` -- censored normal with known bounds: the interior normal log density on
  ``L < y < U``, the left tail mass ``Phi((L-eta)/sigma)`` at ``y = L``, and the
  right tail mass ``Phi((eta-U)/sigma)`` at ``y = U``. A non-finite bound disables
  that side (no censoring there).
- ``ORDERED_PROBIT`` -- ``y in {0, ..., K-1}`` with latent ``y* = eta + e``,
  ``e ~ N(0, 1)``, and ``y = k`` iff ``c_k < y* <= c_{k+1}``, where ``c_0 = -inf``,
  ``c_K = +inf`` and ``c_1 = 0``. The free cutpoints ``c_2 < ... < c_{K-1}`` are
  parameters of the measurement (category `"cutpoints"`).
"""

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm

from skillmodels._beartype_conf import MODEL_SPEC_CONF, beartype_init


@dataclass(frozen=True)
class GaussianMeasurement:
    """Continuous measure with a free normal measurement-error SD (the default)."""


@dataclass(frozen=True)
class ProbitMeasurement:
    """Binary 0/1 measure with a standard-normal latent error (scale fixed to 1)."""


@dataclass(frozen=True)
class TobitMeasurement:
    """Censored-normal measure (not truncated, selected, or zero-inflated).

    `lower` / `upper` are the known censoring bounds; `None` disables that side.
    At least one bound must be finite. Observations equal to a bound are treated as
    censored (tail mass); interior observations use the normal density.
    """

    lower: float | None = 0.0
    """Lower censoring bound, or `None` for no lower censoring."""
    upper: float | None = None
    """Upper censoring bound, or `None` for no upper censoring."""

    def __post_init__(self) -> None:  # noqa: D105
        if self.lower is None and self.upper is None:
            msg = "TobitMeasurement needs at least one finite censoring bound."
            raise ValueError(msg)
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            msg = (
                f"TobitMeasurement lower bound ({self.lower}) must be strictly below "
                f"the upper bound ({self.upper})."
            )
            raise ValueError(msg)


@beartype_init(MODEL_SPEC_CONF)
@dataclass(frozen=True)
class OrderedProbitMeasurement:
    """Ordered categorical measure with integer codes `0, ..., n_categories - 1`.

    The latent index has a standard-normal error (scale fixed to 1). Category `k`
    is observed iff the index lies in `(c_k, c_{k+1}]`, with `c_0 = -inf`,
    `c_K = +inf` and the first cutpoint `c_1` fixed to 0, so the intercept stays
    free. With `n_categories=2` the model equals `ProbitMeasurement`.
    """

    n_categories: int
    """Number of categories `K`; at least 2."""

    def __post_init__(self) -> None:  # noqa: D105
        if isinstance(self.n_categories, bool) or not isinstance(
            self.n_categories, int
        ):
            msg = (
                "OrderedProbitMeasurement.n_categories must be an int, not "
                f"{type(self.n_categories).__name__}."
            )
            raise TypeError(msg)
        if self.n_categories < 2:
            msg = (
                "OrderedProbitMeasurement needs at least 2 categories, got "
                f"{self.n_categories}."
            )
            raise ValueError(msg)


# A measurement's observation model, attached per variable on `ModelSpec`.
MeasurementModel = (
    GaussianMeasurement
    | ProbitMeasurement
    | TobitMeasurement
    | OrderedProbitMeasurement
)


class MeasurementFamily(enum.IntEnum):
    """Observation model attached to a measurement variable.

    Integer-valued so the code can be stored in a JAX array aligned with the
    measurement system and compared inside traced/jitted code. A distinct name from
    `MeasurementType` (which marks state vs endogenous-factor augmented periods).
    """

    GAUSSIAN = 0
    PROBIT = 1
    TOBIT = 2
    ORDERED_PROBIT = 3


def resolve_measurement_family(
    model: MeasurementModel,
) -> tuple[MeasurementFamily, float, float]:
    """Map a public `MeasurementModel` to its internal `(family, lower, upper)`.

    `lower` / `upper` are the censoring bounds threaded to `measurement_loglik`;
    non-Tobit families and open Tobit sides use `-inf` / `+inf`.
    """
    if isinstance(model, ProbitMeasurement):
        return MeasurementFamily.PROBIT, -math.inf, math.inf
    if isinstance(model, TobitMeasurement):
        lower = -math.inf if model.lower is None else float(model.lower)
        upper = math.inf if model.upper is None else float(model.upper)
        return MeasurementFamily.TOBIT, lower, upper
    if isinstance(model, OrderedProbitMeasurement):
        return MeasurementFamily.ORDERED_PROBIT, -math.inf, math.inf
    return MeasurementFamily.GAUSSIAN, -math.inf, math.inf


def measurement_family_arrays(
    measurement_models: Mapping[str, MeasurementModel],
    measure_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(family_codes, lowers, uppers)` aligned to `measure_names`.

    A name absent from `measurement_models` resolves to Gaussian. The arrays line
    up row-for-row with a measurement system's loadings / SD arrays so the shared
    kernel can be vmapped over them. Each code is a `MeasurementFamily` integer.
    """
    resolved = [
        resolve_measurement_family(measurement_models.get(name, GaussianMeasurement()))
        for name in measure_names
    ]
    codes = np.array([int(family) for family, _lo, _hi in resolved], dtype=np.int64)
    lowers = np.array([lo for _f, lo, _hi in resolved], dtype=np.float64)
    uppers = np.array([hi for _f, _lo, hi in resolved], dtype=np.float64)
    return codes, lowers, uppers


class ObservationEvent(enum.IntEnum):
    """Event of one observed value of a measurement.

    Decided once on the host from the submitted float64 value and the declared
    float64 censoring bounds, so later dtype choices cannot change it:

    - `MISSING` -- the value is NaN.
    - `DENSITY` -- a continuous value: Gaussian, or Tobit strictly inside its bounds.
    - `LOWER_TAIL` -- the latent index lies below a threshold: probit `y = 0`
      (threshold 0), Tobit `y <= lower`, or ordered probit `y = 0` (below `c_1`).
    - `UPPER_TAIL` -- the latent index lies above a threshold: probit `y = 1`
      (threshold 0), Tobit `y >= upper`, or ordered probit `y = K - 1` (above
      `c_{K-1}`).
    - `INTERVAL` -- the latent index lies between two finite cutpoints: ordered
      probit `0 < y < K - 1`.
    """

    MISSING = 0
    DENSITY = 1
    LOWER_TAIL = 2
    UPPER_TAIL = 3
    INTERVAL = 4


def classify_observations(
    values: np.ndarray,
    family: MeasurementFamily,
    lower: float,
    upper: float,
    *,
    n_categories: int | None = None,
) -> np.ndarray:
    """Return the `ObservationEvent` code of each value of one measurement.

    Bounds are compared exactly, without tolerance, as in `measurement_loglik`:
    a Tobit value at or beyond a finite bound is censored there.

    Args:
        values: 1d float64 array of submitted values; NaN marks a missing value.
        family: The measurement's family.
        lower: Tobit lower censoring bound (`-inf` if none).
        upper: Tobit upper censoring bound (`+inf` if none).
        n_categories: Number of categories of an ordered-probit measurement.

    Return:
        1d int8 array of `ObservationEvent` codes.

    Raises:
        ValueError: If a probit value is not 0 or 1, an ordered-probit value is not
            an integer code in `0..n_categories-1`, or a limited-family value is
            infinite.

    """
    values = np.asarray(values, dtype=np.float64)
    missing = np.isnan(values)
    events = np.where(missing, ObservationEvent.MISSING, ObservationEvent.DENSITY)
    if family == MeasurementFamily.GAUSSIAN:
        events = np.where(np.isfinite(values), events, ObservationEvent.MISSING)
        return events.astype(np.int8)
    observed = values[~missing]
    if not np.isfinite(observed).all():
        name = family.name.lower().replace("_", " ")
        msg = f"Observed {name} measurements must be finite."
        raise ValueError(msg)
    if family == MeasurementFamily.PROBIT:
        invalid = sorted(set(observed[(observed != 0.0) & (observed != 1.0)].tolist()))
        if invalid:
            msg = f"A probit measurement must be 0, 1, or missing; found {invalid[:5]}."
            raise ValueError(msg)
        lower_tail = values == 0.0
        upper_tail = values == 1.0
    elif family == MeasurementFamily.ORDERED_PROBIT:
        if n_categories is None:
            msg = "An ordered probit measurement needs its number of categories."
            raise ValueError(msg)
        is_code = (observed == np.round(observed)) & (observed >= 0)
        is_code &= observed <= n_categories - 1
        invalid = sorted(set(observed[~is_code].tolist()))
        if invalid:
            msg = (
                "An ordered probit measurement must be an integer code in "
                f"0..{n_categories - 1} or missing; found {invalid[:5]}."
            )
            raise ValueError(msg)
        lower_tail = values == 0.0
        upper_tail = values == n_categories - 1
        events = np.where(
            ~missing & ~lower_tail & ~upper_tail, ObservationEvent.INTERVAL, events
        )
    else:
        lower_tail = values <= lower
        upper_tail = values >= upper
    events = np.where(lower_tail, ObservationEvent.LOWER_TAIL, events)
    events = np.where(upper_tail, ObservationEvent.UPPER_TAIL, events)
    return events.astype(np.int8)


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


# Vmapped over a 1-D measurement vector (all six arguments share axis 0). Use this
# form INSIDE another jitted/vmapped function (e.g. the AF integrand); it is not
# itself jitted, so it composes without nesting `jax.jit`.
measurement_loglik_vec = jax.vmap(measurement_loglik, in_axes=(0, 0, 0, 0, 0, 0))

# Standalone pre-jitted batch form for callers not already inside a jit.
measurement_loglik_batch = jax.jit(measurement_loglik_vec)
