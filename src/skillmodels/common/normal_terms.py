"""Tail and interval terms of the standard normal distribution.

Shared by the CHS limited-measurement update (`skillmodels.chs.halfline`) and the
conditional measurement kernel (`skillmodels.common.measurement_models`). All
functions are elementwise in float64 and differentiable, with finite values and
derivatives far in the tails and for arbitrarily narrow intervals.
"""

import jax
import jax.numpy as jnp
from jax import Array

LOG_SQRT_2PI = 0.91893853320467274178
_TAIL_SWITCH = -8.0
_CENTRAL_CAP = 40.0
_HALF_RANGE = 2.0**1023
_CONTINUED_FRACTION_DEPTH = 24
_NARROW_SWITCH = 0.25
_SERIES_ORDER = 21


def log_normal_interval_mass(lower: Array, upper: Array) -> Array:
    """Return `log(Phi(upper) - Phi(lower))` for `lower < upper`; ends may be infinite.

    - Both ends finite: the log mass of `normal_interval_terms`, which integrates the
      density over a narrow interval instead of subtracting endpoint log-CDFs, so
      every representable interval keeps its finite mass and endpoint derivatives.
    - `lower = -inf`: `log Phi(upper)`.
    - `upper = +inf`: `log Phi(-lower)`.

    Non-finite ends are replaced by finite placeholders before any arithmetic, so the
    value and the gradient stay finite; the gradient with respect to an infinite end
    is 0.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    lower_finite = jnp.isfinite(lower)
    upper_finite = jnp.isfinite(upper)
    both_finite = lower_finite & upper_finite
    interval = normal_interval_terms(
        jnp.where(both_finite, lower, -1.0), jnp.where(both_finite, upper, 1.0)
    )[0]
    below_upper = jax.scipy.special.log_ndtr(jnp.where(upper_finite, upper, 0.0))
    above_lower = jax.scipy.special.log_ndtr(-jnp.where(lower_finite, lower, 0.0))
    return jnp.where(
        both_finite,
        interval,
        jnp.where(upper_finite, below_upper, jnp.where(lower_finite, above_lower, 0.0)),
    )


def normal_interval_terms(
    lower: Array, upper: Array
) -> tuple[Array, Array, Array, Array]:
    """Return `log Z`, `E[z]`, `d = 1 - Var[z]` and `Var[z]` of a truncated normal.

    `z ~ N(0, 1)` is truncated to `(lower, upper]` with finite `lower < upper`, and
    `Z = Phi(upper) - Phi(lower)`. Elementwise in float64. The interval is reflected
    to `(a, b] = (-upper, -lower]` when `lower + upper > 0` (which flips the sign of
    the mean), so that `a + b <= 0`. With `c = (a + b) / 2` and `h = (b - a) / 2`:

    - `h max(1, |c|) <= 0.25`: a Hermite series around `c` in the half-width,
      `Z = phi(c) * 2h * E_U[g]` with `g(u) = exp(-c u - u^2 / 2)` and `u` uniform on
      `(-h, h]`, together with `E_U[u g]` and `E_U[u^2 g]`, which keeps `Var[z]`
      accurate for narrow intervals;
    - otherwise: the interval as the difference of the half-lines below `b` and `a`,
      with `q = Phi(a) / Phi(b)` and the half-line terms of `normal_tail_terms`:
      `Z = Phi(b) (1 - q)`, `E[z] = (q r_a - r_b) / (1 - q)` and
      `Var[z] = (V_b - q V_a) / (1 - q) - q (r_b - r_a)^2 / (1 - q)^2`, where `r_x`
      and `V_x = 1 - d(x)` are the Mills ratio and the variance below `x`. All
      terms stay finite and accurate far in the tails.

    The interval is the one between the two given ends; `source_interval_terms`
    evaluates an interval given by source values and a location and scale instead.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    flip = lower + upper > 0.0
    a = jnp.where(flip, -upper, lower)
    b = jnp.where(flip, -lower, upper)
    centre = 0.5 * (a + b)
    half = 0.5 * (b - a)
    narrow = half * jnp.maximum(1.0, -centre) <= _NARROW_SWITCH

    log_z_n, mean_n, var_n = _narrow_interval_moments(
        jnp.where(narrow, centre, 0.0), jnp.where(narrow, half, 0.1)
    )

    a_w = jnp.where(narrow, -1.0, a)
    b_w = jnp.where(narrow, 1.0, b)
    log_cdf_a, mills_a, _, var_a = normal_tail_terms(a_w)
    log_cdf_b, mills_b, _, var_b = normal_tail_terms(b_w)
    one_minus_q = -jnp.expm1(log_cdf_a - log_cdf_b)
    q = jnp.exp(log_cdf_a - log_cdf_b)
    log_z_w = log_cdf_b + jnp.log(one_minus_q)
    mean_w = (q * mills_a - mills_b) / one_minus_q
    spread = (mills_b - mills_a) / one_minus_q
    var_w = (var_b - q * var_a) / one_minus_q - q * spread * spread

    log_z = jnp.where(narrow, log_z_n, log_z_w)
    mean = jnp.where(narrow, mean_n, mean_w)
    var = jnp.where(narrow, var_n, var_w)
    return log_z, jnp.where(flip, -mean, mean), 1.0 - var, var


def source_interval_terms(
    lower: Array, upper: Array, location: Array, scale: Array
) -> tuple[Array, Array, Array, Array]:
    """Return `log Z`, `E[z]`, `d = 1 - Var[z]` and `Var[z]` of a source interval.

    `y ~ N(location, scale^2)` is truncated to `(lower, upper]` with finite
    `lower < upper`, finite `location` and finite `scale > 0`; `z = (y - location) /
    scale`. Elementwise in float64. The interval is represented by its standardized
    ends `a = (lower - location) / scale`, `b = (upper - location) / scale` and its
    log width `log(upper - lower) - log(scale)`, each formed from the source values
    with one rounding per operation and without an intermediate that can overflow
    or underflow (`_scaled_difference`, `_log_difference`); in particular the
    standardized width itself is never formed. After the reflection of
    `normal_interval_terms`, with `c = (a + b) / 2` and `h = w / 2` from the log
    width `log w`:

    - `h max(1, |c|) <= 0.25`: the Hermite series of `_narrow_interval_moments`
      around `c`, with the log width in `log Z`;
    - otherwise: the difference of the half-lines below `b` and `a` as in
      `normal_interval_terms`, with `log q = log Phi(a) - log Phi(b)` evaluated as
      `c w + log r(b) - log r(a)` (`r` the Mills ratio): the leading term
      `(b^2 - a^2) / 2 = c w` comes from the centre and the log width, and the log
      Mills ratios vary slowly in their arguments, so `log q` keeps the span of the
      source interval when the two rounded standardized ends coincide (there a
      difference of their log-CDFs would be 0).

    The two branches agree to rounding at their boundary. Values and derivatives are
    finite whenever the correctly rounded log mass is a finite float, that is for
    `|c|` up to about `2^512`; beyond, `log Z` is `-inf`.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    location = jnp.asarray(location, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    a = _scaled_difference(lower, location, scale)
    b = _scaled_difference(upper, location, scale)
    log_width = _log_difference(upper, lower) - jnp.log(scale)
    midpoint = 0.5 * a + 0.5 * b
    flip = midpoint > 0.0
    a, b = jnp.where(flip, -b, a), jnp.where(flip, -a, b)
    centre = jnp.where(flip, -midpoint, midpoint)
    width = jnp.exp(log_width)
    half = 0.5 * width
    narrow = half * jnp.maximum(1.0, -centre) <= _NARROW_SWITCH

    log_z_n, mean_n, var_n = _narrow_interval_moments(
        jnp.where(narrow, centre, 0.0),
        jnp.where(narrow, half, 0.1),
        log_width=jnp.where(narrow, log_width, 0.0),
    )

    a_w = jnp.where(narrow, -1.0, a)
    b_w = jnp.where(narrow, 1.0, b)
    _, mills_a, log_mills_a, var_a = _tail_terms_with_log_mills(a_w)
    log_cdf_b, mills_b, log_mills_b, var_b = _tail_terms_with_log_mills(b_w)
    log_q = (
        jnp.where(narrow, 0.0, centre) * jnp.where(narrow, 2.0, width)
        + log_mills_b
        - log_mills_a
    )
    one_minus_q = -jnp.expm1(log_q)
    q = jnp.exp(log_q)
    log_z_w = log_cdf_b + jnp.log(one_minus_q)
    mean_w = (q * mills_a - mills_b) / one_minus_q
    spread = (mills_b - mills_a) / one_minus_q
    var_w = (var_b - q * var_a) / one_minus_q - q * spread * spread

    log_z = jnp.where(narrow, log_z_n, log_z_w)
    mean = jnp.where(narrow, mean_n, mean_w)
    var = jnp.where(narrow, var_n, var_w)
    return log_z, jnp.where(flip, -mean, mean), 1.0 - var, var


def _halving_factor(x: Array, y: Array) -> Array:
    """Return 0.5 where `x - y` could overflow, else 1.

    The difference of two floats overflows only if one of them has a magnitude of
    at least `2^1023`. Halving such operands is exact; halving a subnormal operand
    next to one of that magnitude changes the difference by less than one part in
    `2^2000`.
    """
    return jnp.where(jnp.maximum(jnp.abs(x), jnp.abs(y)) >= _HALF_RANGE, 0.5, 1.0)


def _scaled_difference(minuend: Array, subtrahend: Array, scale: Array) -> Array:
    """Return `(minuend - subtrahend) / scale` without overflow of the difference.

    Where no operand reaches `2^1023` the operations are literally
    `(minuend - subtrahend) / scale`; elsewhere the operands are halved and the
    quotient is doubled, which rounds once in the subtraction and once in the
    division, like the plain form. A quotient that is not a finite float is `inf`.
    """
    factor = _halving_factor(minuend, subtrahend)
    return (minuend * factor - subtrahend * factor) / scale / factor


def _log_difference(upper: Array, lower: Array) -> Array:
    """Return `log(upper - lower)` for `lower < upper` without overflow.

    The difference of two distinct finite floats is a positive float, so the
    logarithm is finite; operands that reach `2^1023` are halved first.
    """
    factor = _halving_factor(upper, lower)
    return jnp.log(upper * factor - lower * factor) - jnp.log(factor)


def _narrow_interval_moments(
    centre: Array, half: Array, *, log_width: Array | None = None
) -> tuple[Array, Array, Array]:
    """Return `log Z`, `E[z]` and `Var[z]` on `(centre - half, centre + half]`.

    With `exp(-c u - u^2 / 2) = sum_n He_n(c) (-u)^n / n!` and the uniform moments
    `E_U[u^k] = h^k / (k + 1)` for even `k`, the three integrals are power series in
    `h` with probabilists' Hermite coefficients; for `h max(1, |c|) <= 0.25` the
    truncation after order 21 is below double precision. `log_width`, if given, is
    `log(2 h)` and replaces the logarithm of `2 h` in `log Z`, which keeps `log Z`
    finite when `2 h` underflows.
    """
    he_prev = jnp.ones_like(centre)
    he = centre
    power = jnp.ones_like(centre)
    e0 = jnp.ones_like(centre)
    e1 = jnp.zeros_like(centre)
    e2 = half * half / 3.0
    for n in range(1, _SERIES_ORDER + 1):
        power = power * half / n
        if n % 2:
            e1 = e1 - he * power * half / (n + 2)
        else:
            e0 = e0 + he * power / (n + 1)
            e2 = e2 + he * power * half * half / (n + 3)
        he_prev, he = he, centre * he - n * he_prev
    mean_u = e1 / e0
    log_z = (
        -0.5 * centre * centre
        - LOG_SQRT_2PI
        + (jnp.log(2.0 * half * e0) if log_width is None else log_width + jnp.log(e0))
    )
    return log_z, centre + mean_u, e2 / e0 - mean_u * mean_u


def _normal_tail_terms_impl(t: Array) -> tuple[Array, Array, Array, Array]:
    """Return `log Phi(t)`, `r = phi(t) / Phi(t)`, `d = r (r + t)` and `1 - d`.

    Elementwise in float64, with the first derivatives `r`, `-d`, `d'` and `-d'`,
    where `d' = r - d (t + 2 r)`. The terms come from

    - `t <= -8`: a 24-level continued fraction of the Mills ratio, which yields
      `1 - d` and `d'` without cancellation;
    - `-8 < t <= 0`: `log_ndtr`;
    - `t > 0`: `log1p(-ndtr(-t))`.

    Above `t = 40`, `r`, `d` and `d'` underflow to 0 and `log Phi(t)` to 0.
    """
    log_cdf, mills, contraction, complement, _ = _normal_tail_terms_and_derivative(t)
    return log_cdf, mills, contraction, complement


def _normal_tail_terms_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
    (t,), (t_dot,) = primals, tangents
    log_cdf, mills, contraction, complement, contraction_prime = (
        _normal_tail_terms_and_derivative(t)
    )
    return (log_cdf, mills, contraction, complement), (
        mills * t_dot,
        -contraction * t_dot,
        contraction_prime * t_dot,
        -contraction_prime * t_dot,
    )


normal_tail_terms = jax.custom_jvp(_normal_tail_terms_impl)
normal_tail_terms.defjvp(_normal_tail_terms_jvp)


def _tail_terms_with_log_mills_impl(t: Array) -> tuple[Array, Array, Array, Array]:
    """Return `log Phi(t)`, `r`, `log r` and `1 - d` of `normal_tail_terms`.

    `log r` is the log Mills ratio with the derivative `-(t + r)`; it is finite also
    where `r` underflows, which `normal_tail_terms` leaves to the caller. Used by
    `source_interval_terms`; the half-line update keeps `normal_tail_terms`.
    """
    log_cdf, mills, _, complement, _, log_mills, _ = _normal_tail_terms_and_derivative(
        t, with_log_mills=True
    )
    return log_cdf, mills, log_mills, complement


def _tail_terms_with_log_mills_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
    (t,), (t_dot,) = primals, tangents
    log_cdf, mills, contraction, complement, prime, log_mills, log_mills_prime = (
        _normal_tail_terms_and_derivative(t, with_log_mills=True)
    )
    return (log_cdf, mills, log_mills, complement), (
        mills * t_dot,
        -contraction * t_dot,
        log_mills_prime * t_dot,
        -prime * t_dot,
    )


_tail_terms_with_log_mills = jax.custom_jvp(_tail_terms_with_log_mills_impl)
_tail_terms_with_log_mills.defjvp(_tail_terms_with_log_mills_jvp)


def _normal_tail_terms_and_derivative(
    t: Array, *, with_log_mills: bool = False
) -> tuple[Array, ...]:
    """Evaluate both regions on safe arguments and select per element.

    With `with_log_mills`, append the log Mills ratio and its derivative: the
    continued fraction gives `r` directly, and the central region forms `r` as the
    exponential of `log r`. The operations of the plain call are kept in their order.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    in_tail = t <= _TAIL_SWITCH

    tc = jnp.where(in_tail, 0.0, jnp.minimum(t, _CENTRAL_CAP))
    # log1p(-Phi(-t)) keeps the relative accuracy of log Phi(t) for t > 0.
    log_cdf_c = jnp.where(
        tc > 0.0,
        jnp.log1p(-jax.scipy.special.ndtr(-tc)),
        jax.scipy.special.log_ndtr(tc),
    )
    log_mills_c = -0.5 * tc * tc - LOG_SQRT_2PI - log_cdf_c
    mills_c = jnp.exp(log_mills_c)
    contraction_c = mills_c * (mills_c + tc)
    complement_c = 1.0 - contraction_c
    prime_c = mills_c - contraction_c * (tc + 2.0 * mills_c)

    x = jnp.where(in_tail, -t, -_TAIL_SWITCH)
    mills_n, complement_n, prime_n = _negative_tail_continued_fraction(x)
    log_cdf_n = -0.5 * x * x - LOG_SQRT_2PI - jnp.log(mills_n)
    contraction_n = 1.0 - complement_n

    terms = (
        jnp.where(in_tail, log_cdf_n, log_cdf_c),
        jnp.where(in_tail, mills_n, mills_c),
        jnp.where(in_tail, contraction_n, contraction_c),
        jnp.where(in_tail, complement_n, complement_c),
        jnp.where(in_tail, prime_n, prime_c),
    )
    if not with_log_mills:
        return terms
    # d log r / dt = -d / r = -(t + r); the first form avoids the cancellation of
    # `t + r` in the tail, the second the underflow of `r` in the central region.
    return (
        *terms,
        jnp.where(in_tail, jnp.log(mills_n), log_mills_c),
        jnp.where(in_tail, -contraction_n / mills_n, -(tc + mills_c)),
    )


def _negative_tail_continued_fraction(x: Array) -> tuple[Array, Array, Array]:
    """Return `r(-x)`, `1 - d(-x)` and `d'(-x)` for `x >= 8`.

    With `u = 1/x` and `y = u^2`, `r(-x) = x + u B(y)` where
    `B = 1 / (1 + 2y / (1 + 3y / (1 + ...)))`. The recurrence carries `B`, `B_y` and
    `B_yy`, from which `1 - d = y (B + 2 y B_y)` and
    `d' = -2 u y (B + 5 y B_y + 2 y^2 B_yy)` follow without subtracting near-equal
    numbers.
    """
    u = 1.0 / x
    y = u * u
    f = jnp.ones_like(x)
    f_y = jnp.zeros_like(x)
    f_yy = jnp.zeros_like(x)
    for n in range(_CONTINUED_FRACTION_DEPTH, 1, -1):
        inv = 1.0 / f
        new_f = 1.0 + n * y * inv
        new_f_y = n * (inv - y * f_y * inv * inv)
        new_f_yy = n * (
            -2.0 * f_y * inv * inv
            - y * f_yy * inv * inv
            + 2.0 * y * f_y * f_y * inv * inv * inv
        )
        f, f_y, f_yy = new_f, new_f_y, new_f_yy
    b = 1.0 / f
    b_y = -f_y * b * b
    b_yy = 2.0 * f_y * f_y * b * b * b - f_yy * b * b
    complement = y * (b + 2.0 * y * b_y)
    prime = -2.0 * u * y * (b + 5.0 * y * b_y + 2.0 * y * y * b_yy)
    return x + u * b, complement, prime
