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


def _narrow_interval_moments(centre: Array, half: Array) -> tuple[Array, Array, Array]:
    """Return `log Z`, `E[z]` and `Var[z]` on `(centre - half, centre + half]`.

    With `exp(-c u - u^2 / 2) = sum_n He_n(c) (-u)^n / n!` and the uniform moments
    `E_U[u^k] = h^k / (k + 1)` for even `k`, the three integrals are power series in
    `h` with probabilists' Hermite coefficients; for `h max(1, |c|) <= 0.25` the
    truncation after order 21 is below double precision.
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
    log_z = -0.5 * centre * centre - LOG_SQRT_2PI + jnp.log(2.0 * half * e0)
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


def _normal_tail_terms_and_derivative(
    t: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate both regions on safe arguments and select per element."""
    t = jnp.asarray(t, dtype=jnp.float64)
    in_tail = t <= _TAIL_SWITCH

    tc = jnp.where(in_tail, 0.0, jnp.minimum(t, _CENTRAL_CAP))
    # log1p(-Phi(-t)) keeps the relative accuracy of log Phi(t) for t > 0.
    log_cdf_c = jnp.where(
        tc > 0.0,
        jnp.log1p(-jax.scipy.special.ndtr(-tc)),
        jax.scipy.special.log_ndtr(tc),
    )
    mills_c = jnp.exp(-0.5 * tc * tc - LOG_SQRT_2PI - log_cdf_c)
    contraction_c = mills_c * (mills_c + tc)
    complement_c = 1.0 - contraction_c
    prime_c = mills_c - contraction_c * (tc + 2.0 * mills_c)

    x = jnp.where(in_tail, -t, -_TAIL_SWITCH)
    mills_n, complement_n, prime_n = _negative_tail_continued_fraction(x)
    log_cdf_n = -0.5 * x * x - LOG_SQRT_2PI - jnp.log(mills_n)
    contraction_n = 1.0 - complement_n

    return (
        jnp.where(in_tail, log_cdf_n, log_cdf_c),
        jnp.where(in_tail, mills_n, mills_c),
        jnp.where(in_tail, contraction_n, contraction_c),
        jnp.where(in_tail, complement_n, complement_c),
        jnp.where(in_tail, prime_n, prime_c),
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
