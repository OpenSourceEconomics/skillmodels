"""CHS measurement update for probit and Tobit (limited) measurements.

For one person and mixture component let the predicted state be `theta ~ N(m, P)` with
`P = U'U`, and let the latent index be `y* = a + lambda'theta + eps`,
`eps ~ N(0, sigma^2)`, with `a` the control part. Write `v = U lambda`,
`s^2 = sigma^2 + v'v` and `mu = a + lambda'm`.

A half-line event `direction * (y* - threshold) > 0` -- probit `y = 1` / `y = 0` with
threshold 0 and `sigma = 1`, or a Tobit value at its upper / lower bound -- has the
exact probability `Phi(t)`, `t = direction * (mu - threshold) / s`, and the exact
posterior moments

- `m+ = m + direction * U'w * r`, `w = v / s`,
- `P+ = P - P lambda lambda' P * d / s^2`,

with `r = phi(t) / Phi(t)` and `d = r (r + t)`. The posterior is replaced by the
Gaussian with these moments (assumed-density filtering): the likelihood contribution
is exact given the Gaussian predicted state, the Gaussian shape of the filtered state is
an approximation. A Tobit value strictly inside its bounds is an ordinary Gaussian
observation, which is the same update with `d = 1` and the standardized innovation
`(y - mu) / s` in place of `direction * r`.

The covariance factor is updated as the R factor of `U - b w (w'U)` with
`b = d / (1 + c)` and `c = sqrt((sigma / s)^2 + (1 - d) w'w)`, which squares to the
posterior covariance. No step divides by `d`, takes `sqrt(d)`, or divides by
`lambda'P lambda`, so the update stays finite (and becomes the identity) when the event
is certain or the loadings are zero.
"""

import jax
import jax.numpy as jnp
from jax import Array

from skillmodels.chs.kalman_filters import reset_negligible_mixture_components
from skillmodels.common.measurement_models import ObservationEvent

_LOG_SQRT_2PI = 0.91893853320467274178
_TAIL_SWITCH = -8.0
_CENTRAL_CAP = 40.0
_CONTINUED_FRACTION_DEPTH = 24


def kalman_update_limited(
    states: Array,
    upper_chols: Array,
    loadings: Array,
    control_params: Array,
    meas_sd: Array,
    measurements: Array,
    events: Array,
    thresholds: Array,
    controls: Array,
    log_mixture_weights: Array,
) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
    """Condition each person's Gaussian mixture on one limited measurement.

    Args:
        states: Array of shape (n_obs, n_mixtures, n_states) with pre-update means.
        upper_chols: Array of shape (n_obs, n_mixtures, n_states, n_states) with
            upper Cholesky factors `U` of the pre-update covariances, `P = U'U`.
        loadings: 1d array of length n_states with factor loadings.
        control_params: 1d array of length n_controls.
        meas_sd: Measurement-error SD `sigma` (1 for a probit measurement).
        measurements: 1d array of length n_obs; only `DENSITY` entries are read.
        events: 1d int8 array of length n_obs with `ObservationEvent` codes.
        thresholds: Array `[lower, upper]` with the thresholds of the
            `LOWER_TAIL` and `UPPER_TAIL` events (0 for probit, the Tobit bounds;
            any finite value for a side without censoring).
        controls: Array of shape (n_obs, n_controls).
        log_mixture_weights: Array of shape (n_obs, n_mixtures).

    Return:
        - new_states: Same shape as `states`.
        - new_upper_chols: Same shape as `upper_chols`.
        - new_log_mixture_weights: Same shape as `log_mixture_weights`.
        - new_loglikes: 1d array of length n_obs; 0 for missing observations.
        - debug_info: `residuals` and `residual_sds` of shape (n_obs, n_mixtures)
          with `y - mu` and `s` for `DENSITY` observations and NaN otherwise.

    """
    n_obs, n_mixtures, n_states = states.shape
    not_missing = events != ObservationEvent.MISSING
    is_density = (events == ObservationEvent.DENSITY)[:, None]
    is_upper = (events == ObservationEvent.UPPER_TAIL)[:, None]
    is_tail = is_upper | (events == ObservationEvent.LOWER_TAIL)[:, None]

    # Placeholders keep every intermediate and its derivative finite for persons
    # without an observation; their results are discarded below.
    safe_controls = jnp.where(not_missing[:, None], controls, 0.0)
    safe_chols = jnp.where(
        not_missing.reshape(n_obs, 1, 1, 1),
        upper_chols,
        jnp.eye(n_states, dtype=upper_chols.dtype),
    )

    v = jnp.einsum("nkij,j->nki", safe_chols, loadings)
    # Accumulating hypot from sigma > 0 keeps s differentiable at v = 0.
    s = jnp.broadcast_to(meas_sd, (n_obs, n_mixtures))
    for i in range(n_states):
        s = jnp.hypot(s, v[..., i])
    w = v / s[..., None]
    alpha = jnp.sum(w * w, axis=-1)
    mu = jnp.dot(states, loadings) + jnp.dot(safe_controls, control_params)[:, None]

    safe_y = jnp.where(is_density, measurements[:, None], mu)
    innovation = (safe_y - mu) / s
    direction = jnp.where(is_upper, 1.0, -1.0)
    threshold = jnp.where(is_upper, thresholds[1], thresholds[0])
    t = jnp.where(is_tail, direction * (mu - threshold) / s, 0.0)
    log_cdf, mills, contraction, complement = normal_tail_terms(t)

    coefficient = jnp.where(is_tail, direction * mills, innovation)
    d = jnp.where(is_tail, contraction, 1.0)
    one_minus_d = jnp.where(is_tail, complement, 0.0)
    loglikes_per_dist = jnp.where(
        is_tail,
        log_cdf,
        -0.5 * innovation * innovation - jnp.log(s) - _LOG_SQRT_2PI,
    )

    _new_states = (
        states + jnp.einsum("nkji,nkj->nki", safe_chols, w) * coefficient[..., None]
    )
    c = jnp.sqrt((meas_sd / s) ** 2 + one_minus_d * alpha)
    b = d / (1.0 + c)
    w_u = jnp.einsum("nkj,nkji->nki", w, safe_chols)
    factor = safe_chols - b[..., None, None] * w[..., :, None] * w_u[..., None, :]
    _new_upper_chols = jnp.linalg.qr(factor)[1]

    if n_mixtures >= 2:
        weighted = loglikes_per_dist + log_mixture_weights
        _loglikes = jax.scipy.special.logsumexp(weighted, axis=1)
        _new_log_mixture_weights = weighted - _loglikes[:, None]
    else:
        _loglikes = loglikes_per_dist[:, 0]
        _new_log_mixture_weights = log_mixture_weights

    new_states = jnp.where(not_missing.reshape(n_obs, 1, 1), _new_states, states)
    new_upper_chols = jnp.where(
        not_missing.reshape(n_obs, 1, 1, 1), _new_upper_chols, upper_chols
    )
    new_loglikes = jnp.where(not_missing, _loglikes, 0.0)
    new_log_mixture_weights = jnp.where(
        not_missing[:, None], _new_log_mixture_weights, log_mixture_weights
    )
    if n_mixtures >= 2:
        new_states, new_upper_chols = reset_negligible_mixture_components(
            states=new_states,
            upper_chols=new_upper_chols,
            log_mixture_weights=new_log_mixture_weights,
        )

    observed_density = is_density & not_missing[:, None]
    debug_info = {
        "residuals": jnp.where(observed_density, safe_y - mu, jnp.nan),
        "residual_sds": jnp.where(observed_density, s, jnp.nan),
        "log_mixture_weights": new_log_mixture_weights,
    }
    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )


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
    mills_c = jnp.exp(-0.5 * tc * tc - _LOG_SQRT_2PI - log_cdf_c)
    contraction_c = mills_c * (mills_c + tc)
    complement_c = 1.0 - contraction_c
    prime_c = mills_c - contraction_c * (tc + 2.0 * mills_c)

    x = jnp.where(in_tail, -t, -_TAIL_SWITCH)
    mills_n, complement_n, prime_n = _negative_tail_continued_fraction(x)
    log_cdf_n = -0.5 * x * x - _LOG_SQRT_2PI - jnp.log(mills_n)
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
