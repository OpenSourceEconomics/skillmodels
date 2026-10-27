"""The CHS half-line (probit / censored Tobit) measurement update.

For one Gaussian component `theta ~ N(m, P)` and the latent index
`y* = a + lambda'theta + eps`, `eps ~ N(0, sigma^2)`, a half-line event
`direction * (y* - threshold) > 0` has the exact probability `Phi(t)` and the exact
posterior moments

- `m+ = m + direction * P lambda * r / s`,
- `P+ = P - P lambda lambda' P * d / s^2`,

with `s^2 = sigma^2 + lambda'P lambda`,
`t = direction * (a + lambda'm - threshold) / s`, `r = phi(t) / Phi(t)` and
`d = r (r + t)`. A Tobit value strictly inside its bounds is an ordinary Gaussian
observation. The references below are independent of the
production formulas: high-precision `decimal` arithmetic for the normal-tail terms and
adaptive quadrature over the Gaussian prior for the update.
"""

import math
from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from skillmodels.chs.halfline import kalman_update_limited, normal_tail_terms
from skillmodels.chs.kalman_filters import kalman_update
from skillmodels.common.measurement_models import ObservationEvent

jax.config.update("jax_enable_x64", True)

E = ObservationEvent
_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")


def _decimal_tail_terms(t: float) -> dict[str, float]:
    """Return log Phi, r, d, 1 - d and d' at `t` with 100-digit arithmetic.

    - `t <= -8`: the Laplace continued fraction of the Mills ratio, evaluated to a
      depth at which doubling it changes nothing at 60 digits.
    - `|t| < 8`: the power series `Phi(t) = 1/2 + phi(t) sum t^(2n+1)/(2n+1)!!`.
    - `t >= 8`: `Phi(t) = 1 - phi(t) / R(t)` with the continued fraction `R`.
    """
    getcontext().prec = 200
    td = Decimal(repr(t))
    log_sqrt_2pi = (2 * _PI).sqrt().ln()

    def mills_excess(x: Decimal, depth: int) -> Decimal:
        """Return `R(x) - x = 1 / (x + 2 / (x + 3 / ...))`."""
        tail = x
        for n in range(depth, 1, -1):
            tail = x + n / tail
        return 1 / tail

    def converged_mills_excess(x: Decimal) -> Decimal:
        depth = 2000
        value = mills_excess(x, depth)
        while abs(mills_excess(x, 2 * depth) - value) > Decimal("1e-90") * value:
            depth *= 2
            value = mills_excess(x, depth)
        return value

    if td <= -8:
        x = -td
        r_plus_t = converged_mills_excess(x)
        r = x + r_plus_t
        log_cdf = -x * x / 2 - log_sqrt_2pi - r.ln()
    else:
        phi = (-td * td / 2 - log_sqrt_2pi).exp()
        if td < 8:
            total, term, n = Decimal(0), td, 0
            while abs(term) > Decimal("1e-190"):
                total += term
                n += 1
                term = term * td * td / (2 * n + 1)
            cdf = Decimal("0.5") + phi * total
            log_cdf = cdf.ln()
        else:
            q = phi / (td + converged_mills_excess(td))
            cdf = 1 - q
            log_cdf = -q - q * q / 2 - q * q * q / 3
        r = phi / cdf
        r_plus_t = r + td
    d = r * r_plus_t
    dprime = r - d * (td + 2 * r)
    return {
        "log_cdf": float(log_cdf),
        "r": float(r),
        "d": float(d),
        "h": float(1 - d),
        "dprime": float(dprime),
    }


_TAIL_POINTS = [
    -1e20,
    -1e8,
    -1e4,
    -100.0,
    -40.0,
    -20.0,
    math.nextafter(-8.0, -math.inf),
    -8.0,
    math.nextafter(-8.0, math.inf),
    -7.0,
    -3.0,
    -1.0,
    0.0,
    0.5,
    3.0,
    8.0,
    20.0,
    37.0,
    40.0,
    100.0,
]


@pytest.mark.parametrize("t", _TAIL_POINTS)
def test_normal_tail_terms_match_high_precision_reference(t) -> None:
    expected = _decimal_tail_terms(t)
    log_cdf, r, d, h = (float(x) for x in normal_tail_terms(jnp.array(t)))
    np.testing.assert_allclose(log_cdf, expected["log_cdf"], rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(r, expected["r"], rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(d, expected["d"], rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(h, expected["h"], rtol=1e-11, atol=1e-300)


@pytest.mark.parametrize("t", _TAIL_POINTS)
def test_normal_tail_term_derivatives_match_identities(t) -> None:
    """`(log Phi)' = r`, `r' = -d`, `d' = r - d (t + 2r)`, `(1 - d)' = -d'`."""
    expected = _decimal_tail_terms(t)
    _, tangents = jax.jvp(normal_tail_terms, (jnp.array(t),), (jnp.array(1.0),))
    dlog_cdf, dr, dd, dh = (float(x) for x in tangents)
    np.testing.assert_allclose(dlog_cdf, expected["r"], rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(dr, -expected["d"], rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(dd, expected["dprime"], rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(dh, -expected["dprime"], rtol=1e-9, atol=1e-300)


def test_normal_tail_terms_are_identical_under_jit_and_vmap() -> None:
    t = jnp.array(_TAIL_POINTS)
    eager = jnp.stack([jnp.stack(normal_tail_terms(x)) for x in t])
    batched = jnp.stack(jax.jit(normal_tail_terms)(t), axis=1)
    np.testing.assert_allclose(eager, batched, rtol=1e-14, atol=0)
    grads = jax.vmap(jax.grad(lambda x: normal_tail_terms(x)[0]))(t)
    np.testing.assert_allclose(grads, normal_tail_terms(t)[1], rtol=1e-14, atol=0)


def _quadrature_update(
    *,
    mean,
    covariance,
    loading,
    offset,
    error_sd,
    kind,
    threshold=0.0,
    direction=1.0,
    observed_value=0.0,
):
    """Integrate the observation model over the Gaussian prior projected on `lambda`.

    `z = lambda'(theta - m) / sqrt(q)` is standard normal; the posterior of `theta`
    given `z` is Gaussian, so integrating the scalar observation model times the
    density of `z` (centred at its mode, with log scaling against underflow) gives the
    likelihood and the first two posterior moments of `theta` without any Mills ratio.
    """
    mu = offset + loading @ mean
    cross = covariance @ loading
    q = float(loading @ cross)
    sqrt_q = math.sqrt(q)

    def log_observation(z):
        index = mu + sqrt_q * z
        if kind == "tail":
            return float(log_ndtr(direction * (index - threshold) / error_sd))
        u = (observed_value - index) / error_sd
        return -0.5 * u * u - 0.5 * math.log(2 * math.pi) - math.log(error_sd)

    def log_integrand(z):
        return -0.5 * z * z - 0.5 * math.log(2 * math.pi) + log_observation(z)

    centre = threshold if kind == "tail" else observed_value
    radius = 12.0 + abs((centre - mu) / sqrt_q)
    mode = float(
        minimize_scalar(
            lambda z: -log_integrand(z),
            bounds=(-radius, radius),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 500},
        ).x
    )
    peak = log_integrand(mode)

    def moment(power):
        def integrand(u):
            exponent = log_integrand(mode + u) - peak
            return (math.exp(exponent) if exponent > -746.0 else 0.0) * u**power

        return quad(integrand, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13, limit=200)[
            0
        ]

    i0, i1, i2 = moment(0), moment(1), moment(2)
    z_mean = i1 / i0
    z_var = i2 / i0 - z_mean**2
    direction_vector = cross / sqrt_q
    post_mean = mean + direction_vector * (mode + z_mean)
    post_cov = covariance - np.outer(direction_vector, direction_vector) * (1 - z_var)
    return {"loglike": peak + math.log(i0), "mean": post_mean, "cov": post_cov}


def _prior(dim):
    upper = np.eye(dim) * 0.8 + np.triu(np.full((dim, dim), 0.15), 1)
    mean = np.linspace(-0.2, 0.3, dim)
    loading = np.linspace(0.3, 0.8, dim)
    return mean, upper, loading


def _run_update(
    *, mean, upper, loading, offset, sigma, event, value=0.0, thresholds=(0.0, 0.0)
):
    """Run the production update for one person and one mixture component."""
    states, chols, _, loglikes, _ = kalman_update_limited(
        states=jnp.asarray(mean)[None, None, :],
        upper_chols=jnp.asarray(upper)[None, None, :, :],
        loadings=jnp.asarray(loading),
        control_params=jnp.array([offset]),
        meas_sd=jnp.array(sigma),
        measurements=jnp.array([value]),
        events=jnp.array([event], dtype=jnp.int8),
        thresholds=jnp.asarray(thresholds, dtype=float),
        controls=jnp.ones((1, 1)),
        log_mixture_weights=jnp.zeros((1, 1)),
    )
    chol = np.asarray(chols[0, 0])
    return {
        "loglike": float(loglikes[0]),
        "mean": np.asarray(states[0, 0]),
        "cov": chol.T @ chol,
    }


def _assert_update_close(actual, expected, prior_cov, *, loglike_atol=1e-10):
    """Mean errors in prior SDs, covariance errors in products of prior SDs."""
    scales = np.sqrt(np.diag(prior_cov))
    assert abs(actual["loglike"] - expected["loglike"]) < loglike_atol
    mean_error = np.max(np.abs(actual["mean"] - expected["mean"]) / scales)
    cov_error = np.max(
        np.abs(actual["cov"] - expected["cov"]) / np.outer(scales, scales)
    )
    assert mean_error < 1e-8
    assert cov_error < 1e-8


_TAIL_CASES = [
    # Columns: threshold, direction, sigma, t, event, thresholds.
    (0.0, -1.0, 1.0, -0.7, E.LOWER_TAIL, (0.0, 0.0)),  # probit y = 0
    (0.0, 1.0, 1.0, 1.1, E.UPPER_TAIL, (0.0, 0.0)),  # probit y = 1
    (-0.4, -1.0, 0.7, -0.8, E.LOWER_TAIL, (-0.4, 0.9)),  # Tobit at lower bound
    (0.9, 1.0, 1.4, 0.9, E.UPPER_TAIL, (-0.4, 0.9)),  # Tobit at upper bound
]


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    ("threshold", "direction", "sigma", "t", "event", "thresholds"), _TAIL_CASES
)
def test_halfline_update_matches_prior_quadrature(
    dim, threshold, direction, sigma, t, event, thresholds
) -> None:
    mean, upper, loading = _prior(dim)
    cov = upper.T @ upper
    s = math.sqrt(sigma**2 + loading @ cov @ loading)
    offset = threshold + direction * t * s - loading @ mean
    actual = _run_update(
        mean=mean,
        upper=upper,
        loading=loading,
        offset=offset,
        sigma=sigma,
        event=event,
        thresholds=thresholds,
    )
    expected = _quadrature_update(
        mean=mean,
        covariance=cov,
        loading=loading,
        offset=offset,
        error_sd=sigma,
        kind="tail",
        threshold=threshold,
        direction=direction,
    )
    _assert_update_close(actual, expected, cov)


@pytest.mark.parametrize("t", [-40.0, -20.0, 20.0, 40.0])
@pytest.mark.parametrize("event", [E.LOWER_TAIL, E.UPPER_TAIL])
def test_halfline_update_far_in_the_tails_matches_quadrature(t, event) -> None:
    mean, upper, loading = _prior(3)
    cov = upper.T @ upper
    sigma, threshold = 0.9, 0.2
    direction = 1.0 if event == E.UPPER_TAIL else -1.0
    s = math.sqrt(sigma**2 + loading @ cov @ loading)
    offset = threshold + direction * t * s - loading @ mean
    actual = _run_update(
        mean=mean,
        upper=upper,
        loading=loading,
        offset=offset,
        sigma=sigma,
        event=event,
        thresholds=(threshold, threshold),
    )
    expected = _quadrature_update(
        mean=mean,
        covariance=cov,
        loading=loading,
        offset=offset,
        error_sd=sigma,
        kind="tail",
        threshold=threshold,
        direction=direction,
    )
    # A fixed absolute tolerance on log Phi(-40) ~ -804 is a relative 1e-13.
    _assert_update_close(actual, expected, cov, loglike_atol=1e-10 * max(1, abs(t)))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_interior_tobit_update_matches_quadrature(dim) -> None:
    mean, upper, loading = _prior(dim)
    cov = upper.T @ upper
    actual = _run_update(
        mean=mean,
        upper=upper,
        loading=loading,
        offset=0.1,
        sigma=0.7,
        event=E.DENSITY,
        value=0.2,
        thresholds=(-0.4, 0.9),
    )
    expected = _quadrature_update(
        mean=mean,
        covariance=cov,
        loading=loading,
        offset=0.1,
        error_sd=0.7,
        kind="density",
        observed_value=0.2,
    )
    _assert_update_close(actual, expected, cov)


def test_square_root_update_equals_dense_covariance_formula() -> None:
    """`B'B = P - d P lambda lambda' P / s^2` over generated priors and scales."""
    rng = np.random.default_rng(123)
    for scale in (1e-8, 1e-3, 1.0, 1e3):
        for dim in (1, 2, 3, 5):
            for _ in range(5):
                a = rng.normal(size=(dim, dim))
                upper = np.linalg.qr(a)[1] * scale
                mean = rng.normal(size=dim) * scale
                loading = rng.normal(size=dim)
                sigma = float(rng.uniform(0.2, 2.0))
                t = float(rng.normal() * 3)
                cov = upper.T @ upper
                s2 = sigma**2 + loading @ cov @ loading
                offset = t * math.sqrt(s2) - loading @ mean
                actual = _run_update(
                    mean=mean,
                    upper=upper,
                    loading=loading,
                    offset=offset,
                    sigma=sigma,
                    event=E.UPPER_TAIL,
                )
                ref = _decimal_tail_terms(t)
                pl = cov @ loading
                dense_cov = cov - np.outer(pl, pl) * ref["d"] / s2
                dense_mean = mean + pl * ref["r"] / math.sqrt(s2)
                np.testing.assert_allclose(
                    actual["cov"], dense_cov, rtol=1e-9, atol=1e-12 * scale**2
                )
                np.testing.assert_allclose(
                    actual["mean"], dense_mean, rtol=1e-9, atol=1e-12 * scale
                )
                np.testing.assert_allclose(
                    actual["loglike"], ref["log_cdf"], rtol=1e-12
                )


def test_zero_loading_leaves_state_unchanged_and_scores_the_intercept() -> None:
    mean, upper, _ = _prior(2)
    loading = np.zeros(2)
    actual = _run_update(
        mean=mean,
        upper=upper,
        loading=loading,
        offset=0.3,
        sigma=1.0,
        event=E.UPPER_TAIL,
    )
    np.testing.assert_array_equal(actual["mean"], mean)
    np.testing.assert_allclose(actual["cov"], upper.T @ upper, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(actual["loglike"], log_ndtr(0.3), rtol=1e-14)


def _sum_loglike(loadings, control_params, meas_sd, *, events, measurements):
    mean, upper, _ = _prior(2)
    rng = np.random.default_rng(5)
    n_obs = len(events)
    states = jnp.asarray(mean + 0.3 * rng.normal(size=(n_obs, 2, 2)))
    chols = jnp.broadcast_to(jnp.asarray(upper), (n_obs, 2, 2, 2))
    out = kalman_update_limited(
        states=states,
        upper_chols=chols,
        loadings=loadings,
        control_params=control_params,
        meas_sd=meas_sd,
        measurements=jnp.asarray(measurements),
        events=jnp.asarray(events, dtype=jnp.int8),
        thresholds=jnp.array([-0.3, 1.2]),
        controls=jnp.asarray(np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])),
        log_mixture_weights=jnp.log(jnp.full((n_obs, 2), 0.5)),
    )
    _, new_chols, new_weights, loglikes, _ = out
    # Include downstream quantities so their derivatives are checked as well.
    return loglikes.sum() + new_chols.sum() + new_weights.sum() + out[0].sum()


_EVENTS = [E.LOWER_TAIL, E.UPPER_TAIL, E.DENSITY, E.MISSING, E.LOWER_TAIL]
_VALUES = [-0.3, 1.2, 0.4, np.nan, -0.3]


@pytest.mark.parametrize("loading_scale", [0.0, 1.0])
def test_gradient_matches_step_swept_finite_differences(loading_scale) -> None:
    args = (
        jnp.array([0.6, -0.4]) * loading_scale,
        jnp.array([0.1, 0.5]),
        jnp.array(0.8),
    )

    def func(*a):
        return _sum_loglike(*a, events=_EVENTS, measurements=_VALUES)

    grads = jax.grad(func, argnums=(0, 1, 2))(*args)
    flat_args = np.concatenate([np.atleast_1d(np.asarray(a)) for a in args])
    flat_grad = np.concatenate([np.atleast_1d(np.asarray(g)) for g in grads])
    assert np.isfinite(flat_grad).all()

    def flat_func(x):
        return float(func(jnp.asarray(x[:2]), jnp.asarray(x[2:4]), jnp.asarray(x[4])))

    for i in range(len(flat_args)):
        errors = []
        for step in (1e-3, 1e-4, 1e-5, 1e-6):
            up, down = flat_args.copy(), flat_args.copy()
            up[i] += step
            down[i] -= step
            fd = (flat_func(up) - flat_func(down)) / (2 * step)
            errors.append(abs(fd - flat_grad[i]) / max(1.0, abs(flat_grad[i])))
        assert min(errors) < 1e-5


def test_interior_and_gaussian_updates_agree_including_mixtures_and_missing() -> None:
    rng = np.random.default_rng(11)
    n_obs, n_mix, dim = 6, 2, 3
    states = jnp.asarray(rng.normal(size=(n_obs, n_mix, dim)))
    chols = jnp.asarray(
        np.stack(
            [
                [np.linalg.qr(rng.normal(size=(dim, dim)))[1] for _ in range(n_mix)]
                for _ in range(n_obs)
            ]
        )
    )
    measurements = rng.normal(size=n_obs)
    measurements[2] = np.nan
    kwargs = {
        "states": states,
        "upper_chols": chols,
        "loadings": jnp.array([0.4, 0.9, -0.3]),
        "control_params": jnp.array([0.2, -0.1]),
        "meas_sd": jnp.array(0.6),
        "measurements": jnp.asarray(measurements),
        "controls": jnp.asarray(
            np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])
        ),
        "log_mixture_weights": jnp.log(jnp.asarray(rng.dirichlet([1, 1], size=n_obs))),
    }
    gaussian = kalman_update(**kwargs)
    events = np.where(np.isnan(measurements), E.MISSING, E.DENSITY).astype(np.int8)
    limited = kalman_update_limited(
        **kwargs, events=jnp.asarray(events), thresholds=jnp.array([-9.0, 9.0])
    )
    for got, expected in zip(limited[:4], gaussian, strict=True):
        np.testing.assert_allclose(
            _covariances(got), _covariances(expected), rtol=1e-12, atol=1e-13
        )


def _covariances(arr):
    """Compare Cholesky factors through `U'U`; other arrays pass through."""
    arr = np.asarray(arr)
    if arr.ndim == 4:
        return np.einsum("nkji,nkjl->nkil", arr, arr)
    return arr


def test_mixture_weights_are_bayes_updated_with_event_probabilities() -> None:
    mean0, mean1 = np.array([0.0]), np.array([1.5])
    upper = np.array([[0.7]])
    states = jnp.asarray(np.stack([mean0, mean1])[None])
    chols = jnp.asarray(np.stack([upper, upper])[None])
    log_w = jnp.log(jnp.array([[0.3, 0.7]]))
    _, _, new_log_w, loglikes, _ = kalman_update_limited(
        states=states,
        upper_chols=chols,
        loadings=jnp.array([1.0]),
        control_params=jnp.array([-0.5]),
        meas_sd=jnp.array(1.0),
        measurements=jnp.array([1.0]),
        events=jnp.array([E.UPPER_TAIL], dtype=jnp.int8),
        thresholds=jnp.array([0.0, 0.0]),
        controls=jnp.ones((1, 1)),
        log_mixture_weights=log_w,
    )
    s = math.sqrt(1 + 0.49)
    probs = np.array([math.erfc(-(m - 0.5) / s / math.sqrt(2)) / 2 for m in (0.0, 1.5)])
    joint = np.array([0.3, 0.7]) * probs
    np.testing.assert_allclose(float(loglikes[0]), math.log(joint.sum()), rtol=1e-13)
    np.testing.assert_allclose(
        np.exp(np.asarray(new_log_w[0])), joint / joint.sum(), rtol=1e-12
    )


def test_missing_observation_keeps_the_component_and_contributes_zero() -> None:
    mean, upper, loading = _prior(2)
    out = kalman_update_limited(
        states=jnp.asarray(mean)[None, None],
        upper_chols=jnp.asarray(upper)[None, None],
        loadings=jnp.asarray(loading),
        control_params=jnp.array([0.0]),
        meas_sd=jnp.array(1.0),
        measurements=jnp.array([np.nan]),
        events=jnp.array([E.MISSING], dtype=jnp.int8),
        thresholds=jnp.array([0.0, 0.0]),
        controls=jnp.array([[np.nan]]),
        log_mixture_weights=jnp.zeros((1, 1)),
    )
    np.testing.assert_array_equal(out[0][0, 0], mean)
    np.testing.assert_array_equal(out[1][0, 0], upper)
    assert float(out[3][0]) == 0.0
