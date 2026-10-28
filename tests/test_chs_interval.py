"""The CHS interval (ordered-probit) measurement update.

For a standard normal `z` truncated to `(a, b]`, with finite `a < b`, the references
use 200-digit `decimal` arithmetic on the closed forms

- `Z = Phi(b) - Phi(a)`,
- `E[z] = (phi(a) - phi(b)) / Z`,
- `Var[z] = 1 + (a phi(a) - b phi(b)) / Z - E[z]^2`,

and the derivative identities `dE[f]/db = p_b (f(b) - E[f])` and
`dE[f]/da = p_a (E[f] - f(a))` with `p_x = phi(x) / Z`. The update of one Gaussian
component on the event `lo < y* <= hi` is checked against adaptive quadrature over the
prior, as for the half-line events.
"""

import math
from decimal import Decimal, getcontext
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr
from test_chs_halfline import _assert_update_close, _prior, _quadrature_update

from skillmodels.chs.halfline import kalman_update_limited, normal_interval_terms
from skillmodels.common.measurement_models import ObservationEvent

jax.config.update("jax_enable_x64", True)

E = ObservationEvent
_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")


def _decimal_phi(x: Decimal) -> Decimal:
    return (-x * x / 2).exp() / (2 * _PI).sqrt()


def _decimal_cdf(x: Decimal) -> Decimal:
    """Normal CDF: power series for `|x| < 8`, Laplace continued fraction beyond."""
    if abs(x) < 8:
        total, term, n = Decimal(0), x, 0
        while abs(term) > Decimal("1e-190"):
            total += term
            n += 1
            term = term * x * x / (2 * n + 1)
        return Decimal("0.5") + _decimal_phi(x) * total
    y = abs(x)
    depth = 4000
    tail = y
    for n in range(depth, 0, -1):
        tail = y + n / tail
    lower_tail = _decimal_phi(y) / tail
    return lower_tail if x < 0 else 1 - lower_tail


def _decimal_interval(a: float, b: float) -> dict[str, float]:
    # Phi(45) - Phi(40) needs about 450 digits before the difference is exact.
    getcontext().prec = 600
    # The exact binary values: narrow widths are sensitive to the last bit.
    ad, bd = Decimal(a), Decimal(b)
    z = _decimal_cdf(bd) - _decimal_cdf(ad)
    pa, pb = _decimal_phi(ad) / z, _decimal_phi(bd) / z
    mean = pa - pb
    second = 1 + ad * pa - bd * pb
    var = second - mean * mean
    return {
        "log_z": float(z.ln()),
        "mean": float(mean),
        "var": float(var),
        "d": float(1 - var),
        "dlogz_da": float(-pa),
        "dlogz_db": float(pb),
        "dmean_da": float(pa * (mean - ad)),
        "dmean_db": float(pb * (bd - mean)),
        "dvar_da": float(pa * (var - (ad - mean) ** 2)),
        "dvar_db": float(pb * ((bd - mean) ** 2 - var)),
    }


_INTERVALS = [
    (-1.0, 0.5),
    (-3.0, 4.0),
    (0.2, 2.5),
    (-0.25, 0.25),
    (-1e-6, 1e-6),
    (3.0, 3.0 + 1e-10),
    (-40.0, -39.99),
    (-45.0, -40.0),
    (40.0, 45.0),
    (-8.001, -8.0),
    (7.5, 9.0),
    (-12.0, 30.0),
    (-200.0, -150.0),
    (5.0, 5.5),
    # Either side of the switch between the series and the tail formulas.
    (-0.25, math.nextafter(0.25, math.inf)),
    (-0.25, math.nextafter(0.25, -math.inf)),
    (-10.0, -10.0 + 0.05),
    (-10.0, -10.0 + math.nextafter(0.05, math.inf)),
]


def _terms(a, b):
    log_z, mean, d, var = normal_interval_terms(jnp.asarray(a), jnp.asarray(b))
    return {"log_z": log_z, "mean": mean, "d": d, "var": var}


@pytest.mark.parametrize(("a", "b"), _INTERVALS)
def test_interval_terms_match_high_precision_reference(a, b) -> None:
    expected = _decimal_interval(a, b)
    got = _terms(a, b)
    width = b - a
    np.testing.assert_allclose(got["log_z"], expected["log_z"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        got["mean"], expected["mean"], rtol=0, atol=1e-12 * max(1, abs(a), abs(b))
    )
    np.testing.assert_allclose(got["var"], expected["var"], rtol=1e-8, atol=0)
    np.testing.assert_allclose(got["d"], expected["d"], rtol=0, atol=1e-12)
    assert 0 < float(got["var"]) <= min(1.0, width**2 / 4)


@jax.jit
def _term_gradients(a, b):
    """Return the derivatives of `log Z`, `E[z]` and `Var[z]` w.r.t. both ends."""

    def stacked(x, y):
        log_z, mean, _, var = normal_interval_terms(x, y)
        return jnp.stack([log_z, mean, var])

    return jax.jacrev(stacked, argnums=(0, 1))(a, b)


# One case per regime runs by default: central, narrow series, wide far tail, and
# both sides of the series/tail switch.
_FAST_DERIVATIVE_CASES = {(-1.0, 0.5), (-1e-6, 1e-6), (-45.0, -40.0), (40.0, 45.0)}


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (a, b)
        if (a, b) in _FAST_DERIVATIVE_CASES
        else pytest.param(a, b, marks=pytest.mark.long_running)
        for a, b in _INTERVALS
    ],
)
def test_interval_term_derivatives_match_identities(a, b) -> None:
    expected = _decimal_interval(a, b)
    gradients_a, gradients_b = _term_gradients(jnp.asarray(a), jnp.asarray(b))
    for i, name in enumerate(("log_z", "mean", "var")):
        da, db = float(gradients_a[i]), float(gradients_b[i])
        scale = max(1.0, abs(expected[f"d{name.replace('_', '')}_da"]))
        np.testing.assert_allclose(
            da, expected[f"d{name.replace('_', '')}_da"], rtol=1e-7, atol=1e-9 * scale
        )
        scale = max(1.0, abs(expected[f"d{name.replace('_', '')}_db"]))
        np.testing.assert_allclose(
            db, expected[f"d{name.replace('_', '')}_db"], rtol=1e-7, atol=1e-9 * scale
        )


# The compiled, vectorized terms are also checked against the decimal reference through
# `log_normal_interval_mass` in `test_normal_interval_mass.py`.
@pytest.mark.long_running
def test_interval_terms_agree_under_jit_and_vmap() -> None:
    a = jnp.array([x for x, _ in _INTERVALS])
    b = jnp.array([y for _, y in _INTERVALS])
    eager = [normal_interval_terms(x, y) for x, y in zip(a, b, strict=True)]
    batched = jax.jit(jax.vmap(normal_interval_terms))(a, b)
    for i in range(len(eager)):
        for k in range(4):
            np.testing.assert_allclose(batched[k][i], eager[i][k], rtol=1e-13)


def _run_interval_update(*, mean, upper, loading, offset, lo, hi, event):
    states, chols, _, loglikes, _ = kalman_update_limited(
        states=jnp.asarray(mean)[None, None, :],
        upper_chols=jnp.asarray(upper)[None, None, :, :],
        loadings=jnp.asarray(loading),
        control_params=jnp.array([offset]),
        meas_sd=jnp.array(1.0),
        measurements=jnp.array([1.0]),
        events=jnp.array([event], dtype=jnp.int8),
        thresholds=jnp.array([[hi, lo]]),
        controls=jnp.ones((1, 1)),
        log_mixture_weights=jnp.zeros((1, 1)),
        has_intervals=True,
    )
    chol = np.asarray(chols[0, 0])
    return {
        "loglike": float(loglikes[0]),
        "mean": np.asarray(states[0, 0]),
        "cov": chol.T @ chol,
    }


def _interval_quadrature(*, mean, cov, loading, offset, lo, hi):
    """Quadrature with the observation probability `Phi(hi - y*) - Phi(lo - y*)`."""
    return _quadrature_update(
        mean=mean,
        covariance=cov,
        loading=loading,
        offset=offset,
        error_sd=1.0,
        kind="interval",
        threshold=lo,
        upper_threshold=hi,
    )


@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(1, marks=pytest.mark.long_running),
        2,
        pytest.param(3, marks=pytest.mark.long_running),
    ],
)
@pytest.mark.parametrize(
    ("lo", "hi", "offset"),
    [(0.0, 0.8, 0.1), (-0.5, 0.4, 2.5), (0.3, 0.31, -0.2), (1.0, 3.0, -6.0)],
)
def test_interval_update_matches_prior_quadrature(dim, lo, hi, offset) -> None:
    mean, upper, loading = _prior(dim)
    cov = upper.T @ upper
    actual = _run_interval_update(
        mean=mean,
        upper=upper,
        loading=loading,
        offset=offset,
        lo=lo,
        hi=hi,
        event=E.INTERVAL,
    )
    expected = _interval_quadrature(
        mean=mean, cov=cov, loading=loading, offset=offset, lo=lo, hi=hi
    )
    _assert_update_close(actual, expected, cov)


@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(1, marks=pytest.mark.long_running),
        3,
        pytest.param(5, marks=pytest.mark.long_running),
    ],
)
def test_square_root_interval_update_equals_dense_covariance_formula(dim) -> None:
    """`B'B = P - d P lambda lambda' P / s^2` with `d` from the decimal reference."""
    rng = np.random.default_rng(321 + dim)
    for scale in (1e-6, 1.0, 1e3):
        for _ in range(2):
            upper = np.linalg.qr(rng.normal(size=(dim, dim)))[1] * scale
            mean = rng.normal(size=dim) * scale
            loading = rng.normal(size=dim)
            cov = upper.T @ upper
            s = math.sqrt(1.0 + loading @ cov @ loading)
            a = float(rng.normal() * 3)
            b = a + float(rng.exponential())
            mu = float(rng.normal())
            offset = mu - loading @ mean
            actual = _run_interval_update(
                mean=mean,
                upper=upper,
                loading=loading,
                offset=offset,
                lo=mu + a * s,
                hi=mu + b * s,
                event=E.INTERVAL,
            )
            ref = _decimal_interval(a, b)
            pl = cov @ loading
            dense_cov = cov - np.outer(pl, pl) * ref["d"] / s**2
            dense_mean = mean + pl * ref["mean"] / s
            np.testing.assert_allclose(
                actual["cov"], dense_cov, rtol=1e-8, atol=1e-11 * scale**2
            )
            np.testing.assert_allclose(
                actual["mean"], dense_mean, rtol=1e-8, atol=1e-11 * scale
            )
            np.testing.assert_allclose(actual["loglike"], ref["log_z"], rtol=1e-11)


@pytest.mark.parametrize(("event", "far"), [(E.UPPER_TAIL, 1e4), (E.LOWER_TAIL, -1e4)])
def test_interval_with_a_far_end_equals_the_halfline_update(event, far) -> None:
    """`(c, c + 1e4]` is the upper half-line beyond `c` up to `Phi(-1e4)`."""
    mean, upper, loading = _prior(3)
    c = 0.4
    lo, hi = (c, c + far) if far > 0 else (c + far, c)
    halfline = kalman_update_limited(
        states=jnp.asarray(mean)[None, None, :],
        upper_chols=jnp.asarray(upper)[None, None, :, :],
        loadings=jnp.asarray(loading),
        control_params=jnp.array([0.2]),
        meas_sd=jnp.array(1.0),
        measurements=jnp.array([0.0]),
        events=jnp.array([event], dtype=jnp.int8),
        thresholds=jnp.array([c, c]),
        controls=jnp.ones((1, 1)),
        log_mixture_weights=jnp.zeros((1, 1)),
    )
    interval = kalman_update_limited(
        states=jnp.asarray(mean)[None, None, :],
        upper_chols=jnp.asarray(upper)[None, None, :, :],
        loadings=jnp.asarray(loading),
        control_params=jnp.array([0.2]),
        meas_sd=jnp.array(1.0),
        measurements=jnp.array([0.0]),
        events=jnp.array([E.INTERVAL], dtype=jnp.int8),
        thresholds=jnp.array([[hi, lo]]),
        controls=jnp.ones((1, 1)),
        log_mixture_weights=jnp.zeros((1, 1)),
        has_intervals=True,
    )
    for got, expected in zip(interval[:4], halfline[:4], strict=True):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.long_running
def test_halfline_events_are_unchanged_by_the_interval_branch() -> None:
    """Tail and density events give bitwise the same update with `has_intervals`."""
    mean, upper, loading = _prior(2)
    events = jnp.array(
        [E.LOWER_TAIL, E.UPPER_TAIL, E.DENSITY, E.MISSING], dtype=jnp.int8
    )
    kwargs: dict[str, Any] = {
        "states": jnp.broadcast_to(jnp.asarray(mean), (4, 1, 2)),
        "upper_chols": jnp.broadcast_to(jnp.asarray(upper), (4, 1, 2, 2)),
        "loadings": jnp.asarray(loading),
        "control_params": jnp.array([0.2]),
        "meas_sd": jnp.array(0.9),
        "measurements": jnp.array([-0.3, 1.2, 0.4, np.nan]),
        "events": events,
        "thresholds": jnp.array([-0.3, 1.2]),
        "controls": jnp.ones((4, 1)),
        "log_mixture_weights": jnp.zeros((4, 1)),
    }
    without = kalman_update_limited(**kwargs)
    with_intervals = kalman_update_limited(**kwargs, has_intervals=True)
    for got, expected in zip(with_intervals[:4], without[:4], strict=True):
        np.testing.assert_array_equal(got, expected)


def _interval_sum_loglike(loadings, control_params, cuts):
    mean, upper, _ = _prior(2)
    rng = np.random.default_rng(7)
    n_obs = 5
    states = jnp.asarray(mean + 0.3 * rng.normal(size=(n_obs, 2, 2)))
    chols = jnp.broadcast_to(jnp.asarray(upper), (n_obs, 2, 2, 2))
    # Codes 0..3 of a four-category measure; the table is [-inf, c1, c2, c3, inf].
    codes = jnp.array([0, 1, 2, 3, 1])
    table = jnp.concatenate([jnp.array([-jnp.inf]), cuts, jnp.array([jnp.inf])])
    events = jnp.array(
        [E.LOWER_TAIL, E.INTERVAL, E.INTERVAL, E.UPPER_TAIL, E.INTERVAL],
        dtype=jnp.int8,
    )
    thresholds = jnp.stack([table[codes + 1], table[codes]], axis=1)
    out = kalman_update_limited(
        states=states,
        upper_chols=chols,
        loadings=loadings,
        control_params=control_params,
        meas_sd=jnp.array(1.0),
        measurements=codes.astype(float),
        events=events,
        thresholds=thresholds,
        controls=jnp.asarray(np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])),
        log_mixture_weights=jnp.log(jnp.full((n_obs, 2), 0.5)),
        has_intervals=True,
    )
    new_states, new_chols, new_weights, loglikes, _ = out
    return loglikes.sum() + new_chols.sum() + new_weights.sum() + new_states.sum()


@pytest.mark.long_running
def test_interval_gradient_matches_step_swept_finite_differences() -> None:
    args = (jnp.array([0.6, -0.4]), jnp.array([0.1, 0.5]), jnp.array([0.0, 0.7, 1.3]))
    grads = jax.grad(_interval_sum_loglike, argnums=(0, 1, 2))(*args)
    flat_args = np.concatenate([np.asarray(a) for a in args])
    flat_grad = np.concatenate([np.asarray(g) for g in grads])
    assert np.isfinite(flat_grad).all()

    def flat_func(x):
        return float(
            _interval_sum_loglike(
                jnp.asarray(x[:2]), jnp.asarray(x[2:4]), jnp.asarray(x[4:])
            )
        )

    for i in range(len(flat_args)):
        errors = []
        for step in (1e-3, 1e-4, 1e-5, 1e-6):
            up, down = flat_args.copy(), flat_args.copy()
            up[i] += step
            down[i] -= step
            fd = (flat_func(up) - flat_func(down)) / (2 * step)
            errors.append(abs(fd - flat_grad[i]) / max(1.0, abs(flat_grad[i])))
        assert min(errors) < 1e-6


def test_interval_probability_is_the_cdf_difference_at_zero_loadings() -> None:
    mean, upper, _ = _prior(2)
    got = _run_interval_update(
        mean=mean,
        upper=upper,
        loading=np.zeros(2),
        offset=0.3,
        lo=-0.2,
        hi=0.9,
        event=E.INTERVAL,
    )
    np.testing.assert_allclose(
        got["loglike"], math.log(ndtr(0.9 - 0.3) - ndtr(-0.2 - 0.3)), rtol=1e-14
    )
    np.testing.assert_array_equal(got["mean"], mean)
