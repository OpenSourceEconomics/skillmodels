"""Tests for the shared measurement-family log-likelihood kernel.

`measurement_loglik` returns one log density/probability contribution per
measurement, dispatching on a `MeasurementFamily` code:

- Gaussian: `log N(y; eta, sigma)`.
- Probit: `log Phi((2y-1) eta)` for `y in {0, 1}` (standard-normal latent error).
- Tobit: censored normal -- the interior normal density, or the tail mass
  `Phi((L-eta)/sigma)` / `Phi((eta-U)/sigma)` at a censoring bound.

The kernel is the single source of truth shared by AF estimation, CHS, simulation
and posterior-state reweighting, so it is validated against SciPy closed forms and
its JAX gradient against finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from skillmodels.common.measurement_models import (
    MeasurementFamily,
    measurement_loglik,
)

jax.config.update("jax_enable_x64", True)

_INF = float("inf")


def _ll(y, eta, sigma, family, lower=-_INF, upper=_INF):
    return float(
        measurement_loglik(
            jnp.asarray(float(y)),
            jnp.asarray(float(eta)),
            jnp.asarray(float(sigma)),
            jnp.asarray(int(family)),
            jnp.asarray(float(lower)),
            jnp.asarray(float(upper)),
        )
    )


@pytest.mark.parametrize("eta", [-2.0, 0.0, 1.5])
@pytest.mark.parametrize("y", [-1.0, 0.7, 3.0])
def test_gaussian_kernel_matches_scipy(y: float, eta: float) -> None:
    sigma = 0.8
    got = _ll(y, eta, sigma, MeasurementFamily.GAUSSIAN)
    assert got == pytest.approx(norm.logpdf(y, loc=eta, scale=sigma))


@pytest.mark.parametrize("eta", [-3.0, -0.5, 0.0, 0.5, 3.0])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_probit_kernel_matches_scipy(y: float, eta: float) -> None:
    got = _ll(y, eta, 1.0, MeasurementFamily.PROBIT)
    expected = norm.logcdf((2.0 * y - 1.0) * eta)
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("eta", [-40.0, 40.0])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_probit_kernel_finite_in_tails(y: float, eta: float) -> None:
    got = _ll(y, eta, 1.0, MeasurementFamily.PROBIT)
    assert np.isfinite(got)
    assert got == pytest.approx(norm.logcdf((2.0 * y - 1.0) * eta), abs=1e-6)


def test_tobit_interior_matches_scipy() -> None:
    y, eta, sigma = 2.3, 1.0, 0.9
    got = _ll(y, eta, sigma, MeasurementFamily.TOBIT, lower=0.0)
    expected = norm.logpdf(y, loc=eta, scale=sigma)
    assert got == pytest.approx(expected)


def test_tobit_left_censored_matches_scipy() -> None:
    lower, eta, sigma = 0.0, 1.0, 0.9
    got = _ll(lower, eta, sigma, MeasurementFamily.TOBIT, lower=lower)
    expected = norm.logcdf((lower - eta) / sigma)
    assert got == pytest.approx(expected)


def test_tobit_right_censored_matches_scipy() -> None:
    upper, eta, sigma = 5.0, 4.0, 1.1
    got = _ll(upper, eta, sigma, MeasurementFamily.TOBIT, lower=0.0, upper=upper)
    expected = norm.logcdf((eta - upper) / sigma)
    assert got == pytest.approx(expected)


def test_tobit_one_sided_lower_does_not_censor_at_plus_inf() -> None:
    # With no upper bound, a large interior value uses the normal density, never
    # the (absent) upper tail mass.
    y, eta, sigma = 100.0, 1.0, 0.9
    got = _ll(y, eta, sigma, MeasurementFamily.TOBIT, lower=0.0, upper=_INF)
    assert got == pytest.approx(norm.logpdf(y, loc=eta, scale=sigma))


def test_probit_loading_gradient_matches_finite_difference() -> None:
    # eta = c + lambda * theta; differentiate the probit contribution wrt lambda.
    y, theta, sigma = 1.0, 0.6, 1.0

    def ll_of_lambda(lam: jax.Array) -> jax.Array:
        eta = 0.2 + lam * theta
        return measurement_loglik(
            jnp.asarray(y),
            eta,
            jnp.asarray(sigma),
            jnp.asarray(int(MeasurementFamily.PROBIT)),
            jnp.asarray(-_INF),
            jnp.asarray(_INF),
        )

    lam0 = jnp.asarray(0.8)
    grad = float(jax.grad(ll_of_lambda)(lam0))
    h = 1e-6
    fd = (float(ll_of_lambda(lam0 + h)) - float(ll_of_lambda(lam0 - h))) / (2 * h)
    assert grad == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_tobit_scale_gradient_matches_finite_difference() -> None:
    # Differentiate a left-censored Tobit contribution wrt sigma.
    lower, eta = 0.0, 1.0

    def ll_of_sigma(sig: jax.Array) -> jax.Array:
        return measurement_loglik(
            jnp.asarray(lower),
            jnp.asarray(eta),
            sig,
            jnp.asarray(int(MeasurementFamily.TOBIT)),
            jnp.asarray(lower),
            jnp.asarray(_INF),
        )

    sig0 = jnp.asarray(0.9)
    grad = float(jax.grad(ll_of_sigma)(sig0))
    h = 1e-6
    fd = (float(ll_of_sigma(sig0 + h)) - float(ll_of_sigma(sig0 - h))) / (2 * h)
    assert grad == pytest.approx(fd, rel=1e-5, abs=1e-7)


_ORDERED_CASES = [
    # Columns: eta, lower, upper -- the latent interval of the observed category.
    (0.3, -_INF, 0.0),
    (0.3, 0.0, 0.8),
    (0.3, 0.8, _INF),
    (-2.0, 0.5, 0.6),
    (6.0, -_INF, 0.0),
    (-6.0, 1.0, _INF),
    (45.0, 0.0, 0.7),
    (-45.0, 0.0, 0.7),
]


@pytest.mark.parametrize(("eta", "lower", "upper"), _ORDERED_CASES)
def test_ordered_probit_kernel_is_the_log_interval_probability(
    eta, lower, upper
) -> None:
    got = _ll(1.0, eta, 1.0, MeasurementFamily.ORDERED_PROBIT, lower, upper)
    # Reference in the tail where both cdfs are accurate: by symmetry the interval
    # (lower - eta, upper - eta] has the mass of (eta - upper, eta - lower].
    a, b = lower - eta, upper - eta
    if a + b > 0:
        a, b = -b, -a
    expected = norm.logcdf(b) + np.log(-np.expm1(norm.logcdf(a) - norm.logcdf(b)))
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("eta", [-40.0, -0.7, 0.0, 2.0, 40.0])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_two_category_ordered_probit_kernel_equals_probit(y, eta) -> None:
    lower, upper = (-_INF, 0.0) if y == 0.0 else (0.0, _INF)
    ordered = _ll(y, eta, 1.0, MeasurementFamily.ORDERED_PROBIT, lower, upper)
    assert ordered == _ll(y, eta, 1.0, MeasurementFamily.PROBIT)


@pytest.mark.parametrize(("eta", "lower", "upper"), _ORDERED_CASES[:4])
def test_ordered_probit_kernel_gradient_matches_finite_difference(
    eta, lower, upper
) -> None:
    def func(e, lo, hi):
        return measurement_loglik(
            jnp.asarray(1.0),
            e,
            jnp.asarray(1.0),
            jnp.asarray(int(MeasurementFamily.ORDERED_PROBIT)),
            lo,
            hi,
        )

    args = (jnp.asarray(eta), jnp.asarray(lower), jnp.asarray(upper))
    grads = jax.grad(func, argnums=(0, 1, 2))(*args)
    step = 1e-6
    fd = float(func(args[0] + step, *args[1:])) - float(func(args[0] - step, *args[1:]))
    np.testing.assert_allclose(grads[0], fd / (2 * step), rtol=1e-6)
    for end, grad in ((lower, grads[1]), (upper, grads[2])):
        assert np.isfinite(float(grad))
        if np.isinf(end):
            assert float(grad) == 0.0
