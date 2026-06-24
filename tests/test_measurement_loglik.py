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
