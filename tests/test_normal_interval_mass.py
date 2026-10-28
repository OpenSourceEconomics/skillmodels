"""Log mass of a normal interval in the shared measurement kernel.

`log(Phi(b) - Phi(a))` for finite `a < b` must stay finite and accurate for every
representable interval, however narrow, together with its endpoint derivatives. The
reference is `test_interval_mass_oracle.interval_reference`, a 220-digit decimal
integral of the normal density at the exact binary64 endpoints. A difference of rounded
endpoint log-CDFs cannot meet this: at `(0, 1e-17]` both endpoint log-CDFs round to the
same binary64 value (`test_interval_mass_oracle.quantization_case`).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.stats import norm
from test_interval_mass_oracle import interval_reference

from skillmodels.af.likelihood import af_per_obs_loglike_initial
from skillmodels.chs.halfline import normal_interval_terms
from skillmodels.common.measurement_models import MeasurementFamily, measurement_loglik
from skillmodels.common.normal_terms import log_normal_interval_mass

jax.config.update("jax_enable_x64", True)

_WIDTH = 1e-17
_ORDERED = int(MeasurementFamily.ORDERED_PROBIT)
_TOBIT = int(MeasurementFamily.TOBIT)


def _cases() -> list[tuple[float, float]]:
    """Return distinct binary64 intervals over widths, reflections and translations."""
    pairs = set()
    for centre in (-6.0, -3.0, 0.0, 3.0, 6.0):
        for width in (1e-17, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.1, 1.0):
            a, b = centre, centre + width
            if a < b:
                pairs.add((a, b))
                pairs.add((-b, -a))
    pairs.update(
        [
            (-0.25, 0.25),
            (-0.25, float(np.nextafter(0.25, np.inf))),
            (-0.25, float(np.nextafter(0.25, -np.inf))),
        ]
    )
    return sorted(pairs)


def _ordered(width: jax.Array) -> jax.Array:
    return measurement_loglik(
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(_ORDERED),
        jnp.array(0.0),
        width,
    )


def test_narrow_ordered_category_has_its_finite_log_mass() -> None:
    actual = float(_ordered(jnp.array(_WIDTH)))
    assert abs(actual - interval_reference(0.0, _WIDTH)["log_mass"]) <= 1e-10


def test_narrow_interval_endpoint_derivatives() -> None:
    derivative = jax.grad(log_normal_interval_mass, argnums=(0, 1))
    actual = np.asarray(derivative(jnp.array(0.0), jnp.array(_WIDTH)))
    reference = interval_reference(0.0, _WIDTH)
    np.testing.assert_allclose(
        actual, [reference["dlog_da"], reference["dlog_db"]], rtol=1e-8, atol=0.0
    )


def test_af_initial_period_integrates_the_narrow_category_mass() -> None:
    """A narrow ordered category next to a Gaussian measure in the AF initial period.

    The ordered measure has a zero loading and intercept, so its category probability
    `Phi(1e-17) - Phi(0)` is constant over the state and factors out of the
    node-weighted Gaussian likelihood exactly.
    """
    mu, chol, loading_g, sd_g = 0.5, 1.1, 1.0, 0.5
    params = jnp.array([1.0, mu, chol, 0.1, 0.0, loading_g, 0.0, sd_g, 1.0])
    cuts = np.array([-np.inf, 0.0, _WIDTH, 1.4, np.inf])
    rng = np.random.default_rng(11)
    y_gauss = rng.normal(0, 1, 6)
    controls = rng.normal(0, 1, (6, 1))
    raw_nodes = np.array([-1.5, -0.5, 0.5, 1.5])
    node_w = np.exp(-0.5 * raw_nodes**2)
    node_w = node_w / node_w.sum()
    actual = np.asarray(
        af_per_obs_loglike_initial(
            params,
            n_factors=1,
            n_mixture_components=1,
            n_measures=2,
            n_controls=1,
            measurements=jnp.asarray(np.column_stack([y_gauss, np.ones(6)])),
            controls=jnp.asarray(controls),
            loading_mask=jnp.array([[True], [True]]),
            nodes=jnp.asarray(raw_nodes.reshape(-1, 1)),
            weights=jnp.asarray(node_w),
            stability_floor=0.0,
            n_latent_factors=1,
            measurement_families=jnp.array([int(MeasurementFamily.GAUSSIAN), _ORDERED]),
            measurement_lowers=jnp.array([-np.inf, -np.inf]),
            measurement_uppers=jnp.array([np.inf, np.inf]),
            measurement_cut_tables=jnp.asarray(np.vstack([np.zeros_like(cuts), cuts])),
        )
    )
    eta = 0.1 * controls + loading_g * (mu + chol * raw_nodes[None, :])
    gaussian = np.log(norm.pdf(y_gauss[:, None], loc=eta, scale=sd_g) @ node_w)
    expected = gaussian + interval_reference(0.0, _WIDTH)["log_mass"]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


def test_ordered_terms_do_not_poison_narrow_tobit_gradients() -> None:
    """A Tobit with bounds `(0, 1e-17)` at its lower bound, inside, and at its upper."""
    y = jnp.array([0.0, 0.5, 1.0]) * _WIDTH

    def loglike(y, eta, sigma):
        return measurement_loglik(
            y, eta, sigma, jnp.array(_TOBIT), jnp.array(0.0), jnp.array(_WIDTH)
        )

    grad = jax.jit(jax.vmap(jax.grad(loglike, argnums=(1, 2)), in_axes=(0, None, None)))
    d_eta, d_sigma = grad(y, jnp.array(0.0), jnp.array(1.0))
    root_two_over_pi = math.sqrt(2.0 / math.pi)
    np.testing.assert_allclose(
        np.asarray(d_eta),
        [-root_two_over_pi, 0.5 * _WIDTH, root_two_over_pi],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(d_sigma),
        [0.0, -1.0, root_two_over_pi * _WIDTH],
        rtol=1e-8,
        atol=1e-10,
    )


def test_log_mass_and_gradients_over_finite_and_half_line_intervals() -> None:
    """One compiled, vectorized evaluation of every finite case and four half-lines.

    Finite cases: values against the decimal reference, endpoint derivatives for the
    narrow ones. Half-lines: `log Phi`, the Mills ratio as the derivative with respect
    to the finite end, and 0 with respect to the infinite end.
    """
    pairs = _cases()
    half_lower = [-np.inf, 0.3, -np.inf, 40.0]
    half_upper = [0.3, np.inf, -40.0, np.inf]
    a = jnp.asarray([pair[0] for pair in pairs] + half_lower)
    b = jnp.asarray([pair[1] for pair in pairs] + half_upper)
    value_and_grad = jax.jit(
        jax.vmap(jax.value_and_grad(log_normal_interval_mass, argnums=(0, 1)))
    )
    value, (da, db) = value_and_grad(a, b)
    value, da, db = np.asarray(value), np.asarray(da), np.asarray(db)
    n = len(pairs)

    references = [interval_reference(*pair) for pair in pairs]
    np.testing.assert_allclose(
        value[:n], [ref["log_mass"] for ref in references], rtol=0.0, atol=1e-10
    )
    narrow = np.array([upper - lower < 1e-3 for lower, upper in pairs])
    np.testing.assert_allclose(
        da[:n][narrow],
        [ref["dlog_da"] for ref, m in zip(references, narrow, strict=True) if m],
        rtol=1e-8,
        atol=0.0,
    )
    np.testing.assert_allclose(
        db[:n][narrow],
        [ref["dlog_db"] for ref, m in zip(references, narrow, strict=True) if m],
        rtol=1e-8,
        atol=0.0,
    )

    expected = [log_ndtr(0.3), log_ndtr(-0.3), log_ndtr(-40.0), log_ndtr(-40.0)]
    np.testing.assert_allclose(value[n:], expected, rtol=1e-13)
    np.testing.assert_array_equal(da[n:][[0, 2]], 0.0)
    np.testing.assert_array_equal(db[n:][[1, 3]], 0.0)
    np.testing.assert_allclose(
        db[n:][[0, 2]],
        np.exp(norm.logpdf([0.3, -40.0]) - log_ndtr([0.3, -40.0])),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        da[n:][[1, 3]],
        -np.exp(norm.logpdf([0.3, 40.0]) - log_ndtr([-0.3, -40.0])),
        rtol=1e-10,
    )


@pytest.mark.long_running
def test_shared_and_chs_interval_log_masses_agree() -> None:
    pairs = _cases()
    a = jnp.asarray([pair[0] for pair in pairs])
    b = jnp.asarray([pair[1] for pair in pairs])
    shared = np.asarray(jax.jit(jax.vmap(log_normal_interval_mass))(a, b))
    chs = np.asarray(
        jax.jit(jax.vmap(lambda x, y: normal_interval_terms(x, y)[0]))(a, b)
    )
    np.testing.assert_allclose(shared, chs, rtol=0.0, atol=1e-10)
