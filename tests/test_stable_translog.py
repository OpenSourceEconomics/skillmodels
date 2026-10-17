"""stable_translog against an independent decimal oracle (round-0 audit RT1 + MT2)."""

from __future__ import annotations

import math
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from numpy.testing import assert_allclose, assert_array_equal
from stable_translog_reference import clip, clip_derivative, transition

from skillmodels.common import transition_functions as tf

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def api():
    # Absence is a failure, not a skip. This suite is for the implemented proposal.
    assert callable(getattr(tf, "stable_translog", None))
    assert callable(getattr(tf, "_soft_clip", None))
    f = tf.stable_translog
    return f, jax.jit(f), jax.jit(jax.vmap(f, in_axes=(0, None)))


def test_finite_tail_witness_and_limits(api):
    f, compiled, batch = api
    x = jnp.array([[-1e80], [-1e76], [-50.0], [0.0], [50.0], [1e76], [1e80]])
    p = jnp.array([0.0, 1.0, 0.0])
    expected = np.array([float(transition(row, p)) for row in np.asarray(x)])  # ty: ignore[invalid-argument-type]
    assert_allclose(batch(x, p), expected, rtol=3e-14, atol=2e-14)
    assert_allclose(compiled(jnp.array([1e80]), p), 9.0, rtol=3e-14)
    assert_allclose(f(jnp.array([-1e80]), p), 9.0, rtol=3e-14)
    q = tf._soft_clip(x[:, 0], 3.0, 4)
    assert bool(jnp.all(jnp.isfinite(q)))
    assert bool(jnp.all(jnp.diff(q) >= 0))
    assert bool(jnp.all(jnp.abs(q) <= 3.0 + 3e-14))
    assert_allclose(tf._soft_clip(jnp.array([jnp.inf, -jnp.inf]), 3.0, 4), [3.0, -3.0])
    assert bool(jnp.isnan(tf._soft_clip(jnp.array(jnp.nan), 3.0, 4)))
    # Recovering a helper's limit does not promise to recover nonfinite linear input.
    assert not bool(jnp.isfinite(f(jnp.array([jnp.inf]), jnp.array([1.0, 0.0, 0.0]))))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_shape_order_permutations_and_parameter_basis(api, n):
    f, compiled, batch = api
    rng = np.random.default_rng(0)
    states = rng.uniform(-6, 6, size=(8, n))
    k = 2 * n + n * (n - 1) // 2 + 1
    params = rng.normal(size=k)
    expected = np.array([float(transition(row, params)) for row in states])
    assert_allclose(
        batch(jnp.asarray(states), jnp.asarray(params)),
        expected,
        rtol=4e-13,
        atol=4e-13,
    )
    assert_allclose(
        compiled(jnp.asarray(states[0]), jnp.asarray(params)),
        expected[0],
        rtol=4e-13,
        atol=4e-13,
    )
    # Isolate every coefficient at one asymmetric state. No large linear subtraction.
    x = np.arange(1, n + 1, dtype=float) * 1.1
    basis_expected = np.array([float(transition(x, row)) for row in np.eye(k)])  # ty: ignore[invalid-argument-type]
    values = jax.vmap(f, in_axes=(None, 0))(jnp.asarray(x), jnp.eye(k))
    assert_allclose(values, basis_expected, rtol=4e-13, atol=4e-13)
    # Simultaneously relabel states and coefficients: numerical output is invariant.
    pairs = list(combinations(range(n), 2))
    permutation = list(reversed(range(n)))
    perm_params = np.zeros(k)
    for new, old in enumerate(permutation):
        perm_params[new] = params[old]
        perm_params[n + new] = params[n + old]
    for j, (a, b) in enumerate(pairs):
        old_pair = tuple(sorted((permutation[a], permutation[b])))
        perm_params[2 * n + j] = params[2 * n + pairs.index(old_pair)]  # ty: ignore[invalid-argument-type]
    perm_params[-1] = params[-1]
    assert_allclose(
        f(jnp.asarray(states[0, permutation]), jnp.asarray(perm_params)),
        expected[0],
        rtol=4e-13,
        atol=4e-13,
    )


@pytest.mark.parametrize(("x", "budget"), [(1.0, 0.01), (2.0, 0.10)])
def test_default_profile_fidelity_for_squares_and_interactions(api, x, budget):
    f = api[0]
    square = float(f(jnp.array([x]), jnp.array([0.0, 1.0, 0.0])))
    cross = float(f(jnp.array([x, -x]), jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])))
    assert abs(square - x * x) / (x * x) <= budget
    assert abs(cross + x * x) / (x * x) <= budget
    # Checks the actual default equation, not just a loose upper bound.
    assert_allclose(square, float(clip(x) ** 2), rtol=3e-14, atol=2e-14)


def test_all_200_iterates_and_fixed_point(api):
    f = api[0]
    p = jnp.array([0.769, -0.103, -0.003])

    def step(x, _):
        y = f(jnp.array([x]), p)
        return y, y

    final, trace = jax.jit(
        lambda: jax.lax.scan(step, jnp.array(-3.0), None, length=200)
    )()
    all_states = jnp.concatenate((jnp.array([-3.0]), trace))
    assert bool(jnp.all(jnp.isfinite(all_states)))
    assert float(jnp.max(jnp.abs(all_states))) <= 5.0
    assert bool(jnp.all(jnp.diff(all_states) >= -1e-13))
    assert_allclose(final, -0.013063101278542104, rtol=0, atol=2e-13)
    # Root count is additionally established by reference.default_certificate(),
    # not inferred from this one initial condition or a sign-change grid.


def test_derivatives_at_zero_clip_knee_and_tails(api):
    f = api[0]
    knee = 3.0
    x = np.array(
        [
            -50.0,
            -knee,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            np.nextafter(knee, -np.inf),
            knee,
            np.nextafter(knee, np.inf),
            50.0,
        ]
    )
    params = jnp.array([0.4, 0.2, 0.1])

    def objective(z):
        return f(jnp.array([z]), params)

    result = jax.jit(jax.vmap(jax.grad(objective)))(jnp.asarray(x))
    expected = np.array([0.4 + 0.4 * float(clip(v) * clip_derivative(v)) for v in x])
    assert bool(jnp.all(jnp.isfinite(result)))
    assert_allclose(result, expected, rtol=2e-12, atol=2e-13)
    gradient_params = jax.grad(f, argnums=1)(jnp.array([2.0]), params)
    assert_allclose(gradient_params, [2.0, float(clip(2) ** 2), 1.0], rtol=3e-14)
    # Gradient correctness is separate from zero nonlinear coefficients.
    identity_grad = jax.grad(f)(jnp.array([1e80]), jnp.array([1.0, 0.0, 0.0]))
    assert_allclose(identity_grad, [1.0], rtol=0, atol=2e-14)


@pytest.mark.parametrize(("scale", "power"), [(0.5, 2), (3.0, 4), (9.0, 8)])
def test_static_configuration_and_scaling(api, scale, power):
    f = api[0]
    compiled = jax.jit(f, static_argnames=("scale", "power"))
    params = jnp.array([0.0, 1.0, 0.0])
    for x in (-2 * scale, 0.0, 0.5 * scale, 2 * scale):
        value = compiled(jnp.array([x]), params, scale=scale, power=power)
        assert_allclose(
            value, float(clip(x, scale, power) ** 2), rtol=5e-14, atol=5e-14
        )
    # Algebraic covariance requires transforming the clip scale with the units.
    g1 = tf._soft_clip(jnp.array(2.0), 3.0, power)
    g2 = tf._soft_clip(jnp.array(10.0), 15.0, power)
    assert_allclose(g2, 5 * g1, rtol=4e-14)


def test_identity_and_name_companions(api):
    f = api[0]
    factors = ("a", "b", "c")
    names = tf.params_translog(factors)
    assert tf.params_stable_translog(factors) == names
    for own in range(3):
        p = jnp.zeros(len(names)).at[own].set(1.0)
        x = jnp.array([1e80, -25.0, 7.0])
        assert_array_equal(f(x, p), x[own])
        got = tf.identity_constraints_stable_translog(factors[own], 0, factors)
        want = tf.identity_constraints_translog(factors[own], 0, factors)
        assert [(v.loc, v.value) for v in got] == [(v.loc, v.value) for v in want]


@pytest.mark.parametrize("scale", [0.0, -3.0, math.inf, math.nan])
def test_invalid_scale_fails_loud(api, scale):
    with pytest.raises(ValueError, match="scale"):
        api[0](jnp.array([0.0]), jnp.array([1.0, 0.0, 0.0]), scale=scale)


@pytest.mark.parametrize("power", [0, 1, 3, 4.0, True])
def test_invalid_power_fails_loud(api, power):
    # Under the tests' beartype claw, power=4.0 violates the `int` hint and is
    # rejected by beartype before the function body runs.
    with pytest.raises(
        (ValueError, TypeError, BeartypeCallHintParamViolation), match="power"
    ):
        api[0](jnp.array([0.0]), jnp.array([1.0, 0.0, 0.0]), power=power)


@pytest.mark.parametrize(("scale", "power"), [(0.5, 4), (3.0, 4), (9.0, 8)])
def test_finite_power_overflow_neighborhood(scale, power):
    limit = np.finfo(np.float64).max
    threshold = scale * math.exp(math.log(limit) / power)
    samples = [
        0.0,
        scale,
        np.nextafter(scale, 0.0),
        np.nextafter(scale, np.inf),
        50.0,
        threshold / 2,
        np.nextafter(threshold, 0.0),
        threshold,
        np.nextafter(threshold, np.inf),
        threshold * 2,
        1e80,
        1e150,
        1e300,
    ]
    samples = sorted({v for x in samples for v in (x, -x) if np.isfinite(v)})
    x = jnp.array(samples)
    helper = jax.jit(lambda z: tf._soft_clip(z, scale, power))
    got = helper(x)
    expected = np.array([float(clip(v, scale, power)) for v in samples])
    assert bool(jnp.all(jnp.isfinite(got)))
    assert bool(jnp.all(jnp.diff(got) >= -2e-13 * scale))
    assert_allclose(got, expected, rtol=8e-14, atol=8e-14 * scale)
    f = jax.jit(tf.stable_translog, static_argnames=("scale", "power"))
    p = jnp.array([0.0, 1.0, 0.0])
    vals = jax.vmap(lambda z: f(jnp.array([z]), p, scale=scale, power=power))(x)
    assert_allclose(vals, expected**2, rtol=2e-13, atol=2e-13 * scale**2)
    # Losing the nonlinear term must not be hidden by an enormous linear part.
    slopes = jax.vmap(
        jax.grad(
            lambda z: f(
                jnp.array([z]), jnp.array([1.0, 0.0, 0.0]), scale=scale, power=power
            )
        )
    )(x)
    assert bool(jnp.all(jnp.isfinite(slopes)))
    assert_allclose(slopes, 1.0, rtol=0, atol=2e-13)


def test_observed_input_beyond_scale_is_clipped_in_interaction(api) -> None:
    f = api[0]
    # latent at 1.0, observed input at 50: only the interaction coefficient is on
    params = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    value = float(f(jnp.array([1.0, 50.0]), params))
    assert_allclose(value, float(clip(1.0) * clip(50.0)), rtol=3e-14)
    assert value < 3.0  # translog would give 50
