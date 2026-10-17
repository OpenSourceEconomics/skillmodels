"""Native acceptance tests. Run in the project's Python 3.14 pixi environment.

These are NOT executed by the reviewer. They import the real implementation; they
never substitute a shim. The only required numerical profile is CPU float64.
"""
from __future__ import annotations

import math
import sys
from functools import partial
from itertools import combinations
from pathlib import Path

import pytest

if sys.version_info[:2] != (3, 14):
    raise RuntimeError('Native audit tests require the declared Python 3.14 environment.')

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from skillmodels.common import transition_functions as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_oracles.reference import clip, clip_derivative, transition

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def api():
    # Absence is a failure, not a skip. This suite is for the implemented proposal.
    assert callable(getattr(tf, 'stable_translog', None))
    assert callable(getattr(tf, '_soft_clip', None))
    f = tf.stable_translog
    return f, jax.jit(f), jax.jit(jax.vmap(f, in_axes=(0, None)))


def test_finite_tail_witness_and_limits(api):
    f, compiled, batch = api
    x = jnp.array([[-1e80], [-1e76], [-50.0], [0.0], [50.0], [1e76], [1e80]])
    p = jnp.array([0.0, 1.0, 0.0])
    expected = np.array([float(transition(row, p)) for row in np.asarray(x)])
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
    assert not bool(jnp.isfinite(f(jnp.array([jnp.inf]), jnp.array([1., 0., 0.]))))


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_shape_order_permutations_and_parameter_basis(api, n):
    f, compiled, batch = api
    rng = np.random.default_rng(0)
    states = rng.uniform(-6, 6, size=(8, n))
    k = 2*n+n*(n-1)//2+1
    params = rng.normal(size=k)
    expected = np.array([float(transition(row, params)) for row in states])
    assert_allclose(batch(jnp.asarray(states), jnp.asarray(params)), expected,
                    rtol=4e-13, atol=4e-13)
    assert_allclose(compiled(jnp.asarray(states[0]), jnp.asarray(params)), expected[0],
                    rtol=4e-13, atol=4e-13)
    # Isolate every coefficient at one asymmetric state. No large linear subtraction.
    x = np.arange(1, n+1, dtype=float)*1.1
    basis_expected = np.array([float(transition(x, row)) for row in np.eye(k)])
    values = jax.vmap(f, in_axes=(None, 0))(jnp.asarray(x), jnp.eye(k))
    assert_allclose(values, basis_expected, rtol=4e-13, atol=4e-13)
    # Simultaneously relabel states and coefficients: numerical output is invariant.
    pairs = list(combinations(range(n), 2))
    permutation = list(reversed(range(n)))
    perm_params = np.zeros(k)
    for new, old in enumerate(permutation):
        perm_params[new] = params[old]
        perm_params[n+new] = params[n+old]
    for j, (a,b) in enumerate(pairs):
        old_pair = tuple(sorted((permutation[a], permutation[b])))
        perm_params[2*n+j] = params[2*n+pairs.index(old_pair)]
    perm_params[-1] = params[-1]
    assert_allclose(f(jnp.asarray(states[0, permutation]), jnp.asarray(perm_params)),
                    expected[0], rtol=4e-13, atol=4e-13)


@pytest.mark.parametrize('x,budget', [(1.0, .01), (2.0, .10)])
def test_default_profile_fidelity_for_squares_and_interactions(api, x, budget):
    f = api[0]
    square = float(f(jnp.array([x]), jnp.array([0., 1., 0.])))
    cross = float(f(jnp.array([x, -x]), jnp.array([0., 0., 0., 0., 1., 0.])))
    assert abs(square-x*x)/(x*x) <= budget
    assert abs(cross+x*x)/(x*x) <= budget
    # Checks the actual default equation, not just a loose upper bound.
    assert_allclose(square, float(clip(x)**2), rtol=3e-14, atol=2e-14)


def test_all_200_iterates_and_fixed_point(api):
    f = api[0]
    p = jnp.array([.769, -.103, -.003])
    def step(x, _):
        y = f(jnp.array([x]), p)
        return y, y
    final, trace = jax.jit(lambda: jax.lax.scan(step, jnp.array(-3.), None, length=200))()
    all_states = jnp.concatenate((jnp.array([-3.]), trace))
    assert bool(jnp.all(jnp.isfinite(all_states)))
    assert float(jnp.max(jnp.abs(all_states))) <= 5.0
    assert bool(jnp.all(jnp.diff(all_states) >= -1e-13))
    assert_allclose(final, -.013063101278542104, rtol=0, atol=2e-13)
    # Root count is additionally established by reference.default_certificate(),
    # not inferred from this one initial condition or a sign-change grid.


def test_derivatives_at_zero_clip_knee_and_tails(api):
    f=api[0]
    knee = 3.
    x = np.array([-50., -knee, -2., -1., -0., 0., 1., 2.,
                  np.nextafter(knee, -np.inf), knee,
                  np.nextafter(knee, np.inf), 50.])
    params = jnp.array([.4, .2, .1])
    def objective(z):
        return f(jnp.array([z]), params)
    result = jax.jit(jax.vmap(jax.grad(objective)))(jnp.asarray(x))
    expected = np.array([.4+.4*float(clip(v)*clip_derivative(v)) for v in x])
    assert bool(jnp.all(jnp.isfinite(result)))
    assert_allclose(result, expected, rtol=2e-12, atol=2e-13)
    gradient_params = jax.grad(f, argnums=1)(jnp.array([2.]), params)
    assert_allclose(gradient_params, [2., float(clip(2)**2), 1.], rtol=3e-14)
    # Gradient correctness is separate from zero nonlinear coefficients.
    identity_grad = jax.grad(f)(jnp.array([1e80]), jnp.array([1., 0., 0.]))
    assert_allclose(identity_grad, [1.], rtol=0, atol=2e-14)


@pytest.mark.parametrize('scale,power', [(.5,2), (3.,4), (9.,8)])
def test_static_configuration_and_scaling(api, scale, power):
    f = api[0]
    compiled = jax.jit(f, static_argnames=('scale','power'))
    params=jnp.array([0., 1., 0.])
    for x in (-2*scale, 0., .5*scale, 2*scale):
        value=compiled(jnp.array([x]), params, scale=scale, power=power)
        assert_allclose(value, float(clip(x, scale, power)**2), rtol=5e-14, atol=5e-14)
    # Algebraic covariance requires transforming the clip scale with the units.
    g1=tf._soft_clip(jnp.array(2.), 3., power)
    g2=tf._soft_clip(jnp.array(10.), 15., power)
    assert_allclose(g2, 5*g1, rtol=4e-14)


def test_identity_and_name_companions(api):
    f=api[0]
    factors=('a','b','c')
    names=tf.params_translog(factors)
    assert tf.params_stable_translog(factors)==names
    for own in range(3):
        p=jnp.zeros(len(names)).at[own].set(1.)
        x=jnp.array([1e80, -25., 7.])
        assert_array_equal(f(x,p),x[own])
        got=tf.identity_constraints_stable_translog(factors[own],0,factors)
        want=tf.identity_constraints_translog(factors[own],0,factors)
        assert [(v.loc,v.value) for v in got]==[(v.loc,v.value) for v in want]


@pytest.mark.parametrize('scale', [0., -3., math.inf, math.nan])
def test_invalid_scale_fails_loud(api, scale):
    with pytest.raises(ValueError, match='scale'):
        api[0](jnp.array([0.]), jnp.array([1.,0.,0.]), scale=scale)


@pytest.mark.parametrize('power', [0, 1, 3, 4.0, True])
def test_invalid_power_fails_loud(api, power):
    with pytest.raises((ValueError, TypeError), match='power'):
        api[0](jnp.array([0.]), jnp.array([1.,0.,0.]), power=power)
