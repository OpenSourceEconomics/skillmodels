"""Kernel tests for asymmetric_translog."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from skillmodels.common.transition_functions import (
    asymmetric_translog,
    identity_constraints_asymmetric_translog,
    params_asymmetric_translog,
    translog,
)

jax.config.update("jax_enable_x64", True)

# Toy parameters of tests/test_transition_functions.py::test_translog.
LIN = jnp.array([0.2, 0.1, 0.12])
SQ = jnp.array([0.08, 0.04, 0.05])
INTER = jnp.array([0.05, 0.03, 0.06])
CONST = 0.04
TRANSLOG_PARAMS = jnp.concatenate([LIN, SQ, INTER, jnp.array([CONST])])

STATES = jnp.array(
    [
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 4],
        [0, 0, 0],
        [1, 1, 1],
        [0, -3, 0],
        [-1, -1, -1],
        [1.5, -2, 1.8],
        [12, -34, 48],
        *itertools.product([-2.5, -0.3, 0.4, 2.2], repeat=3),
    ],
    dtype=float,
)


def _asym_params(lin, above, below, knots, inter, const):
    return jnp.concatenate(
        [lin, above, below, knots, inter, jnp.atleast_1d(jnp.asarray(const))]
    )


def _eval(params):
    return jax.vmap(asymmetric_translog, in_axes=(0, None))(STATES, params)


def test_equal_curvature_zero_knots_is_translog() -> None:
    params = _asym_params(LIN, SQ, SQ, jnp.zeros(3), INTER, CONST)
    expected = jax.vmap(translog, in_axes=(0, None))(STATES, TRANSLOG_PARAMS)
    assert_allclose(_eval(params), expected, rtol=1e-12, atol=1e-12)


def test_equal_curvature_nonzero_knots_is_shifted_translog() -> None:
    knots = jnp.array([0.7, -1.3, 0.25])
    params = _asym_params(LIN, SQ, SQ, knots, INTER, CONST)
    shifted = jnp.concatenate(
        [LIN - 2 * SQ * knots, SQ, INTER, jnp.array([CONST + jnp.sum(SQ * knots**2)])]
    )
    expected = jax.vmap(translog, in_axes=(0, None))(STATES, shifted)
    assert_allclose(_eval(params), expected, rtol=1e-12, atol=1e-12)


def test_one_sided_curvature_derivative() -> None:
    below = jnp.array([0.3, 0.2, 0.15])
    knots = jnp.array([0.5, -0.4, 1.0])
    params = _asym_params(LIN, jnp.zeros(3), below, knots, jnp.zeros(3), CONST)
    grad = jax.grad(asymmetric_translog)
    for j in range(3):
        for offset in (0.01, 0.8, 3.0):
            x_above = jnp.zeros(3).at[j].set(knots[j] + offset)
            assert_allclose(grad(x_above, params)[j], LIN[j], rtol=0, atol=1e-15)
            x_below = jnp.zeros(3).at[j].set(knots[j] - offset)
            expected = LIN[j] + 2 * below[j] * (x_below[j] - knots[j])
            assert_allclose(grad(x_below, params)[j], expected, rtol=1e-14)


def test_c1_at_the_knot() -> None:
    above = jnp.array([-0.07, 0.11, 0.02])
    below = jnp.array([0.2, -0.05, 0.09])
    knots = jnp.array([0.5, -0.4, 1.0])
    params = _asym_params(LIN, above, below, knots, INTER, CONST)
    grad_x = jax.grad(asymmetric_translog, argnums=0)
    grad_p = jax.grad(asymmetric_translog, argnums=1)
    base = jnp.array([0.3, -0.2, 0.6])
    for j in range(3):
        lo = base.at[j].set(knots[j] - 1e-7)
        hi = base.at[j].set(knots[j] + 1e-7)
        assert_allclose(
            asymmetric_translog(lo, params),
            asymmetric_translog(hi, params),
            atol=1e-6,
        )
        assert_allclose(grad_x(lo, params), grad_x(hi, params), atol=1e-6)
        knot_slice = slice(9, 12)
        assert_allclose(
            grad_p(lo, params)[knot_slice], grad_p(hi, params)[knot_slice], atol=1e-6
        )


def test_param_names_and_identity_constraints() -> None:
    factors = ("a", "b", "c")
    names = params_asymmetric_translog(factors)
    nfac = len(factors)
    assert len(names) == 4 * nfac + nfac * (nfac - 1) // 2 + 1
    assert names == [
        "a",
        "b",
        "c",
        "a ** 2 (above)",
        "b ** 2 (above)",
        "c ** 2 (above)",
        "a ** 2 (below)",
        "b ** 2 (below)",
        "c ** 2 (below)",
        "a knot",
        "b knot",
        "c knot",
        "a * b",
        "a * c",
        "b * c",
        "constant",
    ]
    constraints = identity_constraints_asymmetric_translog("b", 3, factors)
    assert [c.loc for c in constraints] == [("transition", 3, "b", n) for n in names]
    assert [c.value for c in constraints] == [float(n == "b") for n in names]


def _law_params():
    rng = np.random.default_rng(0)
    out = []
    for j in range(3):
        lin = jnp.zeros(3).at[j].set(0.8)
        p = _asym_params(
            lin,
            jnp.array(rng.uniform(-0.05, 0.05, 3)),
            jnp.array(rng.uniform(-0.05, 0.05, 3)),
            jnp.array(rng.uniform(-0.5, 0.5, 3)),
            jnp.array(rng.uniform(-0.02, 0.02, 3)),
            0.01,
        )
        out.append(p)
    return jnp.stack(out)


def _step(x, law):
    return jax.vmap(asymmetric_translog, in_axes=(None, 0))(x, law)


def test_jit_vmap_and_scan_match_eager() -> None:
    params = _asym_params(LIN, SQ, -SQ, jnp.array([0.7, -1.3, 0.25]), INTER, CONST)
    eager = np.array([float(asymmetric_translog(s, params)) for s in STATES])
    assert_allclose(_eval(params), eager, rtol=1e-14)
    jitted = jax.jit(asymmetric_translog)
    assert_allclose([jitted(s, params) for s in STATES], eager, rtol=1e-14)

    law = _law_params()
    x0 = jnp.array([0.4, -0.6, 1.1])
    x = x0
    path = []
    for _ in range(50):
        x = _step(x, law)
        path.append(x)

    def body(carry, _):
        new = _step(carry, law)
        return new, new

    _, scanned = jax.lax.scan(body, x0, None, length=50)
    assert np.all(np.isfinite(scanned))
    assert_allclose(scanned, jnp.stack(path), rtol=1e-12, atol=1e-14)


def test_fixed_point_witness_for_application_pattern() -> None:
    params = jnp.array([0.97, 0.0, 0.1, -0.5, -0.03])
    # Below the knot: 0.1 x^2 + 0.07 x - 0.005 = 0; the root below the knot.
    fixed_point = (-0.07 - np.sqrt(0.07**2 + 4 * 0.1 * 0.005)) / 0.2

    def body(x, _):
        return asymmetric_translog(jnp.atleast_1d(x), params), None

    for start in (-3.0, -1.0, 0.0, 0.7, 2.0):
        end, _ = jax.lax.scan(body, jnp.asarray(start), None, length=200)
        assert np.isfinite(end)
        assert_allclose(end, fixed_point, atol=1e-3)
