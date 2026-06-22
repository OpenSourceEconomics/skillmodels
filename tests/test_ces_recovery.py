"""Tests for Freyberger's restricted-CES primitive scale recovery (audit F2).

AMN's restricted-CES Stage 3 fits the production function in *transformed*
(tilde) coordinates. The transformed CES (Freyberger eq prod_fn_ces_norm, psi=1)

    ln theta~_{t+1} = (lambda_{theta,t+1,1}/sigma_t)
                      * ln( g1 theta~_t^{sigma_t/lambda_{theta,t,1}}
                            + g2 I~_t^{sigma_t/lambda_{I,t,1}} )

is exactly the functional form of `log_ces_general`
(`tfp * log(sum_i gamma_i * state_i^{sigma_i})`) with

    tfp        = lambda_{theta,t+1,1}/sigma_t        (outside coefficient)
    sigma_skills = sigma_t/lambda_{theta,t,1}        (theta exponent)
    sigma_inv    = sigma_t/lambda_{I,t,1}            (investment exponent).

`recover_primitive_ces_scales` inverts that map: from the per-period
(tfp, theta_exp, inv_exp) and the single scale anchor lambda_{theta,0,1}=1 it
recovers the primitive sigma_t, lambda_{theta,t,1}, lambda_{I,t,1} recursively
(paper lines 1357-1366).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from skillmodels.amn.ces_recovery import (
    CESTransformedCoeffs,
    recover_primitive_ces_scales,
)
from skillmodels.common.transition_functions import log_ces_general


def _transformed_coeffs_from_primitives(
    lambda_theta: list[float],  # lambda_{theta,t,1} for t = 0..T   (len T+1)
    lambda_inv: list[float],  # lambda_{I,t,1}     for t = 0..T-1 (len T)
    sigma: list[float],  # sigma_t            for t = 0..T-1 (len T)
) -> list[CESTransformedCoeffs]:
    """Forward map: primitive scales -> per-period transformed-form coefficients."""
    coeffs = []
    for t in range(len(sigma)):
        coeffs.append(
            CESTransformedCoeffs(
                outside=lambda_theta[t + 1] / sigma[t],
                theta_exponent=sigma[t] / lambda_theta[t],
                inv_exponent=sigma[t] / lambda_inv[t],
            )
        )
    return coeffs


def test_recover_primitive_ces_scales_inverts_known_primitives() -> None:
    # A known primitive CES with non-trivial, period-varying scales.
    lambda_theta = [1.0, 0.82, 0.9]  # lambda_{theta,0,1}=1 is the anchor
    lambda_inv = [1.0, 1.1]
    sigma = [-0.5, -0.4]
    coeffs = _transformed_coeffs_from_primitives(lambda_theta, lambda_inv, sigma)

    recovered = recover_primitive_ces_scales(coeffs, lambda_theta_0=1.0)

    assert [r.sigma for r in recovered] == pytest.approx(sigma)
    assert [r.lambda_theta for r in recovered] == pytest.approx(lambda_theta[:-1])
    assert [r.lambda_inv for r in recovered] == pytest.approx(lambda_inv)
    assert recovered[-1].lambda_theta_next == pytest.approx(lambda_theta[-1])


def test_recover_raises_on_zero_anchor() -> None:
    coeffs = [
        CESTransformedCoeffs(outside=-2.0, theta_exponent=-0.5, inv_exponent=-0.5)
    ]
    with pytest.raises(ValueError, match=r"nonzero|finite"):
        recover_primitive_ces_scales(coeffs, lambda_theta_0=0.0)


def test_recover_raises_on_zero_inv_exponent() -> None:
    coeffs = [CESTransformedCoeffs(outside=-2.0, theta_exponent=-0.5, inv_exponent=0.0)]
    with pytest.raises(ValueError, match=r"nonzero|finite"):
        recover_primitive_ces_scales(coeffs, lambda_theta_0=1.0)


def test_recover_raises_on_nonfinite_coeff() -> None:
    coeffs = [
        CESTransformedCoeffs(
            outside=float("inf"), theta_exponent=-0.5, inv_exponent=-0.5
        )
    ]
    with pytest.raises(ValueError, match=r"nonzero|finite"):
        recover_primitive_ces_scales(coeffs, lambda_theta_0=1.0)


def test_recover_raises_on_overflowing_derived_scale() -> None:
    # Finite nonzero inputs whose products overflow to inf must raise, not
    # silently return infinite primitive scales (Pro F5).
    coeffs = [CESTransformedCoeffs(outside=2.0, theta_exponent=2.0, inv_exponent=1.0)]
    with pytest.raises(ValueError, match=r"nonzero|finite"):
        recover_primitive_ces_scales(coeffs, lambda_theta_0=1e308)


def test_log_ces_general_represents_transformed_ces() -> None:
    # The audit counterexample: lambda_theta=2, lambda_inv=1, lambda_next=1,
    # sigma=-0.5. The single-rho form cannot represent it (best-fit max abs
    # error ~0.544); log_ces_general with the mapped params represents it
    # exactly.
    lam_theta, lam_inv, lam_next, sig = 2.0, 1.0, 1.0, -0.5
    g1, g2 = 0.65, 0.35
    tfp = lam_next / sig
    s_theta = sig / lam_theta
    s_inv = sig / lam_inv

    rng = np.random.default_rng(42)
    states = rng.normal(size=(50, 2))  # [ln theta~, ln I~]

    # Direct transformed-CES target.
    target = tfp * np.log(
        g1 * np.exp(states[:, 0] * s_theta) + g2 * np.exp(states[:, 1] * s_inv)
    )
    # log_ces_general with the mapped params, evaluated row-wise.
    params = jnp.asarray([g1, g2, s_theta, s_inv, tfp])
    got = np.array([float(log_ces_general(jnp.asarray(row), params)) for row in states])

    # Exact up to JAX float precision (x64 not enabled in this unit test).
    np.testing.assert_allclose(got, target, rtol=1e-6, atol=1e-6)
