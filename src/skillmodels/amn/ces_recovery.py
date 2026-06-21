"""Freyberger's restricted-CES primitive scale recovery (audit F2).

AMN fits the restricted-CES production function in *transformed* (tilde)
coordinates: it estimates the joint distribution of the transformed factors
under the internal anchors lambda~_{theta,t,1}=lambda~_{I,t,1}=1, then fits the
transformed production function (Freyberger eq prod_fn_ces_norm, with psi_t=1)

    ln theta~_{t+1} = (lambda_{theta,t+1,1}/sigma_t)
                      * ln( gamma~_1t theta~_t^{sigma_t/lambda_{theta,t,1}}
                            + gamma~_2t I~_t^{sigma_t/lambda_{I,t,1}} )
                      + kappa~_t eta~_{I,t}.

This is exactly the functional form of `log_ces_general`
(`tfp * log(sum_i gamma_i * state_i^{sigma_i})`), with the per-period
coefficients

    outside (tfp)   = lambda_{theta,t+1,1} / sigma_t
    theta_exponent  = sigma_t / lambda_{theta,t,1}
    inv_exponent    = sigma_t / lambda_{I,t,1}.

The single-rho form previously used by AMN's `_fit_log_ces` (delta + (1/rho) *
log(sum gamma_i exp(X_i rho)), one rho inside and out) CANNOT represent this
when lambda_{theta,t,1} != lambda_{I,t,1}: the audit's counterexample
(lambda_theta=2, lambda_inv=1, lambda_next=1, sigma=-0.5) has best-fit max abs
error ~0.544. Hence restricted CES must be fit through the `log_ces_general`
functional form.

Given a `log_ces_general` fit in transformed coordinates and the *single* scale
anchor lambda_{theta,0,1}=1 (with psi_t=1; paper line 831), this module recovers
the primitive sigma_t, lambda_{theta,t,1} and lambda_{I,t,1} recursively
(paper lines 1357-1366):

    sigma_t              = theta_exponent_t * lambda_{theta,t,1}
    lambda_{I,t,1}       = sigma_t / inv_exponent_t
    lambda_{theta,t+1,1} = outside_t * sigma_t          (carried into period t+1)

The remaining recovery rescalings (paper 1357-1366), which are applied by the
integrator once the scales are known, are:

    lambda_{theta,t,m} = lambda~_{theta,t,m} * lambda_{theta,t,1}
    lambda_{I,t,m}     = lambda~_{I,t,m}     * lambda_{I,t,1}
    beta_0t = beta~_0t / lambda_{I,t,1}
    beta_1t = beta~_1t * lambda_{theta,t,1} / lambda_{I,t,1}
    beta_2t = beta~_2t / lambda_{I,t,1}
    rho_0   = rho~_0
    rho_1   = rho~_1 * lambda_{theta,T,1}       (TERMINAL skill scale; see below)
    ln theta_t = ln theta~_t / lambda_{theta,t,1}
    ln I_t     = ln I~_t     / lambda_{I,t,1}

and, with the same scales, the scale-dependent shock and control categories
(Pro F9):

    sigma_{eta_I,t}     = sigma_{eta~_I,t}     / abs(lambda_{I,t,1})
    sigma_{eta_theta,t} = sigma_{eta~_theta,t} / abs(lambda_{theta,t+1,1})
    kappa_t             = kappa~_t * lambda_{I,t,1} / lambda_{theta,t+1,1}

and the Stage-2 joint mixture: with the block-diagonal map A whose latent
entries are the reciprocal scales 1/lambda_{*,t,1} and whose observed (income,
outcome) entries are 1, every component transforms as m_k = A m~_k and
Sigma_k = A Sigma~_k A^T (this handles latent-latent AND latent-observed
cross-covariances); the mixture weights are unchanged. gamma~ = gamma when all
first-measure locations mu are zero (eq 751).

rho_1 (Pro F10/T1): the paper's recovery paragraph writes a generic t, but the
anchor Q = rho_0 + rho_1 ln(theta_T) is on the TERMINAL skill, and
rho~_1 = rho_1 / lambda_{theta,T,1}, so rho_1 = rho~_1 * lambda_{theta,T,1}. The
generic t is a source transcription artifact; the forward recursion below
correctly produces lambda_{theta,T,1} as its last carried scale.

Scope (Pro T2/F9): this module implements ONLY the scale recursion (unit-tested,
randomized-inversion verified by the reviewer to ~9e-16). The downstream
rescalings above -- including the shock SDs, kappa, and the full joint-mixture
transform -- are NOT yet wired into the AMN pipeline, so a standalone restricted-
CES result would still mix coordinate systems. Because AMN is start-values-only
in production (CHS/AF re-fit every parameter) the standalone guard is KEPT until
the complete affine transformation is implemented and verified; the reviewer
agreed (returning transformed-coordinate parameters in the primitive params
schema would be misleading and is acceptable only behind a separately named API).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _require_nonzero_finite(value: float, what: str) -> None:
    """Raise ValueError unless `value` is finite and nonzero (psi_t=1 domain)."""
    if not math.isfinite(value) or value == 0.0:
        msg = (
            f"recover_primitive_ces_scales requires a nonzero, finite {what} "
            f"(got {value}); the restricted-CES recovery assumes psi_t=1 with "
            "nonzero scales and exponents. The exact Cobb-Douglas limit sigma_t=0 "
            "is outside this parameterization and needs a separate limit form."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class CESTransformedCoeffs:
    """Per-period coefficients of the transformed CES (a `log_ces_general` fit)."""

    outside: float
    """`tfp` = lambda_{theta,t+1,1} / sigma_t (the outside coefficient)."""
    theta_exponent: float
    """`sigma_skills` = sigma_t / lambda_{theta,t,1} (the skills exponent)."""
    inv_exponent: float
    """`sigma_investment` = sigma_t / lambda_{I,t,1} (the investment exponent)."""


@dataclass(frozen=True)
class CESPrimitiveScales:
    """Recovered primitive scales for one transition period t."""

    sigma: float
    """The CES elasticity sigma_t."""
    lambda_theta: float
    """The skills scale lambda_{theta,t,1} at period t."""
    lambda_inv: float
    """The investment scale lambda_{I,t,1} at period t."""
    lambda_theta_next: float
    """The skills scale lambda_{theta,t+1,1} carried into period t+1."""


def recover_primitive_ces_scales(
    coeffs: Sequence[CESTransformedCoeffs],
    *,
    lambda_theta_0: float = 1.0,
) -> list[CESPrimitiveScales]:
    """Recover primitive CES scales from transformed-form coefficients.

    Walks the periods forward from the scale anchor `lambda_theta_0`
    (= lambda_{theta,0,1}, normally 1), inverting the transformed-form map at
    each period and carrying the implied lambda_{theta,t+1,1} into the next.

    Args:
        coeffs: Per-period transformed-CES coefficients, period 0 first.
        lambda_theta_0: The single primitive scale anchor lambda_{theta,0,1}.

    Return:
        One `CESPrimitiveScales` per period, in order.
    """
    _require_nonzero_finite(lambda_theta_0, "lambda_theta_0 anchor")
    lambda_theta = lambda_theta_0
    recovered: list[CESPrimitiveScales] = []
    for coeff in coeffs:
        _require_nonzero_finite(coeff.theta_exponent, "theta_exponent")
        _require_nonzero_finite(coeff.inv_exponent, "inv_exponent")
        _require_nonzero_finite(coeff.outside, "outside coefficient")
        # Forward recursion (paper 1357-1366), confirmed faithful by the Pro
        # review (randomized inversion of 1000 systems, max err ~9e-16):
        # theta_exp = sigma/lambda_theta => sigma = theta_exp * lambda_theta;
        # inv_exp = sigma/lambda_inv => lambda_inv = sigma/inv_exp;
        # outside = lambda_theta_next/sigma => lambda_theta_next = outside*sigma.
        # Anchor is lambda_theta,0,1 (psi_t=1, paper line 831). The domain
        # checks above exclude the sigma_t -> 0 Cobb-Douglas limit (separate
        # parameterization needed).
        sigma = coeff.theta_exponent * lambda_theta
        lambda_inv = sigma / coeff.inv_exponent
        lambda_theta_next = coeff.outside * sigma
        recovered.append(
            CESPrimitiveScales(
                sigma=sigma,
                lambda_theta=lambda_theta,
                lambda_inv=lambda_inv,
                lambda_theta_next=lambda_theta_next,
            )
        )
        lambda_theta = lambda_theta_next
    return recovered
