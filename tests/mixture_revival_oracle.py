"""Independent joint Gaussian oracle for the component-revival counterexample.

This file was written and statically inspected by the reviewer, not executed.
It deliberately does not import skillmodels, JAX, or a Kalman-filter routine.

Model: X ~ N(0, initial_sd**2) in every initial component;
Y_first = intercept + X + error_first[k];
Y_second = intercept + X + shock + error_second.
All errors are independent. A nonzero shock_sd represents an intervening identity
transition. Missing outcomes are integrated out analytically.
"""

import math

import numpy as np
from scipy.special import logsumexp


def joint_reference(
    observed,
    first_sds,
    second_sd,
    weights,
    *,
    initial_sd=1.0,
    shock_sd=0.0,
    intercept=0.0,
):
    """Return contributions, posterior weights and analytical SD scores.

    ``observed`` is (n_individuals, 2); ``first_sds`` and ``weights`` are
    (n_components,). The first-SD score has one column per component. The
    second-SD score is for the single pooled second SD.

    For V=Var(X), a=Var(error_first), b=Var(shock+error_second),
    S=[[V+a,V],[V,V+b]]. Its determinant V*(a+b)+a*b and quadratic form
    [V*(z1-z2)**2+b*z1**2+a*z2**2]/det avoid cancellation. This computes the
    joint density directly; no sequence of measurement updates is reproduced.
    """
    observed = np.asarray(observed, dtype=np.float64)
    first_sds = np.asarray(first_sds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("observed must have shape (n_individuals, 2)")
    if first_sds.shape != weights.shape or first_sds.ndim != 1:
        raise ValueError("first_sds and weights must be aligned vectors")
    if np.any(first_sds <= 0) or second_sd <= 0 or initial_sd <= 0:
        raise ValueError("standard deviations must be positive")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")

    n_individuals, n_components = len(observed), len(weights)
    comp_logpdf = np.zeros((n_individuals, n_components))
    first_score = np.zeros_like(comp_logpdf)
    second_score = np.zeros_like(comp_logpdf)
    v = float(initial_sd) ** 2
    b = float(second_sd) ** 2 + float(shock_sd) ** 2
    log_2pi = math.log(2.0 * math.pi)
    for i, row in enumerate(observed):
        first_seen, second_seen = np.isfinite(row)
        z1, z2 = row - intercept
        for k, first_sd in enumerate(first_sds):
            a = first_sd**2
            if first_seen and second_seen:
                det = v * (a + b) + a * b
                quadratic = (v * (z1 - z2) ** 2 + b * z1**2 + a * z2**2) / det
                comp_logpdf[i, k] = -log_2pi - 0.5 * (math.log(det) + quadratic)
                u1 = (v * (z1 - z2) + b * z1) / det
                u2 = (v * (z2 - z1) + a * z2) / det
                first_score[i, k] = first_sd * (u1**2 - (v + b) / det)
                second_score[i, k] = second_sd * (u2**2 - (v + a) / det)
            elif first_seen:
                variance = v + a
                comp_logpdf[i, k] = -0.5 * (log_2pi + math.log(variance) + z1**2 / variance)
                first_score[i, k] = first_sd * (z1**2 / variance**2 - 1.0 / variance)
            elif second_seen:
                variance = v + b
                comp_logpdf[i, k] = -0.5 * (log_2pi + math.log(variance) + z2**2 / variance)
                second_score[i, k] = second_sd * (z2**2 / variance**2 - 1.0 / variance)

    log_weights = np.full(n_components, -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive])
    weighted = comp_logpdf + log_weights
    contributions = logsumexp(weighted, axis=1)
    posterior = np.exp(weighted - contributions[:, None])
    contributions[~np.isfinite(observed).any(axis=1)] = 0.0
    return {
        "contributions": contributions,
        "value": float(contributions.sum()),
        "posterior_weights": posterior,
        "first_sd_scores": posterior * first_score,
        "second_sd_scores": np.sum(posterior * second_score, axis=1),
    }


def revival_formula(amplitude=60.0, small_sd=0.01):
    """Analytical log odds, exact joint ratio and loss from the historical reset.

    Applies to equal weights, initial N(0,1), first SDs (1,small_sd),
    second SD small_sd, and observed (amplitude,amplitude).
    """
    r = small_sd**2
    first_log_odds = 0.5 * math.log(2.0 / (1.0 + r)) - 0.5 * amplitude**2 * (
        1.0 / (1.0 + r) - 0.5
    )
    joint_log_ratio = 0.5 * math.log((1.0 + 2.0 * r) / (r * (2.0 + r))) - (
        amplitude**2 * r * (1.0 - r) / (2.0 * (2.0 + r) * (1.0 + 2.0 * r))
    )
    return {
        "first_log_odds": first_log_odds,
        "first_log_posterior_small": first_log_odds - float(np.logaddexp(0.0, first_log_odds)),
        "joint_log_ratio": joint_log_ratio,
        "joint_posterior_small": 1.0 / (1.0 + math.exp(-joint_log_ratio)),
        "reset_loglike_loss": float(np.logaddexp(0.0, joint_log_ratio) - np.logaddexp(0.0, first_log_odds)),
    }
