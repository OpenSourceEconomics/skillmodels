"""Accuracy of the CHS likelihood with probit and Tobit measurements.

Each limited update is exact given its Gaussian predicted state; replacing the
non-Gaussian posterior by a Gaussian with the same moments (assumed-density
filtering) makes the likelihood of later measurements approximate. The reference
here integrates the latent path exactly: condition the joint Gaussian path on the
Gaussian and interior-Tobit values, then integrate the probit and censoring
probabilities with tensor Gauss-Hermite quadrature. Its convergence in the quadrature
order is checked, not certified by an error bound.
"""

import math
from functools import cache

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import log_ndtr, logsumexp

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.measurement_models import ProbitMeasurement, TobitMeasurement
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations

_INITIAL_MEAN, _INITIAL_VAR = 0.2, 0.7
_RHO, _TRANS_CONST, _SHOCK_SD = (0.8, 0.7), (0.1, -0.1), (0.4, 0.5)
_G_SD, _T_SD, _T_CONST = 0.7, 0.7, 0.4
_P_CONST = (0.2, -0.1, 0.1)
_LOWER, _UPPER = 0.0, 1.4
_DATA = {
    # person: per period (g1, p1, t1)
    0: ((-0.4, 1.0, 0.0), (0.3, 0.0, 0.6), (0.6, 1.0, 1.4)),
    1: ((0.1, 0.0, 0.7), (-0.2, 1.0, 0.0), (0.9, 1.0, 0.3)),
}


def _path_moments():
    mean = np.zeros(3)
    factor = np.zeros((3, 3))
    mean[0] = _INITIAL_MEAN
    factor[0, 0] = math.sqrt(_INITIAL_VAR)
    for t in range(1, 3):
        mean[t] = _TRANS_CONST[t - 1] + _RHO[t - 1] * mean[t - 1]
        factor[t] = _RHO[t - 1] * factor[t - 1]
        factor[t, t] = _SHOCK_SD[t - 1]
    return mean, factor @ factor.T


@cache
def _grid(order):
    nodes, weights = hermegauss(order)
    log_w = np.log(weights / math.sqrt(2 * math.pi))
    mesh = np.meshgrid(*([nodes] * 3), indexing="ij")
    grid = np.stack([x.ravel() for x in mesh], axis=1)
    log_weights = sum(
        np.meshgrid(*([log_w] * 3), indexing="ij")[i].ravel() for i in range(3)
    )
    return grid, log_weights


def _exact_path_loglike(person, p_loading, t_loading, order):
    mean, cov = _path_moments()
    rows, values, sds, tails = [], [], [], []
    for t, (g, p, y) in enumerate(_DATA[person]):
        rows.append((t, 1.0, 0.0))
        values.append(g)
        sds.append(_G_SD)
        tails.append((t, p_loading, _P_CONST[t], 0.0, 2 * p - 1, 1.0))
        if _LOWER < y < _UPPER:
            rows.append((t, t_loading, _T_CONST))
            values.append(y)
            sds.append(_T_SD)
        else:
            direction, threshold = (-1.0, _LOWER) if y <= _LOWER else (1.0, _UPPER)
            tails.append((t, t_loading, _T_CONST, threshold, direction, _T_SD))
    h = np.zeros((len(rows), 3))
    offsets = np.zeros(len(rows))
    for i, (t, loading, const) in enumerate(rows):
        h[i, t] = loading
        offsets[i] = const
    variance = h @ cov @ h.T + np.diag(np.square(sds))
    chol = np.linalg.cholesky(variance)
    white = np.linalg.solve(chol, np.array(values) - offsets - h @ mean)
    density = -0.5 * (
        len(rows) * math.log(2 * math.pi)
        + 2 * np.log(np.diag(chol)).sum()
        + white @ white
    )
    cross = cov @ h.T
    post_mean = mean + cross @ np.linalg.solve(chol.T, white)
    solved = np.linalg.solve(chol, cross.T)
    post_cov = cov - solved.T @ solved
    grid, log_weights = _grid(order)
    latent = post_mean + grid @ np.linalg.cholesky(post_cov).T
    log_integrand = np.zeros(len(grid))
    for t, loading, const, threshold, direction, sd in tails:
        index = const + loading * latent[:, t]
        log_integrand += log_ndtr(direction * (index - threshold) / sd)
    return density + float(logsumexp(log_weights + log_integrand))


def _converged_exact_loglike(person, p_loading, t_loading):
    values = [
        _exact_path_loglike(person, p_loading, t_loading, n) for n in (24, 32, 48)
    ]
    assert max(abs(np.diff(values))) < 1e-11
    return values[-1]


def _model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("g1", "p1", "t1"),) * 3,
                normalizations=Normalizations(
                    loadings=({"g1": 1},) * 3, intercepts=({"g1": 0},) * 3
                ),
                transition_function="linear",
            ),
        },
        measurement_models={
            "p1": ProbitMeasurement(),
            "t1": TobitMeasurement(lower=_LOWER, upper=_UPPER),
        },
    )


def _chs_loglikes(p_loading, t_loading):
    rows = [
        {"caseid": i, "period": t, "g1": g, "p1": p, "t1": y}
        for i, periods in _DATA.items()
        for t, (g, p, y) in enumerate(periods)
    ]
    inputs = get_maximization_inputs(
        model_spec=_model(),
        data=pd.DataFrame(rows).set_index(["caseid", "period"]),
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    params = inputs["params_template"].copy()
    values = {}
    for t in range(3):
        values |= {
            ("controls", t, "g1", "constant"): 0.0,
            ("controls", t, "p1", "constant"): _P_CONST[t],
            ("controls", t, "t1", "constant"): _T_CONST,
            ("loadings", t, "g1", "skills"): 1.0,
            ("loadings", t, "p1", "skills"): p_loading,
            ("loadings", t, "t1", "skills"): t_loading,
            ("meas_sds", t, "g1", "-"): _G_SD,
            ("meas_sds", t, "p1", "-"): 1.0,
            ("meas_sds", t, "t1", "-"): _T_SD,
        }
    for t in range(2):
        values |= {
            ("shock_sds", t, "skills", "-"): _SHOCK_SD[t],
            ("transition", t, "skills", "skills"): _RHO[t],
            ("transition", t, "skills", "constant"): _TRANS_CONST[t],
        }
    values |= {
        ("initial_states", 0, "mixture_0", "skills"): _INITIAL_MEAN,
        ("initial_cholcovs", 0, "mixture_0", "skills-skills"): math.sqrt(_INITIAL_VAR),
        ("mixture_weights", 0, "mixture_0", "-"): 1.0,
    }
    for loc, value in values.items():
        params.loc[loc, "value"] = value
    return inputs["loglikeobs"](params)


def test_zero_limited_loadings_give_the_exact_path_likelihood() -> None:
    """Without loadings the limited events are independent of the state."""
    exact = [_converged_exact_loglike(i, 0.0, 0.0) for i in sorted(_DATA)]
    np.testing.assert_allclose(_chs_loglikes(0.0, 0.0), exact, atol=1e-10)


def test_assumed_density_gap_shrinks_with_the_limited_loadings() -> None:
    gaps = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        p_loading, t_loading = 0.8 * scale, 0.6 * scale
        chs = _chs_loglikes(p_loading, t_loading)
        exact = [_converged_exact_loglike(i, p_loading, t_loading) for i in _DATA]
        gaps.append(np.abs(chs - exact).max())
    assert gaps[0] < 1e-2
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3]
