"""stable_translog through the CHS likelihood, against an independent same-law oracle.

Fixture: the two-latent, two-observed, three-period panel of
tests/test_missing_observed_factors.py (with missing observed-factor cells, so the
sigma-point integration over N(m, v) fills is exercised too) and its endogenous
two-period model with carry-forward substeps.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_missing_observed_factors import (
    LATENT,
    MOMENTS,
    OPTIONS,
    _data,
    _endogenous_data,
    _endogenous_model,
    _loglikeobs,
    _model,
    _params,
    _predict_inputs,
)

from skillmodels.chs.kalman_filters import (
    _calculate_sigma_points,
    calculate_sigma_scaling_factor_and_weights,
    kalman_predict,
)
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.decorators import register_params
from skillmodels.common.transition_functions import params_translog, stable_translog

jax.config.update("jax_enable_x64", True)


def _g(x):
    return x / (1.0 + jnp.abs(x / 3.0) ** 4) ** 0.25


ALL_FACTORS = ("fac1", "fac2", "ob1", "ob2")


def _reference_formula(x, names, params):
    """Original formula: translog on soft-clipped states; linear part untouched."""
    g = [_g(v) for v in x]
    out = params["constant"]
    for f, v in zip(names, x, strict=True):
        out = out + params[f] * v
    for f, gv in zip(names, g, strict=True):
        out = out + params[f"{f} ** 2"] * gv**2
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            out = out + params[f"{names[i]} * {names[j]}"] * g[i] * g[j]
    return out


@register_params(params=params_translog(ALL_FACTORS))
def REFERENCE(fac1, fac2, ob1, ob2, params):  # noqa: N802
    """Original-formula translog-on-clipped-states as a registered custom callable.

    dags passes factors by argument name, so the signature names them explicitly.
    """
    return _reference_formula([fac1, fac2, ob1, ob2], ALL_FACTORS, params)


def _loglikeobs_and_grad(transition, data, nonlinear):
    inputs = get_maximization_inputs(
        _model(2, transition, MOMENTS), data, chs_options=OPTIONS
    )
    params = _params(inputs["params_template"], nonlinear=nonlinear)
    _, grad = inputs["loglike_and_gradient"](params)
    return np.asarray(inputs["loglikeobs"](params)), np.asarray(grad)


def test_zero_nonlinear_params_reproduce_linear_filter() -> None:
    data = _data()
    linear = _loglikeobs(_model(2, "linear", MOMENTS), data)
    stable = _loglikeobs(_model(2, "stable_translog", MOMENTS), data)
    assert_allclose(stable, linear, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "ob_scale", [1.0, 10.0], ids=["inputs_inside_scale", "inputs_beyond_scale"]
)
def test_contributions_and_gradient_match_same_law_reference(ob_scale) -> None:
    data = _data()
    data[["ob1", "ob2"]] *= ob_scale
    stable, g_stable = _loglikeobs_and_grad("stable_translog", data, nonlinear=0.05)
    reference, g_reference = _loglikeobs_and_grad(REFERENCE, data, nonlinear=0.05)
    translog, _ = _loglikeobs_and_grad("translog", data, nonlinear=0.05)
    assert np.all(np.isfinite(stable))
    assert np.all(np.isfinite(g_stable))
    assert_allclose(stable, reference, rtol=1e-10, atol=1e-10)
    assert_allclose(g_stable, g_reference, rtol=1e-8, atol=1e-10)
    # The sigma points reach the nonlinear region: an alias to translog must fail here.
    assert np.max(np.abs(stable - translog)) > 1e-6


ENDOGENOUS_FACTORS = ("s", "inv", "x", "z")


@register_params(params=params_translog(ENDOGENOUS_FACTORS))
def REFERENCE_ENDOGENOUS(s, inv, x, z, params):  # noqa: N802
    return _reference_formula([s, inv, x, z], ENDOGENOUS_FACTORS, params)


def _endogenous_inputs(transition, nonlinear: float):
    model = _endogenous_model(None)
    # with_transition_functions requires every factor; replace only `s`.
    factors = dict(model.factors) | {
        "s": replace(model.factors["s"], transition_function=transition)
    }
    model = model._replace(factors=factors)
    inputs = get_maximization_inputs(model, _endogenous_data(), chs_options=OPTIONS)
    params = inputs["params_template"].copy()
    cat = params.index.get_level_values("category")
    name2 = params.index.get_level_values(3).astype(str)
    params["value"] = np.where(
        cat.isin(["meas_sds", "shock_sds", "loadings"]), 0.5, 0.1
    )
    params.loc[(cat == "transition") & name2.str.contains(r"\*"), "value"] = nonlinear
    # Carry-forward substep of `s` (aug_period 0): identity, zero on everything else.
    # Built-ins get this from identity_constraints_*; the custom reference callable has
    # no such hook, so the values are written explicitly for every law.
    carry = (cat == "transition") & (params.index.get_level_values("aug_period") == 0)
    carry &= params.index.get_level_values("name1") == "s"
    params.loc[carry, "value"] = np.where(name2[carry] == "s", 1.0, 0.0)
    for idx in params.query("category == 'initial_cholcovs'").index:
        row, col = idx[3].split("-")
        params.loc[idx, "value"] = float(row == col)
    return inputs, params


def _endogenous_loglike(transition, nonlinear: float) -> float:
    inputs, params = _endogenous_inputs(transition, nonlinear)
    return float(inputs["loglike"](params))


def _endogenous_loglikeobs_and_grad(transition, nonlinear: float):
    inputs, params = _endogenous_inputs(transition, nonlinear)
    _, grad = inputs["loglike_and_gradient"](params)
    return np.asarray(inputs["loglikeobs"](params)), np.asarray(grad)


def test_endogenous_carry_forward_substeps_accept_stable_translog() -> None:
    """Carry-forward substeps keep their identity/zero-kappa constraints."""
    linear = _endogenous_loglike("linear", nonlinear=0.0)
    assert_allclose(
        _endogenous_loglike("stable_translog", nonlinear=0.0), linear, rtol=1e-10
    )
    stable = _endogenous_loglike("stable_translog", nonlinear=0.1)
    assert np.isfinite(stable)
    assert abs(stable - _endogenous_loglike("translog", nonlinear=0.1)) > 1e-8


def test_endogenous_contributions_and_gradient_match_same_law_reference() -> None:
    stable, g_stable = _endogenous_loglikeobs_and_grad("stable_translog", 0.1)
    reference, g_reference = _endogenous_loglikeobs_and_grad(REFERENCE_ENDOGENOUS, 0.1)
    translog, _ = _endogenous_loglikeobs_and_grad("translog", 0.1)
    assert np.all(np.isfinite(g_stable))
    assert_allclose(stable, reference, rtol=1e-10, atol=1e-10)
    assert_allclose(g_stable, g_reference, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(stable - translog)) > 1e-8


def _g_np(x):
    return x / (1.0 + np.abs(x / 3.0) ** 4) ** 0.25


def _stable_translog_transition(params, states):
    """`kalman_predict` transition signature: (trans_coeffs, states (N, n_fac))."""
    return jnp.column_stack(
        [
            jax.vmap(stable_translog, in_axes=(0, None))(states, params[f])
            for f in LATENT
        ]
    )


def test_unscented_predict_moments_equal_sigma_weighted_reference_law() -> None:
    """Predicted moments are the sigma-weighted moments of the clipped law.

    The fixture's own sigma points and weights are pushed through an independent
    NumPy evaluation of the original formula; weighted mean and covariance plus the
    shock variance must equal what `kalman_predict` returns for `stable_translog`.
    """
    inp = _predict_inputs()
    rng = np.random.default_rng(1)
    n_fac = 4
    names = (*LATENT, "ob1", "ob2")
    n_par = len(params_translog(names))
    coeffs = {f: jnp.array(rng.normal(scale=0.3, size=n_par)) for f in LATENT}
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(2, 2)
    shock_sds = inp["shock_sds"] / inp["scaling"][1, :2]
    means, chols = kalman_predict(
        _stable_translog_transition,
        jnp.array(inp["states"]),
        jnp.array(inp["chols"]),
        scaling_factor,
        weights,
        coeffs,
        jnp.array(shock_sds),
        jnp.array(inp["scaling"]),
        jnp.array(inp["constants"]),
        jnp.array(inp["observed"]),
    )
    covs = np.einsum("nkja,nkjb->nkab", chols, chols)
    sigma = np.asarray(
        _calculate_sigma_points(
            jnp.array(inp["states"]),
            jnp.array(inp["chols"]),
            scaling_factor,
            jnp.array(inp["observed"]),
        )
    )
    w = np.asarray(weights)
    s_in, s_out = inp["scaling"][0, :2], inp["scaling"][1, :2]
    c_in, c_out = inp["constants"][0, :2], inp["constants"][1, :2]
    n_obs, n_mix, n_sigma, _ = sigma.shape
    for i in range(n_obs):
        for k in range(n_mix):
            out = np.empty((n_sigma, 2))
            for q in range(n_sigma):
                x = sigma[i, k, q].copy()
                x[:2] = x[:2] * s_in + c_in
                g = _g_np(x)
                for j, f in enumerate(LATENT):
                    p = np.asarray(coeffs[f])
                    val = p[-1] + p[:n_fac] @ x + p[n_fac : 2 * n_fac] @ g**2
                    pos = 2 * n_fac
                    for a in range(n_fac):
                        for b in range(a + 1, n_fac):
                            val += p[pos] * g[a] * g[b]
                            pos += 1
                    out[q, j] = (val - c_out[j]) / s_out[j]
            mean = w @ out
            dev = out - mean
            cov = (dev * w[:, None]).T @ dev + np.diag(shock_sds**2)
            assert_allclose(means[i, k], mean, rtol=1e-10, atol=1e-12)
            assert_allclose(covs[i, k], cov, rtol=1e-10, atol=1e-12)
