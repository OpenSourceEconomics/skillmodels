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
    MOMENTS,
    OPTIONS,
    _data,
    _endogenous_data,
    _endogenous_model,
    _loglikeobs,
    _model,
    _params,
)

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.decorators import register_params
from skillmodels.common.transition_functions import params_translog

jax.config.update("jax_enable_x64", True)


def _g(x):
    return x / (1.0 + jnp.abs(x / 3.0) ** 4) ** 0.25


ALL_FACTORS = ("fac1", "fac2", "ob1", "ob2")


@register_params(params=params_translog(ALL_FACTORS))
def REFERENCE(fac1, fac2, ob1, ob2, params):  # noqa: N802
    """Original-formula translog-on-clipped-states as a registered custom callable.

    dags passes factors by argument name, so the signature names them explicitly.
    """
    x = [fac1, fac2, ob1, ob2]
    g = [_g(v) for v in x]
    out = params["constant"]
    for f, v in zip(ALL_FACTORS, x, strict=True):
        out = out + params[f] * v
    for f, gv in zip(ALL_FACTORS, g, strict=True):
        out = out + params[f"{f} ** 2"] * gv**2
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            out = out + params[f"{ALL_FACTORS[i]} * {ALL_FACTORS[j]}"] * g[i] * g[j]
    return out


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


def _endogenous_loglike(transition, nonlinear: float) -> float:
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
    for idx in params.query("category == 'initial_cholcovs'").index:
        row, col = idx[3].split("-")
        params.loc[idx, "value"] = float(row == col)
    return float(inputs["loglike"](params))


def test_endogenous_carry_forward_substeps_accept_stable_translog() -> None:
    """Carry-forward substeps keep their identity/zero-kappa constraints."""
    linear = _endogenous_loglike("linear", nonlinear=0.0)
    assert_allclose(
        _endogenous_loglike("stable_translog", nonlinear=0.0), linear, rtol=1e-10
    )
    stable = _endogenous_loglike("stable_translog", nonlinear=0.1)
    assert np.isfinite(stable)
    assert abs(stable - _endogenous_loglike("translog", nonlinear=0.1)) > 1e-8
