"""asymmetric_translog through the CHS likelihood, against a same-law oracle.

Fixture: the two-latent, two-observed, three-period panel of
tests/test_missing_observed_factors.py and its endogenous two-period model with
carry-forward substeps.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from test_missing_observed_factors import (
    MOMENTS,
    OPTIONS,
    _data,
    _endogenous_data,
    _endogenous_model,
    _model,
    _params,
)

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.decorators import register_params
from skillmodels.common.transition_functions import params_asymmetric_translog

jax.config.update("jax_enable_x64", True)

ALL_FACTORS = ("fac1", "fac2", "ob1", "ob2")


def _reference_formula(x, names, params):
    """Translog with own squares split at the knot, written term by term."""
    out = params["constant"]
    for f, v in zip(names, x, strict=True):
        d = v - params[f"{f} knot"]
        out = out + params[f] * v
        out = out + params[f"{f} ** 2 (above)"] * jnp.where(d > 0, d, 0.0) ** 2
        out = out + params[f"{f} ** 2 (below)"] * jnp.where(d < 0, d, 0.0) ** 2
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            out = out + params[f"{names[i]} * {names[j]}"] * x[i] * x[j]
    return out


@register_params(params=params_asymmetric_translog(ALL_FACTORS))
def REFERENCE(fac1, fac2, ob1, ob2, params):  # noqa: N802
    """Registered custom callable; dags passes factors by argument name."""
    return _reference_formula([fac1, fac2, ob1, ob2], ALL_FACTORS, params)


def _fill(template, *, above: float, below: float, knot: float):
    """`_params` of the fixture for everything but the asymmetric-only entries."""
    name2 = template.index.get_level_values(3).astype(str)
    is_knot = np.asarray(name2.str.endswith(" knot"))
    params = template.copy()
    params.loc[~is_knot, "value"] = _params(template[~is_knot], nonlinear=0.05)["value"]
    params.loc[is_knot, "value"] = knot
    params.loc[np.asarray(name2.str.endswith("(above)")), "value"] = above
    params.loc[np.asarray(name2.str.endswith("(below)")), "value"] = below
    return params


def _loglikeobs_and_grad(transition, data, **values):
    inputs = get_maximization_inputs(
        _model(2, transition, MOMENTS), data, chs_options=OPTIONS
    )
    template = inputs["params_template"]
    params = _fill(template, **values) if values else _params(template, 0.05)
    _, grad = inputs["loglike_and_gradient"](params)
    return np.asarray(inputs["loglikeobs"](params)), np.asarray(grad)


def test_contributions_and_gradient_match_same_law_reference() -> None:
    data = _data()
    values = {"above": 0.08, "below": -0.04, "knot": 0.3}
    asym, g_asym = _loglikeobs_and_grad("asymmetric_translog", data, **values)
    reference, g_reference = _loglikeobs_and_grad(REFERENCE, data, **values)
    assert np.all(np.isfinite(asym))
    assert np.all(np.isfinite(g_asym))
    assert_allclose(asym, reference, rtol=1e-10, atol=1e-10)
    assert_allclose(g_asym, g_reference, rtol=1e-8, atol=1e-10)
    # Sigma points straddle the knot: swapping the two sides must change the result.
    swapped, _ = _loglikeobs_and_grad(
        "asymmetric_translog", data, above=-0.04, below=0.08, knot=0.3
    )
    assert np.max(np.abs(asym - swapped)) > 1e-6


def test_equal_curvature_zero_knots_reproduce_translog() -> None:
    data = _data()
    asym, _ = _loglikeobs_and_grad(
        "asymmetric_translog", data, above=0.05, below=0.05, knot=0.0
    )
    translog, _ = _loglikeobs_and_grad("translog", data)
    assert_allclose(asym, translog, rtol=1e-10, atol=1e-10)


def test_unequal_curvature_differs_from_translog() -> None:
    data = _data()
    asym, _ = _loglikeobs_and_grad(
        "asymmetric_translog", data, above=0.05, below=0.02, knot=0.0
    )
    translog, _ = _loglikeobs_and_grad("translog", data)
    assert np.max(np.abs(asym - translog)) > 1e-8


ENDOGENOUS_FACTORS = ("s", "inv", "x", "z")


@register_params(params=params_asymmetric_translog(ENDOGENOUS_FACTORS))
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
    # Sides of the knot differ (knots stay at 0.1).
    params.loc[(cat == "transition") & name2.str.endswith("(below)"), "value"] = (
        nonlinear / 2
    )
    # Carry-forward substep of `s` (aug_period 0): identity, zero on everything else,
    # knots included. Built-ins get this from identity_constraints_*; the custom
    # reference callable has no such hook, so the values are written explicitly.
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


def test_endogenous_carry_forward_substeps_accept_asymmetric_translog() -> None:
    linear = _endogenous_loglike("linear", nonlinear=0.0)
    assert_allclose(
        _endogenous_loglike("asymmetric_translog", nonlinear=0.0), linear, rtol=1e-10
    )
    asym = _endogenous_loglike("asymmetric_translog", nonlinear=0.1)
    assert np.isfinite(asym)
    assert abs(asym - _endogenous_loglike("translog", nonlinear=0.1)) > 1e-8


def test_endogenous_contributions_and_gradient_match_same_law_reference() -> None:
    asym, g_asym = _endogenous_loglikeobs_and_grad("asymmetric_translog", 0.1)
    reference, g_reference = _endogenous_loglikeobs_and_grad(REFERENCE_ENDOGENOUS, 0.1)
    assert np.all(np.isfinite(g_asym))
    assert_allclose(asym, reference, rtol=1e-10, atol=1e-10)
    assert_allclose(g_asym, g_reference, rtol=1e-8, atol=1e-10)
