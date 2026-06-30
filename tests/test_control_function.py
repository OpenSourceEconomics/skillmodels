"""Tests for the control-function (cf) DAG-node construction."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from skillmodels.common.control_function import (
    build_cf_node,
    build_kappa_addition_node,
    build_kappa_term_evaluators,
    build_prediction_node,
    compute_investment_residual_sds,
    generate_kappa_terms,
)
from skillmodels.common.model_spec import CorrectionSpec
from skillmodels.common.params_index import get_params_index
from skillmodels.common.parse_params import create_parsing_info, parse_params
from skillmodels.common.process_model import process_model
from skillmodels.test_data.model2 import MODEL2


def _corr_model_with_instrument():
    """MODEL2 with fac3 endogenous, an observed instrument, and a correction."""
    fac3 = MODEL2.factors["fac3"]
    corr = CorrectionSpec(instruments=("inv_z",))
    new_fac3 = replace(fac3, is_endogenous=True, correction=corr)
    new_factors = dict(MODEL2.factors) | {"fac3": new_fac3}
    model = MODEL2._replace(factors=new_factors)._replace(stagemap=None)
    model = model._replace(observed_factors=("inv_z",))
    return process_model(model)


def _period0_trans_coeffs(processed):
    index = get_params_index(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        transition_info=processed.transition_info,
        endogenous_factors_info=processed.endogenous_factors_info,
    )
    parsing_info = create_parsing_info(
        params_index=index,
        update_info=processed.update_info,
        labels=processed.labels,
        anchoring=processed.anchoring,
        has_endogenous_factors=True,
    )
    params_vec = jnp.arange(len(index)).astype(float)
    _, _, _, parsed = parse_params(
        params_vec, parsing_info, processed.dimensions, processed.labels, n_obs=5
    )
    return {key: value[0] for key, value in parsed.transition.items()}


def test_transition_adds_kappa_times_cf_to_target_output() -> None:
    """The wired transition adds kappa*cf to a target factor's base output.

    Invariant test independent of the base production form: zeroing kappa
    recovers the base, and the difference equals kappa * cf, where cf is formed
    contemporaneously from the INPUT investment state and the first-stage
    prediction of it.
    """
    processed = _corr_model_with_instrument()
    all_factors = processed.labels.all_factors  # ('fac1','fac2','fac3','inv_z')
    trans = _period0_trans_coeffs(processed)

    states = jnp.array([1.0, 2.0, 3.0, 4.0])  # one anchored sigma-point row
    target = "fac2"
    individual = processed.transition_info.individual_functions[target]

    out_full = individual(trans, states[None, :])
    trans_zero_kappa = {
        **trans,
        "__kappa_fac2__": jnp.zeros_like(trans["__kappa_fac2__"]),
    }
    out_base = individual(trans_zero_kappa, states[None, :])

    # Reconstruct cf = ln_inv(fac3) - first-stage prediction.
    betas = trans["__first_stage_fac3__"]  # order: fac1, fac2, inv_z, constant
    predictor_positions = [all_factors.index(f) for f in ("fac1", "fac2", "inv_z")]
    x = jnp.concatenate([states[jnp.array(predictor_positions)], jnp.array([1.0])])
    prediction = jnp.dot(x, betas)
    cf = states[all_factors.index("fac3")] - prediction
    kappa = trans["__kappa_fac2__"][0]  # single 'cf' coefficient

    assert_allclose(float(out_full[0] - out_base[0]), float(kappa * cf), rtol=1e-6)
    # Sanity: kappa is non-zero here, so the correction actually moved the output.
    assert abs(float(kappa * cf)) > 1e-6


def test_prediction_node_is_contemporaneous_linear_combination() -> None:
    # all_factors order: [health_mom, health_kid, ln_inv, z1]; predict ln_inv from
    # the state factors (positions 0, 1) and the instrument z1 (position 3).
    prediction = build_prediction_node(
        beta_key="__first_stage_ln_inv__", predictor_positions=(0, 1, 3)
    )
    states = jnp.array([2.0, 3.0, 99.0, 5.0])  # ln_inv value (99) must be ignored
    params = {"__first_stage_ln_inv__": jnp.array([1.0, 0.5, 2.0, 0.1])}
    # 1.0*2 + 0.5*3 + 2.0*5 + 0.1(constant) = 2 + 1.5 + 10 + 0.1
    assert_allclose(float(prediction(states, params)), 13.6)


def test_cf_node_is_investment_minus_prediction() -> None:
    cf = build_cf_node(inv_pos=2)
    states = jnp.array([2.0, 3.0, 7.0, 5.0])
    out = cf(states, jnp.array(4.0))
    assert_allclose(float(out), 3.0)  # ln_inv 7 - pred 4


def test_kappa_term_evaluators_cover_cf_square_and_interaction() -> None:
    evals = build_kappa_term_evaluators(
        kappa_terms=("cf", "cf ** 2", "cf * health_mom"),
        factor_positions={"health_mom": 0, "health_kid": 1},
    )
    states = jnp.array([2.0, 3.0, 7.0])
    cf_val = jnp.array(4.0)
    assert_allclose(float(evals[0](cf_val, states)), 4.0)
    assert_allclose(float(evals[1](cf_val, states)), 16.0)
    assert_allclose(float(evals[2](cf_val, states)), 8.0)  # cf * health_mom = 4*2


def test_kappa_term_evaluators_handle_higher_order_monomials() -> None:
    evals = build_kappa_term_evaluators(
        kappa_terms=("cf ** 2 * health_mom * health_kid", "cf * health_mom ** 2"),
        factor_positions={"health_mom": 0, "health_kid": 1},
    )
    states = jnp.array([2.0, 3.0])
    cf_val = jnp.array(4.0)
    # cf**2 * mom * kid = 16 * 2 * 3 = 96
    assert_allclose(float(evals[0](cf_val, states)), 96.0)
    # cf * mom**2 = 4 * 4 = 16
    assert_allclose(float(evals[1](cf_val, states)), 16.0)


def test_kappa_term_evaluators_reject_term_without_cf() -> None:
    with pytest.raises(ValueError, match="cf"):
        build_kappa_term_evaluators(
            kappa_terms=("health_mom",), factor_positions={"health_mom": 0}
        )


def test_generate_kappa_terms_reproduces_translog_set_at_degree_two() -> None:
    terms = generate_kappa_terms(("health_mom", "health_kid"), max_degree=2)
    assert set(terms) == {
        "cf",
        "cf * health_mom",
        "cf * health_kid",
        "cf ** 2",
    }


def test_generate_kappa_terms_degree_one_is_just_cf() -> None:
    assert generate_kappa_terms(("health_mom", "health_kid"), max_degree=1) == ("cf",)


def test_generate_kappa_terms_includes_higher_order_and_squares() -> None:
    terms = set(generate_kappa_terms(("a", "b"), max_degree=4))
    # the user's example, plus a state square, must be present at degree 4
    assert "cf ** 2 * a * b" in terms
    assert "cf * a ** 2" in terms
    # every term has cf power >= 1 and total degree <= 4
    assert all(t.startswith("cf") for t in terms)


def test_kappa_addition_node_adds_kappa_times_cf_to_base_output() -> None:
    # Kappa lives in its own key; the base node's output is passed in unchanged.
    evals = build_kappa_term_evaluators(
        kappa_terms=("cf", "cf ** 2"), factor_positions={}
    )
    add_kappa = build_kappa_addition_node(
        kappa_key="__kappa_health_mom__", kappa_evaluators=evals
    )
    states = jnp.array([2.0, 3.0])
    params = {"__kappa_health_mom__": jnp.array([0.2, 0.3])}
    base_value = jnp.array(5.5)
    cf_val = jnp.array(4.0)
    # base unchanged + kappa = 5.5 + 0.2*4 + 0.3*16
    expected = 5.5 + 0.2 * 4.0 + 0.3 * 16.0
    out = add_kappa(base_value, params, cf_val, states)
    assert_allclose(float(out), expected)


def test_investment_residual_sds_recovers_known_noise_sd() -> None:
    rng = np.random.default_rng(7)
    n_obs, n_periods, n_pred = 20_000, 3, 2
    betas = jnp.array([[1.0, -0.5, 0.3], [0.8, 0.2, 1.0], [0.5, 0.5, -0.4]])
    predictors = jnp.asarray(rng.normal(size=(n_obs, n_periods, n_pred)))
    ones = jnp.ones((n_obs, n_periods, 1))
    design = jnp.concatenate([predictors, ones], axis=-1)
    mean = jnp.einsum("opk,pk->op", design, betas)
    true_sds = jnp.array([0.5, 1.0, 1.5])
    noise = jnp.asarray(rng.normal(size=(n_obs, n_periods))) * true_sds
    investment = mean + noise

    sds = compute_investment_residual_sds(investment, predictors, betas)
    assert sds.shape == (n_periods,)
    assert_allclose(np.asarray(sds), np.asarray(true_sds), rtol=0.05)


def test_kappa_addition_node_is_identity_when_kappa_is_zero() -> None:
    evals = build_kappa_term_evaluators(kappa_terms=("cf",), factor_positions={})
    add_kappa = build_kappa_addition_node(
        kappa_key="__kappa_health_mom__", kappa_evaluators=evals
    )
    params = {"__kappa_health_mom__": jnp.array([0.0])}
    out = add_kappa(jnp.array(5.5), params, jnp.array(4.0), jnp.array([2.0, 3.0]))
    assert_allclose(float(out), 5.5)
