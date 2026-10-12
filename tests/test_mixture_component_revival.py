"""Public-API acceptance tests for finite log-weight component preservation.

Reviewer status: written and statically inspected; NOT RUN. Run with the project's
native Python 3.14/JAX environment, preserving the sibling ``oracles`` directory:

    PYTHONPATH=src pixi run -e tests-cpu pytest /path/to/tests/test_mixture_component_revival.py -q

The unequal-SD tests require the target feature and are not representable at base
52bbca1. ``-k option_off`` exercises a pooled control on both base and head.
The suite uses at most ten small public model fixtures; mutations of parameter
values within one fixture reuse the same shape and compiled likelihood.
"""

import importlib.util
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from skillmodels.chs import kalman_filters, kalman_filters_debug
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.process_model import process_model


ORACLE_PATH = Path(__file__).resolve().parent / "mixture_revival_oracle.py"
ORACLE_SPEC = importlib.util.spec_from_file_location("mixture_revival_oracle", ORACLE_PATH)
assert ORACLE_SPEC is not None and ORACLE_SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(ORACLE_SPEC)
ORACLE_SPEC.loader.exec_module(ORACLE)
jax.config.update("jax_enable_x64", True)


def _observed_cases():
    """One data array covers the witness, threshold neighbours and missingness."""
    r = 0.01**2
    threshold = math.sqrt(
        2.0 * (0.5 * math.log(2.0 / (1.0 + r)) - math.log(np.finfo(np.float64).tiny))
        / (1.0 / (1.0 + r) - 0.5)
    )
    rounded = np.float32(threshold)
    below = np.nextafter(rounded, np.float32(-np.inf))
    above = np.nextafter(rounded, np.float32(np.inf))
    assert float(below) < threshold < float(above)
    return np.array(
        [
            [60.0, 60.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [below, below],
            [above, above],
            [60.0, np.nan],
            [np.nan, 60.0],
            [np.nan, np.nan],
        ],
        dtype=np.float64,
    )


def _make_case(
    *,
    sequence=("first", "second"),
    first_sds=(1.0, 0.01),
    weights=(0.5, 0.5),
    scale=1.0,
    intercept=0.0,
    across_periods=False,
    transition_name="linear",
    option_on=True,
):
    """Construct the same supported two-period model for every mutation."""
    factor = FactorSpec(
        measurements=(sequence, sequence),
        normalizations=Normalizations(
            loadings=({"first": 1.0},) * 2,
            intercepts=({"first": intercept},) * 2,
        ),
        transition_function=transition_name,
    )
    model = ModelSpec(
        factors={"skills": factor},
        n_mixtures=len(weights),
        mixture_specific_meas_sds=("first",) if option_on else (),
    )
    observed = _observed_cases() * scale + intercept
    # Match process_data's documented float32 storage exactly, including after
    # a unit transformation. Integer witness values 0, 1, 60 remain exact.
    observed = observed.astype(np.float32).astype(np.float64)
    rows = []
    for i, (first, second) in enumerate(observed):
        for period in (0, 1):
            row = {name: np.nan for name in sequence}
            if period == 0:
                row["first"] = first
            if period == int(across_periods):
                row["second"] = second
            rows.append({"caseid": i, "period": period, **row})
    data = pd.DataFrame(rows).set_index(["caseid", "period"])
    inputs = get_maximization_inputs(
        model,
        data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    params = inputs["params_template"].copy()
    values = []
    for category, _period, name1, name2 in params.index:
        if category == "controls":
            value = intercept
        elif category == "loadings":
            value = 1.0
        elif category == "meas_sds":
            if name1 == "first":
                k = 0 if name2 == "-" else int(name2.rsplit("_", 1)[1])
                value = first_sds[k] * scale
            else:
                value = 0.01 * scale
        elif category == "initial_states":
            value = 0.0
        elif category == "initial_cholcovs":
            value = scale
        elif category == "mixture_weights":
            value = weights[int(name1.rsplit("_", 1)[1])]
        elif category == "shock_sds":
            value = 0.01 * scale
        elif category == "transition":
            # In the translog mutation the square coefficient is zero, so the
            # unscented predictor represents the same identity transition.
            value = 1.0 if name2 == "skills" else 0.0
        else:
            raise AssertionError(category)
        values.append(value)
    params["value"] = np.asarray(values, dtype=np.float64)
    actual_sds = first_sds if option_on else (first_sds[0],) * len(weights)
    oracle_kwargs = {
        "observed": observed,
        "first_sds": np.asarray(actual_sds) * scale,
        "second_sd": 0.01 * scale,
        "weights": weights,
        "initial_sd": scale,
        "shock_sd": 0.01 * scale if across_periods else 0.0,
        "intercept": intercept,
    }
    return {
        "model": model,
        "inputs": inputs,
        "params": params,
        "oracle_kwargs": oracle_kwargs,
        "reference": ORACLE.joint_reference(**oracle_kwargs),
    }


def _assert_public_matches(case):
    inputs, params, ref = case["inputs"], case["params"], case["reference"]
    actual = inputs["loglike"](params)
    assert actual == pytest.approx(ref["value"], rel=0.0, abs=1e-10)
    np.testing.assert_allclose(
        inputs["loglikeobs"](params),
        ref["contributions"],
        rtol=0.0,
        atol=1e-10,
    )


@pytest.fixture(scope="module")
def revival_case():
    return _make_case()


def test_structural_formula_exceeds_the_fast_budget():
    witness = ORACLE.revival_formula()
    assert witness["first_log_posterior_small"] < math.log(np.finfo(np.float64).tiny)
    assert witness["first_log_odds"] < -899.0
    assert witness["joint_log_ratio"] > math.log(60.0)
    assert witness["joint_posterior_small"] > 60.0 / 61.0
    assert witness["reset_loglike_loss"] > 4.1


def test_public_likelihood_preserves_a_component_that_revives(revival_case):
    _assert_public_matches(revival_case)


def test_debug_and_compiled_likelihood_match_joint_reference(revival_case):
    case = revival_case
    result = case["inputs"]["debug_loglike"](case["params"])
    assert result["value"] == pytest.approx(case["reference"]["value"], rel=0, abs=1e-10)


def test_sd_gradients_match_analytical_joint_scores(revival_case):
    case = revival_case
    value, gradient = case["inputs"]["loglike_and_gradient"](case["params"])
    ref = case["reference"]
    assert value == pytest.approx(ref["value"], rel=0, abs=1e-10)
    gradient = pd.Series(gradient, index=case["params"].index)
    actual = np.array(
        [
            gradient.loc[("meas_sds", 0, "first", "mixture_0")],
            gradient.loc[("meas_sds", 0, "first", "mixture_1")],
            gradient.loc[("meas_sds", 0, "second", "-")],
        ]
    )
    expected = np.r_[
        ref["first_sd_scores"].sum(axis=0),
        ref["second_sd_scores"].sum(),
    ]
    relative_error = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-8)
    assert np.isfinite(actual).all()
    assert relative_error.max() <= 1e-5


@pytest.mark.parametrize(
    "variant",
    ["measurement_order", "component_order", "missing_gap", "later_period", "three_components", "unscented"],
)
def test_systematic_supported_mutations(variant):
    kwargs = {}
    if variant == "measurement_order":
        kwargs["sequence"] = ("second", "first")
    elif variant == "component_order":
        kwargs["first_sds"] = (0.01, 1.0)
    elif variant == "missing_gap":
        kwargs["sequence"] = ("first", "gap", "second")
    elif variant == "later_period":
        kwargs["across_periods"] = True
    elif variant == "three_components":
        kwargs["first_sds"] = (1.0, 0.01, 0.03)
        kwargs["weights"] = (0.2, 0.3, 0.5)
    elif variant == "unscented":
        kwargs["transition_name"] = "translog"
        kwargs["across_periods"] = True
    case = _make_case(**kwargs)
    if variant == "measurement_order":
        actual_order = process_model(case["model"]).update_info.index[:2].tolist()
        assert actual_order == [(0, "second"), (0, "first")]
    _assert_public_matches(case)


def test_scale_and_translation_equivariance():
    # SD .02 exceeds the default .001 lower bound; transformed measurements are
    # pre-rounded exactly as in production before the independent comparison.
    _assert_public_matches(_make_case(scale=2.0, intercept=7.0))


def test_sd_neighbourhood_without_another_model_compile(revival_case):
    original = revival_case
    for small_sd in (0.005, 0.02, 0.03):
        params = original["params"].copy()
        params.loc[("meas_sds", 0, "first", "mixture_1"), "value"] = small_sd
        oracle_kwargs = dict(original["oracle_kwargs"], first_sds=(1.0, small_sd))
        case = dict(original, params=params, reference=ORACLE.joint_reference(**oracle_kwargs))
        _assert_public_matches(case)


def test_option_off_pooled_control():
    case = _make_case(option_on=False)
    sd_rows = [row for row in case["params"].index if row[0] == "meas_sds"]
    assert sd_rows == [
        ("meas_sds", period, name, "-")
        for period in (0, 1)
        for name in ("first", "second")
    ]
    _assert_public_matches(case)


def test_exact_zero_component_does_not_change_likelihood():
    _assert_public_matches(_make_case(weights=(1.0, 0.0)))


@pytest.mark.parametrize("update", [kalman_filters.kalman_update, kalman_filters_debug.kalman_update])
def test_missing_vector_update_preserves_finite_logweight_states_eager_and_jit(update):
    # A finite weight that is tiny in level units is a valid carried filtering
    # state. Missing data must not change either component's moments.
    states = jnp.array([[[0.0], [5.0]], [[2.0], [7.0]]])
    chols = jnp.array([[[[1.0]], [[0.2]]], [[[0.7]], [[0.3]]]])
    log_weights = jnp.array([[0.0, -900.0], [0.0, -1000.0]])
    kwargs = {
        "states": states,
        "upper_chols": chols,
        "loadings": jnp.array([1.0]),
        "control_params": jnp.array([0.0]),
        "meas_sd": jnp.array([1.0, 0.01]),
        "measurements": jnp.full(2, jnp.nan),
        "controls": jnp.ones((2, 1)),
        "log_mixture_weights": log_weights,
    }
    for call in (update, jax.jit(update)):
        new_states, new_chols, new_weights, loglikes = call(**kwargs)[:4]
        np.testing.assert_array_equal(new_states, states)
        np.testing.assert_array_equal(new_chols, chols)
        np.testing.assert_array_equal(new_weights, log_weights)
        np.testing.assert_array_equal(loglikes, np.zeros(2))
