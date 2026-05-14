"""Tests for the jaxopt optimizer backend in AF estimation.

The jaxopt backend (`optimizer_backend="jaxopt"`) keeps the parameter
vector on device through L-BFGS-B iterations, avoiding the
host<->device transfer that optimagic incurs once per likelihood call.
It supports `FixedConstraintWithValue` plus bounds; probability and
equality constraints raise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.af.jaxopt_backend import (
    JaxoptResult,
    minimize_with_jaxopt,
)
from skillmodels.common.constraints import FixedConstraintWithValue
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _linear_single_factor_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("m1", "m2", "m3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"m1": 1},) * 2,
                    intercepts=({"m1": 0},) * 2,
                ),
                transition_function="linear",
            ),
        },
    )


def _linear_single_factor_data(n_obs: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_obs)
    rows = []
    for i in range(n_obs):
        for t in range(2):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "m1": theta[i] + rng.normal(0, 0.3),
                    "m2": 0.5 + 0.8 * theta[i] + rng.normal(0, 0.4),
                    "m3": -0.2 + 1.2 * theta[i] + rng.normal(0, 0.35),
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def test_optimizer_backend_defaults_to_auto() -> None:
    """The default backend is `"auto"`; resolved inside `estimate_af`.

    Resolution picks `"jaxopt"` when a JAX GPU is visible and the
    model is jaxopt-compatible (no `log_ces*` transitions, no
    user-supplied constraints); otherwise falls back to
    `"optimagic"`. See ``af.estimate._resolve_optimizer_backend``.
    """
    options = AFEstimationOptions()
    assert options.optimizer_backend == "auto"


def test_optimizer_backend_rejects_unknown_value() -> None:
    """Typos in the backend name fail fast via the beartype perimeter."""
    from skillmodels.exceptions import OptionsInitializationError  # noqa: PLC0415

    with pytest.raises(OptionsInitializationError, match="optimizer_backend"):
        AFEstimationOptions(optimizer_backend="lbfgsb")  # ty: ignore[invalid-argument-type]


def test_minimize_with_jaxopt_recovers_quadratic_minimum() -> None:
    """Smoke-test the wrapper on a small quadratic loss."""
    target = jnp.array([1.0, -2.0, 0.5])

    def loglike_and_grad(x):
        loss = jnp.sum((x - target) ** 2)
        return loss, jax.grad(lambda y: jnp.sum((y - target) ** 2))(x)

    df = pd.DataFrame(
        {
            "value": [0.0, 0.0, 0.0],
            "lower_bound": [-np.inf, -np.inf, -np.inf],
            "upper_bound": [np.inf, np.inf, np.inf],
        },
        index=pd.MultiIndex.from_tuples(
            [("a", 0, "x", "-"), ("a", 0, "y", "-"), ("a", 0, "z", "-")],
            names=["category", "period", "name1", "name2"],
        ),
    )

    result = minimize_with_jaxopt(
        loglike_and_grad=loglike_and_grad,
        full_params_df=df,
        constraints=[],
        optimizer_options={"maxiter": 200, "tol": 1e-8},
    )

    assert isinstance(result, JaxoptResult)
    np.testing.assert_allclose(
        result.params["value"].to_numpy(),
        [1.0, -2.0, 0.5],
        atol=1e-4,
    )
    assert result.fun < 1e-6


def test_minimize_with_jaxopt_respects_pinned_values() -> None:
    """Pinned coordinates keep their target value, others are optimized."""
    target = jnp.array([3.0, -1.5, 7.0])

    def loglike_and_grad(x):
        loss = jnp.sum((x - target) ** 2)
        return loss, jax.grad(lambda y: jnp.sum((y - target) ** 2))(x)

    locs = [("a", 0, "x", "-"), ("a", 0, "y", "-"), ("a", 0, "z", "-")]
    df = pd.DataFrame(
        {
            "value": [0.0, 0.0, 0.0],
            "lower_bound": [-np.inf, -np.inf, -np.inf],
            "upper_bound": [np.inf, np.inf, np.inf],
        },
        index=pd.MultiIndex.from_tuples(
            locs, names=["category", "period", "name1", "name2"]
        ),
    )
    # Pin the second coordinate to a value that is NOT the unconstrained
    # minimum; the optimizer should leave it alone.
    constraints: list[om.constraints.Constraint] = [
        FixedConstraintWithValue(loc=locs[1], value=2.0)
    ]

    result = minimize_with_jaxopt(
        loglike_and_grad=loglike_and_grad,
        full_params_df=df,
        constraints=constraints,
        optimizer_options={"maxiter": 200, "tol": 1e-8},
    )

    values = result.params["value"].to_numpy()
    assert values[1] == pytest.approx(2.0)
    assert values[0] == pytest.approx(3.0, abs=1e-4)
    assert values[2] == pytest.approx(7.0, abs=1e-4)


def test_minimize_with_jaxopt_rejects_unsupported_constraints() -> None:
    """Probability / equality constraints raise a clear error."""

    def loglike_and_grad(x):
        loss = jnp.sum(x**2)
        return loss, 2 * x

    df = pd.DataFrame(
        {
            "value": [0.0, 0.0],
            "lower_bound": [-np.inf] * 2,
            "upper_bound": [np.inf] * 2,
        },
        index=pd.MultiIndex.from_tuples(
            [("a", 0, "x", "-"), ("a", 0, "y", "-")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    with pytest.raises(NotImplementedError, match="optimagic"):
        minimize_with_jaxopt(
            loglike_and_grad=loglike_and_grad,
            full_params_df=df,
            constraints=[om.EqualityConstraint(selector=lambda p: p.loc[("a", 0)])],
            optimizer_options=None,
        )


@pytest.mark.end_to_end
def test_estimate_af_jaxopt_matches_optimagic_on_linear_model() -> None:
    """Both backends converge to similar measurement params on a linear model."""
    model = _linear_single_factor_model()
    data = _linear_single_factor_data(n_obs=200)

    res_optimagic = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=25,
            n_halton_points_shock=10,
            n_mixture_components=1,
            optimizer_backend="optimagic",
            optimizer_algorithm="scipy_lbfgsb",
            initialization_strategy="spearman",
        ),
    )
    res_jaxopt = estimate_af(
        model_spec=model,
        data=data,
        af_options=AFEstimationOptions(
            n_halton_points=25,
            n_halton_points_shock=10,
            n_mixture_components=1,
            optimizer_backend="jaxopt",
            optimizer_options={"maxiter": 500, "tol": 1e-7},
            initialization_strategy="spearman",
        ),
    )

    # Period-level log-likelihoods should match within optimizer tolerance.
    for i in range(2):
        ll_opt = res_optimagic.period_results[i].loglikelihood
        ll_jax = res_jaxopt.period_results[i].loglikelihood
        assert np.isfinite(ll_opt)
        assert np.isfinite(ll_jax)
        # 0.5 nats per period: enough slack for stochastic line searches but
        # tight enough to catch a real divergence.
        assert abs(ll_opt - ll_jax) < 0.5, (
            f"period {i}: optimagic={ll_opt:.4f} vs jaxopt={ll_jax:.4f}"
        )

    # Period-0 free loadings should land in the same neighbourhood.
    free_loadings = res_optimagic.all_params.query(
        "category == 'loadings' and period == 0"
    )
    free_loadings = free_loadings[
        ~free_loadings.index.get_level_values("name2").isin(["m1"])
    ]
    for idx in free_loadings.index:
        v_opt = float(free_loadings.loc[idx, "value"])
        v_jax = float(res_jaxopt.all_params.loc[idx, "value"])
        assert abs(v_opt - v_jax) < 0.1, (
            f"loading {idx}: optimagic={v_opt:.3f} vs jaxopt={v_jax:.3f}"
        )


@pytest.mark.end_to_end
def test_estimate_af_jaxopt_rejects_log_ces_model() -> None:
    """Models with log_ces transitions (probability constraints) must reject."""
    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("m1", "m2", "m3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"m1": 1},) * 2,
                    intercepts=({"m1": 0},) * 2,
                ),
                transition_function="log_ces",
            ),
        },
    )
    rng = np.random.default_rng(0)
    rows = []
    for i in range(100):
        theta = rng.normal(0, 1, size=2)
        for t in range(2):
            rows.append(
                {
                    "caseid": i,
                    "period": t,
                    "m1": theta[t] + rng.normal(0, 0.3),
                    "m2": 0.8 * theta[t] + rng.normal(0, 0.4),
                    "m3": 1.2 * theta[t] + rng.normal(0, 0.35),
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    af_options = AFEstimationOptions(
        n_halton_points=20,
        n_halton_points_shock=10,
        n_mixture_components=1,
        optimizer_backend="jaxopt",
        initialization_strategy="spearman",
    )

    with pytest.raises(NotImplementedError, match="optimagic"):
        estimate_af(model_spec=model, data=data, af_options=af_options)
