"""Tests for `skillmodels.amn.simulate_and_regress` (AMN Stage 3)."""

import numpy as np
import pandas as pd

from skillmodels.amn.simulate_and_regress import (
    _draw_factor_panel,
    _fit_linear,
    _fit_log_ces,
    simulate_and_regress,
)
from skillmodels.amn.types import MinimumDistanceResult
from skillmodels.common.model_spec import (
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.process_model import process_model


def _linear_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
        },
        estimation_options=EstimationOptions(
            robust_bounds=True, bounds_distance=0.001, n_mixtures=1
        ),
    )


def _make_structural(
    means: np.ndarray,
    covs: np.ndarray,
    slots: tuple[tuple[int, str], ...],
) -> MinimumDistanceResult:
    return MinimumDistanceResult(
        loadings=pd.DataFrame(),
        measurement_intercepts=pd.DataFrame(),
        measurement_sds=pd.DataFrame(),
        factor_mixture_means=means,
        factor_mixture_covariances=covs,
        factor_period_slots=slots,
        objective_value=0.0,
        success=True,
    )


def test_fit_linear_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 1000
    x_design = rng.normal(size=(n, 2))
    y = 0.5 * x_design[:, 0] - 0.3 * x_design[:, 1] + 1.2 + rng.normal(0, 0.1, size=n)

    params, sd = _fit_linear(y, x_design, ["a", "b"])

    assert params["a"] == _pytest_approx(0.5, 0.05)
    assert params["b"] == _pytest_approx(-0.3, 0.05)
    assert params["constant"] == _pytest_approx(1.2, 0.05)
    assert sd == _pytest_approx(0.1, abs_tol=0.02)


def _pytest_approx(target: float, rel: float = 0.05, *, abs_tol: float | None = None):
    import pytest  # noqa: PLC0415

    if abs_tol is not None:
        return pytest.approx(target, abs=abs_tol)
    return pytest.approx(target, rel=rel)


def test_fit_log_ces_recovers_known_rho_and_share():
    rng = np.random.default_rng(1)
    n = 2000
    x_design = rng.normal(0, 0.5, size=(n, 2))
    rho_true = -0.5
    gammas_true = np.array([0.65, 0.35])
    exponents = x_design * rho_true
    log_inside = np.log(
        gammas_true[0] * np.exp(exponents[:, 0])
        + gammas_true[1] * np.exp(exponents[:, 1])
    )
    y = log_inside / rho_true + rng.normal(0, 0.05, size=n)

    params, sd = _fit_log_ces(y, x_design, ["a", "b"], with_constant=False)

    assert params["a"] == _pytest_approx(0.65, 0.15)
    assert params["b"] == _pytest_approx(0.35, 0.15)
    assert params["phi"] == _pytest_approx(rho_true, abs_tol=0.15)
    assert sd == _pytest_approx(0.05, abs_tol=0.05)


def test_draw_factor_panel_yields_expected_shape_and_moments():
    slots = ((0, "skills"), (1, "skills"))
    truth_means = np.array([[-0.5, -0.2], [0.5, 0.3]])
    truth_covs = np.array(
        [
            [[1.0, 0.3], [0.3, 1.1]],
            [[0.9, 0.1], [0.1, 1.0]],
        ]
    )
    structural = _make_structural(truth_means, truth_covs, slots)

    panel = _draw_factor_panel(structural, np.array([0.4, 0.6]), n_draws=20000, seed=0)

    assert panel.shape == (20000, 2)
    # Sample-mean on slot 0: 0.4 * (-0.5) + 0.6 * 0.5 = 0.1
    # Sample-mean on slot 1: 0.4 * (-0.2) + 0.6 * 0.3 = 0.1
    np.testing.assert_allclose(panel.mean().to_numpy(), [0.1, 0.1], atol=0.05)


def test_simulate_and_regress_returns_linear_transition_for_simple_model():
    model = _linear_model()
    processed = process_model(model)

    # Build a structural result where both periods have a single
    # factor; truth coefficient for the period-0 -> period-1 transition
    # is 0.7 with intercept 0.1.
    slots = ((0, "skills"), (1, "skills"))
    truth_means = np.array([[0.0, 0.0]])
    truth_covs = np.array([[[1.0, 0.7], [0.7, 1.0 * 0.7**2 + 0.51]]])
    structural = _make_structural(truth_means, truth_covs, slots)

    result = simulate_and_regress(
        structural,
        processed,
        model,
        mixture_weights=np.array([1.0]),
        n_draws=5000,
        seed=0,
    )

    params = result.production_params
    slope = float(
        params.loc[("transition", 0, "skills", "skills"), "value"]  # ty: ignore[invalid-argument-type]
    )
    assert slope == _pytest_approx(0.7, abs_tol=0.05)


def test_simulate_and_regress_handles_translog():
    """Generic NLS path recovers translog params via the function callable."""
    from skillmodels.common.model_spec import (  # noqa: PLC0415
        EstimationOptions,
        FactorSpec,
        ModelSpec,
        Normalizations,
    )
    from skillmodels.common.process_model import process_model  # noqa: PLC0415

    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="translog",
            ),
        },
        estimation_options=EstimationOptions(
            robust_bounds=True, bounds_distance=0.001, n_mixtures=1
        ),
    )
    processed = process_model(model)
    slots = ((0, "skills"), (1, "skills"))
    # Cov(period0, period1) chosen so OLS slope ≈ 0.6.
    truth_means = np.array([[0.0, 0.0]])
    truth_covs = np.array([[[1.0, 0.6], [0.6, 1.0 * 0.6**2 + 0.4]]])
    structural = MinimumDistanceResult(
        loadings=pd.DataFrame(),
        measurement_intercepts=pd.DataFrame(),
        measurement_sds=pd.DataFrame(),
        factor_mixture_means=truth_means,
        factor_mixture_covariances=truth_covs,
        factor_period_slots=slots,
        objective_value=0.0,
        success=True,
    )

    result = simulate_and_regress(
        structural,
        processed,
        model,
        mixture_weights=np.array([1.0]),
        n_draws=5000,
        seed=0,
    )

    params = result.production_params
    # translog params: linear coefficient on `skills` plus `skills ** 2`
    # plus `constant`. The linear coefficient should approach the
    # cov / var slope (≈ 0.6); the square coefficient should be small.
    assert ("transition", 0, "skills", "skills") in params.index
    assert ("transition", 0, "skills", "skills ** 2") in params.index
    assert ("transition", 0, "skills", "constant") in params.index
    slope = float(
        params.loc[("transition", 0, "skills", "skills"), "value"]  # ty: ignore[invalid-argument-type]
    )
    assert slope == _pytest_approx(0.6, abs_tol=0.1)
