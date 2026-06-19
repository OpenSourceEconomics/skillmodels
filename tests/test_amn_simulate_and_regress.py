"""Tests for `skillmodels.amn.simulate_and_regress` (AMN Stage 3)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn.simulate_and_regress import (
    _draw_factor_panel,
    _fit_investment_residual,
    _fit_linear,
    _fit_log_ces,
    simulate_and_regress,
)
from skillmodels.amn.types import MinimumDistanceResult
from skillmodels.common.model_spec import (
    CorrectionSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.process_model import process_model


def _endogenous_model() -> ModelSpec:
    """2-period model with an endogenous investment factor and a state factor."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2"), ("y1", "y2")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1", "i2"), ("i1", "i2")),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}),
                    intercepts=({"i1": 0}, {}),
                ),
                transition_function="linear",
                is_endogenous=True,
            ),
        },
    )


def _endogenous_structural() -> MinimumDistanceResult:
    """Structural mixture covering the endogenous model's (period, factor) slots."""
    slots = (
        (0, "skills"),
        (0, "investment"),
        (1, "skills"),
        (1, "investment"),
    )
    n = len(slots)
    means = np.zeros((1, n))
    covs = np.array([np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n))])
    return _make_structural(means, covs, slots)


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


def test_simulate_and_regress_no_investment_eq_category():
    model = _endogenous_model()
    processed = process_model(model)
    structural = _endogenous_structural()

    result = simulate_and_regress(
        structural,
        processed,
        model,
        mixture_weights=np.array([1.0]),
        n_draws=3000,
        seed=0,
    )

    # Production params are produced; the misleading 'investment_eq' rows
    # are gone and investment_params is empty.
    assert not result.production_params.empty
    assert len(result.investment_params) == 0
    categories = set(result.production_params.index.get_level_values("category")) | set(
        result.investment_params.index.get_level_values("category")
    )
    assert "investment_eq" not in categories


# --- Control-function (investment-endogeneity) regression fixtures ----------
#
# Closed-form LINEAR DGP for the AMN/AF control-function correction. One
# skill (state) factor + one endogenous investment factor + an observed
# instrument `income`, over two periods. Primitives are independent
# N(0, .): theta0 (var 1.0), income Y0 (var 1.0), eta_I (var 0.50),
# eps_C (var 0.30), and an extra independent primitive feeding the
# irrelevant f[1|investment] slot. The DGP sets investment I0 to
# b_I times theta0 plus b_Y times Y0 plus eta_I (b_I=0.50, b_Y=0.70),
# and next-period skills theta1 to lam times theta0 plus psi times I0
# plus kappa times eta_I plus eps_C (lam=0.40, psi=0.60, kappa=0.80).
# OLS of theta1 on theta0 and I0 is biased upward on the I0 coefficient
# (I0 is correlated with eta_I, which enters theta1); adding the
# first-stage residual eta_I as a control function `cf` recovers
# psi=0.60 and identifies kappa=0.80.

_CF_B_I = 0.50
_CF_B_Y = 0.70
_CF_LAM = 0.40
_CF_PSI = 0.60
_CF_KAPPA = 0.80
_CF_SLOTS = (
    (0, "skills"),
    (0, "investment"),
    (0, "income"),
    (1, "skills"),
    (1, "investment"),
)


def _cf_model() -> ModelSpec:
    """Skill + endogenous investment factor with an observed instrument."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2"), ("y1", "y2")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1", "i2"), ("i1", "i2")),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}),
                    intercepts=({"i1": 0}, {}),
                ),
                transition_function="linear",
                is_endogenous=True,
                correction=CorrectionSpec(
                    state_predictors=("skills",),
                    instruments=("income",),
                    targets=("skills",),
                ),
            ),
        },
        observed_factors=("income",),
    )


def _cf_structural() -> MinimumDistanceResult:
    """Structural mixture built from the closed-form control-function DGP."""
    # Primitive variances: theta0, Y0, eta_I, eps_C, extra.
    prim_var = np.array([1.0, 1.0, 0.50, 0.30, 1.0])
    # Rows = slots; cols = primitives [theta0, Y0, eta_I, eps_C, extra].
    # Reduced form of theta1: loading (lam + psi*b_I) on theta0, psi*b_Y
    # on Y0, (psi + kappa) on eta_I, and 1 on eps_C.
    b_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # (0, skills) = theta0
            [_CF_B_I, _CF_B_Y, 1.0, 0.0, 0.0],  # (0, investment) = I0
            [0.0, 1.0, 0.0, 0.0, 0.0],  # (0, income) = Y0
            [
                _CF_LAM + _CF_PSI * _CF_B_I,
                _CF_PSI * _CF_B_Y,
                _CF_PSI + _CF_KAPPA,
                1.0,
                0.0,
            ],  # (1, skills) = theta1
            [0.0, 0.0, 0.0, 0.0, 1.0],  # (1, investment) = extra
        ]
    )
    cov = b_matrix @ np.diag(prim_var) @ b_matrix.T
    means = np.zeros((1, len(_CF_SLOTS)))
    return _make_structural(means, cov[None, :, :], _CF_SLOTS)


def test_fit_investment_residual_recovers_first_stage():
    rng = np.random.default_rng(0)
    n = 5000
    theta0 = rng.normal(size=n)
    income = rng.normal(size=n)
    eta_i = rng.normal(0, np.sqrt(0.50), size=n)
    y_invest = _CF_B_I * theta0 + _CF_B_Y * income + eta_i
    x_determinants = np.column_stack([theta0, income])

    coefs, sd, residual = _fit_investment_residual(
        y_invest, x_determinants, ["skills", "income"]
    )

    assert coefs["skills"] == _pytest_approx(_CF_B_I, abs_tol=0.03)
    assert coefs["income"] == _pytest_approx(_CF_B_Y, abs_tol=0.03)
    assert sd == _pytest_approx(np.sqrt(0.50), abs_tol=0.05)
    corr = np.corrcoef(residual, eta_i)[0, 1]
    assert corr == _pytest_approx(1.0, abs_tol=0.02)


def test_simulate_and_regress_control_function_recovers_psi_and_kappa():
    model = _cf_model()
    processed = process_model(model)
    structural = _cf_structural()

    result = simulate_and_regress(
        structural,
        processed,
        model,
        mixture_weights=np.array([1.0]),
        n_draws=4000,
        seed=0,
    )

    prod = result.production_params
    psi = float(prod.loc[("transition", 0, "skills", "investment"), "value"])  # ty: ignore[invalid-argument-type]
    lam = float(prod.loc[("transition", 0, "skills", "skills"), "value"])  # ty: ignore[invalid-argument-type]
    kappa = float(prod.loc[("transition", 0, "skills", "cf"), "value"])  # ty: ignore[invalid-argument-type]
    assert psi == _pytest_approx(_CF_PSI, abs_tol=0.08)
    assert lam == _pytest_approx(_CF_LAM, abs_tol=0.08)
    assert kappa == _pytest_approx(_CF_KAPPA, abs_tol=0.10)

    inv = result.investment_params
    assert float(
        inv.loc[("investment_eq", 0, "investment", "income"), "value"]  # ty: ignore[invalid-argument-type]
    ) == _pytest_approx(_CF_B_Y, abs_tol=0.08)
    assert float(
        inv.loc[("investment_eq", 0, "investment", "skills"), "value"]  # ty: ignore[invalid-argument-type]
    ) == _pytest_approx(_CF_B_I, abs_tol=0.08)
    assert ("investment_sds", 0, "investment", "-") in inv.index

    # Gating: the investment factor's own transition gets no cf row.
    assert ("transition", 0, "investment", "cf") not in prod.index


_CF_SES = 0.5  # true production coefficient on the non-instrument control `ses`


def _cf_model_with_control() -> ModelSpec:
    """Corrected model with an observed instrument AND a non-instrument control."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2"), ("y1", "y2")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1", "i2"), ("i1", "i2")),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}),
                    intercepts=({"i1": 0}, {}),
                ),
                transition_function="linear",
                is_endogenous=True,
                correction=CorrectionSpec(
                    state_predictors=("skills",),
                    instruments=("income",),
                    targets=("skills",),
                ),
            ),
        },
        observed_factors=("income", "ses"),
    )


def _cf_control_structural() -> MinimumDistanceResult:
    """CF DGP where skills1 also loads on the non-instrument control ses0."""
    slots = ((0, "skills"), (0, "investment"), (0, "income"), (0, "ses"), (1, "skills"))
    # Primitives: [theta0, Y0, eta_I, ses0, eps_C].
    prim_var = np.array([1.0, 1.0, 0.50, 1.0, 0.30])
    b_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # (0, skills) = theta0
            [_CF_B_I, _CF_B_Y, 1.0, 0.0, 0.0],  # (0, investment) = I0
            [0.0, 1.0, 0.0, 0.0, 0.0],  # (0, income) = Y0
            [0.0, 0.0, 0.0, 1.0, 0.0],  # (0, ses) = ses0
            [
                _CF_LAM + _CF_PSI * _CF_B_I,
                _CF_PSI * _CF_B_Y,
                _CF_PSI + _CF_KAPPA,
                _CF_SES,
                1.0,
            ],  # (1, skills) = theta1, with a direct ses0 effect
        ]
    )
    cov = b_matrix @ np.diag(prim_var) @ b_matrix.T
    means = np.zeros((1, len(slots)))
    return _make_structural(means, cov[None, :, :], slots)


def test_simulate_and_regress_keeps_non_instrument_observed_controls():
    """A non-instrument observed factor stays a production control under a CF.

    Regression for audit F8: under an active control function the production
    design dropped ALL observed factors (treating every one as an excluded
    instrument), so a genuine observed control silently vanished from the
    production equation (its column index was clamped under jax.vmap). Only
    `CorrectionSpec.instruments` must be excluded; other observed factors remain
    production inputs and their coefficients must be recovered.
    """
    model = _cf_model_with_control()
    processed = process_model(model)
    structural = _cf_control_structural()

    result = simulate_and_regress(
        structural,
        processed,
        model,
        mixture_weights=np.array([1.0]),
        n_draws=8000,
        seed=0,
    )

    prod = result.production_params
    # The non-instrument control `ses` is a production input and is recovered.
    assert ("transition", 0, "skills", "ses") in prod.index
    ses_coef = float(prod.loc[("transition", 0, "skills", "ses"), "value"])  # ty: ignore[invalid-argument-type]
    assert ses_coef == _pytest_approx(_CF_SES, abs_tol=0.08)
    # The excluded instrument `income` is NOT a production input.
    assert ("transition", 0, "skills", "income") not in prod.index


def test_simulate_and_regress_naive_path_is_biased():
    model = _cf_model()
    structural = _cf_structural()
    naive_model = model.without_correction()

    naive = simulate_and_regress(
        structural,
        process_model(naive_model),
        naive_model,
        mixture_weights=np.array([1.0]),
        n_draws=4000,
        seed=0,
    )
    corrected = simulate_and_regress(
        structural,
        process_model(model),
        model,
        mixture_weights=np.array([1.0]),
        n_draws=4000,
        seed=0,
    )

    naive_psi = float(
        naive.production_params.loc[("transition", 0, "skills", "investment"), "value"]  # ty: ignore[invalid-argument-type]
    )
    corrected_psi = float(
        corrected.production_params.loc[  # ty: ignore[invalid-argument-type]
            ("transition", 0, "skills", "investment"), "value"
        ]
    )
    # The naive OLS on I_t is biased well above the true psi=0.60.
    assert naive_psi > 0.90
    assert abs(naive_psi - _CF_PSI) > 0.25
    # The control-function estimate is close to the truth.
    assert abs(corrected_psi - _CF_PSI) < 0.10


def test_simulate_and_regress_raises_on_higher_order_kappa():
    # AMN implements only the linear cf term; a degree-2 (translog) CorrectionSpec
    # basis must raise rather than silently estimate a linear correction.
    model = _cf_model()
    inv = model.factors["investment"]
    assert inv.correction is not None
    hi_correction = replace(inv.correction, kappa_degree=2)
    model = model._replace(
        factors=dict(model.factors)
        | {"investment": replace(inv, correction=hi_correction)}
    )
    processed = process_model(model)
    structural = _cf_structural()

    with pytest.raises(NotImplementedError, match="linear control function"):
        simulate_and_regress(
            structural,
            processed,
            model,
            mixture_weights=np.array([1.0]),
            n_draws=200,
            seed=0,
        )
