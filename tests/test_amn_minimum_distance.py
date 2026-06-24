"""Tests for `skillmodels.amn.minimum_distance` (AMN Stage 2)."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn.minimum_distance import (
    _build_structure,
    _initial_guess,
    _make_objective_and_grad,
    _objective,
    _pack_layout,
    solve_minimum_distance,
)
from skillmodels.amn.mixture_em import (
    build_augmented_measure_layout,
    build_augmented_measure_matrix,
    fit_mixture_em,
)
from skillmodels.amn.types import (
    AugmentedMeasureLayout,
    MixtureFitResult,
)
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.process_model import process_model


def _tiny_model() -> ModelSpec:
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


def _build_oracle_mixture(
    *,
    n_components: int = 2,
    n_aug: int = 6,
    seed: int = 0,
    layout: AugmentedMeasureLayout | None = None,
) -> tuple[MixtureFitResult, dict[str, np.ndarray]]:
    """Build a synthetic MixtureFitResult with known structural moments.

    Layout: 2 periods x 3 measurements on a single latent factor, anchor
    measurement loading=1, others = (1.0, 0.8, 1.2). Mean-zero on the
    period-0 factor.
    """
    del seed
    if layout is None:
        layout = AugmentedMeasureLayout(
            columns=tuple(
                f"y[{t}|skills|{m}]" for t in (0, 1) for m in ("y1", "y2", "y3")
            ),
            measurement_slots=tuple(range(n_aug)),
            observed_factor_slots=(),
            control_slots=(),
            measurement_meta=tuple(
                (t, "skills", m) for t in (0, 1) for m in ("y1", "y2", "y3")
            ),
            observed_factor_meta=(),
            control_meta=(),
        )

    truth_lambda = np.zeros((6, 2))
    truth_lambda[0, 0] = 1.0
    truth_lambda[1, 0] = 0.8
    truth_lambda[2, 0] = 1.2
    truth_lambda[3, 1] = 1.0
    truth_lambda[4, 1] = 0.8
    truth_lambda[5, 1] = 1.2
    truth_intercept = np.array([0.0, 0.1, -0.2, 0.5, 0.3, 0.4])
    truth_sigma2 = np.array([0.3, 0.25, 0.4, 0.35, 0.2, 0.5]) ** 2

    truth_mu = np.array([[-0.6, 0.4], [0.4, -0.3]])  # period-0 enforces sum-to-zero
    # Enforce sum-to-zero on column 0 (period-0 latent slot) with
    # weights 0.5/0.5.
    truth_mu[1, 0] = -truth_mu[0, 0]
    truth_omega = np.array(
        [
            [[1.0, 0.4], [0.4, 1.2]],
            [[0.9, 0.2], [0.2, 1.1]],
        ]
    )

    means = np.empty((n_components, n_aug))
    covs = np.empty((n_components, n_aug, n_aug))
    for m in range(n_components):
        means[m] = truth_intercept + truth_lambda @ truth_mu[m]
        covs[m] = truth_lambda @ truth_omega[m] @ truth_lambda.T + np.diag(truth_sigma2)

    weights = np.array([0.5, 0.5])

    return MixtureFitResult(
        weights=weights,
        means=means,
        covariances=covs,
        loglikelihood=-100.0,
        n_iter=10,
        converged=True,
        layout=layout,
    ), {
        "lambda": truth_lambda,
        "intercept": truth_intercept,
        "sigma2": truth_sigma2,
        "mu": truth_mu,
        "omega": truth_omega,
    }


def _mean_zero_model() -> ModelSpec:
    """Canonical AMN: free period-0 intercept, location set by weighted mean-zero."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({}, {}),
                ),
                transition_function="linear",
            ),
        },
    )


def test_build_structure_drops_mean_zero_when_initial_intercept_pinned():
    """A pinned period-0 intercept removes that factor's mean-zero baseline slot.

    `_tiny_model` pins the period-0 anchor intercept to 0, which supplies the
    initial location. Imposing the weighted mean-zero on top would be a second
    normalization of the same orbit direction (audit F7), so the period-0 factor
    mean stays free and is NOT a baseline slot.
    """
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)

    struct = _build_structure(layout, processed)

    # 2 latent-factor-period slots: (0, skills) and (1, skills).
    assert len(struct.factor_period_slots) == 2
    assert (0, "skills") in struct.factor_period_slots
    assert (1, "skills") in struct.factor_period_slots
    # 6 measurement slots; 2 of them (y1 at periods 0,1) have
    # normalized loading=1, so lambda has 4 free entries.
    assert struct.lambda_free_mask.sum() == 4
    # y1 at period 0 has normalized intercept=0; the other 5 are free.
    assert struct.intercept_free_mask.sum() == 5
    # All 6 measurement slots have free sigma2 (no obs factors, no controls).
    assert struct.sigma2_free_mask.sum() == 6
    # The pinned period-0 intercept already anchors location, so there is no
    # mean-zero baseline slot.
    assert struct.baseline_mean_zero_slots == ()


def test_build_structure_keeps_mean_zero_when_initial_intercept_free():
    """With a free period-0 intercept, the weighted mean-zero anchors location.

    The canonical AMN convention leaves the measurement intercepts free and sets
    the initial location through the period-0 weighted mean-zero, so the period-0
    factor slot IS a baseline slot.
    """
    model = _mean_zero_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)

    struct = _build_structure(layout, processed)

    # No pinned intercept anywhere, so all 6 intercepts are free.
    assert struct.intercept_free_mask.sum() == 6
    baseline_slot = struct.factor_period_slots.index((0, "skills"))
    assert baseline_slot in struct.baseline_mean_zero_slots


def test_pack_layout_returns_consistent_total():
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    struct = _build_structure(layout, processed)

    n_total, slices = _pack_layout(struct, n_components=2)

    # `_tiny_model` pins the period-0 intercept, so there is no mean-zero
    # baseline slot and every mu entry is free (audit F7).
    # sigma2: 6 free; chol_0+chol_1: 2*3=6; mu: 2*2 - 0 baseline = 4;
    # lambda: 4 free; intercept: 5 free => 6+6+4+4+5 = 25.
    assert n_total == 25
    assert slices["sigma2"] == slice(0, 6)


def _numpy_fd_grad(fn, x, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of a scalar function."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        grad[i] = (fn(xp) - fn(xm)) / (2 * h)
    return grad


def test_md_analytical_gradient_matches_numerical():
    """The JAX objective + gradient match the numpy criterion and its FD gradient.

    `solve_minimum_distance` passes this analytical gradient to the optimizer so
    each L-BFGS-B step costs one backward pass instead of `n_params` finite-
    difference objective evaluations -- the difference between seconds and hours
    once the factor-period block (and thus the parameter vector) is large.
    """
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)
    struct = _build_structure(layout, processed)
    n_components = mixture.weights.shape[0]
    n_total, slices = _pack_layout(struct, n_components)

    value_fn, grad_fn = _make_objective_and_grad(
        struct,
        slices,
        n_components=n_components,
        mixture_weights=mixture.weights,
        target_means=mixture.means,
        target_covs=mixture.covariances,
    )

    rng = np.random.default_rng(1)
    flat = _initial_guess(
        struct,
        slices,
        n_components=n_components,
        n_total=n_total,
        target_means=mixture.means,
        target_covs=mixture.covariances,
    ) + 0.05 * rng.standard_normal(n_total)

    def numpy_obj(x: np.ndarray) -> float:
        return _objective(
            x,
            struct,
            slices,
            n_components=n_components,
            mixture_weights=mixture.weights,
            target_means=mixture.means,
            target_covs=mixture.covariances,
        )

    assert value_fn(flat) == pytest.approx(numpy_obj(flat), rel=1e-6)
    np.testing.assert_allclose(
        grad_fn(flat), _numpy_fd_grad(numpy_obj, flat), rtol=1e-4, atol=1e-5
    )


def test_solve_minimum_distance_respects_maxiter_cap():
    """A `stopping_maxiter` cap is honoured (the seeding budget bound).

    With the exact analytical gradient each L-BFGS-B step is cheap, but a seed
    does not need full convergence; the cap bounds the Stage-2 cost on a large
    factor-period block. A tiny cap must return a valid result, not error.
    """
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)

    result = solve_minimum_distance(
        mixture, processed, algo_options={"stopping_maxiter": 2}
    )

    assert isinstance(result.objective_value, float)
    assert result.loadings.shape[0] == 6


def test_solve_minimum_distance_recovers_oracle():
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _truth = _build_oracle_mixture(layout=layout)

    result = solve_minimum_distance(mixture, processed)

    # The minimum-distance criterion should be near zero on oracle moments.
    assert result.objective_value < 1e-3

    # Loadings should match truth within tolerance.
    loadings = result.loadings.reset_index().set_index(["period", "measurement"])
    assert loadings.loc[(0, "y1"), "loading"] == pytest.approx(1.0, abs=1e-6)
    assert loadings.loc[(0, "y2"), "loading"] == pytest.approx(0.8, abs=5e-2)
    assert loadings.loc[(0, "y3"), "loading"] == pytest.approx(1.2, abs=5e-2)


def _nonzero_initial_mean_oracle(
    layout: AugmentedMeasureLayout,
) -> MixtureFitResult:
    """Oracle moments whose period-0 factor mean is NONZERO in every component.

    The period-0 intercept of the anchor measurement is pinned to 0 (see
    `_tiny_model`), so that pin already supplies the initial location: the
    factor mean must stay free to absorb the true nonzero level. Imposing the
    AMN weighted mean-zero restriction on top would force the weighted mean to
    0 and contradict these moments, so a faithful Stage-2 fit must reach SSE 0
    only when it does NOT double-normalize the location.
    """
    n_components, n_aug = 2, 6
    truth_lambda = np.zeros((6, 2))
    truth_lambda[0, 0] = 1.0
    truth_lambda[1, 0] = 0.8
    truth_lambda[2, 0] = 1.2
    truth_lambda[3, 1] = 1.0
    truth_lambda[4, 1] = 0.8
    truth_lambda[5, 1] = 1.2
    truth_intercept = np.array([0.0, 0.1, -0.2, 0.5, 0.3, 0.4])
    truth_sigma2 = np.array([0.3, 0.25, 0.4, 0.35, 0.2, 0.5]) ** 2
    # Period-0 latent mean is 0.4 in BOTH mixtures => weighted mean 0.4 != 0.
    truth_mu = np.array([[0.4, 0.4], [0.4, -0.3]])
    truth_omega = np.array(
        [
            [[1.0, 0.4], [0.4, 1.2]],
            [[0.9, 0.2], [0.2, 1.1]],
        ]
    )

    means = np.empty((n_components, n_aug))
    covs = np.empty((n_components, n_aug, n_aug))
    for m in range(n_components):
        means[m] = truth_intercept + truth_lambda @ truth_mu[m]
        covs[m] = truth_lambda @ truth_omega[m] @ truth_lambda.T + np.diag(truth_sigma2)

    return MixtureFitResult(
        weights=np.array([0.5, 0.5]),
        means=means,
        covariances=covs,
        loglikelihood=-100.0,
        n_iter=10,
        converged=True,
        layout=layout,
    )


def test_pinned_initial_intercept_frees_factor_mean_from_mean_zero():
    """A pinned period-0 intercept disables the weighted mean-zero on that factor.

    Pinning the anchor intercept and imposing the AMN weighted mean-zero
    constraint are two location normalizations on the same orbit direction.
    With a nonzero true initial mean only ONE can hold, so a faithful fit drops
    mean-zero (the intercept pin supplies location) and recovers the moments at
    SSE 0; double-normalizing would leave a large irreducible residual.
    """
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture = _nonzero_initial_mean_oracle(layout)

    result = solve_minimum_distance(mixture, processed)

    assert result.objective_value < 1e-3


def test_solve_minimum_distance_rejects_unknown_weighting():
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)

    with pytest.raises(ValueError, match="Unknown weighting"):
        solve_minimum_distance(mixture, processed, weighting="bogus")


def _observed_factor_model() -> ModelSpec:
    """One latent factor (2 measurements, 1 period) plus an observed factor."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2"),),
                normalizations=Normalizations(
                    loadings=({"y1": 1},),
                    intercepts=({"y1": 0},),
                ),
                transition_function="linear",
            ),
        },
        observed_factors=("inv",),
        controls=("momed",),
    )


def test_build_structure_pins_observed_and_control_intercepts():
    model = _observed_factor_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)

    struct = _build_structure(layout, processed)

    # Observed-factor and control slots must have pinned (not free) zero
    # intercepts; this is the direct guard against the regression.
    for idx in (*layout.observed_factor_slots, *layout.control_slots):
        assert struct.intercept_free_mask[idx] == np.False_
        assert struct.intercept_value[idx] == 0.0

    # A latent measurement slot with a non-normalized intercept (y2) is
    # still free, confirming we only pinned the observed/control slots.
    y2_slot = next(
        slot
        for slot, meta in zip(
            layout.measurement_slots, layout.measurement_meta, strict=True
        )
        if meta[2] == "y2"
    )
    assert struct.intercept_free_mask[y2_slot] == np.True_


def test_minimum_distance_recovers_observed_factor_level():
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2"),),
                normalizations=Normalizations(
                    loadings=({"y1": 1},),
                    intercepts=({"y1": 0},),
                ),
                transition_function="linear",
            ),
        },
        observed_factors=("inv",),
    )
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)

    # Augmented vector columns: y1, y2, obs_factor inv.
    # Factor-period slots: (0, skills), (0, inv).
    n_aug = 3
    n_components = 2
    truth_lambda = np.zeros((n_aug, 2))
    truth_lambda[0, 0] = 1.0  # y1 loads on skills, normalized.
    truth_lambda[1, 0] = 0.8  # y2 loads on skills, free.
    truth_lambda[2, 1] = 1.0  # obs_factor inv loads on its own slot.
    truth_intercept = np.array([0.0, 0.1, 0.0])
    truth_sigma2 = np.array([0.3, 0.25, 0.0]) ** 2

    pi = np.array([1.3, 2.1])  # true per-component reduced-form level of inv.
    truth_mu = np.zeros((n_components, 2))
    truth_mu[0, 0] = -0.6  # skills period-0 mean-zero (weights 0.5/0.5).
    truth_mu[1, 0] = 0.6
    truth_mu[:, 1] = pi
    truth_omega = np.array(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.9, 0.0], [0.0, 0.4]],
        ]
    )

    means = np.empty((n_components, n_aug))
    covs = np.empty((n_components, n_aug, n_aug))
    for m in range(n_components):
        means[m] = truth_intercept + truth_lambda @ truth_mu[m]
        covs[m] = truth_lambda @ truth_omega[m] @ truth_lambda.T + np.diag(truth_sigma2)

    mixture = MixtureFitResult(
        weights=np.array([0.5, 0.5]),
        means=means,
        covariances=covs,
        loglikelihood=-1.0,
        n_iter=1,
        converged=True,
        layout=layout,
    )

    result = solve_minimum_distance(mixture, processed)

    inv_col = result.factor_period_slots.index((0, "inv"))
    np.testing.assert_allclose(result.factor_mixture_means[:, inv_col], pi, atol=1e-3)
    assert result.objective_value < 1e-3


def test_solve_minimum_distance_runs_on_fitted_mixture():
    """End-to-end: simulate 1-component data, fit, then recover Lambda."""
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)

    rng = np.random.default_rng(0)
    n = 1500
    # period-0 factor mean-zero (sum-to-zero with itself => 0).
    period0 = rng.normal(0.0, 1.0, size=n)
    period1 = 0.7 * period0 + rng.normal(0.0, 0.6, size=n)

    rows = []
    for caseid in range(n):
        for period, f in [(0, period0[caseid]), (1, period1[caseid])]:
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": f + rng.normal(0, 0.3),
                    "y2": 0.8 * f + rng.normal(0, 0.4),
                    "y3": 1.2 * f + rng.normal(0, 0.35),
                }
            )
    data = pd.DataFrame(rows).set_index(["caseid", "period"])
    augmented = build_augmented_measure_matrix(data, processed, layout)

    mixture = fit_mixture_em(augmented, n_components=2, n_init=2, seed=0, layout=layout)

    result = solve_minimum_distance(mixture, processed)

    # Just verifying it runs and produces a finite objective.
    assert np.isfinite(result.objective_value)
    assert result.loadings.shape[0] == 6


def _ces_model(*, loadings: tuple, intercepts: tuple) -> ModelSpec:
    """CES (`log_ces`) model with 3 measurements over 2 periods."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=loadings,
                    intercepts=intercepts,
                ),
                transition_function="log_ces",
            ),
        },
    )


def test_solve_minimum_distance_rejects_ces_overnormalization():
    model = _ces_model(
        loadings=({"y1": 1, "y2": 1}, {"y1": 1}),
        intercepts=({"y1": 0}, {}),
    )
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)

    with pytest.raises(ValueError, match="loading normalizations"):
        solve_minimum_distance(mixture, processed)


def test_solve_minimum_distance_allows_ces_overnormalization_when_opted_in():
    model = _ces_model(
        loadings=({"y1": 1, "y2": 1}, {"y1": 1}),
        intercepts=({"y1": 0}, {}),
    )
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)

    result = solve_minimum_distance(mixture, processed, allow_overnormalization=True)

    assert np.isfinite(result.objective_value)


def test_solve_minimum_distance_rejects_ces_missing_normalization():
    model = _ces_model(
        loadings=({"y1": 1}, {}),
        intercepts=({"y1": 0}, {}),
    )
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)

    with pytest.raises(ValueError, match="no loading normalization"):
        solve_minimum_distance(mixture, processed)


def test_solve_minimum_distance_rejects_documented_optimal():
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    mixture, _ = _build_oracle_mixture(layout=layout)

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        solve_minimum_distance(mixture, processed, weighting="optimal")
