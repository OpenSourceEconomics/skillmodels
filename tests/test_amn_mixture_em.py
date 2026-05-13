"""Tests for `skillmodels.amn.mixture_em` (AMN Stage 1)."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn.mixture_em import (
    build_augmented_measure_layout,
    build_augmented_measure_matrix,
    fit_mixture_em,
)
from skillmodels.common.model_spec import (
    CHSEstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.process_model import process_model


def _tiny_model() -> ModelSpec:
    """Return a 2-period, 1-latent-factor model with 3 indicators per period."""
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
        chs_estimation_options=CHSEstimationOptions(
            robust_bounds=True, bounds_distance=0.001, n_mixtures=1
        ),
    )


def _tiny_long_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two periods, three measurements each, drawn from N(0, 1) + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        factor_0 = rng.normal()
        factor_1 = 0.6 * factor_0 + rng.normal(0, 0.5)
        for period, f in [(0, factor_0), (1, factor_1)]:
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": f + rng.normal(0, 0.3),
                    "y2": 0.9 * f + rng.normal(0, 0.4),
                    "y3": 1.1 * f + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def test_layout_has_one_slot_per_measurement_update():
    model = _tiny_model()
    processed = process_model(model)

    layout = build_augmented_measure_layout(processed)

    # 2 periods x 3 measurements = 6 measurement slots, no observed factors
    # or controls.
    assert len(layout.measurement_slots) == 6
    assert layout.observed_factor_slots == ()
    assert layout.control_slots == ()
    assert len(layout.columns) == 6


def test_layout_records_period_factor_and_measurement_names():
    model = _tiny_model()
    processed = process_model(model)

    layout = build_augmented_measure_layout(processed)

    assert set(layout.measurement_meta) == {
        (0, "skills", "y1"),
        (0, "skills", "y2"),
        (0, "skills", "y3"),
        (1, "skills", "y1"),
        (1, "skills", "y2"),
        (1, "skills", "y3"),
    }


def test_layout_skips_anchoring_rows():
    """Anchoring outcomes (purpose != measurement) must not become slots."""
    base = _tiny_model()
    from skillmodels.common.model_spec import AnchoringSpec  # noqa: PLC0415

    anchored = base.with_anchoring(
        AnchoringSpec(
            outcomes={"skills": "outcome"},
            free_controls=False,
            free_constant=False,
            free_loadings=True,
            ignore_constant_when_anchoring=True,
        )
    )
    processed = process_model(anchored)

    layout = build_augmented_measure_layout(processed)

    # 6 measurement slots; anchoring update rows are filtered out.
    for _, factor, _ in layout.measurement_meta:
        assert factor == "skills"
    assert len(layout.measurement_slots) == 6


def test_matrix_fills_each_slot_from_the_right_period():
    model = _tiny_model()
    processed = process_model(model)
    data = _tiny_long_data(n=50, seed=1)

    layout = build_augmented_measure_layout(processed)
    matrix = build_augmented_measure_matrix(data, processed, layout)

    assert matrix.shape == (50, 6)

    # Period 0 slot for y1 must equal data.loc[(*, 0), "y1"].
    period0_y1_slot = next(
        slot
        for slot, meta in zip(
            layout.measurement_slots, layout.measurement_meta, strict=True
        )
        if meta == (0, "skills", "y1")
    )
    expected = data.xs(0, level="period")["y1"].to_numpy()
    np.testing.assert_allclose(matrix[:, period0_y1_slot], expected)


def test_matrix_marks_missing_caseids_as_nan():
    model = _tiny_model()
    processed = process_model(model)
    layout = build_augmented_measure_layout(processed)
    data = _tiny_long_data(n=10, seed=2)
    # Drop period 1 for caseid 0 entirely.
    data = data.drop(index=(0, 1))

    matrix = build_augmented_measure_matrix(data, processed, layout)

    # The first row corresponds to caseid 0; period-1 slots should be NaN.
    period1_slots = [
        slot
        for slot, (period, _, _) in zip(
            layout.measurement_slots, layout.measurement_meta, strict=True
        )
        if period == 1
    ]
    assert np.all(np.isnan(matrix[0, period1_slots]))
    # Period-0 slots for the same caseid stay finite.
    period0_slots = [
        slot
        for slot, (period, _, _) in zip(
            layout.measurement_slots, layout.measurement_meta, strict=True
        )
        if period == 0
    ]
    assert np.all(np.isfinite(matrix[0, period0_slots]))


def _simulate_two_component_panel(
    *,
    n: int,
    weights: tuple[float, float],
    means: tuple[np.ndarray, np.ndarray],
    chols: tuple[np.ndarray, np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], size=n, p=list(weights))
    samples = np.empty((n, means[0].shape[0]))
    for k in (0, 1):
        idx = labels == k
        if idx.any():
            standard = rng.normal(size=(idx.sum(), means[k].shape[0]))
            samples[idx] = standard @ chols[k].T + means[k]
    return samples


def test_fit_mixture_em_recovers_two_components_within_tolerance():
    truth_weights = (0.4, 0.6)
    truth_means = (np.array([-1.5, 1.0]), np.array([1.5, -1.0]))
    truth_chols = (
        np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.2]])),
        np.linalg.cholesky(np.array([[0.8, -0.2], [-0.2, 1.0]])),
    )
    augmented = _simulate_two_component_panel(
        n=4000,
        weights=truth_weights,
        means=truth_means,
        chols=truth_chols,
        seed=11,
    )

    result = fit_mixture_em(augmented, n_components=2, n_init=3, seed=11)

    assert result.converged
    # Order of components is arbitrary; line them up to the truth by
    # nearest-mean.
    order = np.argsort(result.means[:, 0])
    truth_order = np.argsort([truth_means[0][0], truth_means[1][0]])

    np.testing.assert_allclose(
        result.weights[order],
        np.array(truth_weights)[truth_order],
        atol=0.05,
    )
    for fitted_k, truth_k in zip(order, truth_order, strict=True):
        np.testing.assert_allclose(
            result.means[fitted_k],
            truth_means[truth_k],
            atol=0.15,
        )


def test_fit_mixture_em_drops_incomplete_rows():
    rng = np.random.default_rng(3)
    augmented = rng.normal(size=(200, 4))
    augmented[:50, 2] = np.nan  # 50 rows incomplete

    result = fit_mixture_em(augmented, n_components=2, n_init=2, seed=3)

    # n_complete = 150; loglikelihood should be ~150 * per-row mean.
    # The check we actually want is that it runs without error and the
    # iteration count is sensible.
    assert result.n_iter >= 1
    assert result.weights.shape == (2,)


def test_fit_mixture_em_raises_when_too_few_complete_rows():
    augmented = np.array([[np.nan, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="complete-case"):
        fit_mixture_em(augmented, n_components=3, n_init=1, seed=0)
