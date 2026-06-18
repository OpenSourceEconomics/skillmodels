"""Tests for `skillmodels.amn.estimate.estimate_amn` (end-to-end orchestration)."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn import estimate_amn
from skillmodels.amn.types import AMNEstimationOptions
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


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
        n_mixtures=2,
    )


def _tiny_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        f0 = rng.normal()
        f1 = 0.6 * f0 + rng.normal(0, 0.5)
        for period, f in [(0, f0), (1, f1)]:
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


def test_estimate_amn_produces_combined_params_dataframe():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    result = estimate_amn(model, data, options)

    assert result.params.index.names == [
        "category",
        "aug_period",
        "name1",
        "name2",
    ]
    cats = set(result.params.index.get_level_values("category"))
    assert {"loadings", "meas_sds", "transition", "shock_sds"} <= cats
    # 6 measurement loadings, 6 meas_sds, 1 transition (slope on skills) +
    # constant for period 0, 1 shock_sds for period 0.
    assert "controls" in cats  # measurement intercepts collapse to controls


def _subsample_model() -> ModelSpec:
    """Model whose `skills` factor has one rotating-subsample measurement.

    Measured by full-sample y1 (normalization) and y2, plus y_sub, which is
    missing for almost every individual.
    """
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y_sub"), ("y1", "y2", "y_sub")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
        },
        n_mixtures=2,
    )


def _subsample_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        f0 = rng.normal()
        f1 = 0.6 * f0 + rng.normal(0, 0.5)
        for period, f in [(0, f0), (1, f1)]:
            # y_sub observed only for the very first individual: the full
            # complete-case count (1) is below the 2 mixture components, exactly
            # the rotating-subsample regime that makes complete-case EM infeasible.
            y_sub = f + rng.normal(0, 0.3) if caseid == 0 else np.nan
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": f + rng.normal(0, 0.3),
                    "y2": 0.9 * f + rng.normal(0, 0.4),
                    "y_sub": y_sub,
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def test_estimate_amn_seeds_on_observed_subset_with_subsample_measurement():
    """AMN seeds on the always-observed measurements under subsample missingness.

    With a rotating-subsample measurement the full augmented vector has too few
    complete cases to fit the mixture (the complete-case EM would otherwise
    raise). AMN must drop the subsample measurement, seed the mixture on the
    always-observed subset, and still return structural params -- omitting the
    dropped measurement's loadings.
    """
    model = _subsample_model()
    data = _subsample_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    result = estimate_amn(model, data, options)

    meas = result.params.xs("loadings", level="category").index.get_level_values(
        "name1"
    )
    assert "y_sub" not in set(meas)  # subsample measurement dropped from seeding
    assert "y2" in set(meas)  # always-observed measurement retained


def test_estimate_amn_honors_fixed_params():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    pin_loc = ("loadings", 1, "y2", "skills")
    fixed = pd.DataFrame(
        {"value": [0.42]},
        index=pd.MultiIndex.from_tuples(
            [pin_loc], names=["category", "aug_period", "name1", "name2"]
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=fixed)

    assert result.params.loc[pin_loc, "value"] == pytest.approx(0.42)


def test_estimate_amn_returns_success_flag():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=1)

    result = estimate_amn(model, data, options)

    assert isinstance(result.success, bool)
    assert result.stages.mixture.weights.shape == (2,)
    assert result.stages.structural.factor_period_slots == (
        (0, "skills"),
        (1, "skills"),
    )


def test_estimate_amn_honors_fixed_params_keyed_by_period():
    """`fixed_params` keyed by `period` (the public level name) must pin.

    AMN's combined `params` uses `aug_period` internally; users
    supply overrides keyed by `period`. `align_index_names` should
    rename the override's level so `MultiIndex.union` keeps the
    level names intact and the pin survives. Regression for the
    silent-strip behaviour that produced anonymous-level params
    frames and broke `decompose_measurement_variance` downstream.
    """
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    pin_loc = ("loadings", 1, "y2", "skills")
    fixed = pd.DataFrame(
        {"value": [0.42]},
        index=pd.MultiIndex.from_tuples(
            [pin_loc],
            names=["category", "period", "name1", "name2"],
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=fixed)

    assert list(result.params.index.names) == [
        "category",
        "aug_period",
        "name1",
        "name2",
    ]
    assert result.params.loc[pin_loc, "value"] == pytest.approx(0.42)
