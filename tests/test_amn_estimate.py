"""Tests for `skillmodels.amn.estimate.estimate_amn` (end-to-end orchestration)."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn import estimate_amn
from skillmodels.amn.types import AMNEstimationOptions
from skillmodels.common.model_spec import (
    CHSEstimationOptions,
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
        chs_estimation_options=CHSEstimationOptions(
            robust_bounds=True, bounds_distance=0.001, n_mixtures=1
        ),
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
    options = AMNEstimationOptions(
        n_mixture_components=2, n_simulation_draws=5000, seed=0
    )

    result = estimate_amn(model, data, options)

    assert result.all_params.index.names == [
        "category",
        "aug_period",
        "name1",
        "name2",
    ]
    cats = set(result.all_params.index.get_level_values("category"))
    assert {"loadings", "meas_sds", "transition", "shock_sds"} <= cats
    # 6 measurement loadings, 6 meas_sds, 1 transition (slope on skills) +
    # constant for period 0, 1 shock_sds for period 0.
    assert "controls" in cats  # measurement intercepts collapse to controls


def test_estimate_amn_honors_fixed_params():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(
        n_mixture_components=2, n_simulation_draws=5000, seed=0
    )

    pin_loc = ("loadings", 1, "y2", "skills")
    fixed = pd.DataFrame(
        {"value": [0.42]},
        index=pd.MultiIndex.from_tuples(
            [pin_loc], names=["category", "aug_period", "name1", "name2"]
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=fixed)

    assert result.all_params.loc[pin_loc, "value"] == pytest.approx(0.42)


def test_estimate_amn_returns_success_flag():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(
        n_mixture_components=2, n_simulation_draws=2000, seed=1
    )

    result = estimate_amn(model, data, options)

    assert isinstance(result.success, bool)
    assert result.stages.mixture.weights.shape == (2,)
    assert result.stages.structural.factor_period_slots == (
        (0, "skills"),
        (1, "skills"),
    )
