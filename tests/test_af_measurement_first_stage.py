"""Tests for the AF Stage-1 measurement system estimator."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.af.measurement_first_stage import (
    estimate_measurement_system,
    merge_with_user_fixed_params,
)
from skillmodels.model_spec import FactorSpec, ModelSpec, Normalizations


def _build_synthetic_model_spec(n_periods: int = 2) -> ModelSpec:
    skills_meas = ("skill_1", "skill_2", "skill_3")
    skills = FactorSpec(
        measurements=tuple(skills_meas for _ in range(n_periods)),
        normalizations=Normalizations(
            loadings=tuple({"skill_1": 1} for _ in range(n_periods)),
            intercepts=tuple({"skill_1": 0} for _ in range(n_periods)),
        ),
        transition_function="linear",
    )
    return ModelSpec(factors={"skills": skills})


def _simulate_data(
    *,
    n_obs: int,
    n_periods: int,
    loadings: np.ndarray,
    meas_sds: np.ndarray,
    factor_var: float,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n_obs):
        for period in range(n_periods):
            factor = rng.normal(0.0, np.sqrt(factor_var))
            row: dict = {
                "skill_1": loadings[0] * factor + rng.normal(0.0, meas_sds[0]),
                "skill_2": loadings[1] * factor + rng.normal(0.0, meas_sds[1]),
                "skill_3": loadings[2] * factor + rng.normal(0.0, meas_sds[2]),
            }
            rows.append({"caseid": caseid, "period": period, **row})
    df = pd.DataFrame(rows)
    return df.set_index(["caseid", "period"])


def test_recovers_known_measurement_system():
    truth_loadings = np.array([1.0, 1.3, 0.8])
    truth_meas_sds = np.array([0.4, 0.5, 0.3])
    factor_var = 1.5
    n_periods = 2
    data = _simulate_data(
        n_obs=2000,
        n_periods=n_periods,
        loadings=truth_loadings,
        meas_sds=truth_meas_sds,
        factor_var=factor_var,
    )
    model_spec = _build_synthetic_model_spec(n_periods=n_periods)

    result = estimate_measurement_system(model_spec, data)

    for period in range(n_periods):
        for k, meas in enumerate(("skill_1", "skill_2", "skill_3")):
            load_loc = ("loadings", period, meas, "skills")
            sd_loc = ("meas_sds", period, meas, "-")
            assert load_loc in result.index
            assert sd_loc in result.index
            assert result.loc[load_loc, "value"] == pytest.approx(
                truth_loadings[k], rel=0.30
            )
            assert result.loc[sd_loc, "value"] == pytest.approx(
                truth_meas_sds[k], rel=0.30
            )


def test_anchor_loading_pinned_to_one():
    """First-loading normalization is honored — anchor stays at 1.0."""
    truth_loadings = np.array([1.0, 1.5, 0.7])
    truth_meas_sds = np.array([0.3, 0.4, 0.5])
    n_periods = 2
    data = _simulate_data(
        n_obs=1500,
        n_periods=n_periods,
        loadings=truth_loadings,
        meas_sds=truth_meas_sds,
        factor_var=1.0,
        seed=1,
    )
    model_spec = _build_synthetic_model_spec(n_periods=n_periods)

    result = estimate_measurement_system(model_spec, data)

    for period in range(n_periods):
        loc = ("loadings", period, "skill_1", "skills")
        # Anchor loading must be exactly 1.0 (Spearman scale convention).
        assert result.loc[loc, "value"] == pytest.approx(1.0, abs=1e-12)


def test_honors_user_fixed_params():
    n_periods = 2
    data = _simulate_data(
        n_obs=1000,
        n_periods=n_periods,
        loadings=np.array([1.0, 1.2, 0.8]),
        meas_sds=np.array([0.4, 0.4, 0.4]),
        factor_var=1.0,
    )
    model_spec = _build_synthetic_model_spec(n_periods=n_periods)

    user_pinned_idx = pd.MultiIndex.from_tuples(
        [
            ("loadings", 0, "skill_2", "skills"),
            ("meas_sds", 1, "skill_3", "-"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    user_fixed = pd.DataFrame({"value": [99.0, 99.0]}, index=user_pinned_idx)

    result = estimate_measurement_system(model_spec, data, user_fixed_params=user_fixed)

    # User-pinned indices must NOT appear in the Stage-1 output.
    assert ("loadings", 0, "skill_2", "skills") not in result.index
    assert ("meas_sds", 1, "skill_3", "-") not in result.index
    # Other rows still produced.
    assert ("loadings", 0, "skill_3", "skills") in result.index
    assert ("meas_sds", 0, "skill_2", "-") in result.index


def test_emits_warning_for_factor_with_one_indicator():
    skills = FactorSpec(
        measurements=(("skill_1",),),
        normalizations=Normalizations(
            loadings=({"skill_1": 1},),
            intercepts=({"skill_1": 0},),
        ),
        transition_function=None,
    )
    model_spec = ModelSpec(factors={"skills": skills})

    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "caseid": range(200),
            "period": [0] * 200,
            "skill_1": rng.normal(0.0, 1.0, 200),
        }
    ).set_index(["caseid", "period"])

    with pytest.warns(UserWarning, match="fewer than two measurements"):
        result = estimate_measurement_system(model_spec, data)

    # No rows produced — the AF optimizer keeps fitting the single
    # measurement with standard initialization.
    assert len(result) == 0


def test_skips_factor_below_min_n_per_factor():
    """Skips with warning when fewer than `min_n_per_factor` complete cases."""
    n_periods = 1
    data = _simulate_data(
        n_obs=20,  # very small
        n_periods=n_periods,
        loadings=np.array([1.0, 1.2, 0.8]),
        meas_sds=np.array([0.3, 0.3, 0.3]),
        factor_var=1.0,
    )
    model_spec = _build_synthetic_model_spec(n_periods=n_periods)

    with pytest.warns(UserWarning, match="below min_n_per_factor"):
        result = estimate_measurement_system(model_spec, data, min_n_per_factor=50)

    assert len(result) == 0


def test_merge_with_user_fixed_params_user_wins():
    user_idx = pd.MultiIndex.from_tuples(
        [("loadings", 0, "skill_1", "skills")],
        names=["category", "period", "name1", "name2"],
    )
    stage1_idx = pd.MultiIndex.from_tuples(
        [
            ("loadings", 0, "skill_1", "skills"),
            ("loadings", 0, "skill_2", "skills"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    user = pd.DataFrame({"value": [42.0]}, index=user_idx)
    stage1 = pd.DataFrame({"value": [1.0, 2.0]}, index=stage1_idx)

    merged = merge_with_user_fixed_params(user, stage1)

    assert merged.loc[("loadings", 0, "skill_1", "skills"), "value"] == 42.0
    assert merged.loc[("loadings", 0, "skill_2", "skills"), "value"] == 2.0


def test_merge_with_user_fixed_params_handles_none_user():
    stage1 = pd.DataFrame(
        {"value": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "skill_1", "skills")],
            names=["category", "period", "name1", "name2"],
        ),
    )

    merged = merge_with_user_fixed_params(None, stage1)

    assert len(merged) == 1
    assert merged.loc[("loadings", 0, "skill_1", "skills"), "value"] == 1.0
