"""Tests for functions in simulate_data module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.model_spec import (
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.params_index import get_params_index
from skillmodels.process_model import process_model
from skillmodels.simulate_data import (
    _collapse_aug_periods_to_periods,
    _get_shock,
    measurements_from_states,
    simulate_dataset,
    simulate_policy_effect,
)
from skillmodels.test_data.model2 import MODEL2

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def test_simulate_dataset(model2, model2_data) -> None:
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    calculated = simulate_dataset(
        model_spec=model2,
        params=params,
        data=model2_data,
    )

    factors = ["fac1", "fac2", "fac3"]
    expected_ratios = [1.187757, 1, 1]
    for factor, expected_ratio in zip(factors, expected_ratios, strict=False):
        anch_ranges = calculated["anchored_states"]["state_ranges"][factor]
        unanch_ranges = calculated["unanchored_states"]["state_ranges"][factor]
        ratio = (anch_ranges / unanch_ranges).to_numpy()
        assert np.allclose(ratio, expected_ratio)


def test_measurements_from_factors() -> None:
    rng = np.random.default_rng(42)
    states = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    controls = np.array([[1, 1], [1, 1]], dtype=np.float64)
    loadings = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    control_params = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    sds = np.zeros(3)
    expected = np.array([[1, 1, 1], [1.9, 1.9, 1.9]])
    aaae(
        measurements_from_states(rng, states, controls, loadings, control_params, sds),
        expected,
    )


def test_collapse_aug_periods_to_periods_with_endogenous_factors(
    model2_with_endogenous,
) -> None:
    """Test that _collapse_aug_periods_to_periods works with endogenous factors.

    This is a regression test for a bug where MeasurementType enum values were
    compared against strings in pandas queries, causing empty results.
    """
    rng = np.random.default_rng(42)
    processed_model = process_model(model2_with_endogenous)
    factors = processed_model.labels.latent_factors

    # Create a mock aug_latent_data DataFrame with aug_period column
    n_obs = 5
    n_aug_periods = (
        processed_model.dimensions.n_aug_periods - 1
    )  # Exclude last half-period
    records = []
    for aug_p in range(n_aug_periods):
        for obs_id in range(n_obs):
            record: dict[str, int | float] = {"id": obs_id, "aug_period": aug_p}
            for fac in factors:
                record[fac] = rng.standard_normal()
            records.append(record)
    aug_latent_data = pd.DataFrame(records)

    result = _collapse_aug_periods_to_periods(
        df=aug_latent_data,
        factors=factors,
        aug_periods_to_periods=processed_model.labels.aug_periods_to_periods,
        endogenous_factors_info=processed_model.endogenous_factors_info,
    )

    # The result should not be empty
    assert len(result) > 0, "Collapsed DataFrame should not be empty"

    # Should have 'period' column, not 'aug_period'
    assert "period" in result.columns
    assert "aug_period" not in result.columns

    # Should have all factor columns
    for fac in factors:
        assert fac in result.columns

    # Should have correct number of unique periods (half of aug_periods)
    expected_n_periods = processed_model.dimensions.n_periods
    assert result["period"].nunique() == expected_n_periods

    # Should have all observations for each period
    assert len(result) == n_obs * expected_n_periods


def test_simulate_dataset_no_data_no_nobs_raises(model2) -> None:
    """Both data=None and n_obs=None should raise ValueError."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])
    with pytest.raises(ValueError, match="Either `data` or `n_obs`"):
        simulate_dataset(model_spec=model2, params=params, data=None, n_obs=None)


def test_simulate_dataset_observed_factors_without_data_raises() -> None:
    """Model with observed factors requires data."""
    model = MODEL2.with_added_observed_factors("obs1")
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])
    with pytest.raises(ValueError, match="observed factors"):
        simulate_dataset(model_spec=model, params=params, data=None, n_obs=100)


def test_simulate_dataset_controls_without_data_raises(model2) -> None:
    """Model with non-constant controls requires data."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])
    with pytest.raises(ValueError, match="controls"):
        simulate_dataset(model_spec=model2, params=params, data=None, n_obs=100)


def test_simulate_policy_effect(model2, model2_data) -> None:
    """Deterministic policy should produce non-zero diffs."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    policies = [
        {
            "aug_period": 0,
            "factor": "fac1",
            "effect_size": 0.5,
            "standard_deviation": 0,
        },
    ]
    result = simulate_policy_effect(
        model_spec=model2,
        params=params,
        data=model2_data,
        policies=policies,
        seed=42,
    )
    assert isinstance(result, pd.DataFrame)
    assert "fac1" in result.columns
    # The policy effect should show up as non-zero differences
    assert not np.allclose(result["fac1"].to_numpy(), 0)


def test_simulate_policy_effect_unanchored(model2, model2_data) -> None:
    """Unanchored states should also work."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    policies = [
        {
            "aug_period": 0,
            "factor": "fac1",
            "effect_size": 0.5,
            "standard_deviation": 0,
        },
    ]
    result = simulate_policy_effect(
        model_spec=model2,
        params=params,
        data=model2_data,
        policies=policies,
        seed=42,
        use_anchored_states=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert not np.allclose(result["fac1"].to_numpy(), 0)


def test_get_shock_deterministic() -> None:
    """sd=0 should produce array of all equal values."""
    rng = np.random.default_rng(42)
    result = _get_shock(rng, mean=1.5, sd=0, size=10)
    aaae(result, np.full(10, 1.5))


def test_get_shock_stochastic() -> None:
    """sd>0 should produce array of correct shape."""
    rng = np.random.default_rng(42)
    result = _get_shock(rng, mean=0.0, sd=1.0, size=20)
    assert result.shape == (20,)


def test_simulate_dataset_nobs_mismatch_warns(model2, model2_data) -> None:
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])
    with pytest.warns(UserWarning, match="n_obs"):
        simulate_dataset(model_spec=model2, params=params, data=model2_data, n_obs=999)


def test_get_shock_negative_sd_raises() -> None:
    """sd<0 should raise ValueError."""
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="negative standard deviation"):
        _get_shock(rng, mean=0.0, sd=-1.0, size=10)


def test_simulate_dataset_no_data_with_nobs() -> None:
    """Simulate with data=None and n_obs should work for controls-free model."""
    model_no_controls = ModelSpec(
        factors={
            "fac1": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 3,
                    intercepts=({},) * 3,
                ),
                transition_function="linear",
            ),
            "fac2": FactorSpec(
                measurements=(("y4", "y5", "y6"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y4": 1},) * 3,
                    intercepts=({},) * 3,
                ),
                transition_function="linear",
            ),
        },
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )

    processed = process_model(model_no_controls)
    p_index = get_params_index(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        transition_info=processed.transition_info,
        endogenous_factors_info=processed.endogenous_factors_info,
    )
    params = pd.DataFrame({"value": np.ones(len(p_index)) * 0.5}, index=p_index)

    result = simulate_dataset(
        model_spec=model_no_controls,
        params=params,
        n_obs=50,
        data=None,
        seed=42,
    )

    assert "unanchored_states" in result
    states = result["unanchored_states"]["states"]
    assert len(states) > 0
    assert "fac1" in states.columns
    assert "fac2" in states.columns
