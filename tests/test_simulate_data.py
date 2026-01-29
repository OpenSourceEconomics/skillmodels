"""Tests for functions in simulate_data module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import model_spec_from_yaml_dict
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.config import TEST_DATA_DIR
from skillmodels.process_model import process_model
from skillmodels.simulate_data import (
    _collapse_aug_periods_to_periods,
    measurements_from_states,
    simulate_dataset,
)

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2():
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        return model_spec_from_yaml_dict(yaml.load(y, Loader=yaml.SafeLoader))


@pytest.fixture
def model2_data():
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


def test_simulate_dataset(model2, model2_data) -> None:
    model = model2
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    calculated = simulate_dataset(
        model_spec=model,
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


@pytest.fixture
def model2_with_endogenous():
    """Model2 with fac3 set as endogenous factor."""
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        model_dict = yaml.load(y, Loader=yaml.SafeLoader)
    model_dict["factors"]["fac3"]["is_endogenous"] = True
    del model_dict["stagemap"]
    del model_dict["anchoring"]
    return model_spec_from_yaml_dict(model_dict)


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
            record = {"id": obs_id, "aug_period": aug_p}
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
