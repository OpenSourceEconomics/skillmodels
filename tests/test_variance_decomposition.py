"""Tests for variance decomposition."""

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.common.variance_decomposition import (
    _compute_variance_decomposition,
    summarize_measurement_reliability,
)


@pytest.fixture
def setup_variance_decomposition():
    """Create test data for variance decomposition."""
    # Filtered states with known variances
    # fac1: variance = 0.0025 (values 0.1, 0.1, 0.1, 0.2)
    # fac2: variance = 0 (all 0.1)
    # fac3: variance = 0.01 (values 0.2, 0.2, 0.2, 0.4)
    filtered_states = pd.DataFrame(
        {
            "fac1": [0.1, 0.1, 0.1, 0.2],
            "fac2": [0.1, 0.1, 0.1, 0.1],
            "fac3": [0.2, 0.2, 0.2, 0.4],
            "period": [0, 0, 0, 0],
            "id": [0, 1, 2, 3],
        }
    )

    # Parameters: loadings and measurement SDs
    # y1: loading=1 on fac1, meas_sd=0.05
    # y2: loading=0.1 on fac2, meas_sd=1.1
    # y3: loading=2 on fac3, meas_sd=0.1

    loadings_data = []
    for aug_period in [0]:
        for meas, factor, loading in [
            ("y1", "fac1", 1.0),
            ("y1", "fac2", 0.0),
            ("y1", "fac3", 0.0),
            ("y2", "fac1", 0.0),
            ("y2", "fac2", 0.1),
            ("y2", "fac3", 0.0),
            ("y3", "fac1", 0.0),
            ("y3", "fac2", 0.0),
            ("y3", "fac3", 2.0),
        ]:
            loadings_data.append((aug_period, meas, factor, loading))

    loadings_df = pd.DataFrame(
        loadings_data, columns=["aug_period", "name1", "name2", "value"]
    )
    loadings_df = loadings_df.set_index(["aug_period", "name1", "name2"])

    meas_sds_data = [(0, "y1", "-", 0.05), (0, "y2", "-", 1.1), (0, "y3", "-", 0.1)]
    meas_sds_df = pd.DataFrame(
        meas_sds_data, columns=["aug_period", "name1", "name2", "value"]
    )
    meas_sds_df = meas_sds_df.set_index(["aug_period", "name1", "name2"])

    params = pd.concat([loadings_df, meas_sds_df], keys=["loadings", "meas_sds"])

    return {
        "filtered_states": filtered_states,
        "params": params,
        "aug_periods_to_periods": {0: 0},
    }


@pytest.fixture
def expected_variance_decomposition():
    """Expected results for variance decomposition test.

    Expected signal/noise fractions:
    - y1: 50% signal, 50% noise (loading 1, var 0.0025, sd 0.05)
    - y2: 0% signal, 100% noise (loading 0.1, var 0, sd 1.1)
    - y3: 80% signal, 20% noise (loading 2, var 0.01, sd 0.1)
    """
    index = pd.MultiIndex.from_tuples(
        [(0, "y1", "fac1"), (0, "y2", "fac2"), (0, "y3", "fac3")],
        names=["period", "measurement", "factor"],
    )
    return pd.DataFrame(
        {
            "loading": [1.0, 0.1, 2.0],
            "factor_variance": [0.0025, 0.0, 0.01],
            "meas_sd": [0.05, 1.1, 0.1],
            "fraction_signal": [0.5, 0.0, 0.8],
            "fraction_noise": [0.5, 1.0, 0.2],
            "signal_to_noise_ratio": [1.0, 0.0, 4.0],
        },
        index=index,
    )


def test_compute_variance_decomposition(
    setup_variance_decomposition, expected_variance_decomposition
):
    """Test that variance decomposition computes correct fractions."""
    result = _compute_variance_decomposition(**setup_variance_decomposition)
    aaae(result.values, expected_variance_decomposition.values)


def test_summarize_measurement_reliability(expected_variance_decomposition):
    """Test summary statistics for measurement reliability."""
    summary = summarize_measurement_reliability(expected_variance_decomposition)

    assert "y3" in summary.index  # Highest signal
    assert "y2" in summary.index  # Lowest signal
    assert summary.loc["y3", "mean_signal"] == pytest.approx(0.8)
    assert summary.loc["y2", "mean_signal"] == pytest.approx(0.0)
    assert summary.loc["y1", "mean_signal"] == pytest.approx(0.5)
