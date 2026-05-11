"""Tests for process_debug_data module."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.chs.process_debug_data import (
    _create_post_update_states,
    _process_residuals,
    create_state_ranges,
)


def test_create_state_ranges_invalid_quantile_raises() -> None:
    states = pd.DataFrame({"fac1": [1.0, 2.0, 3.0, 4.0], "period": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="quantile_cutoff"):
        create_state_ranges(states, factors=["fac1"], quantile_cutoff=1.5)


def test_process_residuals_ids_with_mixtures() -> None:
    """ID column should repeat observation ids, not be sequential."""
    n_obs = 3
    n_mixtures = 2
    residuals = [np.ones((n_obs, n_mixtures)) * i for i in range(2)]
    update_info = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([(0, "m1"), (0, "m2")]),
    )

    result = _process_residuals(residuals=residuals, update_info=update_info)  # ty: ignore[invalid-argument-type]

    # For each update, ids should be [0, 0, 1, 1, 2, 2] not [0, 1, 2, 3, 4, 5]
    for _, group in result.groupby(["aug_period", "measurement"]):
        expected_ids = np.repeat(np.arange(n_obs), n_mixtures)
        np.testing.assert_array_equal(group["id"].to_numpy(), expected_ids)


def test_create_post_update_states_ids_with_mixtures() -> None:
    """ID column should repeat observation ids per mixture."""
    n_obs = 3
    n_mixtures = 2
    n_states = 2
    factors = ("fac1", "fac2")

    filtered_states = np.array(
        [np.ones((n_obs, n_mixtures, n_states)) * i for i in range(2)]
    )
    update_info = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([(0, "m1"), (0, "m2")]),
    )

    result = _create_post_update_states(
        filtered_states=filtered_states,  # ty: ignore[invalid-argument-type]
        factors=factors,
        update_info=update_info,
    )

    for _, group in result.groupby(["aug_period", "measurement"]):
        expected_ids = np.repeat(np.arange(n_obs), n_mixtures)
        np.testing.assert_array_equal(group["id"].to_numpy(), expected_ids)
