"""Tests for process_debug_data module."""

import pandas as pd
import pytest

from skillmodels.process_debug_data import create_state_ranges


def test_create_state_ranges_invalid_quantile_raises() -> None:
    states = pd.DataFrame({"fac1": [1.0, 2.0, 3.0, 4.0], "period": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="quantile_cutoff"):
        create_state_ranges(states, factors=["fac1"], quantile_cutoff=1.5)
