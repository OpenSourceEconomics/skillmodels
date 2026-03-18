"""Shared test fixtures and helpers."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from skillmodels.config import TEST_DATA_DIR
from skillmodels.test_data.model2 import MODEL2

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2():
    """Return MODEL2 test model specification."""
    return MODEL2


@pytest.fixture
def model2_data():
    """Load model2 simulated data with (caseid, period) index."""
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


@pytest.fixture
def model2_params():
    """Load regression vault params for one_stage_anchoring."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    return params.set_index(["category", "period", "name1", "name2"])


@pytest.fixture
def model2_with_endogenous():
    """Model2 with fac3 set as endogenous factor."""
    fac3 = MODEL2.factors["fac3"]
    new_fac3 = replace(fac3, is_endogenous=True)
    new_factors = dict(MODEL2.factors) | {"fac3": new_fac3}
    return MODEL2._replace(
        factors=new_factors,
        stagemap=None,
        anchoring=None,
    )
