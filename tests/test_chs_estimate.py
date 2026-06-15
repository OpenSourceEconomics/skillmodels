"""End-to-end smoke test for `skillmodels.chs.estimate.estimate_chs`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skillmodels.chs.estimate import estimate_chs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.test_data.model2 import MODEL2

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2_data():
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


def test_estimate_chs_returns_conforming_result(model2_data):
    """`estimate_chs` converges from the known optimum and returns a
    `CHSEstimationResult` with a finite log-likelihood.

    Starting at the regression-vault optimum keeps the run cheap: the
    optimiser terminates almost immediately. `start_params_strategy="none"`
    skips the (slow) AMN/Spearman seeding since full start values are
    supplied.
    """
    start_params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    start_params = start_params.set_index(["category", "period", "name1", "name2"])

    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=start_params,
    )

    assert isinstance(result, CHSEstimationResult)
    assert result.success
    assert np.isfinite(result.loglikelihood)
    assert result.md_criterion is None
    assert result.model_spec is MODEL2
    assert "value" in result.params.columns
