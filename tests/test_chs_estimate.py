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


@pytest.fixture
def anchoring_start_params():
    start_params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    return start_params.set_index(["category", "period", "name1", "name2"])


def test_estimate_chs_returns_conforming_result(model2_data, anchoring_start_params):
    """Converge from the known optimum to a conforming `CHSEstimationResult`.

    Starting at the regression-vault optimum keeps the run cheap: the
    optimiser terminates almost immediately. `start_params_strategy="none"`
    skips the (slow) AMN/Spearman seeding since full start values are
    supplied.
    """
    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=anchoring_start_params,
    )

    assert isinstance(result, CHSEstimationResult)
    assert result.success
    assert np.isfinite(result.loglikelihood)
    assert result.md_criterion is None
    assert result.model_spec is MODEL2
    assert "value" in result.params.columns


def test_estimate_chs_provides_ml_inference(model2_data, anchoring_start_params):
    """`estimate_chs` drives `estimate_ml`, so the result carries inference.

    The result exposes the estimagic `LikelihoodResult`, from which standard
    errors (and covariances, summaries) are available — the reason apps that
    need inference can adopt `estimate_chs` instead of hand-rolling
    `estimate_ml` on top of `get_maximization_inputs`.
    """
    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=anchoring_start_params,
    )

    assert result.likelihood_result is not None
    standard_errors = result.likelihood_result.se()
    assert standard_errors is not None
