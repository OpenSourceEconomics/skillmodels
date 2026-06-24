"""End-to-end smoke test for `skillmodels.chs.estimate.estimate_chs`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skillmodels.chs.estimate import estimate_chs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.fixed_constraint import FixedConstraintWithValue
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


def test_estimate_chs_rejects_unidentified_model_by_default(model2_data):
    """The identification gate fires before estimation on an unanchored model.

    MODEL2 normalizes its period-0 loadings but leaves the measurement
    intercepts free and runs a single mixture component, so its initial latent
    mean is a free parameter with no location anchor — a common shift of the
    latent mean against the intercepts leaves the likelihood unchanged. With the
    default `require_identification=True`, `estimate_chs` refuses to run it.
    """
    with pytest.raises(ValueError, match="not identified"):
        estimate_chs(
            MODEL2,
            model2_data,
            CHSEstimationOptions(start_params_strategy="none"),
        )


def test_estimate_chs_returns_conforming_result(model2_data, anchoring_start_params):
    """Converge from the known optimum to a conforming `CHSEstimationResult`.

    Starting at the regression-vault optimum keeps the run cheap: the
    optimiser terminates almost immediately. `start_params_strategy="none"`
    skips the (slow) AMN/Spearman seeding since full start values are
    supplied. MODEL2 follows the CHS convention of a free initial latent mean
    (seeded to 0) rather than an intercept normalization, so it is
    intentionally location-under-identified and the identification gate is
    disabled via `require_identification=False`.
    """
    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=anchoring_start_params,
        require_identification=False,
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

    MODEL2 is intentionally location-under-identified (free initial mean,
    CHS convention), so the identification gate is disabled here.
    """
    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=anchoring_start_params,
        require_identification=False,
    )

    assert result.likelihood_result is not None
    standard_errors = result.likelihood_result.se()
    assert standard_errors is not None


def test_estimate_chs_enforces_user_fixed_constraint_value(
    model2_data, anchoring_start_params
):
    """A user `FixedConstraintWithValue` fixes the param at its requested value.

    `om.FixedConstraint` pins a parameter at its *start* value, so the wrapper's
    `.value` only takes effect if `estimate_chs` writes it into the start vector.
    User `constraints=` are merged after `get_maximization_inputs` (which only
    enforces the internal fixed constraints), so without an explicit enforce the
    parameter was silently fixed at the seeded start value, not the requested
    value. Regression for audit finding F9.

    MODEL2 is intentionally location-under-identified (free initial mean,
    CHS convention), so the identification gate is disabled here.
    """
    loc = ("controls", 0, "y1", "x1")
    start_value = float(anchoring_start_params.loc[loc, "value"])
    target = start_value - 2.0  # clearly different from the seeded start value

    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=anchoring_start_params,
        constraints=[FixedConstraintWithValue(loc=loc, value=target)],
        require_identification=False,
    )

    assert result.params.loc[loc, "value"] == pytest.approx(target)
