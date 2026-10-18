"""End-to-end smoke test for `skillmodels.chs.estimate.estimate_chs`."""

import functools
from pathlib import Path

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.chs.estimate import estimate_chs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.fixed_constraint import FixedConstraintWithValue
from skillmodels.common.selector import select_by_loc
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
    default `require_initial_anchors=True`, `estimate_chs` refuses to run it.
    """
    with pytest.raises(ValueError, match="not anchored"):
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
    disabled via `require_initial_anchors=False`.
    """
    result = estimate_chs(
        MODEL2,
        model2_data,
        CHSEstimationOptions(start_params_strategy="none"),
        start_params=anchoring_start_params,
        require_initial_anchors=False,
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
        require_initial_anchors=False,
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
        require_initial_anchors=False,
    )

    assert result.params.loc[loc, "value"] == pytest.approx(target)


def test_estimate_chs_forwards_error_handling(
    model2_data, anchoring_start_params, monkeypatch
):
    """`error_handling` reaches optimagic so failed evaluations can be penalised."""
    captured = {}

    def fake_estimate_ml(**kwargs):
        captured.update(kwargs["optimize_options"])
        raise StopIteration

    monkeypatch.setattr("skillmodels.chs.estimate.estimate_ml", fake_estimate_ml)
    with pytest.raises(StopIteration):
        estimate_chs(
            MODEL2,
            model2_data,
            CHSEstimationOptions(
                start_params_strategy="none", error_handling="continue"
            ),
            start_params=anchoring_start_params,
            require_initial_anchors=False,
        )
    assert captured["error_handling"] == "continue"


def test_estimate_chs_applies_lower_bounds_and_clips_start(
    model2_data, anchoring_start_params, monkeypatch
):
    """`lower_bounds` tightens the template bounds; the start moves onto them."""
    captured = {}

    def fake_estimate_ml(**kwargs):
        captured.update(kwargs)
        raise StopIteration

    loc = anchoring_start_params.query("category == 'meas_sds'").index[:1]
    start = anchoring_start_params.copy()
    start.loc[loc, "value"] = 0.05
    lower_bounds = pd.Series(0.1, index=loc)

    monkeypatch.setattr("skillmodels.chs.estimate.estimate_ml", fake_estimate_ml)
    with pytest.raises(StopIteration):
        estimate_chs(
            MODEL2,
            model2_data,
            CHSEstimationOptions(start_params_strategy="none"),
            start_params=start,
            lower_bounds=lower_bounds,
            require_initial_anchors=False,
        )
    assert captured["bounds"].lower.loc[loc].tolist() == [0.1]
    assert captured["params"].loc[loc, "value"].tolist() == [0.1]
    other = captured["bounds"].lower.drop(loc)
    assert other[np.isfinite(other)].max() < 0.1


def test_estimate_chs_lower_bounds_can_remove_a_bound(
    model2_data, anchoring_start_params, monkeypatch
):
    """A `-inf` entry in `lower_bounds` replaces the template's bound."""
    captured = {}

    def fake_estimate_ml(**kwargs):
        captured.update(kwargs)
        raise StopIteration

    loc = anchoring_start_params.query("category == 'meas_sds'").index[:1]
    monkeypatch.setattr("skillmodels.chs.estimate.estimate_ml", fake_estimate_ml)
    with pytest.raises(StopIteration):
        estimate_chs(
            MODEL2,
            model2_data,
            CHSEstimationOptions(start_params_strategy="none"),
            start_params=anchoring_start_params,
            lower_bounds=pd.Series(-np.inf, index=loc),
            require_initial_anchors=False,
        )
    assert captured["bounds"].lower.loc[loc].tolist() == [-np.inf]


def test_estimate_chs_applies_upper_bounds_and_clips_start(
    model2_data, anchoring_start_params, monkeypatch
):
    """`upper_bounds` replaces the template bounds; the start moves onto them."""
    captured = {}

    def fake_estimate_ml(**kwargs):
        captured.update(kwargs)
        raise StopIteration

    loc = pd.MultiIndex.from_tuples(
        [("controls", 0, "y1", "x1")], names=anchoring_start_params.index.names
    )
    start = anchoring_start_params.copy()
    start.loc[loc, "value"] = 0.5
    monkeypatch.setattr("skillmodels.chs.estimate.estimate_ml", fake_estimate_ml)
    with pytest.raises(StopIteration):
        estimate_chs(
            MODEL2,
            model2_data,
            CHSEstimationOptions(start_params_strategy="none"),
            start_params=start,
            upper_bounds=pd.Series(0.0, index=loc),
            require_initial_anchors=False,
        )
    assert captured["bounds"].upper.loc[loc].tolist() == [0.0]
    assert captured["params"].loc[loc, "value"].tolist() == [0.0]


def test_estimate_chs_moves_start_onto_linear_equalities(
    model2_data, anchoring_start_params, monkeypatch
):
    """Start values satisfy user `LinearConstraint` equalities (a linear trend)."""
    captured = {}

    def fake_estimate_ml(**kwargs):
        captured.update(kwargs)
        raise StopIteration

    sds = anchoring_start_params.query("category == 'meas_sds' & name1 == 'y1'").index
    start = anchoring_start_params.copy()
    start.loc[sds, "value"] = np.linspace(0.5, 1.5, len(sds)) ** 2
    constraints: list[om.constraints.Constraint] = [
        om.LinearConstraint(
            selector=functools.partial(select_by_loc, loc=list(sds[t : t + 3])),
            weights=np.array([1.0, -2.0, 1.0]),
            value=0.0,
        )
        for t in range(len(sds) - 2)
    ]
    monkeypatch.setattr("skillmodels.chs.estimate.estimate_ml", fake_estimate_ml)
    with pytest.raises(StopIteration):
        estimate_chs(
            MODEL2,
            model2_data,
            CHSEstimationOptions(start_params_strategy="none"),
            start_params=start,
            constraints=constraints,
            lower_bounds=pd.Series(-np.inf, index=sds),
            require_initial_anchors=False,
        )
    values = captured["params"].loc[sds, "value"].to_numpy()
    np.testing.assert_allclose(np.diff(values, n=2), 0.0, atol=1e-12)
