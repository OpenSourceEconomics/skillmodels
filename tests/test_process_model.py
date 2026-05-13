"""Tests for process model."""

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.model_spec import FactorSpec
from skillmodels.common.process_model import get_has_endogenous_factors, process_model
from skillmodels.common.types import Normalizations, TransitionInfo
from skillmodels.test_data.model2 import MODEL2


@pytest.fixture
def model2():
    return MODEL2


def test_has_endogenous_factors(model2) -> None:
    assert not process_model(model2).endogenous_factors_info.has_endogenous_factors


def test_dimensions(model2) -> None:
    res = process_model(model2).dimensions
    assert res.n_latent_factors == 3
    assert res.n_observed_factors == 0
    assert res.n_all_factors == 3
    assert res.n_periods == 8
    assert res.n_controls == 2
    assert res.n_mixtures == 1


def test_labels(model2) -> None:
    res = process_model(model2).labels
    assert res.latent_factors == ("fac1", "fac2", "fac3")
    assert res.observed_factors == ()
    assert res.all_factors == ("fac1", "fac2", "fac3")
    assert res.controls == ("constant", "x1")
    assert res.periods == (0, 1, 2, 3, 4, 5, 6, 7)
    assert res.stagemap == (0, 0, 0, 0, 0, 0, 0)
    assert res.stages == (0,)


def test_chs_estimation_options(model2) -> None:
    res = process_model(model2).chs_estimation_options
    assert res.sigma_points_scale == 2
    assert res.robust_bounds
    assert np.isclose(res.bounds_distance, 0.001)


def test_anchoring(model2) -> None:
    res = process_model(model2).anchoring
    assert res.outcomes == {"fac1": "Q1"}
    assert res.factors == ("fac1",)
    assert res.free_controls
    assert res.free_constant
    assert res.free_loadings


def test_transition_info(model2) -> None:
    res = process_model(model2).transition_info

    assert isinstance(res, TransitionInfo)
    assert callable(res.func)

    assert list(inspect.signature(res.func).parameters) == ["params", "states"]


def test_update_info(model2) -> None:
    res = process_model(model2).update_info
    expected = pd.read_csv(
        TEST_DATA_DIR / "model2_correct_update_info.csv",
        index_col=["aug_period", "variable"],
    )
    assert_frame_equal(res, expected)


def test_normalizations(model2) -> None:
    expected = {
        "fac1": Normalizations(
            loadings=(
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
            ),
            intercepts=({}, {}, {}, {}, {}, {}, {}, {}),
        ),
        "fac2": Normalizations(
            loadings=(
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
            ),
            intercepts=({}, {}, {}, {}, {}, {}, {}, {}),
        ),
        "fac3": Normalizations(
            loadings=({"y7": 1}, {}, {}, {}, {}, {}, {}, {}),
            intercepts=({}, {}, {}, {}, {}, {}, {}, {}),
        ),
    }
    res = process_model(model2).normalizations

    assert res == expected


def _make_fac3_endogenous(model):
    """Return a new model with fac3 set as endogenous."""
    fac3 = model.factors["fac3"]
    new_fac3 = replace(fac3, is_endogenous=True)
    new_factors = dict(model.factors) | {"fac3": new_fac3}
    return model._replace(factors=new_factors)


def test_anchoring_and_endogenous_factors_work_together() -> None:
    model = _make_fac3_endogenous(MODEL2)._replace(stagemap=None)
    # Should not raise - anchoring and endogenous factors now work together
    result = process_model(model)
    # Verify anchoring is enabled
    assert result.anchoring.anchoring
    assert result.anchoring.factors == ("fac1",)
    # Verify endogenous factors are enabled
    assert result.endogenous_factors_info.has_endogenous_factors
    # Verify dimensions
    assert result.dimensions.n_periods == 8
    assert result.dimensions.n_aug_periods == 16
    # Verify update_info has anchoring entries for all aug_periods
    anchoring_updates = result.update_info[result.update_info["purpose"] == "anchoring"]
    assert (
        len(anchoring_updates) == 16
    )  # One per aug_period for the one anchored factor


def test_stagemap_with_endogenous_factors_wrong_labels() -> None:
    model = _make_fac3_endogenous(MODEL2)._replace(
        stagemap=(0, 0, 1, 1, 2, 2, 4),
        anchoring=None,
    )
    with pytest.raises(ValueError, match="Invalid stage map:"):
        process_model(model)


def test_stagemap_with_endogenous_factors() -> None:
    stagemap = (0, 0, 1, 1, 2, 2, 3)
    model = _make_fac3_endogenous(MODEL2)._replace(
        stagemap=stagemap,
        anchoring=None,
    )
    processed = process_model(model)
    assert processed.labels.stagemap == stagemap
    assert processed.labels.stages == (0, 1, 2, 3)
    assert processed.labels.aug_stagemap == (0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7)


@pytest.fixture
def model2_inv():
    return _make_fac3_endogenous(MODEL2)._replace(
        stagemap=None,
        anchoring=None,
    )


def test_with_endog_has_endogenous_factors(model2_inv) -> None:
    assert process_model(model2_inv).endogenous_factors_info.has_endogenous_factors


def test_with_endog_dimensions(model2_inv) -> None:
    res = process_model(model2_inv).dimensions
    assert res.n_latent_factors == 3
    assert res.n_observed_factors == 0
    assert res.n_all_factors == 3
    assert res.n_aug_periods == 16
    assert res.n_periods == 8
    assert res.n_controls == 2
    assert res.n_mixtures == 1


def test_with_endog_labels(model2_inv) -> None:
    res = process_model(model2_inv).labels
    n_aug_periods = 16
    assert res.latent_factors == ("fac1", "fac2", "fac3")
    assert res.observed_factors == ()
    assert res.all_factors == ("fac1", "fac2", "fac3")
    assert res.controls == ("constant", "x1")
    assert res.aug_periods == tuple(range(n_aug_periods))
    assert res.periods == (0, 1, 2, 3, 4, 5, 6, 7)
    assert res.aug_stagemap == tuple(range(n_aug_periods - 2))
    assert res.aug_stages == tuple(range(n_aug_periods - 2))


def test_with_endog_chs_estimation_options(model2_inv) -> None:
    res = process_model(model2_inv).chs_estimation_options
    assert res.sigma_points_scale == 2
    assert res.robust_bounds
    assert np.isclose(res.bounds_distance, 0.001)


def test_with_endog_anchoring_is_empty(model2_inv) -> None:
    res = process_model(model2_inv).anchoring
    assert res.outcomes == {}
    assert res.factors == ()
    assert res.free_controls is False
    assert res.free_constant is False
    assert res.free_loadings is False


def test_with_endog_transition_info(model2_inv) -> None:
    res = process_model(model2_inv).transition_info

    assert isinstance(res, TransitionInfo)
    assert callable(res.func)

    assert list(inspect.signature(res.func).parameters) == ["params", "states"]


def test_with_endog_update_info(model2_inv) -> None:
    res = process_model(model2_inv).update_info
    expected = pd.read_csv(
        TEST_DATA_DIR / "model2_with_endog_correct_update_info.csv",
        index_col=["aug_period", "variable"],
    )
    assert_frame_equal(res, expected)


def test_with_endog_normalizations(model2_inv) -> None:
    e = {}
    expected = {
        "fac1": Normalizations(
            loadings=(
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
            ),
            intercepts=(e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e),
        ),
        "fac2": Normalizations(
            loadings=(
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
            ),
            intercepts=(e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e),
        ),
        "fac3": Normalizations(
            loadings=(
                e,
                {"y7": 1},
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
            ),
            intercepts=(e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e),
        ),
    }
    res = process_model(model2_inv).normalizations

    assert res == expected


def _fspec(**kwargs) -> FactorSpec:
    """Create a minimal FactorSpec for unit tests."""
    return FactorSpec(measurements=((),), **kwargs)


def test_model_has_endogenous_factors_not_specified() -> None:
    factors = {"a": _fspec()}
    assert not get_has_endogenous_factors(factors)


def test_get_has_endogenous_factors_wrong_constellation() -> None:
    factors = {"a": _fspec(is_endogenous=False, is_correction=True)}
    with pytest.raises(ValueError, match="is_endogenous"):
        get_has_endogenous_factors(factors)


def test_get_has_endogenous_factors_indeed() -> None:
    factors = {
        "a": _fspec(is_endogenous=True, is_correction=False),
        "b": _fspec(is_endogenous=False, is_correction=False),
    }
    assert get_has_endogenous_factors(factors)


def test_get_has_endogenous_factors_and_correction() -> None:
    factors = {
        "a": _fspec(is_endogenous=True, is_correction=False),
        "b": _fspec(is_endogenous=False, is_correction=False),
        "c": _fspec(is_endogenous=True, is_correction=True),
    }
    assert get_has_endogenous_factors(factors)
