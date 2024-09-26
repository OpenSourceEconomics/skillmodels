import inspect
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from skillmodels.process_model import get_has_investments, process_model

# ======================================================================================
# Integration test with model2 from the replication files of CHS2010
# ======================================================================================

# importing the TEST_DIR from config does not work for test run in conda build
TEST_DIR = Path(__file__).parent.resolve()


@pytest.fixture
def model2():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    return model_dict


def test_has_investments(model2):
    assert process_model(model2)["has_investments"] == False


def test_dimensions(model2):
    res = process_model(model2)["dimensions"]
    assert res["n_latent_factors"] == 3
    assert res["n_observed_factors"] == 0
    assert res["n_all_factors"] == 3
    assert res["n_periods"] == 8
    assert res["n_controls"] == 2
    assert res["n_mixtures"] == 1


def test_labels(model2):
    res = process_model(model2)["labels"]
    assert res["latent_factors"] == ["fac1", "fac2", "fac3"]
    assert res["observed_factors"] == []
    assert res["all_factors"] == ["fac1", "fac2", "fac3"]
    assert res["controls"] == ["constant", "x1"]
    assert res["periods"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert res["stagemap"] == [0, 0, 0, 0, 0, 0, 0]
    assert res["stages"] == [0]


def test_estimation_options(model2):
    res = process_model(model2)["estimation_options"]
    assert res["sigma_points_scale"] == 2
    assert res["robust_bounds"]
    assert res["bounds_distance"] == 0.001


def test_anchoring(model2):
    res = process_model(model2)["anchoring"]
    assert res["outcomes"] == {"fac1": "Q1"}
    assert res["factors"] == ["fac1"]
    assert res["free_controls"]
    assert res["free_constant"]
    assert res["free_loadings"]


def test_transition_info(model2):
    res = process_model(model2)["transition_info"]

    assert isinstance(res, dict)
    assert callable(res["func"])

    assert list(inspect.signature(res["func"]).parameters) == ["params", "states"]


def test_update_info(model2):
    res = process_model(model2)["update_info"]
    test_dir = Path(__file__).parent.resolve()
    expected = pd.read_csv(
        test_dir / "model2_correct_update_info.csv",
        index_col=["period", "variable"],
    )
    assert_frame_equal(res, expected)


def test_normalizations(model2):
    expected = {
        "fac1": {
            "loadings": [
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
            ],
            "intercepts": [{}, {}, {}, {}, {}, {}, {}, {}],
        },
        "fac2": {
            "loadings": [
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
            ],
            "intercepts": [{}, {}, {}, {}, {}, {}, {}, {}],
        },
        "fac3": {
            "loadings": [{"y7": 1}, {}, {}, {}, {}, {}, {}, {}],
            "intercepts": [{}, {}, {}, {}, {}, {}, {}, {}],
        },
    }
    res = process_model(model2)["normalizations"]

    assert res == expected


# ======================================================================================
# Augment model2 with investments
# ======================================================================================


def test_anchoring_and_investments_conflict():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    # Set fac3 to be an investment
    model_dict["factors"]["fac3"]["is_investment"] = True
    del model_dict["stagemap"]
    with pytest.raises(
        ValueError, match="anchoring is not supported when investments are present."
    ):
        process_model(model_dict)


def test_stages_and_investments_conflict():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    # Set fac3 to be an investment
    model_dict["factors"]["fac3"]["is_investment"] = True
    del model_dict["anchoring"]
    with pytest.raises(
        ValueError, match="Stages currently not supported when investments are present"
    ):
        process_model(model_dict)


@pytest.fixture
def model2_inv():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    # Set fac3 to be an investment
    model_dict["factors"]["fac3"]["is_investment"] = True
    del model_dict["stagemap"]
    del model_dict["anchoring"]
    return model_dict


def test_with_inv_has_investments(model2_inv):
    assert process_model(model2_inv)["has_investments"] == True


def test_with_inv_dimensions(model2_inv):
    res = process_model(model2_inv)["dimensions"]
    assert res["n_latent_factors"] == 3
    assert res["n_observed_factors"] == 0
    assert res["n_all_factors"] == 3
    assert res["n_periods"] == 16
    assert res["n_periods_raw"] == 8
    assert res["n_controls"] == 2
    assert res["n_mixtures"] == 1


def test_with_inv_labels(model2_inv):
    res = process_model(model2_inv)["labels"]
    assert res["latent_factors"] == ["fac1", "fac2", "fac3"]
    assert res["observed_factors"] == []
    assert res["all_factors"] == ["fac1", "fac2", "fac3"]
    assert res["controls"] == ["constant", "x1"]
    assert res["periods"] == list(range(16))
    assert res["periods_raw"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert res["stagemap"] == list(range(15))
    assert res["stages"] == list(range(15))


def test_with_inv_estimation_options(model2_inv):
    res = process_model(model2_inv)["estimation_options"]
    assert res["sigma_points_scale"] == 2
    assert res["robust_bounds"]
    assert res["bounds_distance"] == 0.001


def test_with_inv_anchoring_is_empty(model2_inv):
    res = process_model(model2_inv)["anchoring"]
    assert res["outcomes"] == {}
    assert res["factors"] == []
    assert res["free_controls"] is False
    assert res["free_constant"] is False
    assert res["free_loadings"] is False


def test_with_inv_transition_info(model2_inv):
    res = process_model(model2_inv)["transition_info"]

    assert isinstance(res, dict)
    assert callable(res["func"])

    assert list(inspect.signature(res["func"]).parameters) == ["params", "states"]


def test_with_inv_update_info(model2_inv):
    res = process_model(model2_inv)["update_info"]
    test_dir = Path(__file__).parent.resolve()
    expected = pd.read_csv(
        test_dir / "model2_with_inv_correct_update_info.csv",
        index_col=["period", "variable"],
    )
    assert_frame_equal(res, expected)


def test_with_inv_normalizations(model2_inv):
    expected = {
        "fac1": {
            "loadings": [
                {"y1": 1},
                {},
                {"y1": 1},
                {},
                {"y1": 1},
                {},
                {"y1": 1},
                {},
                {"y1": 1},
                {},
                {"y1": 1},
                {},
                {"y1": 1},
                {},
                {"y1": 1},
                {},
            ],
            "intercepts": [
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
            ],
        },
        "fac2": {
            "loadings": [
                {"y4": 1},
                {},
                {"y4": 1},
                {},
                {"y4": 1},
                {},
                {"y4": 1},
                {},
                {"y4": 1},
                {},
                {"y4": 1},
                {},
                {"y4": 1},
                {},
                {"y4": 1},
                {},
            ],
            "intercepts": [
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
            ],
        },
        "fac3": {
            "loadings": [
                {},
                {"y7": 1},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
            ],
            "intercepts": [
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                {},
            ],
        },
    }
    res = process_model(model2_inv)["normalizations"]

    assert res == expected


# ======================================================================================
# Unit tests
# ======================================================================================


def test_model_has_investments_not_specified():
    factors = {"a": {}}
    assert get_has_investments(factors) == False


def test_get_has_investments_wrong_type():
    factors = {"a": {"is_investment": 3}}
    with pytest.raises(ValueError):
        get_has_investments(factors)


def test_get_has_investments_wrong_constellation():
    factors = {"a": {"is_investment": False, "is_correction": True}}
    with pytest.raises(ValueError):
        get_has_investments(factors)


def test_get_has_investments_indeed():
    factors = {
        "a": {"is_investment": True, "is_correction": False},
        "b": {"is_investment": False, "is_correction": False},
    }
    assert get_has_investments(factors) == True


def test_get_has_investments_and_correction():
    factors = {
        "a": {"is_investment": True, "is_correction": False},
        "b": {"is_investment": False, "is_correction": False},
        "c": {"is_investment": True, "is_correction": True},
    }
    assert get_has_investments(factors) == True
