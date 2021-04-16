from pathlib import Path

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from skillmodels.process_model import process_model

# ======================================================================================
# Integration test with model2 from the replication files of CHS2010
# ======================================================================================


@pytest.fixture
def model2():
    test_dir = Path(__file__).parent.resolve()
    with open(test_dir / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    return model_dict


def test_dimensions(model2):
    res = process_model(model2)["dimensions"]
    assert res["n_states"] == 3
    assert res["n_periods"] == 8
    assert res["n_controls"] == 2
    assert res["n_mixtures"] == 1


def test_labels(model2):
    res = process_model(model2)["labels"]
    assert res["factors"] == ["fac1", "fac2", "fac3"]
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


def test_transition_functions(model2):
    res = process_model(model2)["transition_functions"]
    assert len(res) == 3
    assert [tup[0] for tup in res] == ["log_ces", "linear", "constant"]


def test_update_info(model2):
    res = process_model(model2)["update_info"]
    test_dir = Path(__file__).parent.resolve()
    expected = pd.read_csv(
        test_dir / "model2_correct_update_info.csv", index_col=["period", "variable"]
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
