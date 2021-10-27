from pathlib import Path

import pandas as pd
import pytest
import yaml

from skillmodels.params_index import get_control_params_index_tuples
from skillmodels.params_index import get_initial_cholcovs_index_tuples
from skillmodels.params_index import get_loadings_index_tuples
from skillmodels.params_index import get_meas_sds_index_tuples
from skillmodels.params_index import get_mixture_weights_index_tuples
from skillmodels.params_index import get_params_index
from skillmodels.params_index import get_shock_sds_index_tuples
from skillmodels.params_index import get_transition_index_tuples
from skillmodels.params_index import initial_mean_index_tuples
from skillmodels.process_model import process_model


@pytest.fixture
def model2_inputs():
    test_dir = Path(__file__).parent.resolve()
    with open(test_dir / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    processed = process_model(model_dict)
    return processed["update_info"], processed["labels"], processed["dimensions"]


def test_params_index_with_model2(model2_inputs):
    test_dir = Path(__file__).parent.resolve()
    calculated = get_params_index(*model2_inputs)
    expected = pd.read_csv(
        test_dir / "model2_correct_params_index.csv",
        index_col=["category", "period", "name1", "name2"],
    ).index

    assert calculated.equals(expected)


def test_control_coeffs_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))
    controls = ["constant", "c1"]

    expected = [
        ("controls", 0, "m1", "constant"),
        ("controls", 0, "m1", "c1"),
        ("controls", 0, "m2", "constant"),
        ("controls", 0, "m2", "c1"),
        ("controls", 0, "bla", "constant"),
        ("controls", 0, "bla", "c1"),
        ("controls", 1, "m1", "constant"),
        ("controls", 1, "m1", "c1"),
        ("controls", 1, "m2", "constant"),
        ("controls", 1, "m2", "c1"),
    ]

    calculated = get_control_params_index_tuples(controls, uinfo)
    assert calculated == expected


def test_loading_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))
    factors = ["fac1", "fac2"]
    expected = [
        ("loadings", 0, "m1", "fac1"),
        ("loadings", 0, "m1", "fac2"),
        ("loadings", 0, "m2", "fac1"),
        ("loadings", 0, "m2", "fac2"),
        ("loadings", 0, "bla", "fac1"),
        ("loadings", 0, "bla", "fac2"),
        ("loadings", 1, "m1", "fac1"),
        ("loadings", 1, "m1", "fac2"),
        ("loadings", 1, "m2", "fac1"),
        ("loadings", 1, "m2", "fac2"),
    ]

    calculated = get_loadings_index_tuples(factors, uinfo)
    assert calculated == expected


def test_meas_sd_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))

    expected = [
        ("meas_sds", 0, "m1", "-"),
        ("meas_sds", 0, "m2", "-"),
        ("meas_sds", 0, "bla", "-"),
        ("meas_sds", 1, "m1", "-"),
        ("meas_sds", 1, "m2", "-"),
    ]

    calculated = get_meas_sds_index_tuples(uinfo)
    assert calculated == expected


def test_shock_sd_index_tuples():
    periods = [0, 1, 2]
    factors = ["fac1", "fac2"]

    expected = [
        ("shock_sds", 0, "fac1", "-"),
        ("shock_sds", 0, "fac2", "-"),
        ("shock_sds", 1, "fac1", "-"),
        ("shock_sds", 1, "fac2", "-"),
    ]

    calculated = get_shock_sds_index_tuples(periods, factors)
    assert calculated == expected


def test_initial_mean_index_tuples():
    nmixtures = 3
    factors = ["fac1", "fac2"]

    expected = [
        ("initial_states", 0, "mixture_0", "fac1"),
        ("initial_states", 0, "mixture_0", "fac2"),
        ("initial_states", 0, "mixture_1", "fac1"),
        ("initial_states", 0, "mixture_1", "fac2"),
        ("initial_states", 0, "mixture_2", "fac1"),
        ("initial_states", 0, "mixture_2", "fac2"),
    ]

    calculated = initial_mean_index_tuples(nmixtures, factors)
    assert calculated == expected


def test_mixture_weight_index_tuples():
    nmixtures = 3
    expected = [
        ("mixture_weights", 0, "mixture_0", "-"),
        ("mixture_weights", 0, "mixture_1", "-"),
        ("mixture_weights", 0, "mixture_2", "-"),
    ]
    calculated = get_mixture_weights_index_tuples(nmixtures)
    assert calculated == expected


def test_initial_cov_index_tuples():
    nmixtures = 2
    factors = ["fac1", "fac2", "fac3"]
    expected = [
        ("initial_cholcovs", 0, "mixture_0", "fac1-fac1"),
        ("initial_cholcovs", 0, "mixture_0", "fac2-fac1"),
        ("initial_cholcovs", 0, "mixture_0", "fac2-fac2"),
        ("initial_cholcovs", 0, "mixture_0", "fac3-fac1"),
        ("initial_cholcovs", 0, "mixture_0", "fac3-fac2"),
        ("initial_cholcovs", 0, "mixture_0", "fac3-fac3"),
        ("initial_cholcovs", 0, "mixture_1", "fac1-fac1"),
        ("initial_cholcovs", 0, "mixture_1", "fac2-fac1"),
        ("initial_cholcovs", 0, "mixture_1", "fac2-fac2"),
        ("initial_cholcovs", 0, "mixture_1", "fac3-fac1"),
        ("initial_cholcovs", 0, "mixture_1", "fac3-fac2"),
        ("initial_cholcovs", 0, "mixture_1", "fac3-fac3"),
    ]

    calculated = get_initial_cholcovs_index_tuples(nmixtures, factors)
    assert calculated == expected


def test_trans_coeffs_index_tuples():
    factors = ["fac1", "fac2", "fac3"]
    periods = [0, 1, 2]
    transition_names = ["linear", "constant", "log_ces"]

    expected = [
        ("transition", 0, "fac1", "fac1"),
        ("transition", 0, "fac1", "fac2"),
        ("transition", 0, "fac1", "fac3"),
        ("transition", 0, "fac1", "constant"),
        ("transition", 1, "fac1", "fac1"),
        ("transition", 1, "fac1", "fac2"),
        ("transition", 1, "fac1", "fac3"),
        ("transition", 1, "fac1", "constant"),
        ("transition", 0, "fac3", "fac1"),
        ("transition", 0, "fac3", "fac2"),
        ("transition", 0, "fac3", "fac3"),
        ("transition", 0, "fac3", "phi"),
        ("transition", 1, "fac3", "fac1"),
        ("transition", 1, "fac3", "fac2"),
        ("transition", 1, "fac3", "fac3"),
        ("transition", 1, "fac3", "phi"),
    ]

    calculated = get_transition_index_tuples(factors, periods, transition_names)

    assert calculated == expected
