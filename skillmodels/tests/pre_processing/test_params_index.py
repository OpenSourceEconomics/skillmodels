import pandas as pd

from skillmodels.pre_processing.params_index import _control_coeffs_index_tuples
from skillmodels.pre_processing.params_index import _initial_cov_index_tuples
from skillmodels.pre_processing.params_index import _initial_mean_index_tuples
from skillmodels.pre_processing.params_index import _loading_index_tuples
from skillmodels.pre_processing.params_index import _meas_sd_index_tuples
from skillmodels.pre_processing.params_index import _mixture_weight_index_tuples
from skillmodels.pre_processing.params_index import _shock_sd_index_tuples
from skillmodels.pre_processing.params_index import _trans_coeffs_index_tuples


def test_control_coeffs_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))
    controls = [["c1"], ["c1", "c2"]]

    expected = [
        ("control_coeffs", 0, "m1", "constant"),
        ("control_coeffs", 0, "m1", "c1"),
        ("control_coeffs", 0, "m2", "constant"),
        ("control_coeffs", 0, "m2", "c1"),
        ("control_coeffs", 0, "bla", "constant"),
        ("control_coeffs", 0, "bla", "c1"),
        ("control_coeffs", 1, "m1", "constant"),
        ("control_coeffs", 1, "m1", "c1"),
        ("control_coeffs", 1, "m1", "c2"),
        ("control_coeffs", 1, "m2", "constant"),
        ("control_coeffs", 1, "m2", "c1"),
        ("control_coeffs", 1, "m2", "c2"),
    ]

    calculated = _control_coeffs_index_tuples(controls, uinfo)
    assert calculated == expected


def test_loading_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))
    factors = ["fac1", "fac2"]
    expected = [
        ("loading", 0, "m1", "fac1"),
        ("loading", 0, "m1", "fac2"),
        ("loading", 0, "m2", "fac1"),
        ("loading", 0, "m2", "fac2"),
        ("loading", 0, "bla", "fac1"),
        ("loading", 0, "bla", "fac2"),
        ("loading", 1, "m1", "fac1"),
        ("loading", 1, "m1", "fac2"),
        ("loading", 1, "m2", "fac1"),
        ("loading", 1, "m2", "fac2"),
    ]

    calculated = _loading_index_tuples(factors, uinfo)
    assert calculated == expected


def test_meas_sd_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))

    expected = [
        ("meas_sd", 0, "m1", "-"),
        ("meas_sd", 0, "m2", "-"),
        ("meas_sd", 0, "bla", "-"),
        ("meas_sd", 1, "m1", "-"),
        ("meas_sd", 1, "m2", "-"),
    ]

    calculated = _meas_sd_index_tuples(uinfo)
    assert calculated == expected


def test_shock_sd_index_tuples():
    periods = [0, 1, 2]
    factors = ["fac1", "fac2"]

    expected = [
        ("shock_sd", 0, "fac1", "-"),
        ("shock_sd", 0, "fac2", "-"),
        ("shock_sd", 1, "fac1", "-"),
        ("shock_sd", 1, "fac2", "-"),
    ]

    calculated = _shock_sd_index_tuples(periods, factors)
    assert calculated == expected


def test_initial_mean_index_tuples():
    nmixtures = 3
    factors = ["fac1", "fac2"]

    expected = [
        ("initial_mean", 0, "mixture_0", "fac1"),
        ("initial_mean", 0, "mixture_0", "fac2"),
        ("initial_mean", 0, "mixture_1", "fac1"),
        ("initial_mean", 0, "mixture_1", "fac2"),
        ("initial_mean", 0, "mixture_2", "fac1"),
        ("initial_mean", 0, "mixture_2", "fac2"),
    ]

    calculated = _initial_mean_index_tuples(nmixtures, factors)
    assert calculated == expected


def test_mixture_weight_index_tuples():
    nmixtures = 3
    expected = [
        ("mixture_weight", 0, "mixture_0", "-"),
        ("mixture_weight", 0, "mixture_1", "-"),
        ("mixture_weight", 0, "mixture_2", "-"),
    ]
    calculated = _mixture_weight_index_tuples(nmixtures)
    assert calculated == expected


def test_initial_cov_index_tuples():
    nmixtures = 2
    factors = ["fac1", "fac2", "fac3"]
    expected = [
        ("initial_cov", 0, "mixture_0", "fac1-fac1"),
        ("initial_cov", 0, "mixture_0", "fac2-fac1"),
        ("initial_cov", 0, "mixture_0", "fac2-fac2"),
        ("initial_cov", 0, "mixture_0", "fac3-fac1"),
        ("initial_cov", 0, "mixture_0", "fac3-fac2"),
        ("initial_cov", 0, "mixture_0", "fac3-fac3"),
        ("initial_cov", 0, "mixture_1", "fac1-fac1"),
        ("initial_cov", 0, "mixture_1", "fac2-fac1"),
        ("initial_cov", 0, "mixture_1", "fac2-fac2"),
        ("initial_cov", 0, "mixture_1", "fac3-fac1"),
        ("initial_cov", 0, "mixture_1", "fac3-fac2"),
        ("initial_cov", 0, "mixture_1", "fac3-fac3"),
    ]

    calculated = _initial_cov_index_tuples(nmixtures, factors)
    assert calculated == expected


def test_trans_coeffs_index_tuples():
    factors = ["fac1", "fac2", "fac3"]
    periods = [0, 1, 2]
    transition_names = ["linear", "constant", "log_ces"]
    included_factors = [["fac1", "fac2"], ["fac2"], ["fac2", "fac3"]]

    expected = [
        ("trans", 0, "fac1", "fac1"),
        ("trans", 0, "fac1", "fac2"),
        ("trans", 0, "fac1", "constant"),
        ("trans", 0, "fac3", "fac2"),
        ("trans", 0, "fac3", "fac3"),
        ("trans", 0, "fac3", "phi"),
        ("trans", 1, "fac1", "fac1"),
        ("trans", 1, "fac1", "fac2"),
        ("trans", 1, "fac1", "constant"),
        ("trans", 1, "fac3", "fac2"),
        ("trans", 1, "fac3", "fac3"),
        ("trans", 1, "fac3", "phi"),
    ]

    calculated = _trans_coeffs_index_tuples(
        factors, periods, transition_names, included_factors
    )

    assert calculated == expected
