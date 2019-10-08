import pandas as pd

from skillmodels.pre_processing.params_index import _delta_index_tuples
from skillmodels.pre_processing.params_index import _h_index_tuples
from skillmodels.pre_processing.params_index import _p_index_tuples
from skillmodels.pre_processing.params_index import _q_index_tuples
from skillmodels.pre_processing.params_index import _r_index_tuples
from skillmodels.pre_processing.params_index import _trans_coeffs_index_tuples
from skillmodels.pre_processing.params_index import _w_index_tuples
from skillmodels.pre_processing.params_index import _x_index_tuples


def test_delta_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))
    controls = [["c1"], ["c1", "c2"]]

    expected = [
        ("delta", 0, "m1", "constant"),
        ("delta", 0, "m1", "c1"),
        ("delta", 0, "m2", "constant"),
        ("delta", 0, "m2", "c1"),
        ("delta", 0, "bla", "constant"),
        ("delta", 0, "bla", "c1"),
        ("delta", 1, "m1", "constant"),
        ("delta", 1, "m1", "c1"),
        ("delta", 1, "m1", "c2"),
        ("delta", 1, "m2", "constant"),
        ("delta", 1, "m2", "c1"),
        ("delta", 1, "m2", "c2"),
    ]

    calculated = _delta_index_tuples(controls, uinfo)
    assert calculated == expected


def test_h_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))
    factors = ["fac1", "fac2"]
    expected = [
        ("h", 0, "m1", "fac1"),
        ("h", 0, "m1", "fac2"),
        ("h", 0, "m2", "fac1"),
        ("h", 0, "m2", "fac2"),
        ("h", 0, "bla", "fac1"),
        ("h", 0, "bla", "fac2"),
        ("h", 1, "m1", "fac1"),
        ("h", 1, "m1", "fac2"),
        ("h", 1, "m2", "fac1"),
        ("h", 1, "m2", "fac2"),
    ]

    calculated = _h_index_tuples(factors, uinfo)
    assert calculated == expected


def test_r_index_tuples():
    uinfo_tups = [(0, "m1"), (0, "m2"), (0, "bla"), (1, "m1"), (1, "m2")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_tups))

    expected = [
        ("r", 0, "m1", "-"),
        ("r", 0, "m2", "-"),
        ("r", 0, "bla", "-"),
        ("r", 1, "m1", "-"),
        ("r", 1, "m2", "-"),
    ]

    calculated = _r_index_tuples(uinfo)
    assert calculated == expected


def test_q_index_tuples():
    periods = [0, 1, 2]
    factors = ["fac1", "fac2"]

    expected = [
        ("q", 0, "fac1", "-"),
        ("q", 0, "fac2", "-"),
        ("q", 1, "fac1", "-"),
        ("q", 1, "fac2", "-"),
    ]

    calculated = _q_index_tuples(periods, factors)
    assert calculated == expected


def test_x_index_tuples():
    nmixtures = 3
    factors = ["fac1", "fac2"]

    expected = [
        ("x", 0, "mixture_0", "fac1"),
        ("x", 0, "mixture_0", "fac2"),
        ("x", 0, "mixture_1", "fac1"),
        ("x", 0, "mixture_1", "fac2"),
        ("x", 0, "mixture_2", "fac1"),
        ("x", 0, "mixture_2", "fac2"),
    ]

    calculated = _x_index_tuples(nmixtures, factors)
    assert calculated == expected


def test_w_index_tuples():
    nmixtures = 3
    expected = [
        ("w", 0, "mixture_0", "-"),
        ("w", 0, "mixture_1", "-"),
        ("w", 0, "mixture_2", "-"),
    ]
    calculated = _w_index_tuples(nmixtures)
    assert calculated == expected


def test_p_index_tuples():
    nmixtures = 2
    factors = ["fac1", "fac2", "fac3"]
    expected = [
        ("p", 0, "mixture_0", "fac1-fac1"),
        ("p", 0, "mixture_0", "fac2-fac1"),
        ("p", 0, "mixture_0", "fac2-fac2"),
        ("p", 0, "mixture_0", "fac3-fac1"),
        ("p", 0, "mixture_0", "fac3-fac2"),
        ("p", 0, "mixture_0", "fac3-fac3"),
        ("p", 0, "mixture_1", "fac1-fac1"),
        ("p", 0, "mixture_1", "fac2-fac1"),
        ("p", 0, "mixture_1", "fac2-fac2"),
        ("p", 0, "mixture_1", "fac3-fac1"),
        ("p", 0, "mixture_1", "fac3-fac2"),
        ("p", 0, "mixture_1", "fac3-fac3"),
    ]

    calculated = _p_index_tuples(nmixtures, factors)
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
