"""
Tests for functions in simulate_data module
"""
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
import pytest
import sys

sys.path.append("../../simulation/")
sys.path.append("../../model_functions/")

import simulate_data as sdata

# test measuerments_from_factors
@pytest.fixture
def set_up_meas():
    out = {}
    out["factors"] = np.array([[0, 0, 0], [1, 1, 1]])
    out["controls"] = np.array([[1, 1], [1, 1]])
    out["loadings"] = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    out["deltas"] = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    out["variances"] = np.zeros(3)
    out["nmeas"] = 3
    return out


@pytest.fixture
def expected_meas():
    out = np.array([[1, 1, 1], [1.9, 1.9, 1.9]])
    return out


def test_measurements_from_factors(set_up_meas, expected_meas):
    aaae(sdata.measurements_from_factors(**set_up_meas), expected_meas)


# Test next_period_factors


@pytest.fixture
def set_up_npfac():
    out = {}
    out["factors"] = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1, 1]])
    out["transition_names"] = [
        "translog",
        "linear",
        "log_ces",
        "ar1",
        "constant",
        "linear_with_constant",
    ]
    out["transition_argument_dicts"] = [
        {"coeffs": np.array([0.02, 0.03, 0.02]), "included_positions": [0, 1, 2, 3, 5]},
        {
            "coeffs": np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.2]),
            "included_positions": [0, 1, 2, 3, 4, 5],
        },
        {
            "coeffs": np.array([0.1, 0.1, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.3]),
            "included_positions": [0, 1, 2, 3, 4, 5],
        },
        {"coeffs": np.array([2]), "included_positions": [1]},
        {"coeffs": np.array([1]), "included_positions": [1]},
        {
            "coeffs": np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.05]),
            "included_positions": [0, 1, 2, 3, 4, 5],
        },
    ]
    out["shock_variances"] = np.array([0,0,0,0,0,0])
    return out

@pytest.fixture
def expected_npfac():
    #The values have been computed by using the functions from the module transition_functions.py
    # since the main calculations are made through those functions, what the test actualy tests is whether the loops of getatr in sumlated_next_period_factors work correctly
    d={}
    d['tl']=np.array([[0.055,0.09]])
    d['lin']=np.array([[0.7,1.4]])
    d['lces'] = np.array([[2.8104906, 3.3104906]])
    d['ar1']=np.array([[1, 2]])
    d['constant']=np.array([[0.5, 1 ]])
    d['linwic']=np.array([[0.75, 1.45]])
    npfac=np.concatenate(list(d.values())).T
    return npfac

def test_next_period_factors(set_up_npfac, expected_npfac):
    aaae(sdata.next_period_factors(**set_up_npfac), expected_npfac)


