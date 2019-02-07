"""
Tests for functions in simulate_data module
"""
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
import pytest
import sys 

sys.path.append("../../simulation/")
sys.path.append("../../model_functions/")

import simulate_data as sd

# test measuerments_from_factors
@pytest.fixture
def set_up_meas():
    out={}
    out['factors'] = np.array([[0,0,0],[1,1,1]])
    out['controls'] = np.array([[1,1],[1,1]])
    out['loadings'] = np.array([[0.3,0.3,0.3],[0.3,0.3,0.3],[0.3,0.3,0.3]])
    out['deltas'] = np.array([[0.5,0.5],[0.5,0.5],[0.5,0.5]])
    out['variances'] = np.zeros(3)
    out['nmeas'] = 3
    return out

@pytest.fixture
def expected_meas():
    out = np.array([[1,1,1],[1.9,1.9,1.9]])
    return out

def test_measurements_from_factors(set_up_meas,expected_meas):
    aaae(sd.measurements_from_factors(**set_up_meas),expected_meas)


