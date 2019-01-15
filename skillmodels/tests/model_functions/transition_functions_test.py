from nose.tools import assert_equal
import skillmodels.model_functions.transition_functions as tf
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
import pytest

#test of linear version function
@pytest.fixture
def setup_linear():
    nemf, nind, nsigma, nfac = 2, 10, 7, 3
    sigma_points = np.ones((nemf, nind, nsigma, nfac))
    sigma_points[1] *= 2
    sigma_points[:, :, 0, :] = 3
    sigma_points = sigma_points.reshape(nemf * nind * nsigma, nfac)

    args = {
        'sigma_points': sigma_points,
        'coeffs': np.array([0.5, 1.0, 1.5]),
        'included_positions': np.array([0, 1, 2])
    }
    return args


@pytest.fixture
def expected_linear():
    nemf, nind, nsigma = 2, 10, 7
    expected_result = np.ones((nemf, nind, nsigma)) * 3
    expected_result[1, :, :] *= 2
    expected_result[:, :, 0] = 9
    expected_result = expected_result.flatten()
    return expected_result

def test_linear(setup_linear, expected_linear):
    aaae(tf.linear(**setup_linear), expected_linear)

#test for number of linear coefficients
@pytest.fixture
def setup_nr_coeffs_linear():
    
    args = {
        'included_factors': ['f1', 'f2', 'f3'], 
        'params_type': 'short'
    }
        
    return args

@pytest.fixture
def expected_nr_coeffs_linear():
    included_factors = ['f1', 'f2', 'f3']
    expected_result = len(included_factors)
    return expected_result

def test_nr_coeffs_linear(setup_nr_coeffs_linear, expected_nr_coeffs_linear):
    aaae(tf.nr_coeffs_linear(**setup_nr_coeffs_linear), expected_nr_coeffs_linear)

#test for coefficient names linear
@pytest.fixture
def setup_coeffs_names_linear():
   
    args = {
        'included_factors': ['f1', 'f2', 'f3'], 
        'params_type': 'short',
        'factor': 'f1',
        'stage': 3
    }
    return args

def test_coeff_names_linear(setup_coeffs_names_linear):
    expected = ['lincoeff__3__f1__f1', 'lincoeff__3__f1__f2',
                    'lincoeff__3__f1__f3']  
    assert tf.coeff_names_linear(**setup_coeffs_names_linear) == expected

# **************************************************************************************
#test of linear with constant function
@pytest.fixture
def setup_linear_with_constant():
    nemf, nind, nsigma, nfac = 2, 10, 7, 3
    sigma_points = np.ones((nemf, nind, nsigma, nfac))
    sigma_points[1] *= 2
    sigma_points[:, :, 0, :] = 3
    sigma_points = sigma_points.reshape(nemf * nind * nsigma, nfac)
    
    args = {
        'sigma_points': sigma_points,
        'coeffs': np.array([0.5, 1.0, 1.5, 0.5]),
        'included_positions': np.array([0, 1, 2])
    }
    return args

@pytest.fixture
def expected_linear_with_constant():
    nemf, nind, nsigma = 2, 10, 7
    coeffs = [0.5, 1.0, 1.5, 0.5]
    expected_result = np.ones((nemf, nind, nsigma))*3
    expected_result[1, :, :] *= 2
    expected_result[:, :, 0] = 9
    expected_result = expected_result.flatten()
    
    return coeffs[-1] + expected_result

def test_linear_with_constant(setup_linear_with_constant, expected_linear_with_constant):
    aaae(tf.linear_with_constant(**setup_linear_with_constant), expected_linear_with_constant)

# **************************************************************************************
#tests ar1
    
#test for ar1 transition equation function
@pytest.fixture
def setup_ar1_transition_equation():
    nemf, nind, nsigma, nfac = 2, 10, 7, 3

    args = {
            
        'sigma_points': np.ones((nemf * nind * nsigma, nfac)),
        'coeffs': np.array([3]),
        'included_positions': [1]
        
    }
    
    return args

@pytest.fixture
def expected_ar1_transition_equation():
    nemf, nind, nsigma = 2, 10, 7
    expected_result = np.ones(nemf * nind * nsigma) * 3
        
    return expected_result

def test_ar1_transition_equation(setup_ar1_transition_equation,
                                 expected_ar1_transition_equation):
        aaae(tf.ar1(**setup_ar1_transition_equation), 
                                 expected_ar1_transition_equation)

#test for ar1 coeff names function
@pytest.fixture
def setup_ar1_coeff_names():
    
    args = {
            
        'included_factors': ['f2'],
        'params_type': 'short',
        'factor': 'f2',
        'stage': 3
        
    }
    
    return args

@pytest.fixture
def expected_ar1_coeff_names():
    expected_result = ['ar1_coeff__3__f2__f2']
    return expected_result

def test_ar1_coeff_names(setup_ar1_coeff_names, expected_ar1_coeff_names):
        assert_equal(tf.coeff_names_ar1(**setup_ar1_coeff_names),
                     expected_ar1_coeff_names)
        
# **************************************************************************************
#tests LogCes  

#test LogCes function
@pytest.fixture
def setup_log_ces():
    nsigma = 5
    sigma_points = np.array([[3, 7.5]] * nsigma)

    args = {
        'sigma_points': sigma_points,
        'coeffs': np.array([0.4, 0.6, 2]),
        'included_positions': [0, 1]
    }
    return args


@pytest.fixture
def expected_log_ces():
    nsigma = 5
    expected_result = np.ones(nsigma) * 7.244628323025
    return expected_result        

def test_log_ces(setup_log_ces, expected_log_ces):
        aaae(tf.log_ces(**setup_log_ces), expected_log_ces) 
        
#test for logces number of coeffs short
@pytest.fixture
def setup_log_ces_nr_coeffs_short():
    
    args = {
        'included_factors': ['f1', 'f2'], 
        'params_type': 'short'
    }
    return args

def test_log_ces_nr_coeffs_short(setup_log_ces_nr_coeffs_short):
        aaae(tf.nr_coeffs_log_ces(**setup_log_ces_nr_coeffs_short), 2)
        
#test for logces number of coeffs long
@pytest.fixture
def setup_log_ces_nr_coeffs_long():
    
    args = {
        'included_factors': ['f1', 'f2'], 
        'params_type': 'long'
    }
    return args

def test_log_ces_nr_coeffs_long(setup_log_ces_nr_coeffs_long):
        aaae(tf.nr_coeffs_log_ces(**setup_log_ces_nr_coeffs_long), 3)
   
     


# **************************************************************************************

class TestTranslog:
    def setup(self):
        self.nemf = 1
        self.nind = 10
        self.nfac = 4
        self.nsigma = 9
        self.incl_pos = [0, 1, 3]
        self.incl_fac = ['f1', 'f2', 'f4']

        self.coeffs = np.array(
            [0.2, 0.1, 0.12, 0.08, 0.05, 0.04, 0.03, 0.06, 0.05, 0.04])
        input_values = np.array(
            [[2, 0, 5, 0], [0, 3, 5, 0], [0, 0, 7, 4], [0, 0, 1, 0],
             [1, 1, 10, 1], [0, -3, -100, 0], [-1, -1, -1, -1],
             [1.5, -2, 30, 1.8], [12, -34, 50, 48]])
        self.sp = np.zeros((self.nemf, self.nind, self.nsigma, self.nfac))
        self.sp[:] = input_values
        self.sp = self.sp.reshape(
            self.nemf * self.nind * self.nsigma, self.nfac)

    def test_translog(self):
        expected_result = np.zeros((self.nemf, self.nind, self.nsigma))
        expected_result[:] = np.array(
            [0.76, 0.61, 1.32, 0.04, 0.77, 0.01, -0.07, 0.56, 70.92])
        expected_result = expected_result.reshape(
            self.nemf * self.nind * self.nsigma)
        calculated_result = tf.translog(self.sp, self.coeffs, self.incl_pos)
        aaae(calculated_result, expected_result)

    def test_translog_nr_coeffs_short(self):
        assert_equal(tf.nr_coeffs_translog(self.incl_fac, 'short'), 10)

    def test_coeff_names_translog(self):
        expected_names = \
            ['translog__1__f2__f1',
             'translog__1__f2__f2',
             'translog__1__f2__f4',
             'translog__1__f2__f1-squared',
             'translog__1__f2__f1-f2',
             'translog__1__f2__f1-f4',
             'translog__1__f2__f2-squared',
             'translog__1__f2__f2-f4',
             'translog__1__f2__f4-squared',
             'translog__1__f2__TFP']
        names = tf.coeff_names_translog(self.incl_fac, 'short', 'f2', 1)
        assert_equal(names, expected_names)
