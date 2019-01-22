import skillmodels.model_functions.transition_functions as tf
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
import pytest


# test for linear version function
@pytest.fixture
def setup_linear():
    nemf, nind, nsigma, nfac = 2, 10, 7, 3
    sigma_points = np.ones((nemf, nind, nsigma, nfac))
    sigma_points[1] *= 2
    sigma_points[:, :, 0, :] = 3
    sigma_points = sigma_points.reshape(nemf * nind * nsigma, nfac)

    args = {
        "sigma_points": sigma_points,
        "coeffs": np.array([0.5, 1.0, 1.5]),
        "included_positions": np.array([0, 1, 2]),
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


# test for number of linear coefficients
parameters = [(["f1", "f2", "f3"], "short", 3)]


@pytest.mark.parametrize("included_factors, params_type, result", parameters)
def test_nr_coeffs_linear(included_factors, params_type, result):
    assert tf.nr_coeffs_linear(included_factors, params_type) == result


# test for coefficient names linear
parameters = [(["f1", "f2", "f3"], "short", "f1", 3,
               ["lincoeff__3__f1__f1", "lincoeff__3__f1__f2", "lincoeff__3__f1__f3"])]


@pytest.mark.parametrize("included_factors, params_type, factor, stage, result", parameters)
def test_coeff_names_linear(included_factors, params_type, factor, stage, result):
    assert tf.coeff_names_linear(included_factors, params_type, factor, stage) == result
    
# **************************************************************************************
# test for linear with constant function
@pytest.fixture
def expected_linear_with_constant():
    nemf, nind, nsigma = 2, 10, 7
    coeffs = [0.5, 1.0, 1.5, 0.5]
    expected_result = np.ones((nemf, nind, nsigma)) * 3
    expected_result[1, :, :] *= 2
    expected_result[:, :, 0] = 9
    expected_result = expected_result.flatten()

    return coeffs[-1] + expected_result


def test_linear_with_constant(setup_linear, expected_linear):
    setup_linear["coeffs"] = [0.5, 1.0, 1.5, 0.5]
    expected_linear_with_constant = expected_linear + 0.5
    aaae(tf.linear_with_constant(**setup_linear), expected_linear_with_constant)


# test for number of linear coefficients
parameters = [(["f1", "f2", "f3"], "short", 4)]


@pytest.mark.parametrize("included_factors, params_type, result", parameters)
def test_nr_coeffs_linear_constant(included_factors, params_type, result):
    assert tf.nr_coeffs_linear_with_constant(included_factors, params_type) == result


# test names of coefficients in linear with constant
parameters = [(["f1", "f2", "f3"], "short", "f1", 3,
               [
               "lincoeff__3__f1__f1", 
               "lincoeff__3__f1__f2", 
               "lincoeff__3__f1__f3", 
               "lincoeff__3__f1__constant",
               ])]


@pytest.mark.parametrize("included_factors, params_type, factor, stage, result", parameters)
def test_coeff_names_linear_constant(included_factors, params_type, factor, stage, result):
    assert tf.coeff_names_linear_with_constant(included_factors, params_type, factor, stage) == result


# **************************************************************************************
# tests ar1

# test for ar1 transition equation function
@pytest.fixture
def setup_ar1_transition_equation():
    # nemf_nind_nsigma is nemf*nind*nsigma
    nemf_nind_nsigma = 140

    args = {
        "sigma_points": np.ones((nemf_nind_nsigma, 3)),
        "coeffs": np.array([3]),
        "included_positions": [1],
    }
    return args


@pytest.fixture
def expected_ar1_transition_equation():
    nemf_nind_nsigma = 140
    expected_result = np.ones(nemf_nind_nsigma) * 3

    return expected_result


def test_ar1_transition_equation(
    setup_ar1_transition_equation, expected_ar1_transition_equation
):
    aaae(tf.ar1(**setup_ar1_transition_equation), expected_ar1_transition_equation)


# test for ar1 coeff names function
parameters = [(["f2"], "short", "f2", 3, ["ar1_coeff__3__f2__f2"])]


@pytest.mark.parametrize("included_factors, params_type, factor, stage, result", parameters)
def test_ar1_coeff_names(included_factors, params_type, factor, stage, result):
    assert tf.coeff_names_ar1(included_factors, params_type, factor, stage) == result


# **************************************************************************************
# tests LogCes
    
# test LogCes function
@pytest.fixture
def setup_log_ces():
    nsigma = 5
    sigma_points = np.array([[3, 7.5]] * nsigma)

    args = {
        "sigma_points": sigma_points,
        "coeffs": np.array([0.4, 0.6, 2]),
        "included_positions": [0, 1],
    }
    return args


@pytest.fixture
def expected_log_ces():
    nsigma = 5
    expected_result = np.ones(nsigma) * 7.244628323025
    return expected_result


def test_log_ces(setup_log_ces, expected_log_ces):
    aaae(tf.log_ces(**setup_log_ces), expected_log_ces)


# test for logces number of coeffs short& long
parameters = [(["f1", "f2"], "short", 2), (["f1", "f2"], "long", 3)]


@pytest.mark.parametrize("included_factors, params_type, result", parameters)
def test_log_ces_nr_coeffs_long_short(included_factors, params_type, result):
    assert tf.nr_coeffs_log_ces(included_factors, params_type) == result


# test for transform of logces from short to long function
def test_transform_coeffs_log_ces_short_to_long():
    full_out = np.zeros((2, 3))
    small_out = full_out[0, :]

    coeffs = np.array([1, 3])

    tf.transform_coeffs_log_ces(
        coeffs, ['f1', 'f2'], 'short_to_long', small_out)

    expected = np.zeros((2, 3))
    expected[0, :] = np.array([0.5, 0.5, 3])
    aaae(full_out, expected)


def test_transform_coeffs_log_ces_long_to_short():
    full_out = np.zeros((2, 2))
    small_out = full_out[0, :]
    coeffs = np.array([0.5, 0.5, 3])

    tf.transform_coeffs_log_ces(
        coeffs, ['f1', 'f2'], 'long_to_short', small_out)

    expected = np.zeros((2, 2))
    expected[0, :] = np.array([1, 3])
    aaae(full_out, expected)


# test bound logces
def test_bounds_log_ces():
    expected_lb = [0, None]
    expected_ub = [None, None]

    inlcuded_factors = ["f1", "f2"]
    lb, ub = tf.bounds_log_ces(inlcuded_factors)
    assert (list(lb)) == expected_lb
    assert (list(ub)) == expected_ub


# test coeffs names log ces long
parameters = [(["f1", "f2"], "short", "f1", 0, ["gamma__0__f1__f1", "phi__0__f1__Phi"]),
              (["f1", "f2"], "long", "f1", 0, ["gamma__0__f1__f1", "gamma__0__f1__f2", "phi__0__f1__Phi"])]


@pytest.mark.parametrize("included_factors, params_type, factor, stage, result", parameters)
def test_log_ces_nr_coeffs_names_long_short(included_factors, params_type, factor, stage, result):
    assert tf.coeff_names_log_ces(included_factors, params_type, factor, stage) == result
# **************************************************************************************
# tests for Translog

# test for translog function
@pytest.fixture
def setup_translog():
    sigma_points = np.array(
        [
            [2, 0, 5, 0],
            [0, 3, 5, 0],
            [0, 0, 7, 4],
            [0, 0, 1, 0],
            [1, 1, 10, 1],
            [0, -3, -100, 0],
            [-1, -1, -1, -1],
            [1.5, -2, 30, 1.8],
            [12, -34, 50, 48],
        ]
    )

    coeffs = np.array([0.2, 0.1, 0.12, 0.08, 0.05, 0.04, 0.03, 0.06, 0.05, 0.04])

    included_positions = [0, 1, 3]

    args = {
        "sigma_points": sigma_points,
        "coeffs": coeffs,
        "included_positions": included_positions,
    }
    return args


@pytest.fixture
def expected_translog():
    expected_result = np.array([0.76, 0.61, 1.32, 0.04, 0.77, 0.01, -0.07, 0.56, 70.92])

    return expected_result


def test_translog(setup_translog, expected_translog):
    aaae(tf.translog(**setup_translog), expected_translog)


# test for number of coefficients short & long
parameters = [(["f1", "f2", "f4"], "short", 10)]


@pytest.mark.parametrize("included_factors, params_type, result", parameters)
def test_translog_nr_coeffs_short(included_factors, params_type, result):
    assert tf.nr_coeffs_translog(included_factors, params_type) == result
    

# test for translog coefficient names
parameters = [(["f1", "f2", "f4"], "short", "f2", 1,
               [
                "translog__1__f2__f1",
                "translog__1__f2__f2",
                "translog__1__f2__f4",
                "translog__1__f2__f1-squared",
                "translog__1__f2__f1-f2",
                "translog__1__f2__f1-f4",
                "translog__1__f2__f2-squared",
                "translog__1__f2__f2-f4",
                "translog__1__f2__f4-squared",
                "translog__1__f2__TFP",        
                ])]


@pytest.mark.parametrize("included_factors, params_type, factor, stage, result", parameters)
def test_coeff_names_translog(included_factors, params_type, factor, stage, result):
    assert tf.coeff_names_translog(included_factors, params_type, factor, stage) == result
# **************************************************************************************
