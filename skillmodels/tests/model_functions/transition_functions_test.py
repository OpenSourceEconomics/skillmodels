from nose.tools import assert_equal
import skillmodels.model_functions.transition_functions as tf
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae


class TestLinear:
    def setup(self):
        self.nemf = 2
        self.nind = 10
        self.nfac = 3
        self.nsigma = 7
        self.incl_pos = [0, 1, 2]
        self.incl_fac = ['f1', 'f2', 'f3']

        self.coeffs = np.array([0.5, 1.0, 1.5])

        self.sp = np.ones((self.nemf, self.nind, self.nsigma, self.nfac))
        self.sp[1, :, :, :] *= 2
        self.sp[:, :, 0, :] = 3
        self.sp = self.sp.reshape(
            self.nemf * self.nind * self.nsigma, self.nfac)

    def test_linear_transition_equation(self):

        expected_result = np.ones((self.nemf, self.nind, self.nsigma)) * 3
        expected_result[1, :, :] *= 2
        expected_result[:, :, 0] = 9
        expected_result = expected_result.flatten()
        aaae(tf.linear(self.sp, self.coeffs, self.incl_pos),
             expected_result)

    def test_nr_coeffs_linear(self):
        assert_equal(tf.nr_coeffs_linear(self.incl_fac, 'short'), 3)

    def test_coeff_names_linear(self):
        expected = ['lincoeff__3__f1__f1', 'lincoeff__3__f1__f2',
                    'lincoeff__3__f1__f3']

        assert_equal(tf.coeff_names_linear(self.incl_fac, 'short', 'f1', 3),
                     expected)


class TestAR1:
    def setup(self):
        self.nemf = 2
        self.nind = 10
        self.nfac = 3
        self.nsigma = 7

        self.incl_pos = [1]
        self.incl_fac = ['f2']

        self.coeffs = np.array([3])

        self.sp = np.ones((self.nemf * self.nind * self.nsigma, self.nfac))

    def test_ar1_transition_equation(self):
        expected_result = np.ones((self.nemf * self.nind * self.nsigma)) * 3
        aaae(tf.ar1(self.sp, self.coeffs, self.incl_pos), expected_result)

    def test_ar1_coeff_names(self):
        assert_equal(tf.coeff_names_ar1(self.incl_fac, 'short', 'f2', 3),
                     ['ar1_coeff__3__f2__f2'])


class TestLogCes:

    def setup(self):
        self.nemf = 2
        self.nind = 10
        self.nfac = 2
        self.nsigma = 5
        self.incl_pos = [0, 1]
        self.incl_fac = ['f1', 'f2']

        self.coeffs = np.array([0.4, 0.6, 2])
        self.sp = np.zeros((self.nemf, self.nind, self.nsigma, self.nfac))
        self.sp[:] = np.array([[3, 7.5]] * self.nsigma)
        self.sp = self.sp.reshape(
            self.nemf * self.nind * self.nsigma, self.nfac)

    def test_log_ces(self):
        expected_result = \
            np.ones(self.nemf * self.nind * self.nsigma) * 7.244628323025
        aaae(tf.log_ces(self.sp, self.coeffs, self.incl_pos), expected_result)

    def test_log_ces_nr_coeffs_short(self):
        assert_equal(tf.nr_coeffs_log_ces(self.incl_fac, 'short'), 2)

    def test_log_ces_nr_coeffs_long(self):
        assert_equal(tf.nr_coeffs_log_ces(self.incl_fac, 'long'), 3)

    def test_transform_coeffs_log_ces_short_to_long(self):
        big_out = np.zeros((2, 3))
        small_out = big_out[0, :]

        coeffs = np.array([1, 3])

        tf.transform_coeffs_log_ces(
            coeffs, self.incl_fac, 'short_to_long', small_out)

        expected = np.zeros((2, 3))
        expected[0, :] = np.array([0.5, 0.5, 3])
        aaae(big_out, expected)

    def test_transform_coeffs_log_ces_long_to_short(self):
        big_out = np.zeros((2, 2))
        small_out = big_out[0, :]
        coeffs = np.array([0.5, 0.5, 3])

        tf.transform_coeffs_log_ces(
            coeffs, self.incl_fac, 'long_to_short', small_out)

        expected = np.zeros((2, 2))
        expected[0, :] = np.array([1, 3])
        aaae(big_out, expected)

    def test_bounds_log_ces(self):
        expected_lb = [0, None]
        expected_ub = [None, None]

        lb, ub = tf.bounds_log_ces(self.incl_fac)
        assert_equal(list(lb), expected_lb)
        assert_equal(list(ub), expected_ub)

    def test_coeff_names_log_ces_short(self):
        expected = ['gamma__0__f1__f1', 'phi__0__f1__Phi']

        names = tf.coeff_names_log_ces(self.incl_fac, 'short', 'f1', 0)
        assert_equal(names, expected)

    def test_coeff_names_log_ces_long(self):
        expected = ['gamma__0__f1__f1', 'gamma__0__f1__f2', 'phi__0__f1__Phi']

        names = tf.coeff_names_log_ces(self.incl_fac, 'long', 'f1', 0)
        assert_equal(names, expected)


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
