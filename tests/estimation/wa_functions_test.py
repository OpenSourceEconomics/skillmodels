from skillmodels.estimation import wa_functions as wf
# from numpy.testing import assert_array_equal as aae
from numpy.testing import assert_array_almost_equal as aaae
import numpy as np
import pandas as pd
from nose.tools import assert_almost_equal, assert_equal
from pandas.util.testing import assert_series_equal, assert_frame_equal
from statsmodels.sandbox.regression.gmm import LinearIVGMM
from unittest.mock import patch


class TestInitialLoadingsAndIntercepts:
    def setup(self):
        np.random.seed(84320)
        self.true_loadings = np.array([1., 2., 3, 4])
        factor = np.random.normal(scale=1e-2, size=(10000, 1))
        self.true_intercepts = np.array([4., 0, 2, 1])
        measurement_data = factor * self.true_loadings + self.true_intercepts
        self.data = pd.DataFrame(data=measurement_data,
                                 columns=['m1', 'm2', 'm3', 'm4'])
        self.true_loadings_series = pd.Series(
            data=self.true_loadings, name='loadings', index=self.data.columns)
        self.true_intercepts_series = pd.Series(
            data=self.true_intercepts, name='intercepts',
            index=self.data.columns)

    def test_loadings_from_covs_normalized_to_true_value(self):
        true_normalization = ['m1', 1]
        calculated_loadings = wf.loadings_from_covs(self.data,
                                                    true_normalization)
        expected_loadings = self.true_loadings_series
        assert_series_equal(calculated_loadings, expected_loadings)

    def test_loadings_from_covs_normalized_to_different_value(self):
        multiple = 1.5
        different_normalization = ['m2', multiple * 2]
        calculated_loadings = wf.loadings_from_covs(self.data,
                                                    different_normalization)
        expected_loadings = (multiple * self.true_loadings_series)
        assert_series_equal(calculated_loadings, expected_loadings)

    def test_intercepts_from_means_with_true_normalization(self):
        expected_intercepts = self.true_intercepts_series
        true_normalization = ['m3', 2.0]
        calc_intercepts, calc_mean = wf.intercepts_from_means(
            self.data, true_normalization, self.true_loadings_series)
        assert_series_equal(calc_intercepts, expected_intercepts)
        assert_almost_equal(calc_mean, 0.0, places=2)

    def test_intercepts_from_means_with_different_normalization(self):
        difference = 4
        different_normalization = ['m2', 0 + difference]
        expected_mean = -2.0
        expected_intercepts = pd.Series(
            data=[6., 4, 8, 9], index=self.true_intercepts_series.index,
            name='intercepts')
        calc_intercepts, calc_mean = wf.intercepts_from_means(
            self.data, different_normalization, self.true_loadings_series)
        assert_series_equal(calc_intercepts, expected_intercepts)
        assert_almost_equal(calc_mean, expected_mean, places=2)

    def test_intercepts_from_means_without_normalization(self):
        expected_intercepts = self.true_intercepts_series
        calc_intercepts, calc_mean = wf.intercepts_from_means(
            self.data, [], self.true_loadings_series)
        aaae(calc_intercepts.values, expected_intercepts.values, decimal=3)
        assert_equal(calc_mean, None)


class TestInitialMeasCoeffsIntegrationTest:
    def setup(self):
        self.factors = ['f1', 'f2']
        self.measurements = {
            'f1': [['y1', 'y2', 'y3']],
            'f2': [['y4', 'y5', 'y6']]}
        self.normalizations = {
            'f1': {'intercepts': [['y1', 4]], 'loadings': [['y2', 2]]},
            'f2': {'intercepts': [['y4', 2]], 'loadings': [['y5', 4]]}}

        self.true_loadings = np.array([[1., 2., 0.5], [3, 4, 1]])
        self.true_intercepts = np.array([[4., 0, 1], [2, 1, 0.0]])

        to_concat = []
        for f, factor in enumerate(self.factors):
            factor_data = np.random.normal(size=(10000, 1))
            measurements = factor_data * self.true_loadings[f] + \
                self.true_intercepts[f]
            to_concat.append(pd.DataFrame(
                data=measurements, columns=self.measurements[factor][0]))
        self.y_data = pd.concat(to_concat, axis=1)

        cols = ['loadings', 'intercepts']
        data = np.array([
            [1.0, 4.0],
            [2.0, 0.0],
            [0.5, 1],
            [3.0, 2.0],
            [4.0, 1.0],
            [1.0, 0.0]])

        self.expected_meas_coeffs = pd.DataFrame(
            data=data, columns=cols,
            index=['y1', 'y2', 'y3', 'y4', 'y5', 'y6'])

        self.expected_x_zeros = np.array([0.0, 0.0])
        self.calc_meas_coeffs, self.calc_x_zeros = wf.initial_meas_coeffs(
            self.y_data, self.measurements, self.normalizations)

    def test_initial_meas_coeffs(self):
        assert_frame_equal(self.expected_meas_coeffs, self.calc_meas_coeffs)

    def test_x_zeros(self):
        aaae(self.expected_x_zeros, self.calc_x_zeros, decimal=2)


class TestFactorCovsAndMeasurementErrorVariances:
    def setup(self):
        # nobs = 50000
        self.factors = ['f1', 'f2']
        self.meas_per_factor = {
            'f1': ['y1', 'y2', 'y3'],
            'f2': ['y4', 'y5', 'y6']}

        meas_names = self.meas_per_factor['f1'] + self.meas_per_factor['f2']

        self.loading_series = pd.Series(
            data=np.arange(start=1, step=1, stop=7), index=meas_names)

        meas_cov_data = np.array(
            [[1.1, 2.0, 3.0, 0.4, 0.5, 0.6],
             [2.0, 4.8, 6.0, 0.8, 1.0, 1.2],
             [3.0, 6.0, 11.7, 1.2, 1.5, 1.8],
             [0.4, 0.8, 1.2, 38.4, 40.0, 48.0],
             [0.5, 1.0, 1.5, 40.0, 62.5, 60.0],
             [0.6, 1.2, 1.8, 48.0, 60.0, 93.6]])

        self.meas_cov = pd.DataFrame(data=meas_cov_data, columns=meas_names,
                                     index=meas_names)

        self.true_factor_cov_elements = np.array([1.0, 0.1, 2.0])
        self.true_epsilon_variances = pd.Series(
            data=np.arange(start=0.1, step=0.1, stop=0.7), index=meas_names)

    def test_factor_cov_matrix(self):
        expected_cov_matrix = self.true_factor_cov_elements
        calc_cov_matrix = wf.factor_covs_and_measurement_error_variances(
            self.meas_cov, self.loading_series, self.meas_per_factor)[0]
        aaae(calc_cov_matrix, expected_cov_matrix)

    def test_epsilon_variances(self):
        expected_epsilon_variances = self.true_epsilon_variances
        calc_epsilon_variances = \
            wf.factor_covs_and_measurement_error_variances(
                self.meas_cov, self.loading_series, self.meas_per_factor)[1]
        aaae(calc_epsilon_variances, expected_epsilon_variances)


class TestIVRegArrayDict:
    def setup(self):
        self.depvar_name = 'm1'
        self.indepvar_names = ['m1_resid', 'm3_resid']
        self.instrument_names = [['m2', 'm4']]
        self.transition_name = 'some_func'

        cols = pd.MultiIndex.from_tuples(
            [('y', 'm1'), ('x', 'm1_resid'), ('x', 'm3_resid'),
             ('z', 'm2'), ('z', 'm4')])
        dat = np.arange(15, dtype=float).reshape(3, 5)
        dat[2, 0] = np.nan
        self.data = pd.DataFrame(data=dat, columns=cols)
        self.formula_tuple = (
            'm1_resid + m3_resid + m1_resid:m3_resid - 1',
            'm2 + m4 + np.square(m4) - 1')

    @patch('skillmodels.estimation.wa_functions.tf')
    def test_iv_reg_array_dict_y(self, mock_tf):
        mock_tf.iv_formula_some_func.return_value = self.formula_tuple

        expected_y = np.array([0, 5])
        calculated_y = wf.iv_reg_array_dict(
            self.depvar_name, self.indepvar_names, self.instrument_names,
            self.transition_name, self.data)['depvar_arr']
        aaae(expected_y, calculated_y)

    @patch('skillmodels.estimation.wa_functions.tf')
    def test_iv_reg_array_dict_x(self, mock_tf):
        mock_tf.iv_formula_some_func.return_value = self.formula_tuple

        expected_x = self.data['x'].copy()
        expected_x['m1_resid:m3_resid'] = \
            expected_x['m1_resid'] * expected_x['m3_resid']
        expected_x = expected_x.values[:2, :]

        calculated_x = wf.iv_reg_array_dict(
            self.depvar_name, self.indepvar_names, self.instrument_names,
            self.transition_name, self.data)['indepvars_arr']
        aaae(expected_x, calculated_x)

    @patch('skillmodels.estimation.wa_functions.tf')
    def test_iv_reg_array_dict_z(self, mock_tf):
        mock_tf.iv_formula_some_func.return_value = self.formula_tuple

        expected_z = self.data['z'].copy()
        expected_z['np.square(m4)'] = np.square(expected_z['m4'])
        expected_z = expected_z.values[:2, :]

        calculated_z = wf.iv_reg_array_dict(
            self.depvar_name, self.indepvar_names, self.instrument_names,
            self.transition_name, self.data)['instruments_arr']
        aaae(expected_z, calculated_z)


class TestIVMath:
    def setup(self):
        np.random.seed(40925)
        nobs = 2000
        x_cov = np.array([
            [2.0, 0.3, 0.4, 0.1],
            [0.3, 4.0, 0.2, 0.1],
            [0.4, 0.2, 3.5, 0.2],
            [0.1, 0.1, 0.2, 5.0]])

        # self.y = np.random.normal(loc=20, size=(nobs, 1))
        betas = np.array([0.2, 0.3, 0.4, 0.5])
        self.true_x = np.random.multivariate_normal(
            mean=np.arange(4), cov=x_cov, size=(nobs))
        self.x = self.true_x + np.random.normal(
            loc=0, scale=0.4, size=self.true_x.shape)
        self.y = self.true_x.dot(betas) + np.random.normal(
            loc=0, scale=0.5, size=(nobs))
        self.z = self.true_x + np.random.normal(
            loc=0, scale=0.4, size=self.x.shape)
        self.w = np.eye(4)

    def test_iv_math(self):
        expected_beta = LinearIVGMM(
            endog=self.y, exog=self.x, instrument=self.z).fitgmm(
                start=None, weights=self.w)
        calc_beta = wf._iv_math(self.y, self.x, self.z, self.w)

        aaae(calc_beta.flatten(), expected_beta)


class TestIVGMMWeights:
    def setup(self):
        nobs = 2000
        self.z_large = np.random.normal(size=(nobs, 5))
        self.z_small = np.array(
            [[1, 0.5, 3], [0, 0, 2], [1, 1, 1], [0.1, 0.2, 0.3]])
        self.fake_y = np.ones(nobs)
        self.fake_x = np.ones((nobs, 2))
        self.u = np.array([1, 2, 3, 4])

    def test_iv_gmm_weights_2sls_comparison_with_statsmodels(self):
        mod = LinearIVGMM(endog=self.fake_y, exog=self.fake_x,
                          instrument=self.z_large)
        expected_w = mod.start_weights(inv=False)
        calculated_w = wf._iv_gmm_weights(self.z_large)
        aaae(calculated_w, expected_w)

    def test_iv_gmm_weights_optimal_small_case_calculated_manually(self):
        expected_w = np.array([[11.27417695, -10.54526749, -0.56018519],
                               [-10.54526749, 10.51028807, 0.31481481],
                               [-0.56018519, 0.31481481, 0.20833333]])
        calculated_w = wf._iv_gmm_weights(self.z_small, self.u)
        aaae(calculated_w, expected_w)


class TestLargeDFForIVEquations:
    def setup(self):
        pass


class TestTransitionErrorVarianceFromUCovs:
    def setup(self):
        pass


if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
