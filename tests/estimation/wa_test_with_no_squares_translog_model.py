from skillmodels import SkillModel
import json
import pandas as pd
import numpy as np
from skillmodels.model_functions.transition_functions import \
    no_squares_translog
from numpy.testing import assert_array_almost_equal as aaae
from nose.tools import nottest, assert_almost_equal

with open('no_squares_translog_model.json') as j:
    model_dict = json.load(j)


@nottest
def generate_test_data(nobs, factors, periods, included_positions, meas_names,
                       initial_mean, initial_cov, intercepts, loadings,
                       meas_sd, gammas, trans_sd):

    np.random.seed(12345)
    nfac = len(factors)
    initial_factors = np.random.multivariate_normal(
        mean=initial_mean, cov=initial_cov, size=(nobs))
    factor_data = []
    meas_data = []
    m_to_factor = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    counter = 0
    for t in periods:

        if t == 0:
            new_facs = initial_factors
        else:
            new_facs = np.zeros((nobs, nfac))
            new_facs[:, :nfac - 1] += np.random.normal(
                loc=np.zeros(nfac - 1), scale=trans_sd[t - 1],
                size=(nobs, nfac - 1))
            for f, factor in enumerate(factors):
                if f in [0, 1]:
                    new_facs[:, f] += no_squares_translog(
                        factor_data[t - 1], gammas[f][t - 1],
                        included_positions[f])
                else:
                    new_facs[:, f] = factor_data[t - 1][:, f]
        factor_data.append(new_facs)
        nmeas = 9 if t == 0 else 6
        # noise part of measurements
        measurements = np.random.normal(
            loc=np.zeros(nmeas), scale=meas_sd[counter: counter + nmeas],
            size=(nobs, nmeas))
        # add structural part of measurements
        for m in range(nmeas):
            factor_pos = m_to_factor[m]
            measurements[:, m] *= loadings[counter]
            measurements[:, m] += (new_facs[:, factor_pos] * loadings[counter])
            measurements[:, m] += intercepts[counter]
            counter += 1
        df = pd.DataFrame(data=measurements, columns=meas_names[:nmeas])
        df['period'] = t
        df['id'] = np.arange(nobs)
        meas_data.append(df)
    large_df = pd.concat(meas_data)
    large_df.sort_values(by=['id', 'period'], inplace=True)
    return large_df


class TestOfWAEstimator:
    def setup(self):
        self.factor_names = ['fac1', 'fac2', 'fac3']
        self.nfac = len(self.factor_names)
        self.nperiods = 4
        self.periods = list(range(self.nperiods))
        self.included_positions = [np.arange(3), np.array([1, 2]), []]

        self.meas_names = ['y{}'.format(i + 1) for i in range(9)]

        self.true_gammas = [
            [[0.725, 0.01, 0.02, 0.0015, 0.0018, 0.0014, 0.5],
             [0.750, 0.03, 0.03, 0.0003, 0.0020, 0.0024, 0.6],
             [0.775, 0.05, 0.04, 0.0023, 0.0026, 0.0012, 0.7]],

            [[.90, 0.01, 0.0004, 0.25],
             [.925, 0.04, 0.0014, 0.75],
             [.950, 0.07, 0.0002, 1.25]],

            np.zeros((3, 0))]

        self.true_loadings = np.arange(start=0.5, stop=1.85, step=0.05)
        self.true_intercepts = np.arange(start=-0.665, stop=0.665, step=0.05)
        self.true_X_zero = np.array([5, 7.5, 30])
        self.true_cov_matrix = np.array([[1.44, 0.05, 0.1],
                                         [0.05, 2.25, 0.0],
                                         [0.1, 0.0, 4.0]])
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

    def test_loadings_intercepts_transparams_and_xzeros(self):
        self.nobs = 1000
        self.base_meas_sd = 0.00001
        self.base_trans_sd = 0.00001

        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = self.base_trans_sd * np.arange(
            start=0.2, step=0.1, stop=0.75).reshape(self.nperiods - 1, 2)
        self.true_trans_var = self.true_trans_sd ** 2
        self.true_cov_matrix = np.array([[1.44, 0.05, 0.1],
                                         [0.05, 2.25, 0.0],
                                         [0.1, 0.0, 4.0]])
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

        self.y_data = generate_test_data(
            nobs=self.nobs, factors=self.factor_names, periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero, initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts, loadings=self.true_loadings,
            meas_sd=self.true_meas_sd, gammas=self.true_gammas,
            trans_sd=self.true_trans_sd)

        wa_model = SkillModel(
            model_name='no_squares_translog', dataset_name='test_data',
            model_dict=model_dict, dataset=self.y_data, estimator='wa')

        calc_storage_df, calc_X_zero, calc_P_zero, calc_gammas, trans_vars = \
            wa_model.fit_wa()
        calc_loadings = calc_storage_df['loadings']
        calc_intercepts = calc_storage_df['intercepts']

        aaae(calc_loadings.values, self.true_loadings, decimal=3)
        aaae(calc_intercepts.values, self.true_intercepts, decimal=3)
        aaae(calc_X_zero, self.true_X_zero, decimal=1)
        for arr1, arr2 in zip(calc_gammas, self.true_gammas):
            aaae(arr1, arr2, decimal=3)

    def test_pzero_and_measurement_variances(self):
        self.nobs = 20000
        self.base_meas_sd = 0.3
        self.base_trans_sd = 1e-40

        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = self.base_trans_sd * np.arange(
            start=0.2, step=0.1, stop=0.75).reshape(self.nperiods - 1, 2)
        self.true_trans_var = self.true_trans_sd ** 2

        self.true_cov_matrix = np.array([[0.36, 0.001, -0.001],
                                         [0.001, 0.49, 0.0],
                                         [-0.001, 0.0, 0.64]])
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

        self.y_data = generate_test_data(
            nobs=self.nobs, factors=self.factor_names, periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero, initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts, loadings=self.true_loadings,
            meas_sd=self.true_meas_sd, gammas=self.true_gammas,
            trans_sd=self.true_trans_sd)

        wa_model = SkillModel(
            model_name='no_squares_translog', dataset_name='test_data',
            model_dict=model_dict, dataset=self.y_data, estimator='wa')

        calc_storage_df, calc_X_zero, calc_P_zero, calc_gammas, trans_vars = \
            wa_model.fit_wa()

        calc_epsilon_variances = calc_storage_df['meas_error_variances']
        average_epsilon_diff = \
            (calc_epsilon_variances - self.true_meas_var).mean()
        aaae(calc_P_zero, self.true_P_zero, decimal=2)
        assert_almost_equal(average_epsilon_diff, 0.0, places=1)

    def test_transition_variances(self):
        self.nobs = 5000
        self.base_meas_sd = 0.00001
        self.base_trans_sd = 0.5

        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = self.base_trans_sd * np.arange(
            start=0.2, step=0.1, stop=0.75).reshape(self.nperiods - 1, 2)
        self.true_trans_var = self.true_trans_sd ** 2

        self.true_cov_matrix = np.array([[1.44, 0.05, 0.1],
                                         [0.05, 2.25, 0.0],
                                         [0.1, 0.0, 4.0]])
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

        self.y_data = generate_test_data(
            nobs=self.nobs, factors=self.factor_names, periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero, initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts, loadings=self.true_loadings,
            meas_sd=self.true_meas_sd, gammas=self.true_gammas,
            trans_sd=self.true_trans_sd)

        wa_model = SkillModel(
            model_name='no_squares_translog', dataset_name='test_data',
            model_dict=model_dict, dataset=self.y_data, estimator='wa')

        calc_storage_df, calc_X_zero, calc_P_zero, calc_gammas, \
            calc_trans_vars = wa_model.fit_wa()

        aaae(calc_trans_vars.values, self.true_trans_var, decimal=3)


if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
