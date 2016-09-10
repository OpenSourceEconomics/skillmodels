from skillmodels import SkillModel
import json
import pandas as pd
import numpy as np
from skillmodels.model_functions.transition_functions import \
    no_squares_translog
from numpy.testing import assert_array_almost_equal as aaae
from nose.tools import nottest

with open('no_squares_translog_model.json') as j:
    model_dict = json.load(j)


@nottest
def generate_test_data(nobs, factors, periods, included_positions, meas_names,
                       initial_mean, initial_cov, intercepts, loadings,
                       meas_sd, gammas, trans_sd):
    np.random.seed(69403)
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
            loc=np.zeros(nmeas), scale=meas_sd[:nmeas],
            size=(nobs, nmeas))
        measurements = np.zeros((nobs, nmeas))
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
        self.nobs = 10000
        self.nperiods = 8
        self.periods = list(range(self.nperiods))

        self.factor_names = ['fac1', 'fac2', 'fac3']
        self.nfac = len(self.factor_names)
        self.included_positions = [np.arange(3), np.array([1, 2]), []]

        self.meas_names = ['y{}'.format(i + 1) for i in range(9)]

        self.true_gammas = [
            [[0.925, 0.01, 0.02, 0.0015, 0.0018, 0.0014, 0.5],
             [0.950, 0.03, 0.03, 0.0003, 0.0020, 0.0024, 1.0],
             [0.975, 0.05, 0.04, 0.0023, 0.0026, 0.0012, 1.5],
             [1.000, 0.07, 0.05, 0.0001, 0.0018, 0.0022, 2.0],
             [1.025, 0.09, 0.06, 0.0021, 0.0024, 0.0010, 2.5],
             [1.050, 0.11, 0.07, 0.0001, 0.0016, 0.0020, 3.0],
             [1.075, 0.13, 0.08, 0.0019, 0.0022, 0.0008, 3.5]],

            [[1.100, 0.01, 0.0004, 0.25],
             [1.125, 0.04, 0.0014, 0.75],
             [1.150, 0.07, 0.0002, 1.25],
             [1.175, 0.11, 0.0012, 1.75],
             [1.200, 0.04, 0.0001, 2.25],
             [1.225, 0.07, 0.0010, 2.75],
             [1.250, 0.10, 0.0003, 3.25]],

            np.zeros((7, 0))]

        self.true_loadings = np.arange(start=0.5, stop=3.05, step=0.05)
        self.true_intercepts = np.arange(start=-1.275, stop=1.275, step=0.05)
        self.true_X_zero = np.array([10, 15, 40])
        self.true_cov_matrix = np.array([[4.0, 0.18, 0.4],
                                         [0.18, 9.0, 0.0],
                                         [0.4, 0.0, 16.0]])
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]
        self.base_meas_sd = 0.0001
        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = 0.00001 * np.arange(
            start=2, step=1, stop=18).reshape(self.nperiods, 2)
        self.true_trans_var = self.true_trans_sd ** 2

        self.y_data = generate_test_data(
            nobs=self.nobs, factors=self.factor_names, periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero, initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts, loadings=self.true_loadings,
            meas_sd=self.true_meas_sd, gammas=self.true_gammas,
            trans_sd=self.true_trans_sd)

        self.wa_model = SkillModel(
            model_name='no_squares_translog', dataset_name='test_data',
            model_dict=model_dict, dataset=self.y_data, estimator='wa')

        self.calc_storage_df, self.calc_X_zero, self.calc_P_zero, \
            self.calc_gammas = self.wa_model.fit_wa()
        self.calc_loadings = self.calc_storage_df['loadings']
        self.calc_intercepts = self.calc_storage_df['intercepts']

    def test_fit_wa_results(self):
        aaae(self.calc_loadings.values, self.true_loadings)
        aaae(self.calc_intercepts.values, self.true_intercepts)
        aaae(self.calc_X_zero, self.true_X_zero, decimal=1)
        for arr1, arr2 in zip(self.calc_gammas, self.true_gammas):
            aaae(arr1, arr2, decimal=4)
        # aaae(self.calc_P_zero, self.true_P_zero, decimal=2)

if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
