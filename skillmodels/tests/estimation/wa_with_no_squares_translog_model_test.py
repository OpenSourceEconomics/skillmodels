from skillmodels import SkillModel
import json
import pandas as pd
import numpy as np

from skillmodels.estimation.wa_functions import calculate_wa_estimated_quantities
from skillmodels.model_functions.transition_functions import no_squares_translog
from numpy.testing import assert_array_almost_equal as aaae
from nose.tools import nottest, assert_almost_equal

with open("skillmodels/tests/estimation/no_squares_translog_model.json") as j:
    model_dict = json.load(j)


@nottest
def generate_test_data(
    nobs,
    factors,
    periods,
    included_positions,
    meas_names,
    initial_mean,
    initial_cov,
    intercepts,
    loadings,
    meas_sd,
    gammas,
    trans_sd,
    anch_intercept,
    anch_loadings,
    anch_sd,
):

    # np.random.seed(12345)
    np.random.seed(547185)
    nfac = len(factors)
    initial_factors = np.random.multivariate_normal(
        mean=initial_mean, cov=initial_cov, size=(nobs)
    )
    factor_data = []
    meas_data = []
    m_to_factor = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    counter = 0
    for t in periods:

        if t == 0:
            new_facs = initial_factors
        else:
            new_facs = np.zeros((nobs, nfac))
            new_facs[:, : nfac - 1] += np.random.normal(
                loc=np.zeros(nfac - 1), scale=trans_sd[t - 1], size=(nobs, nfac - 1)
            )
            for f, factor in enumerate(factors):
                if f in [0, 1]:
                    new_facs[:, f] += no_squares_translog(
                        factor_data[t - 1], gammas[f][t - 1], included_positions[f]
                    )
                else:
                    new_facs[:, f] = factor_data[t - 1][:, f]
        factor_data.append(new_facs)
        nmeas = 9 if t == 0 else 6
        # noise part of measurements
        measurements = np.random.normal(
            loc=np.zeros(nmeas),
            scale=meas_sd[counter : counter + nmeas],
            size=(nobs, nmeas),
        )
        # add structural part of measurements
        for m in range(nmeas):
            factor_pos = m_to_factor[m]
            measurements[:, m] += new_facs[:, factor_pos] * loadings[counter]
            measurements[:, m] += intercepts[counter]
            counter += 1
        df = pd.DataFrame(data=measurements, columns=meas_names[:nmeas])
        if t == periods[-1]:
            # add the anchoring outcome to the data
            df["anch_out"] = np.dot(new_facs[:, :-1], anch_loadings)
            df["anch_out"] += anch_intercept
            df["anch_out"] += np.random.normal(loc=0, scale=anch_sd, size=nobs)
        df["period"] = t
        df["id"] = np.arange(nobs)
        meas_data.append(df)
    large_df = pd.concat(meas_data, sort=True)
    large_df.sort_values(by=["id", "period"], inplace=True)
    large_df.set_index(['id', 'period'], inplace=True)
    return large_df


class TestOfWAEstimator:
    def setup(self):
        self.factor_names = ["fac1", "fac2", "fac3"]
        self.nfac = len(self.factor_names)
        self.nperiods = 4
        self.periods = list(range(self.nperiods))
        self.included_positions = [np.arange(3), np.array([1, 2]), []]
        self.anch_intercept = 3.0
        self.anch_loadings = np.array([1.2, 1.3])

        self.meas_names = ["y{}".format(i + 1) for i in range(9)]

        self.true_gammas = [
            [
                [0.725, 0.01, 0.02, 0.0015, 0.0018, 0.0014, 0.5],
                [0.750, 0.03, 0.03, 0.0003, 0.0020, 0.0024, 0.6],
                [0.775, 0.05, 0.04, 0.0023, 0.0026, 0.0012, 0.7],
            ],
            [
                [0.90, 0.01, 0.0004, 0.25],
                [0.925, 0.04, 0.0014, 0.75],
                [0.950, 0.07, 0.0002, 1.25],
            ],
            np.zeros((3, 0)),
        ]

        self.true_loadings = np.arange(start=2.5, stop=3.85, step=0.05)
        self.true_intercepts = np.arange(start=-0.665, stop=0.665, step=0.05)
        self.true_X_zero = np.array([5, 7.5, 30])

    def test_loadings_intercepts_transparams_anchparams_and_xzeros(self):
        self.nobs = 5000
        self.base_meas_sd = 0.00001
        self.base_trans_sd = 0.00001
        self.anch_sd = 0.1

        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = self.base_trans_sd * np.arange(
            start=0.2, step=0.1, stop=0.75
        ).reshape(self.nperiods - 1, 2)
        self.true_trans_var = self.true_trans_sd ** 2
        self.true_cov_matrix = np.array(
            [[1.44, 0.05, 0.1], [0.05, 2.25, 0.0], [0.1, 0.0, 4.0]]
        )
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

        self.y_data = generate_test_data(
            nobs=self.nobs,
            factors=self.factor_names,
            periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero,
            initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts,
            loadings=self.true_loadings,
            meas_sd=self.true_meas_sd,
            gammas=self.true_gammas,
            trans_sd=self.true_trans_sd,
            anch_intercept=self.anch_intercept,
            anch_loadings=self.anch_loadings,
            anch_sd=self.anch_sd,
        )

        wa_model = SkillModel(
            model_name="no_squares_translog",
            dataset_name="test_data",
            model_dict=model_dict,
            dataset=self.y_data,
            estimator="wa",
        )

        calc_storage_df, calc_X_zero, calc_P_zero, calc_gammas, trans_vars, anch_intercept, anch_loadings, anch_variance = calculate_wa_estimated_quantities(
            wa_model.identified_restrictions,
            wa_model.y_data,
            wa_model.measurements,
            wa_model.normalizations,
            wa_model.storage_df,
            wa_model.factors,
            wa_model.transition_names,
            wa_model.included_factors,
            wa_model.nstages,
            wa_model.stages,
            wa_model.periods,
            wa_model.stagemap,
            wa_model.anchored_factors,
            wa_model.anch_outcome,
            wa_model.wa_period_weights,
            wa_model.anchoring,
        )

        calc_loadings = calc_storage_df["loadings"]
        calc_intercepts = calc_storage_df["intercepts"]

        aaae(calc_loadings.to_numpy(), self.true_loadings, decimal=3)
        aaae(calc_intercepts.to_numpy(), self.true_intercepts, decimal=3)
        aaae(calc_X_zero, self.true_X_zero, decimal=1)
        for arr1, arr2 in zip(calc_gammas, self.true_gammas):
            aaae(arr1, arr2, decimal=3)
        assert_almost_equal(anch_intercept, 3.0, places=1)
        aaae(anch_loadings, self.anch_loadings, decimal=2)

    def test_pzero_and_measurement_variances(self):
        self.nobs = 20000

        self.true_gammas = [
            [
                [1.1, 0.01, 0.02, 0.0, 0.0, 0.0, 0.5],
                [1.2, 0.03, 0.03, 0.0, 0.0, 0.0, 0.6],
                [1.3, 0.05, 0.04, 0.0, 0.0, 0.0, 0.7],
            ],
            [[1.05, 0.01, 0.0, 0.25], [1.15, 0.04, 0.0, 0.75], [1.25, 0.07, 0.0, 1.25]],
            np.zeros((3, 0)),
        ]

        self.base_meas_sd = 0.15
        self.base_trans_sd = 1e-50
        self.anch_sd = 0.4

        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = self.base_trans_sd * np.arange(
            start=0.2, step=0.1, stop=0.75
        ).reshape(self.nperiods - 1, 2)
        self.true_trans_var = self.true_trans_sd ** 2

        self.true_cov_matrix = np.array(
            [[1.0, 0.05, 0.05], [0.05, 1.0, 0.05], [0.05, 0.05, 1.0]]
        )
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

        self.y_data = generate_test_data(
            nobs=self.nobs,
            factors=self.factor_names,
            periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero,
            initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts,
            loadings=self.true_loadings,
            meas_sd=self.true_meas_sd,
            gammas=self.true_gammas,
            trans_sd=self.true_trans_sd,
            anch_intercept=self.anch_intercept,
            anch_loadings=self.anch_loadings,
            anch_sd=self.anch_sd,
        )

        wa_model = SkillModel(
            model_name="no_squares_translog",
            dataset_name="test_data",
            model_dict=model_dict,
            dataset=self.y_data,
            estimator="wa",
        )

        calc_storage_df, calc_X_zero, calc_P_zero, calc_gammas, trans_vars, anch_intercept, anch_loadings, anch_variance = calculate_wa_estimated_quantities(
            wa_model.identified_restrictions,
            wa_model.y_data,
            wa_model.measurements,
            wa_model.normalizations,
            wa_model.storage_df,
            wa_model.factors,
            wa_model.transition_names,
            wa_model.included_factors,
            wa_model.nstages,
            wa_model.stages,
            wa_model.periods,
            wa_model.stagemap,
            wa_model.anchored_factors,
            wa_model.anch_outcome,
            wa_model.wa_period_weights,
            wa_model.anchoring,
        )

        # df = calc_storage_df.copy(deep=True)
        # df['true_meas_var'] = self.true_meas_var
        # df['diff'] = df['meas_error_variances'] - df['true_meas_var']
        # df['perc_diff'] = df['diff'] / df['true_meas_var']
        # df['true_loadings'] = self.true_loadings
        # print(df[['meas_error_variances', 'true_meas_var', 'diff', 'perc_diff',
        #           'loadings', 'true_loadings']])
        # print(df['diff'].mean())

        calc_epsilon_variances = calc_storage_df["meas_error_variances"].to_numpy()
        # average_epsilon_diff = \
        #     (calc_epsilon_variances - self.true_meas_var).mean()
        aaae(calc_P_zero, self.true_P_zero, decimal=2)
        aaae(calc_epsilon_variances[:9], self.true_meas_var[:9], decimal=2)
        assert_almost_equal(np.sqrt(anch_variance), self.anch_sd, places=1)

    def test_transition_variances(self):
        self.nobs = 5000
        self.base_meas_sd = 0.00001
        self.base_trans_sd = 0.5
        self.anch_sd = 0.1

        self.true_meas_sd = self.true_loadings * self.base_meas_sd
        self.true_meas_var = self.true_meas_sd ** 2
        self.true_trans_sd = self.base_trans_sd * np.arange(
            start=0.2, step=0.1, stop=0.75
        ).reshape(self.nperiods - 1, 2)
        self.true_trans_var = self.true_trans_sd ** 2

        self.true_cov_matrix = np.array(
            [[1.44, 0.05, 0.1], [0.05, 2.25, 0.0], [0.1, 0.0, 4.0]]
        )
        self.true_P_zero = self.true_cov_matrix[np.triu_indices(self.nfac)]

        self.y_data = generate_test_data(
            nobs=self.nobs,
            factors=self.factor_names,
            periods=self.periods,
            included_positions=self.included_positions,
            meas_names=self.meas_names,
            initial_mean=self.true_X_zero,
            initial_cov=self.true_cov_matrix,
            intercepts=self.true_intercepts,
            loadings=self.true_loadings,
            meas_sd=self.true_meas_sd,
            gammas=self.true_gammas,
            trans_sd=self.true_trans_sd,
            anch_intercept=self.anch_intercept,
            anch_loadings=self.anch_loadings,
            anch_sd=self.anch_sd,
        )

        wa_model = SkillModel(
            model_name="no_squares_translog",
            dataset_name="test_data",
            model_dict=model_dict,
            dataset=self.y_data,
            estimator="wa",
        )

        calc_storage_df, calc_X_zero, calc_P_zero, calc_gammas, calc_trans_vars, anch_intercept, anch_loadings, anch_variance = calculate_wa_estimated_quantities(
            wa_model.identified_restrictions,
            wa_model.y_data,
            wa_model.measurements,
            wa_model.normalizations,
            wa_model.storage_df,
            wa_model.factors,
            wa_model.transition_names,
            wa_model.included_factors,
            wa_model.nstages,
            wa_model.stages,
            wa_model.periods,
            wa_model.stagemap,
            wa_model.anchored_factors,
            wa_model.anch_outcome,
            wa_model.wa_period_weights,
            wa_model.anchoring,
        )

        aaae(calc_trans_vars.to_numpy(), self.true_trans_var, decimal=3)
