from nose.tools import assert_equal, assert_raises
import pandas as pd
from pandas import DataFrame
import numpy as np
from unittest.mock import Mock, call, patch    # noqa
from itertools import cycle
from numpy.testing import assert_array_equal as aae
from skillmodels.pre_processing.model_spec_processor \
    import ModelSpecProcessor as msp


class TestTransitionEquationNames:
    def setup(self):
        self.factors = ['f1', 'f2', 'f3']
        names = ['linear', 'ces', 'ar1']
        self._facinf = {factor: {'trans_eq': {'name': name}} for factor, name
                        in zip(self.factors, names)}

    def test_transition_equation_names(self):
        msp._transition_equation_names(self)
        assert_equal(self.transition_names, ['linear', 'ces', 'ar1'])


class TestTransitionEquationIncludedFactors:
    def setup(self):
        self.factors = ['f1', 'f2']
        self._facinf = {factor: {'trans_eq': {'included_factors': []}}
                        for factor in self.factors}
        self._facinf['f1']['trans_eq']['included_factors'] = ['f2', 'f1']
        self._facinf['f2']['trans_eq']['included_factors'] = ['f2']
        self.nfac = 2

    def test_transition_equation_included_factors(self):
        msp._transition_equation_included_factors(self)
        assert_equal(self.included_factors, [['f1', 'f2'], ['f2']])

    def test_transition_equation_included_factor_positions(self):
        msp._transition_equation_included_factors(self)
        assert_equal(self.included_positions,
                     [[0, 1], [1]])


class TestVariableCheckMethods:
    def setup(self):
        df1 = DataFrame(data=np.zeros((5, 2)), columns=['period', 'var1'])
        df1.loc[1, 'var1'] = 1
        df2 = DataFrame(data=np.ones((5, 2)), columns=['period', 'var2'])
        df2.loc[1, 'var2'] = 5
        self.data = pd.concat([df1, df2], axis=0, sort=True)
        self.missing_variables = 'drop_variable'
        self.variables_without_variance = 'drop_variable'
        self.model_name = 'model'
        self.dataset_name = 'dataset'
        self.period_identifier = 'period'

    def test_present_where_true(self):
        assert_equal(msp._present(self, 'var1', 0), True)

    def test_present_where_false_no_raise(self):
        assert_equal(msp._present(self, 'var1', 1), False)

    def test_present_where_false__raise(self):
        self.missing_variables = 'raise_error'
        assert_raises(KeyError, msp._present, self, 'var1', 1)

    def test_is_dummy_where_true(self):
        assert_equal(msp._is_dummy(self, 'var1', 0), True)

    def test_is_dummy_where_false_non_missing(self):
        assert_equal(msp._is_dummy(self, 'var2', 1), False)

    def test_is_dummy_where_false_missing(self):
        assert_equal(msp._is_dummy(self, 'var1', 1), False)

    def test_has_variance_where_true(self):
        assert_equal(msp._has_variance(self, 'var1', 0), True)

    def test_has_variance_where_false_non_missing(self):
        self.data.loc[1, 'var1'] = 0
        assert_equal(msp._has_variance(self, 'var1', 0), False)

    def test_has_variance_where_false_missing(self):
        assert_equal(msp._has_variance(self, 'var2', 0), False)


class TestCleanMesaurementSpecifications:
    def setup(self):
        self.periods = [0, 1]
        inf = {'f1': {}, 'f2': {}}
        inf['f1']['measurements'] = [['m1', 'm2', 'm3', 'm4']] * 2
        inf['f2']['measurements'] = [['m5', 'm6', 'm7', 'm8']] * 2
        self._facinf = inf
        self.factors = sorted(list(self._facinf.keys()))
        self.transition_names = ['log_ces', 'blubb']
        self.estimator = 'CHS'

    def test_clean_measuremnt_specifications_nothing_to_clean(self):
        self._present = Mock(return_value=True)
        self._has_variance = Mock(return_value=True)
        res = {}
        res['f1'] = self._facinf['f1']['measurements']
        res['f2'] = self._facinf['f2']['measurements']
        msp._clean_measurement_specifications(self)
        assert_equal(self.measurements, res)

    def test_clean_measurement_specifications_half_of_variables_missing(self):
        self._present = Mock(side_effect=cycle([True, False]))
        self._has_variance = Mock(return_value=True)
        res = {}
        res['f1'] = [['m1', 'm3']] * 2
        res['f2'] = [['m5', 'm7']] * 2
        msp._clean_measurement_specifications(self)
        assert_equal(self.measurements, res)

    def test_clean_measurement_specs_half_of_variables_without_variance(self):
        self._present = Mock(return_value=True)
        self._has_variance = Mock(side_effecet=cycle([False, True]))
        res = {}
        res['f1'] = [['m2', 'm4']]
        res['f2'] = [['m6', 'm8']]


class TestCleanControlSpecifications:
    def setup(self):
        self._timeinf = {'controls': [['c1', 'c2'], ['c1', 'c2']]}
        self.periods = [0, 1]
        self.nperiods = 2
        self._present = Mock(return_value=True)
        self._has_variance = Mock(return_value=True)
        self.controls_with_missings = 'drop_variable'
        self.model_name = 'model'
        self.dataset_name = 'data'
        cols = ['period', 'c1', 'c2']
        dat = np.zeros((10, 3))
        dat[5:, 0] = 1
        self.data = DataFrame(data=dat, columns=cols)
        self.estimator = 'chs'
        self.period_identifier = 'period'

    def test_clean_control_specs_nothing_to_clean(self):
        msp._clean_controls_specification(self)
        res = [['c1', 'c2'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_variable(self):
        self._present = Mock(side_effect=[True, False, True, True])
        msp._clean_controls_specification(self)
        res = [['c1'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_observations_drop_variable(self):
        self.data.loc[2, 'c2'] = np.nan
        msp._clean_controls_specification(self)
        res = [['c1'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_observation_drop_observation(self):
        self.data.loc[2, 'c2'] = np.nan
        self.controls_with_missings = 'drop_observations'
        msp._clean_controls_specification(self)
        res = [['c1', 'c2'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.array([True, True, False, True, True]))

    def test_clean_control_specs_missing_observations_error(self):
        self.data.loc[2, 'c2'] = np.nan
        self.controls_with_missings = 'raise_error'
        assert_raises(ValueError, msp._clean_controls_specification, self)


class TestCheckNormalizations:
    def setup(self):
        self.periods = [0, 1]
        self._facinf = {'f1': {'measurements': [['m1', 'm2', 'm3', 'm4']] * 2}}
        self.f1_norm_list = [['m1', 1], ['m1', 1]]
        self.factors = sorted(list(self._facinf.keys()))
        self.measurements = {'f1': [['m1', 'm2', 'm3', 'm4']] * 2}
        self.model_name = 'model'
        self.dataset_name = 'data'
        self.nperiods = len(self.periods)

    def test_check_normalizations_no_error(self):
        result = msp._check_normalizations_list(self, 'f1', self.f1_norm_list)
        assert_equal(result, self.f1_norm_list)

    def test_check_normalizations_not_specified_error(self):
        f1_norm_list = [['m10', 1], ['m1', 1]]
        assert_raises(
            KeyError, msp._check_normalizations_list, self, 'f1', f1_norm_list)

    def test_check_normalizations_dropped_error(self):
        self.measurements = {'f1': [['m2', 'm3', 'm4']] * 2}
        assert_raises(KeyError, msp._check_normalizations_list, self, 'f1',
                      self.f1_norm_list)

    def test_check_normalizations_invalid_length_of_list(self):
        assert_raises(
            AssertionError, msp._check_normalizations_list, self, 'f1',
            self.f1_norm_list * 2)

    def test_check_normalizations_invalid_length_of_sublst(self):
        f1_norm_list = [['m1'], ['m1', 1]]
        assert_raises(
            AssertionError, msp._check_normalizations_list, self, 'f1',
            f1_norm_list)


class TestGenerateNormalizationSpecifications:
    def setup(self):
        self.factors = ['fac1', 'fac2', 'fac3']
        self.transition_names = ['ar1', 'log_ces', 'translog']
        self.estimate_X_zeros = True
        self.nperiods = 9
        self.periods = list(range(self.nperiods))
        self.stagemap = [0, 0, 0, 1, 1, 2, 3, 3, 3]
        self.stages = [0, 1, 2, 3]
        self.stage_length_list = [3, 2, 1, 3]

    def test_has_fixed_start_period_list_loadings(self):
        expected = [False, True, True, False]
        calculated = msp._stage_has_fixed_start_period_list(self, 'loadings')
        assert_equal(expected, calculated)

    def test_has_fixed_start_period_list_intercepts_fixed_x_zeros(self):
        expected = [True, True, True, False]
        self.estimate_X_zeros = False
        calculated = msp._stage_has_fixed_start_period_list(self, 'intercepts')
        assert_equal(expected, calculated)

    def test_has_fixed_start_period_list_intercepts_free_x_zeros(self):
        expected = [False, True, True, False]
        calculated = msp._stage_has_fixed_start_period_list(self, 'intercepts')
        assert_equal(expected, calculated)

    def test_is_first_period_in_stage_true(self):
        assert_equal(msp._first_period_in_stage(self, 3), True)

    def test_is_first_period_in_stage_false(self):
        assert_equal(msp._first_period_in_stage(self, 4), False)

    def test_needs_norm_ar1_case_for_loadings(self):
        expected = [True, True, False, False, False,
                    False, False, False, False]
        result = msp.needs_normalization(self, 'fac1', 'loadings')
        assert_equal(result, expected)

    def test_needs_norm_ar1_case_for_intercept_fixed_x_zeros(self):
        self.estimate_X_zeros = False
        expected = [False, False, False, False, False,
                    False, False, False, False]
        result = msp.needs_normalization(self, 'fac1', 'intercepts')
        assert_equal(result, expected)

    def test_needs_norm_ar1_case_for_intercept_free_x_zeros(self):
        expected = [True, False, False, False, False,
                    False, False, False, False]
        result = msp.needs_normalization(self, 'fac1', 'intercepts')
        assert_equal(result, expected)

    def test_needs_norm_log_ces_case_for_loadings(self):
        expected = [True, False, False, False, False,
                    False, False, False, False]
        result = msp.needs_normalization(self, 'fac2', 'loadings')
        assert_equal(result, expected)

    def test_needs_norm_log_ces_case_for_intercepts_fixed_x_zeros(self):
        expected = [False, False, False, False, False,
                    False, False, False, False]
        self.estimate_X_zeros = False
        result = msp.needs_normalization(self, 'fac2', 'intercepts')
        assert_equal(result, expected)

    def test_needs_norm_log_ces_case_for_intercepts_free_x_zeros(self):
        expected = [True, False, False, False, False,
                    False, False, False, False]
        result = msp.needs_normalization(self, 'fac2', 'intercepts')
        assert_equal(result, expected)

    def test_needs_norm_translog_case_for_loadings(self):
        expected = [True, True, False, False, True, False, True, True, False]
        self._stage_has_fixed_start_period_list = Mock(
            return_value=[False, True, True, False])
        self._first_period_in_stage = Mock(
            side_effect=[False, True, False, False, True, False, True, True,
                         True, False, True, False, False])
        result = msp.needs_normalization(self, 'fac3', 'loadings')
        assert_equal(result, expected)

    def test_needs_norm_translog_case_for_intercepts_fixed_x_zeros(self):
        expected = [False, True, False, False, True, False, True, True, False]
        self.estimate_X_zeros = False
        self._stage_has_fixed_start_period_list = Mock(
            return_value=[True, True, True, False])
        self._first_period_in_stage = Mock(
            side_effect=[False, True, False, False, True, False, True, True,
                         True, False, True, False, False])
        result = msp.needs_normalization(self, 'fac3', 'intercepts')
        assert_equal(result, expected)

    def test_needs_norm_translog_case_for_intercepts_free_x_zeros(self):
        expected = [True, True, False, False, True, False, True, True, False]
        self._stage_has_fixed_start_period_list = Mock(
            return_value=[False, True, True, False])
        self._first_period_in_stage = Mock(
            side_effect=[False, True, False, False, True, False, True, True,
                         True, False, True, False, False])
        result = msp.needs_normalization(self, 'fac3', 'intercepts')
        assert_equal(result, expected)


class TestGenerateNormalizationsList:
    def setup(self):
        self.measurements = {'f1': [['m1', 'm2'], ['m1', 'm2', 'm3'], ['m2'],
                                    ['m4'], ['m4']]}
        self.needs_normalization = Mock(
            return_value=[True, True, False, False, True])
        self.periods = list(range(5))

    def test_generate_normalization_list_loadings(self):
        expected = [['m1', 1], ['m1', 1], [], [], ['m4', 1]]
        result = msp.generate_normalizations_list(self, 'f1', 'loadings')
        assert_equal(result, expected)

    def test_generate_normalization_list_intercepts(self):
        expected = [['m1', 0], ['m1', 0], [], [], ['m4', 0]]
        result = msp.generate_normalizations_list(self, 'f1', 'intercepts')
        assert_equal(result, expected)


class TestCheckOrGenerateNormalizationSpecifications:
    def setup(self):
        self._facinf = {
            'f1': {'normalizations':
                   {'loadings': [['a', 1], ['a', 1], ['a', 1]],
                    'intercepts': [['a', 1], ['a', 1], ['a', 1]]}},
            'f2': {}}

        self.factors = sorted(list(self._facinf.keys()))
        self._check_normalizations_list = Mock(
            return_value=[['a', 1], ['a', 1], ['a', 1]])
        self.generate_normalizations_list = \
            Mock(return_value=[['b', 1], ['b', 1], ['b', 1]])

    def test_check_or_generate_normalization_specifications(self):
        res = {'f1': {'loadings': [['a', 1], ['a', 1], ['a', 1]],
                      'intercepts': [['a', 1], ['a', 1], ['a', 1]]},
               'f2': {'loadings': [['b', 1], ['b', 1], ['b', 1]],
                      'intercepts': [['b', 1], ['b', 1], ['b', 1]]}}
        msp._check_or_generate_normalization_specification(self)
        print('calc', '\n\n', self.normalizations)
        print('exp', '\n\n', res)

        assert_equal(self.normalizations, res)


class TestUpdateInfoTable:
    def setup(self):
        self.factors = ['f1', 'f2', 'f3']
        self.periods = [0, 1, 2, 3]
        self.nperiods = len(self.periods)
        self.stagemap = [0, 0, 1, 1]
        self.probit_measurements = True
        self.estimator = 'chs'

        m = {}
        m['f1'] = [['m1', 'm2']] * 4
        m['f2'] = [['m3', 'm4', 'm5']] * 4
        m['f3'] = [['m5', 'm6']] * 4
        self.measurements = m

        self.anchoring = True
        self.anch_outcome = 'a'
        self.anchored_factors = ['f1', 'f3']

        n = {'f1': {}, 'f2': {}, 'f3': {}}

        n['f1']['loadings'] = [['m1', 2]] * 4
        n['f2']['loadings'] = [['m3', 1], ['m3', 1], ['m3', 1], ['m5', 5]]
        n['f3']['loadings'] = [['m5', 4]] * 4

        n['f1']['intercepts'] = [['m1', 0]] * 4
        n['f2']['intercepts'] = [['m3', 1], ['m3', 1], ['m3', 1], ['m5', 5]]
        n['f3']['intercepts'] = [['m5', 4]] * 4

        self.normalizations = n

        self._is_dummy = Mock(side_effect=cycle([False, False, True]))

        cols = self.factors + [
            '{}_loading_norm_value'.format(f) for f in self.factors]
        cols += ['stage', 'purpose', 'update_type']

        different_meas = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

        ind_tuples = []
        for t in self.periods:
            ind_tuples += [(t, d) for d in different_meas]
        ind_tuples.append((3, 'a'))

        index = pd.MultiIndex.from_tuples(
            ind_tuples, names=['period', 'variable'])

        dat = np.zeros((25, len(cols)))
        df = DataFrame(data=dat, columns=cols, index=index)

        for t in self.periods:
            for factor in self.factors:
                for m in self.measurements[factor][0]:
                    df.loc[(t, m), factor] = 1

        df['has_normalized_loading'] = False
        for factor in self.factors:
            norm_col = '{}_loading_norm_value'.format(factor)
            for t, (n_meas, n_val) in enumerate(
                    self.normalizations[factor]['loadings']):
                df.loc[(t, n_meas), norm_col] = n_val
                df.loc[(t, n_meas), 'has_normalized_loading'] = True

        df['has_normalized_intercept'] = False
        df['intercept_norm_value'] = np.nan
        for factor in self.factors:
            for t, (n_meas, n_val) in enumerate(
                    self.normalizations[factor]['intercepts']):
                df.loc[(t, n_meas), 'intercept_norm_value'] = n_val
                df.loc[(t, n_meas), 'has_normalized_intercept'] = True

        for t in self.periods:
            df.loc[t, 'stage'] = self.stagemap[t]

        update_type = ['linear', 'linear', 'probit'] * 9
        df['update_type'] = np.array(update_type[:len(df)])

        df['purpose'] = np.array(['measurement'] * 24 + ['anchoring'])

        df.loc[(3, 'a'), ['f1', 'f3']] = 1
        self.expected_res = df
        self.calculated_res = msp.update_info(self)

    def test_update_info_factor_columns(self):
        calc = self.calculated_res[self.factors]
        exp = self.expected_res[self.factors]
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_loading_normalization_columns(self):
        normcols = ['{}_loading_norm_value'.format(f) for f in self.factors]
        calc = self.calculated_res[normcols]
        exp = self.expected_res[normcols]
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_has_normalized_loading(self):
        calc = self.calculated_res['has_normalized_loading']
        exp = self.expected_res['has_normalized_loading']
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_intercept_norm_value(self):
        calc = self.calculated_res['intercept_norm_value']
        calc = calc[pd.notnull(calc)]
        exp = self.expected_res['intercept_norm_value']
        exp = exp[pd.notnull(exp)]

        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_has_normalized_intercept(self):
        calc = self.calculated_res['has_normalized_intercept']
        exp = self.expected_res['has_normalized_intercept']
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_update_type(self):
        calc = self.calculated_res['update_type']
        exp = self.expected_res['update_type']
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_stage(self):
        calc = self.calculated_res['stage']
        exp = self.expected_res['stage']
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_update_info_purpose(self):
        calc = self.calculated_res['purpose']
        exp = self.expected_res['purpose']
        assert_equal(calc.to_dict(), exp.to_dict())

    def test_that_anchoring_comes_last(self):
        calc = list(self.calculated_res['purpose'])
        exp = ['measurement'] * 24 + ['anchoring']
        assert_equal(calc, exp)


class TestWAStorageDf:
    def setup(self):
        self.factors = ['fac1', 'fac2']

        cols = ['fac1_loading_norm_value', 'fac2_loading_norm_value',
                'intercept_norm_value',
                'has_normalized_intercept', 'has_normalized_loading']
        index = pd.MultiIndex.from_tuples(
            [(0, 'm1'), (0, 'm2'), (1, 'm1'), (1, 'm2')])
        update_data = np.array([
            [1, 0, np.nan, False, True],
            [0, 0, 2, True, False],
            [0, 3, 0, True, True],
            [0, 0, np.nan, False, False]])
        df = pd.DataFrame(data=update_data, columns=cols, index=index)
        df['purpose'] = 'measurement'
        df['fac1'] = 1
        df['fac2'] = 0

        self.update_info = Mock(return_value=df)

        expected_data = np.array([
            [True, False, 1, 0],
            [False, True, 0, 2],
            [True, True, 3, 0],
            [False, False, 0, 0]])
        expected_cols = ['has_normalized_loading', 'has_normalized_intercept',
                         'loadings', 'intercepts']
        self.expected_res = pd.DataFrame(
            data=expected_data, columns=expected_cols, index=index)

        self.expected_res['meas_error_variances'] = 0

    def test_wa_storage_df(self):
        msp._wa_storage_df(self)
        print('exp', self.expected_res)
        print('calc', self.storage_df)
        assert_equal(self.storage_df.to_dict(), self.expected_res.to_dict())


class TestNewTransitionCoeffs:
    def setup(self):
        self.stages = [0, 1, 2]
        self.nstages = len(self.stages)

        self.factors = ['f1', 'f2', 'f3', 'f4', 'f5']
        self.nfac = len(self.factors)
        self.transition_names = ['constant', 'ar1', 'ces', 'bla', 'blubb']
        arr = np.array([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]], dtype=bool)
        self.enough_measurements_array = arr

    def test_new_transition_parameters(self):
        res = np.array(
            [[-1, 1, 1, 1, 1],
             [-1, 0, 1, 1, 1],
             [-1, 0, 1, 1, 1]])
        aae(msp.new_trans_coeffs(self), res)


if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
