from nose.tools import assert_equal, assert_raises
import pandas as pd
from pandas import DataFrame
import numpy as np
from unittest.mock import Mock, call, patch    # noqa
from itertools import cycle
from numpy.testing import assert_array_equal as aae
from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor as msc


class TestTransitionEquationNames:
    def setup(self):
        self.factors = ['f1', 'f2', 'f3']
        names = ['linear', 'ces', 'ar1']
        self._facinf = {factor: {'trans_eq': {'name': name}} for factor, name
                        in zip(self.factors, names)}

    def test_transition_equation_names(self):
        msc._transition_equation_names(self)
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
        msc._transition_equation_included_factors(self)
        assert_equal(self.included_factors, [['f1', 'f2'], ['f2']])

    def test_transition_equation_included_factor_positions(self):
        msc._transition_equation_included_factors(self)
        assert_equal(self.included_positions,
                     [[0, 1], [1]])


class TestVariableCheckMethods:
    def setup(self):
        df1 = DataFrame(data=np.zeros((5, 2)), columns=['period', 'var1'])
        df1.loc[1, 'var1'] = 1
        df2 = DataFrame(data=np.ones((5, 2)), columns=['period', 'var2'])
        df2.loc[1, 'var2'] = 5
        self._data = pd.concat([df1, df2], axis=0)
        self.missing_variables = 'drop_variable'
        self.variables_without_variance = 'drop_variable'
        self.model_name = 'model'
        self.dataset_name = 'dataset'

    def test_present_where_true(self):
        assert_equal(msc._present(self, 'var1', 0), True)

    def test_present_where_false_no_raise(self):
        assert_equal(msc._present(self, 'var1', 1), False)

    def test_present_where_false__raise(self):
        self.missing_variables = 'raise_error'
        assert_raises(KeyError, msc._present, self, 'var1', 1)

    def test_is_dummy_where_true(self):
        assert_equal(msc._is_dummy(self, 'var1', 0), True)

    def test_is_dummy_where_false_non_missing(self):
        assert_equal(msc._is_dummy(self, 'var2', 1), False)

    def test_is_dummy_where_false_missing(self):
        assert_equal(msc._is_dummy(self, 'var1', 1), False)

    def test_has_variance_where_true(self):
        assert_equal(msc._has_variance(self, 'var1', 0), True)

    def test_has_variance_where_false_non_missing(self):
        self._data.loc[1, 'var1'] = 0
        assert_equal(msc._has_variance(self, 'var1', 0), False)

    def test_has_variance_where_false_missing(self):
        assert_equal(msc._has_variance(self, 'var2', 0), False)


class TestCleanMesaurementSpecifications:
    def setup(self):
        self.periods = [0, 1]
        inf = {'f1': {}, 'f2': {}}
        inf['f1']['measurements'] = [['m1', 'm2', 'm3', 'm4']] * 2
        inf['f2']['measurements'] = [['m5', 'm6', 'm7', 'm8']] * 2
        self._facinf = inf
        self.factors = sorted(list(self._facinf.keys()))

    def test_clean_measuremnt_specifications_nothing_to_clean(self):
        self._present = Mock(return_value=True)
        self._has_variance = Mock(return_value=True)
        res = {}
        res['f1'] = self._facinf['f1']['measurements']
        res['f2'] = self._facinf['f2']['measurements']
        msc._clean_measurement_specifications(self)
        assert_equal(self.measurements, res)

    def test_clean_measurement_specifications_half_of_variables_missing(self):
        self._present = Mock(side_effect=cycle([True, False]))
        self._has_variance = Mock(return_value=True)
        res = {}
        res['f1'] = [['m1', 'm3']] * 2
        res['f2'] = [['m5', 'm7']] * 2
        msc._clean_measurement_specifications(self)
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
        self._data = DataFrame(data=dat, columns=cols)

    def test_clean_control_specs_nothing_to_clean(self):
        msc._clean_controls_specification(self)
        res = [['c1', 'c2'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_variable(self):
        self._present = Mock(side_effect=[True, False, True, True])
        msc._clean_controls_specification(self)
        res = [['c1'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_observations_drop_variable(self):
        self._data.loc[2, 'c2'] = np.nan
        msc._clean_controls_specification(self)
        res = [['c1'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_observation_drop_observation(self):
        self._data.loc[2, 'c2'] = np.nan
        self.controls_with_missings = 'drop_observations'
        msc._clean_controls_specification(self)
        res = [['c1', 'c2'], ['c1', 'c2']]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.array([True, True, False, True, True]))

    def test_clean_control_specs_missing_observations_error(self):
        self._data.loc[2, 'c2'] = np.nan
        self.controls_with_missings = 'raise_error'
        assert_raises(ValueError, msc._clean_controls_specification, self)


class TestCheckNormalizations:
    def setup(self):
        self.periods = [0, 1]
        self._facinf = {'f1': {'measurements': [['m1', 'm2', 'm3', 'm4']] * 2}}
        self._facinf['f1']['normalizations'] = [['m1', 1], ['m1', 1]]
        self.factors = sorted(list(self._facinf.keys()))
        self.measurements = {'f1': [['m1', 'm2', 'm3', 'm4']] * 2}
        self.model_name = 'model'
        self.dataset_name = 'data'
        self.nperiods = len(self.periods)

    def test_check_normalizations_no_error(self):
        assert_equal(msc._check_normalizations_list(self, 'f1'),
                     self._facinf['f1']['normalizations'])

    def test_check_normalizations_not_specified_error(self):
        self._facinf['f1']['normalizations'] = [['m10', 1], ['m1', 1]]
        assert_raises(KeyError, msc._check_normalizations_list, self, 'f1')

    def test_check_normalizations_dropped_error(self):
        self.measurements = {'f1': [['m2', 'm3', 'm4']] * 2}
        assert_raises(KeyError, msc._check_normalizations_list, self, 'f1')

    def test_check_normalizations_invalid_length_of_list(self):
        self._facinf['f1']['normalizations'] = [['m1', 1]] * 3
        assert_raises(
            AssertionError, msc._check_normalizations_list, self, 'f1')

    def test_check_normalizations_invalid_length_of_sublst(self):
        self._facinf['f1']['normalizations'] = [['m1'], ['m1', 1]]
        assert_raises(
            AssertionError, msc._check_normalizations_list, self, 'f1')


class TestGenerateNormalizationSpecifications:
    def setup(self):
        pass


class TestCheckOrGenerateNormalizationSpecifications:
    def setup(self):
        self._facinf = {
            'f1': {'normalizations': [['a', 1], ['a', 1], ['a', 1]]},
            'f2': {}}

        self.factors = sorted(list(self._facinf.keys()))
        self._check_normalizations_list = Mock(
            side_effect=lambda x: self._facinf[x]['normalizations'])
        self.generate_normalizations_list = \
            Mock(return_value=[['b', 1], ['b', 1], ['b', 1]])

    def test_check_or_generate_normalization_specifications(self):
        res = {'f1': [['a', 1], ['a', 1], ['a', 1]],
               'f2': [['b', 1], ['b', 1], ['b', 1]]}
        msc._check_or_generate_normalization_specification(self)

        assert_equal(self.normalizations, res)


class TestUpdateInfoTable:
    def setup(self):
        self.factors = ['f1', 'f2', 'f3']
        self.periods = [0, 1, 2, 3]
        self.nperiods = len(self.periods)
        self.stagemap = [0, 0, 1, 1]

        m = {}
        m['f1'] = [['m1', 'm2']] * 4
        m['f2'] = [['m3', 'm4', 'm5']] * 4
        m['f3'] = [['m5', 'm6']] * 4
        self.measurements = m

        self.anchoring = True
        self.anch_outcome = 'a'
        self.anchored_factors = ['f1', 'f3']

        n = {}
        n['f1'] = [['m1', 2]] * 4
        n['f2'] = [['m3', 1], ['m3', 1], ['m3', 1], ['m5', 5]]
        n['f3'] = [['m5', 4]] * 4
        self.normalizations = n

        self._is_dummy = Mock(side_effect=cycle([False, False, True]))

        cols = self.factors + ['{}_norm_value'.format(f) for f in self.factors]
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

        for factor in self.factors:
            norm_col = '{}_norm_value'.format(factor)
            for t, (n_meas, n_val) in enumerate(self.normalizations[factor]):
                df.loc[(t, n_meas), norm_col] = n_val

        for t in self.periods:
            df.loc[t, 'stage'] = self.stagemap[t]

        update_type = ['linear', 'linear', 'probit'] * 9
        df['update_type'] = np.array(update_type[:len(df)])

        df['purpose'] = np.array(['measurement'] * 24 + ['anchoring'])

        df.loc[(3, 'a'), ['f1', 'f3']] = 1
        self.res = df

    def test_update_info_without_order(self):
        calculated = msc.update_info(self).to_dict()
        expected = self.res.to_dict()
        assert_equal(calculated, expected)

    def test_that_anchoring_comes_list(self):
        purposes = list(msc.update_info(self)['purpose'])
        assert_equal(purposes, ['measurement'] * 24 + ['anchoring'])


class TestEnoughMeasurementsArray:
    def setup(self):

        data = np.array([
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],

            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],

            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 0, 1, 2],

            [3, 1, 1, 2],
            [3, 1, 1, 2],
            [4, 1, 0, 3],
            [4, 0, 1, 3]])
        df = DataFrame(data=data, columns=['period', 'f1', 'f2', 'stage'])
        df['purpose'] = 'measurement'
        df['variable'] = ['m_{}'.format(i) for i in range(len(data))]
        df.set_index(['period', 'variable'], inplace=True)

        self.update_info = Mock(return_value=df)
        self.stagemap = [0, 1, 2, 2, 3]
        self.factors = ['f1', 'f2']
        self.periods = [0, 1, 2, 3, 4]

    def test_enough_measurements_array_no_special_transition_functions(self):
        self.transition_names = ['ces', 'bla']
        res = np.array([
            [True, True],
            [False, True],
            [False, False],
            [False, False]])
        aae(msc.enough_measurements_array(self), res)

    def test_enough_measurements_array_ar1_and_constant(self):
        self.transition_names = ['ar1', 'constant']
        res = np.array([[True, True]] * 4)
        aae(msc.enough_measurements_array(self), res)

    def test_type_of_enough_measurements_array(self):
        self.transition_names = ['ces', 'bla']
        calculated = msc.enough_measurements_array(self)
        assert isinstance(calculated, np.ndarray)


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

    def test_new_transition_parameters_with_merge(self):
        self.check_enough_measurements = 'merge_stages'
        res = np.array(
            [[-1, 1, 1, 1, 1],
             [-1, 0, 1, 1, 0],
             [-1, 0, 0, 1, 1]])

        aae(msc.new_trans_coeffs(self), res)

    def test_new_transition_parameters_without_merge(self):
        self.check_enough_measurements = 'no_check'
        res = np.array(
            [[-1, 1, 1, 1, 1],
             [-1, 0, 1, 1, 1],
             [-1, 0, 1, 1, 1]])
        aae(msc.new_trans_coeffs(self), res)


if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
