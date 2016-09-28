from nose.tools import assert_equal, assert_raises, assert_almost_equal
from unittest.mock import Mock, call, patch
from skillmodels import SkillModel as smo
import numpy as np
from pandas import DataFrame
from numpy.testing import assert_array_equal as aae
from numpy.testing import assert_array_almost_equal as aaae
import json
import pandas as pd
from pandas.util.testing import assert_series_equal
from pandas.util.testing import assert_frame_equal


class TestGeneralParamsSlice:
    def setup(self):
        self.param_counter = 10

    def test_general_params_slice(self):
        assert_equal(smo._general_params_slice(self, 10), slice(10, 20))

    def test_general_params_slice_via_usage(self):
        sl = smo._general_params_slice(self, 10)
        aae(np.arange(20)[sl], np.arange(10) + 10)

    def test_side_effect_of_general_params_slice_on_param_counter(self):
        sl = smo._general_params_slice(self, 10) # noqa
        assert_equal(self.param_counter, 20)


class TestDeltasRelatedMethods:
    def setup(self):
        self.periods = [0, 1, 2]
        self.controls = [['c1', 'c2'], ['c1', 'c2', 'c3'], ['c3', 'c4']]
        self.factors = ['f1', 'f2']
        cols = ['name', 'f1_loading_norm_value', 'f2_loading_norm_value']
        dat = np.zeros((13, 3))
        dat[(0, 6, 11), 1] = 5
        dat[(1, 8, 12), 2] = 3
        df = DataFrame(data=dat, columns=cols)
        df['period'] = [0] * 6 + [1] * 3 + [2] * 4
        df['has_normalized_intercept'] = [
            True, False, False, True, False, False, True, True, False, True,
            False, False, False]
        df['intercept_norm_value'] = [
            3, np.nan, np.nan, 4, np.nan, np.nan, 5, 6, np.nan, 7, np.nan,
            np.nan, np.nan]

        df['name'] = ['m{}'.format(number) for number in range(13)]
        df.set_index(['period', 'name'], inplace=True)
        self.update_info = df

    def test_initial_deltas_without_controls_besides_constant(self):
        self.controls = [[], [], []]
        exp1 = np.array([[3], [0], [0], [4], [0], [0]])
        exp2 = np.array([[5], [6], [0]])
        exp3 = np.array([[7], [0], [0], [0]])
        expected = [exp1, exp2, exp3]
        calculated = smo._initial_deltas(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)

    def test_initial_deltas_with_controls_and_constants(self):
        exp1 = np.array([
            [3, 0, 0], [0, 0, 0], [0, 0, 0], [4, 0, 0], [0, 0, 0], [0, 0, 0]])
        exp2 = np.array([[5, 0, 0, 0], [6, 0, 0, 0], [0, 0, 0, 0]])
        exp3 = np.array([[7, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        expected = [exp1, exp2, exp3]

        calculated = smo._initial_deltas(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)

    def test_deltas_bool_without_controls_besides_constants(self):
        exp1 = np.array([False, True, True, False, True, True]).reshape(6, 1)
        exp2 = np.array([False, False, True]).reshape(3, 1)
        exp3 = np.array([False, True, True, True]).reshape(4, 1)
        expected = [exp1, exp2, exp3]
        self.controls = [[], [], []]

        calculated = smo._deltas_bool(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)

    def test_deltas_bool_with_controls_and_constant(self):
        exp1 = np.ones((6, 3), dtype=bool)
        exp1[(0, 3), 0] = False

        exp2 = np.ones((3, 4), dtype=bool)
        exp2[(0, 1), 0] = False

        exp3 = np.ones((4, 3), dtype=bool)
        exp3[0, 0] = False
        expected = [exp1, exp2, exp3]

        calculated = smo._deltas_bool(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)

    def test_params_slice_deltas(self):
        arr0 = np.ones((4, 2), dtype=bool)
        arr1 = np.ones((6, 3), dtype=bool)
        arr1[(0, 1), :] = 0
        self._deltas_bool = Mock(return_value=[arr0, arr1, arr0])
        self._general_params_slice = Mock()
        smo._params_slice_for_deltas(self, 'short')
        self._general_params_slice.assert_has_calls(
            [call(8), call(12), call(8)])

    def test_deltas_names_without_controls_beside_constant(self):
        self.controls = [[], [], []]
        d_boo = [np.ones((6, 1), dtype=bool), np.ones((3, 1), dtype=bool),
                 np.ones((4, 1), dtype=bool)]
        self._deltas_bool = Mock(return_value=d_boo)
        fs = 'delta__{}__{}__{}'
        expected_names = \
            expected_names = \
            [fs.format(0, 'm0', 'constant'),
             fs.format(0, 'm1', 'constant'),
             fs.format(0, 'm2', 'constant'),
             fs.format(0, 'm3', 'constant'),
             fs.format(0, 'm4', 'constant'),
             fs.format(0, 'm5', 'constant'),
             fs.format(1, 'm6', 'constant'),
             fs.format(1, 'm7', 'constant'),
             fs.format(1, 'm8', 'constant'),
             fs.format(2, 'm9', 'constant'),
             fs.format(2, 'm10', 'constant'),
             fs.format(2, 'm11', 'constant'),
             fs.format(2, 'm12', 'constant')]

        assert_equal(smo._deltas_names(self, params_type='short'),
                     expected_names)

    def test_deltas_names_with_controls_and_constant(self):
        self.add_constant = True

        d_boo = [np.ones((6, 2), dtype=bool), np.ones((3, 3), dtype=bool),
                 np.ones((4, 2), dtype=bool)]
        for i in range(3):
            d_boo[i][0, 0] = False
        self._deltas_bool = Mock(return_value=d_boo)
        fs = 'delta__{}__{}__{}'
        expected_names = \
            [fs.format(0, 'm0', 'c1'), fs.format(0, 'm0', 'c2'),
             fs.format(0, 'm1', 'constant'), fs.format(0, 'm1', 'c1'),
             fs.format(0, 'm1', 'c2'),
             fs.format(0, 'm2', 'constant'), fs.format(0, 'm2', 'c1'),
             fs.format(0, 'm2', 'c2'),
             fs.format(0, 'm3', 'constant'), fs.format(0, 'm3', 'c1'),
             fs.format(0, 'm3', 'c2'),
             fs.format(0, 'm4', 'constant'), fs.format(0, 'm4', 'c1'),
             fs.format(0, 'm4', 'c2'),
             fs.format(0, 'm5', 'constant'), fs.format(0, 'm5', 'c1'),
             fs.format(0, 'm5', 'c2'),
             fs.format(1, 'm6', 'c1'), fs.format(1, 'm6', 'c2'),
             fs.format(1, 'm6', 'c3'),
             fs.format(1, 'm7', 'constant'), fs.format(1, 'm7', 'c1'),
             fs.format(1, 'm7', 'c2'), fs.format(1, 'm7', 'c3'),
             fs.format(1, 'm8', 'constant'), fs.format(1, 'm8', 'c1'),
             fs.format(1, 'm8', 'c2'), fs.format(1, 'm8', 'c3'),
             fs.format(2, 'm9', 'c3'), fs.format(2, 'm9', 'c4'),
             fs.format(2, 'm10', 'constant'), fs.format(2, 'm10', 'c3'),
             fs.format(2, 'm10', 'c4'),
             fs.format(2, 'm11', 'constant'), fs.format(2, 'm11', 'c3'),
             fs.format(2, 'm11', 'c4'),
             fs.format(2, 'm12', 'constant'), fs.format(2, 'm12', 'c3'),
             fs.format(2, 'm12', 'c4')]

        assert_equal(smo._deltas_names(self, params_type='short'),
                     expected_names)


class TestPsiRelatedMethods:
    def setup(self):
        self.factors = ['f1', 'f2', 'f3']
        self.nfac = len(self.factors)
        self.endog_factor = 'f3'

    def test_initial_psi(self):
        aae(smo._initial_psi(self), np.ones(3))

    def test_psi_bool(self):
        aae(smo._psi_bool(self), np.array([True, True, False]))

    def test_params_slice_for_psi(self):
        self._general_params_slice = Mock()
        smo._params_slice_for_psi(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(2)])

    def test_psi_names(self):
        assert_equal(smo._psi_names(self, params_type='short'),
                     ['psi__f1', 'psi__f2'])


class TestHRelatedMethods:
    def setup(self):
        self.factors = ['f1', 'f2']
        cols = self.factors + [
            'f1_loading_norm_value', 'f2_loading_norm_value']
        self.nfac = 2
        dat = np.zeros((20, 4))
        dat[(0, 1, 6, 8, 11, 16, 18), 0] = 1
        dat[(2, 3, 6, 7, 12, 13, 19), 1] = 1
        dat[(1, 8), 2] = 5
        dat[(6, 13, 19), 3] = 3

        res = np.zeros((20, 2), dtype=bool)
        res[(0, 6, 11, 16, 18), 0] = True
        res[(2, 3, 7, 12), 1] = True
        self.res_bool = res

        self.exp_init_H = np.zeros((20, 2))
        self.exp_init_H[(1, 8), 0] = 5
        self.exp_init_H[(6, 13, 19), 1] = 3

        df = DataFrame(data=dat, columns=cols)
        self.update_info = df

    def test_initial_H(self):
        aae(smo._initial_H(self), self.exp_init_H)

    def test_H_bool(self):
        self._initial_H = Mock(return_value=self.exp_init_H)
        aae(smo._H_bool(self), self.res_bool)

    def test_params_slice_for_H(self):
        self._H_bool = Mock(return_value=self.res_bool)
        self._general_params_slice = Mock()
        smo._params_slice_for_H(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(9)])

    def test_helpers_for_h_transformation(self):
        self.endog_correction = True
        self.endog_factor = ['f1']
        res1 = np.zeros(20, dtype=bool)
        for i in (0, 1, 6, 8, 11, 16, 18):
            res1[i] = True
        res2 = np.zeros((7, 1))
        res3 = np.zeros((7, 2))

        calc1, calc2, calc3 = smo._helpers_for_H_transformation_with_psi(self)
        aae(calc1, res1)
        aae(calc2, res2)
        aae(calc3, res3)

    def test_H_names(self):
        self.factors = ['f1', 'f2']
        df = DataFrame(data=np.ones((6, 1)), columns=['col'])
        df['name'] = ['m{}'.format(number) for number in range(6)]
        df['period'] = [0, 0, 0, 1, 1, 1]
        df.set_index(['period', 'name'], inplace=True)

        self.update_info = df

        boo = np.zeros((6, 2))
        boo[(0, 1, 3), 0] = True
        boo[(3, 4), 1] = True
        self._H_bool = Mock(return_value=boo)

        fs = 'H__{}__{}__{}'

        expected_names = [
            fs.format(0, 'f1', 'm0'), fs.format(0, 'f1', 'm1'),
            fs.format(1, 'f1', 'm3'), fs.format(1, 'f2', 'm3'),
            fs.format(1, 'f2', 'm4')]

        assert_equal(smo._H_names(self, params_type='short'), expected_names)


class TestRRelatedMethods:
    def setup(self):
        self.nupdates = 12
        self.estimator = 'chs'

        df = DataFrame(data=np.zeros((12, 1)), columns=['col'])
        df['period'] = np.array([0] * 5 + [1] * 7)
        df['name'] = ['m{}'.format(i) for i in range(12)]
        df.set_index(['period', 'name'], inplace=True)
        self.update_info = df

        self.bounds_distance = 0.001
        self.lower_bound = np.empty(100, dtype=object)
        self.lower_bound[:] = None

    def test_initial_R(self):
        aae(smo._initial_R(self), np.zeros(12))

    def test_params_slice_for_R(self):
        self._general_params_slice = Mock()
        smo._params_slice_for_R(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(12)])

    def test_set_bounds_for_R_not_robust(self):
        self.robust_bounds = False
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[10: 30] = 0.0
        smo._set_bounds_for_R(self, slice(10, 30))
        aae(self.lower_bound, expected)

    def test_set_bounds_for_R_robust(self):
        self.robust_bounds = True
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[10: 30] = 0.001
        smo._set_bounds_for_R(self, slice(10, 30))
        aae(self.lower_bound, expected)

    def test_R_names(self):
        periods = [0] * 5 + [1] * 7
        names = ['m{}'.format(i) for i in range(12)]
        expected = ['R__{}__{}'.format(p, m) for p, m in zip(periods, names)]
        assert_equal(smo._R_names(self, params_type='short'), expected)


class TestQRelatedMethods:
    def setup(self):
        self.nstages = 2
        self.stages = [0, 1]
        self.nfac = 5
        self.factors = ['f{}'.format(i) for i in range(1, 6)]
        params_info = np.ones((2, 5))
        params_info[:, 0] = -1
        params_info[1, (2, 4)] = 0

        self.new_trans_coeffs = params_info

        self.exp_bool = np.zeros((2, 5, 5), dtype=bool)
        self.exp_bool[0, :, :] = np.diag([False, True, True, True, True])
        self.exp_bool[1, :, :] = np.diag([False, True, False, True, False])

        self.bounds_distance = 0.001
        self.lower_bound = np.empty(100, dtype=object)
        self.lower_bound[:] = None

    def test_initial_q(self):
        aae(smo._initial_Q(self), np.zeros((2, 5, 5)))

    def test_q_bool(self):
        aae(smo._Q_bool(self), self.exp_bool)

    def test_params_slice_for_q(self):
        self._Q_bool = Mock(return_value=self.exp_bool)
        self._general_params_slice = Mock()
        smo._params_slice_for_Q(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(6)])

    def test_q_replacements(self):
        expected = [[(1, 2, 2), (0, 2, 2)], [(1, 4, 4), (0, 4, 4)]]
        assert_equal(smo._Q_replacements(self), expected)

    def test_q_names(self):
        expected = ['Q__0__f2', 'Q__0__f3', 'Q__0__f4', 'Q__0__f5',
                    'Q__1__f2', 'Q__1__f4']
        assert_equal(smo._Q_names(self, params_type='short'), expected)

    def test_set_bounds_for_Q_not_robust(self):
        self.robust_bounds = False
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[10: 30] = 0.0
        smo._set_bounds_for_Q(self, slice(10, 30))
        aae(self.lower_bound, expected)

    def test_set_bounds_for_Q_robust(self):
        self.robust_bounds = True
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[10: 30] = 0.001
        smo._set_bounds_for_Q(self, slice(10, 30))
        aae(self.lower_bound, expected)


class TestXZeroRelatedMethods:
    def setup(self):
        self.nemf = 3
        self.nobs = 100
        self.nfac = 4
        self.order_X_zeros = 2
        self.factors = ['f1', 'f2', 'f3', 'f4']

    def test_initial_X_zero(self):
        res1, res2 = smo._initial_X_zero(self)
        aae(res1, np.zeros((100, 3, 4)))
        aae(res2, np.zeros((300, 4)))

    def test_that_initial_X_zeros_are_views_on_same_memory(self):
        res1, res2 = smo._initial_X_zero(self)
        res1[:] = 1
        aae(res2, np.ones((300, 4)))

    def test_X_zero_filler(self):
        aae(smo._X_zero_filler(self), np.zeros((3, 4)))

    def test_params_slice_for_X_zero(self):
        self._general_params_slice = Mock()
        smo._params_slice_for_X_zero(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(12)])

    def test_x_zero_replacements(self):
        expected = [[(1, 2), (0, 2)], [(2, 2), (1, 2)]]
        assert_equal(smo._X_zero_replacements(self), expected)

    def test_set_bounds_for_X_zero(self):
        self.lower_bound = np.empty(100, dtype=object)
        self.lower_bound[:] = None

        params_slice = slice(10, 22)

        expected = self.lower_bound.copy()
        expected[[16, 20]] = 0

        smo._set_bounds_for_X_zero(self, params_slice=params_slice)

        aae(self.lower_bound, expected)

    def test_x_zero_names_short_params(self):
        expected = [
            'X_zero__0__f1', 'X_zero__0__f2', 'X_zero__0__f3', 'X_zero__0__f4',
            'X_zero__1__f1', 'X_zero__1__f2', 'diff_X_zero__1__f3',
            'X_zero__1__f4',
            'X_zero__2__f1', 'X_zero__2__f2', 'diff_X_zero__2__f3',
            'X_zero__2__f4']
        assert_equal(smo._X_zero_names(self, params_type='short'), expected)

    def test_x_zero_names_long_params(self):
        expected = [
            'X_zero__0__f1', 'X_zero__0__f2', 'X_zero__0__f3', 'X_zero__0__f4',
            'X_zero__1__f1', 'X_zero__1__f2', 'X_zero__1__f3', 'X_zero__1__f4',
            'X_zero__2__f1', 'X_zero__2__f2', 'X_zero__2__f3', 'X_zero__2__f4']
        assert_equal(smo._X_zero_names(self, params_type='long'), expected)


class TestWZeroRelatedMethods:
    def setup(self):
        self.nemf = 4
        self.nobs = 100

    def test_initial_w_zero(self):
        aae(smo._initial_W_zero(self),
            np.ones((self.nobs, self.nemf)) / 4)

    def test_params_slice_w_zero(self):
        self._general_params_slice = Mock()
        smo._params_slice_for_W_zero(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(4)])

    def test_w_zero_names(self):
        expected = ['W_zero__0', 'W_zero__1', 'W_zero__2', 'W_zero__3']
        assert_equal(smo._W_zero_names(self, params_type='short'), expected)


class TestPZeroRelatedMethods:
    def setup(self):
        self.nemf = 2
        self.nobs = 100
        self.nfac = 4
        self.helper = np.array([[True, True, True, True],
                                [False, True, True, True],
                                [False, False, True, True],
                                [False, False, False, True]])
        self.lower_bound = np.empty(100, dtype=object)
        self.lower_bound[:] = None
        self.bound_indices = [10, 14, 17, 19, 20, 24, 27, 29]
        self.bounds_distance = 0.001
        self.estimator = 'chs'

    def test_initial_P_zero_no_square_root_filters(self):
        self.square_root_filters = False
        res1, res2 = smo._initial_P_zero(self)
        aae(res1, np.zeros((100, 2, 4, 4)))
        aae(res2, np.zeros((200, 4, 4)))

    def test_initial_P_zero_square_root_filters(self):
        self.square_root_filters = True
        res1, res2 = smo._initial_P_zero(self)
        aae(res1, np.zeros((100, 2, 5, 5)))
        aae(res2, np.zeros((200, 5, 5)))

    def test_P_zero_filler_unrestricted_P_zeros(self):
        self.restrict_P_zeros = False
        aae(smo._P_zero_filler(self), np.zeros((2, 4, 4)))

    def test_P_zero_filler_restricted_P_zeros(self):
        self.restrict_P_zeros = True
        aae(smo._P_zero_filler(self), np.zeros((1, 4, 4)))

    def test_P_zero_filler_bool_unrestricted(self):
        self._P_zero_filler = Mock(return_value=np.zeros((2, 4, 4)))
        self.restrict_P_zeros = False
        expected = np.zeros((2, 4, 4), dtype=bool)
        expected[:] = self.helper
        aae(smo._P_zero_bool(self), expected)

    def test_P_zero_filler_bool_restricted(self):
        self._P_zero_filler = Mock(return_value=np.zeros((1, 4, 4)))
        self.restrict_P_zeros = True
        expected = np.zeros((1, 4, 4), dtype=bool)
        expected[:] = self.helper
        aae(smo._P_zero_bool(self), expected)

    def test_params_slice_P_zero_unrestricted(self):
        self.restrict_P_zeros = False
        self._general_params_slice = Mock()
        smo._params_slice_for_P_zero(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(20)])

    def test_params_slice_P_zero_restricted(self):
        self.restrict_P_zeros = True
        self._general_params_slice = Mock()
        smo._params_slice_for_P_zero(self, params_type='short')
        self._general_params_slice.assert_has_calls([call(10)])

    def test_set_bounds_P_zero_unrestricted_not_robust(self):
        self.robust_bounds = False
        self.restrict_P_zeros = False
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[self.bound_indices] = 0
        smo._set_bounds_for_P_zero(self, slice(10, 30))
        aae(self.lower_bound, expected)

    def test_set_bounds_P_zero_unrestricted_robust(self):
        self.robust_bounds = True
        self.restrict_P_zeros = False
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[self.bound_indices] = 0.001
        smo._set_bounds_for_P_zero(self, slice(10, 30))
        aae(self.lower_bound, expected)

    def test_set_bounds_P_zero_restricted_not_robust(self):
        self.robust_bounds = False
        self.restrict_P_zeros = True
        expected = np.zeros(100, dtype=object)
        expected[:] = None
        expected[self.bound_indices[:4]] = 0.0
        smo._set_bounds_for_P_zero(self, slice(10, 20))
        aae(self.lower_bound, expected)

    def test_set_bounds_P_zero_invalid_params_slice(self):
        self.robust_bounds = False
        self.restrict_P_zeros = False
        assert_raises(
            AssertionError, smo._set_bounds_for_P_zero, self, slice(10, 15))

    def test_P_zero_names_short(self):
        self.nemf = 1
        self.nfac = 3
        self.factors = ['f1', 'f2', 'f3']
        self.restrict_P_zeros = False
        self.cholesky_of_P_zero = False
        fs = 'cholesky_P_zero__0__{}__{}'
        expected = [fs.format('f1', 'f1'), fs.format('f1', 'f2'),
                    fs.format('f1', 'f3'), fs.format('f2', 'f2'),
                    fs.format('f2', 'f3'), fs.format('f3', 'f3')]

        assert_equal(smo._P_zero_names(self, params_type='short'), expected)

    def test_P_zero_names_long(self):
        self.nemf = 1
        self.nfac = 3
        self.factors = ['f1', 'f2', 'f3']
        self.restrict_P_zeros = False
        self.cholesky_of_P_zero = False
        fs = 'P_zero__0__{}__{}'
        expected = [fs.format('f1', 'f1'), fs.format('f1', 'f2'),
                    fs.format('f1', 'f3'), fs.format('f2', 'f2'),
                    fs.format('f2', 'f3'), fs.format('f3', 'f3')]
        assert_equal(smo._P_zero_names(self, params_type='long'), expected)


class TestTransCoeffsRelatedMethods:
    def setup(self):
        self.factors = ['f1', 'f2']
        self.nfac = 2
        self.transition_names = ['first_func', 'second_func']
        self.included_factors = [['f1', 'f2'], ['f2']]
        self.stages = [0, 1]
        self.nstages = 2
        new_params = np.array([[1, 1], [0, 1]])
        self.new_trans_coeffs = new_params

        self.lower_bound = np.empty(100, dtype=object)
        self.lower_bound[:] = None
        self.upper_bound = self.lower_bound.copy()

    @patch('skillmodels.estimation.skill_model.tf')
    def test_initial_trans_coeffs(self, mock_tf):
        mock_tf.nr_coeffs_first_func.return_value = 3
        mock_tf.nr_coeffs_second_func.return_value = 10
        expected = [np.zeros((2, 3)), np.zeros((2, 10))]
        initials = smo._initial_trans_coeffs(self)
        for i, e in zip(initials, expected):
            aae(i, e)

    @patch('skillmodels.estimation.skill_model.tf')
    def test_params_slices_for_trans_coeffs(self, mock_tf):
        mock_tf.nr_coeffs_first_func.return_value = 3
        mock_tf.nr_coeffs_second_func.return_value = 10
        self._general_params_slice = Mock(
            side_effect=[slice(0, 3), slice(3, 13), slice(13, 23)])
        res = smo._params_slice_for_trans_coeffs(self, params_type='short')
        self._general_params_slice.assert_has_calls(
            [call(3), call(10), call(10)])
        mock_tf.nr_coeffs_first_func.assert_has_calls(
            [call(included_factors=['f1', 'f2'], params_type='short')])
        mock_tf.nr_coeffs_second_func.assert_has_calls(
            [call(included_factors=['f2'], params_type='short')] * 2)
        assert_equal([[slice(0, 3)] * 2, [slice(3, 13), slice(13, 23)]], res)

    @patch('skillmodels.estimation.skill_model.tf')
    def test_set_bounds_for_trans_coeffs(self, mock_tf):
        lb = np.array([0, None, None], dtype=object)
        ub = np.array([None, None, 1], dtype=object)
        mock_tf.bounds_first_func.return_value = (lb, ub)
        del mock_tf.bounds_second_func
        sl = [[slice(0, 3)] * 2, [slice(3, 13), slice(13, 23)]]

        expected_lb = self.lower_bound.copy()
        expected_lb[0] = 0

        expected_ub = self.upper_bound.copy()
        expected_ub[2] = 1

        smo._set_bounds_for_trans_coeffs(self, sl)

        aae(self.lower_bound, expected_lb)
        aae(self.upper_bound, expected_ub)

    @patch('skillmodels.estimation.skill_model.tf')
    def test_trans_coeffs_names(self, mock_tf):
        mock_tf.nr_coeffs_second_func.return_value = 2
        mock_tf.coeff_names_first_func.return_value = ['epsilon', 'psi', 'pi']
        del mock_tf.coeff_names_second_func
        expected = [
            'epsilon', 'psi', 'pi', 'trans_coeff__0__f2__0',
            'trans_coeff__0__f2__1', 'trans_coeff__1__f2__0',
            'trans_coeff__1__f2__1']

        assert_equal(smo._trans_coeffs_names(self, params_type='short'),
                     expected)


class TestTransformTransitionParamsFuncs:
    def setup(self):
        self.factors = ['f1', 'f2']
        self.transition_names = ['first_func', 'second_func']

    @patch('skillmodels.estimation.skill_model.tf')
    def test_transform_trans_coeffs_funcs(self, mock_tf):
        del mock_tf.transform_coeffs_second_func
        assert_equal(smo._transform_trans_coeffs_funcs(self),
                     ['transform_coeffs_first_func', None])


class TestParamsSlices:
    def setup(self):
        self.params_quants = ['a', 'b']
        self._params_slice_for_a = Mock(return_value=slice(0, 3))
        self._params_slice_for_b = Mock(return_value=slice(3, 5))

    def test_params_slices(self):
        assert_equal(smo.params_slices(self, params_type='short'),
                     {'a': slice(0, 3), 'b': slice(3, 5)})


class TestLenParams:
    def setup(self):
        self.params_quants = ['a', 'b']
        self.params_slices = Mock(
            return_value={'a': slice(0, 3), 'b': slice(3, 5)})

    def test_len_params(self):
        assert_equal(smo.len_params(self, params_type='short'), 5)


class TestBoundsList:
    def setup(self):
        slices = {'a': slice(0, 3), 'b': slice(3, 5)}
        self.params_slices = Mock(return_value=slices)
        self.len_params = Mock(return_value=5)

        self.params_quants = ['a', 'b']

    # mock function
    def _set_bounds_for_b(self, params_slice):
        self.lower_bound[params_slice] = 99
        self.upper_bound[params_slice][0] = 100

    def test_bounds_list(self):
        expected = [(None, None)] * 3 + [(99, 100), (99, None)]
        assert_equal(smo.bounds_list(self), expected)


class TestParamNames:
    def setup(self):
        self.params_quants = ['a', 'b']
        self._a_names = Mock(return_value=['a1', 'a2'])
        self._b_names = Mock(return_value=['b1', 'b2', 'b3'])
        self.len_params = Mock(return_value=5)

    def test_param_names(self):
        assert_equal(smo.param_names(self, params_type='short'),
                     ['a1', 'a2', 'b1', 'b2', 'b3'])

    def test_param_names_invalid(self):
        self.len_params = Mock(return_value=6)
        assert_raises(
            AssertionError, smo.param_names, self, params_type='short')


class TestExpandparams:
    def setup(self):
        self.params_quants = [
            'deltas', 'trans_coeffs', 'X_zero', 'P_zero']
        slices_dict = {
            'short':
                {'deltas': slice(0, 5),
                 'trans_coeffs': slice(5, 12),
                 'X_zero': slice(12, 15),
                 'P_zero': slice(15, 18)},
            'long':
                {'deltas': slice(0, 5),
                 'trans_coeffs': slice(5, 14),
                 'X_zero': slice(14, 17),
                 'P_zero': slice(17, 20)}}

        self.params_slices = Mock(
            side_effect=lambda params_type: slices_dict[params_type])

        len_dict = {'short': 18, 'long': 20}
        self.len_params = Mock(
            side_effect=lambda params_type: len_dict[params_type])
        self._P_zero_filler = Mock()
        self._P_zero_bool = Mock()
        self.cholesky_of_P_zero = True
        self._X_zero_filler = Mock()
        self._X_zero_replacements = Mock()
        self._initial_trans_coeffs = Mock()
        self._transform_trans_coeffs_funcs = Mock()
        self.included_factors = []

    # @patch('src.model_code.skill_model.pp')
    # def test_expand_params(self, mock_pt):
    #     mock_pt.transform_params_for_X_zero.return_value = np.arange(3)
    #     mock_pt.transform_params_for_trans_coeffs.return_value = np.ones(9)
    #     mock_pt.transform_params_for_P_zero.return_value = np.ones(3) * 17
    #     expected = np.array([0] * 5 + [1] * 9 + [0, 1, 2] + [17] * 3)
    #     aae(smo.expandparams(self, np.zeros(18)), expected)


class TestGenerateStartParams:
    def setup(self):
        self.params_quants = [
            'deltas', 'P_zero', 'W_zero', 'trans_coeffs']
        self.nemf = 3
        self.nfac = 2
        self.stages = [0, 1]
        self.factors = ['f1', 'f2']
        self.included_factors = self.factors
        self.transition_names = ['some_func', 'some_func']
        self.start_values_per_quantity = {
            'deltas': 5, 'P_zero_off_diags': 0, 'P_zero_diags': 0.5}
        self.restrict_P_zeros = False

        slices = {'deltas': slice(0, 4), 'P_zero': slice(4, 13),
                  'W_zero': slice(13, 16),
                  'trans_coeffs': [[slice(16, 17), slice(17, 18)],
                                   [slice(18, 19), slice(19, 20)]]}
        self.params_slices = Mock(return_value=slices)
        self.len_params = Mock(return_value=20)

    @patch('skillmodels.estimation.skill_model.tf')
    def test_generate_start_params(self, mock_tf):
        mock_tf.start_values_some_func.return_value = np.ones(1) * 7.7
        expected = np.array(
            [5] * 4 + [0.5, 0, 0.5] * 3 + [1 / 3] * 3 + [7.7] * 4)
        aae(smo.generate_start_params(self), expected)


class TestSigmaWeightsAndScalingFactor:
    def setup(self):
        self.nemf = 2
        self.nobs = 10
        self.nfac = 4
        self.kappa = 1.5

        # these test results have been calculated with the sigma_point
        # function of the filterpy library
        with open('sigma_points_from_filterpy.json') as f:
            self.fixtures = json.load(f)

    def test_julier_sigma_weight_construction(self):
        expected_sws = self.fixtures['julier_wm']
        aae(smo.sigma_weights(self)[0], expected_sws)

    def test_julier_scaling_factor(self):
        expected_sf = 2.34520787991
        assert_almost_equal(smo.sigma_scaling_factor(self), expected_sf)


class TestLikelihoodArgumentsDict:
    def setup(self):
        pass


class TestAllVariablesForIVEquations:
    def setup(self):
        self.measurements = {
            'f1': [['y01', 'y02'], ['y11', 'y12'], []],
            'f2': [['y04', 'y05'], ['y14', 'y15'], []],
            'f3': [['y07', 'y08'], [], []]}

        self.factors = ['f1', 'f2', 'f3']
        self.included_factors = [['f1', 'f3'], ['f2', 'f3'], []]
        self.transition_names = ['blubb', 'blubb', 'constant']
        self.periods = [0, 1, 2]

    def test_all_variables_for_iv_equations_constant_factor(self):
        calc_meas_list = smo.all_variables_for_iv_equations(
            self, 1, 'f1', 'test')
        expected_meas_list = [
            ['y11_test', 'y12_test'],
            ['y07_copied_test', 'y08_copied_test']]
        assert_equal(calc_meas_list, expected_meas_list)

    def test_all_variables_for_iv_equations_non_constant(self):
        calc_meas_list = smo.all_variables_for_iv_equations(self, 1, 'f2', '')
        expected_meas_list = [
            ['y14', 'y15'], ['y07_copied', 'y08_copied']]
        assert_equal(calc_meas_list, expected_meas_list)

    def test_indepvar_permutations(self):
        ret_val = [['y1', 'y2'], ['y3', 'y4']]
        self.all_variables_for_iv_equations = Mock(return_value=ret_val)

        expected_xs = [
            ['y1', 'y3'], ['y1', 'y4'], ['y2', 'y3'], ['y2', 'y4']]
        calc_xs = smo.variable_permutations_for_iv_equations(self, 1, 1)[0]
        assert_equal(calc_xs, expected_xs)

    def test_instrument_permutations(self):
        ret_val = [['y1_resid', 'y2_resid'], ['y3_resid', 'y4_resid']]
        self.all_variables_for_iv_equations = Mock(return_value=ret_val)

        expected_zs = [
            [['y2'], ['y4']],
            [['y2'], ['y3']],
            [['y1'], ['y4']],
            [['y1'], ['y3']]]
        calc_zs = smo.variable_permutations_for_iv_equations(self, 1, 1)[1]

        assert_equal(calc_zs, expected_zs)


class TestNumberOfIVParameters:
    def setup(self):
        self.factors = ['f1', 'f2', 'f3']
        self.transition_names = ['bla', 'bla', 'blubb']
        ret = (['correct', 'wrong'], ['correct2', 'wrong2'])
        self.variable_permutations_for_iv_equations = Mock(return_value=ret)

    @patch('skillmodels.estimation.skill_model.tf')
    def test_number_of_iv_parameters(self, mock_tf):
        mock_tf.iv_formula_bla.return_value = ('1 + 2 + 3 + 4', '_')
        expected_param_nr = 4
        calc_res = smo.number_of_iv_parameters(self, 'f1')
        assert_equal(calc_res, expected_param_nr)

    @patch('skillmodels.estimation.skill_model.tf')
    def test_right_calls(self, mock_tf):
        mock_tf.iv_formula_bla.return_value = ('1 + 2 + 3 + 4', '_')
        smo.number_of_iv_parameters(self, 'f1')
        mock_tf.iv_formula_bla.assert_has_calls([call('correct', 'correct2')])


class TestExtendedMeasCoeffs:
    def setup(self):
        self.factors = ['f1', 'f2']
        self.transition_names = ['linear', 'constant']
        self.measurements = {
            'f1': [['y01', 'y02'], ['y11', 'y12']],
            'f2': [['y03', 'y04'], []]}

        coeffs = np.arange(0.6, 3.0, 0.2).reshape((6, 2))
        cols = ['loadings', 'intercepts']
        index_tuples = [(0, 'y01'), (0, 'y02'), (0, 'y03'), (0, 'y04'),
                        (1, 'y11'), (1, 'y12')]
        self.index = pd.MultiIndex.from_tuples(index_tuples)
        self.storage_df = pd.DataFrame(coeffs, index=self.index, columns=cols)

    def test_extended_meas_coeffs_no_constant_factor_and_intercepts_case(self):
        coeff_type = 'intercepts'
        calc_intercepts = smo.extended_meas_coeffs(self, coeff_type, 0)
        expected_intercepts = pd.Series(
            data=[0.8, 1.2, 1.6, 2.0],
            name='intercepts', index=['y01', 'y02', 'y03', 'y04'])
        assert_series_equal(calc_intercepts, expected_intercepts)

    def test_extendend_meas_coeffs_constant_factor_and_loadings_case(self):
        coeff_type = 'loadings'
        calc_loadings = smo.extended_meas_coeffs(self, coeff_type, 1)
        expected_loadings = pd.Series(
            data=[2.2, 2.6, 1.4, 1.8],
            name='loadings',
            index=['y11', 'y12', 'y03_copied', 'y04_copied'])
        assert_series_equal(calc_loadings, expected_loadings)


class TestResidualMeasurements:
    def setup(self):
        intercepts = pd.Series(
            [3.0, 2.0], name='intercepts', index=['m2', 'm1'])
        loadings = pd.Series(
            [2.0, 0.5], name='loadings', index=['m1', 'm2'])
        self.extended_meas_coeffs = Mock(side_effect=[loadings, intercepts])

        d = pd.DataFrame(data=np.array([[5, 4], [3, 2]]), columns=['m1', 'm2'])

        self.y_data = ['dummy', d, 'dummy']

    def test_residual_measurements(self):
        expected_data = np.array([
            [1.5, 2],
            [0.5, -2]])
        expected_resid = pd.DataFrame(
            expected_data, columns=['m1_resid', 'm2_resid'])
        calc_resid = smo.residual_measurements(self, period=1)
        assert_frame_equal(calc_resid, expected_resid)


class TestWANorminfoDict:
    def setup(self):
        n = {}
        n['f1'] = {'loadings': [['y1', 4], ['y2', 5], ['y3', 6]],
                   'intercepts': [['y4', 7], ['y5', 8]]}

        df = pd.DataFrame(data=[[None]] * 3, columns=['f1'])
        self.identified_restrictions = {
            'coeff_sum_value': df, 'trans_intercept_value': df}
        self.normalizations = n
        self.stagemap = [0, 1, 2, 2]

    def test_wa_norminfo_dict(self):
        expected = {'loading_norminfo': ['y2', 5],
                    'intercept_norminfo': ['y5', 8]}
        calculated = smo.model_coeffs_from_iv_coeffs_args_dict(self, 1, 'f1')
        assert_equal(calculated, expected)


class TestBSMethods:
    def setup(self):
        self.bootstrap_samples = [
            ['id_0', 'id_1', 'id_1'],
            ['id_0', 'id_1', 'id_0'],
            ['id_1', 'id_0', 'id_0']]
        self.bootstrap_nreps = 3
        self.model_name = 'test_check_bs_sample'
        self.dataset_name = 'test_data'
        self.person_identifier = 'id'
        self.period_identifier = 'period'
        self.periods = [0, 1, 2]
        self.data = pd.DataFrame(
            data=np.array([
                [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]).T,
            columns=['period', 'arange'])
        self.data['id'] = pd.Series(dtype='str', data=[
            'id_0', 'id_0', 'id_0', 'id_1', 'id_1', 'id_1', 'id_2', 'id_2', 'id_2']) # noqa
        self.bootstrap_sample_size = 3
        self.nobs = 3
        self.save_bootstrap_params = False
        self.bootstrap_nprocesses = 2

    def test__check_bs_samples_accepts_iterable(self):
        smo._check_bs_samples(self)

    def test_rejects_non_iterable(self):
        self.bootstrap_samples = 240
        assert_raises(
            AssertionError, smo._check_bs_samples, self)

    def test_raises_error_with_unknown_identifier(self):
        self.bootstrap_samples[2][0] = 'a'
        assert_raises(
            AssertionError, smo._check_bs_samples, self)

    def test_only_bootstrap_samples_with_enough_samples(self):
        self.bootstrap_nreps = 10
        assert_raises(
            AssertionError, smo._check_bs_samples, self)

    def test__select_bs_samples(self):
        np.random.seed(495)
        expected_samples = [
            ['id_1', 'id_1', 'id_1'],
            ['id_0', 'id_2', 'id_2'],
            ['id_2', 'id_2', 'id_1']]
        calc_samples = smo._select_bs_samples(self)
        assert_equal(calc_samples, expected_samples)

    def test__generate_bootstrap_data(self):
        expected_data = pd.DataFrame(
            data=np.array([
                [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]]).T,
            columns=['period', 'arange'])
        expected_data['id'] = [
            'id_0', 'id_0', 'id_0', 'id_1', 'id_1', 'id_1', 'id_1', 'id_1', 'id_1'] # noqa
        calc_data = smo._generate_bootstrap_data(self, 0)
        assert_frame_equal(calc_data, expected_data)

    def _bs_fit(self, rep, params, params_type):
        return rep * np.ones(3)

    def test_all_bootstrap_params(self):
        calc_params = smo.all_bootstrap_params(
            self, params=np.ones(3), params_type='short')
        expected_params = pd.DataFrame(
            data=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            columns=['rep_0', 'rep_1', 'rep_2'])
        assert_frame_equal(calc_params, expected_params)


class TestBootstrapParamsToConfIn:
    def setup(self):
        np.random.seed(3904)
        self.bootstrap_params = pd.DataFrame(
            np.random.normal(loc=0, scale=1, size=(3, 100000)))

    def test_bootstrap_parmas_to_conf_int(self):
        expected_conf_int = pd.DataFrame(
            data=[[-1.96, 1.96], [-1.96, 1.96], [-1.96, 1.96]],
            columns=['lower_ci_bound', 'upper_ci_bound'])
        calc_conf_int = smo.bootstrap_params_to_conf_int(
            self, self.bootstrap_params)
        aaae(calc_conf_int, expected_conf_int, decimal=2)


class TestBootstrapCovMatrix:
    def setup(self):
        np.random.seed(94355)
        self.expected_cov = np.array([
            [3.0, 0.5, 0.4],
            [0.5, 2.0, 0.1],
            [0.4, 0.1, 1.0]])
        self.params = np.arange(3)
        self.params_type = 'short'
        self.model_name = 'test_bootstrap_cov_matrix'
        self.dataset_name = 'data_for_testing_covmatrix'

    def len_params(self, params_type):
        return 3

    def all_bootstrap_params(self, params, params_type):
        return pd.DataFrame(
            np.random.multivariate_normal(mean=self.params,
                                          cov=self.expected_cov, size=10000))

    def test_bootstrap_cov_matrix(self):
        calc_cov = smo.bootstrap_cov_matrix(
            self, self.params, self.params_type)
        aaae(calc_cov, self.expected_cov, decimal=1)

if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
