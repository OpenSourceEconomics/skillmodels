from skillmodels.pre_processing.data_processor import DataProcessor as dc
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy.testing import assert_array_equal as aae


class TestCData:
    def setup(self):
        self.controls = [['c1', 'c2'], ['c1', 'c2', 'c3']]
        df = DataFrame(data=np.array([0] * 5 + [1] * 5).reshape(10, 1),
                       columns=['period'])
        df['c1'] = ['c1_t0_{}'.format(i) for i in range(5)] + \
                   ['c1_t1_{}'.format(i) for i in range(5)]

        df['c2'] = ['c2_t0_{}'.format(i) for i in range(5)] + \
                   ['c2_t1_{}'.format(i) for i in range(5)]

        df['c3'] = ['blubb'] * 5 + ['c3_t1_{}'.format(i) for i in range(5)]

        self.data = df

        self.periods = [0, 1]

        self.obs_to_keep = np.array([True, True, True, False, True])

    def test_c_data_without_constants(self):
        self.add_constant = False
        res1 = [['c1_t0_0', 'c2_t0_0'], ['c1_t0_1', 'c2_t0_1'],
                ['c1_t0_2', 'c2_t0_2'], ['c1_t0_4', 'c2_t0_4']]

        res2 = [['c1_t1_0', 'c2_t1_0', 'c3_t1_0'],
                ['c1_t1_1', 'c2_t1_1', 'c3_t1_1'],
                ['c1_t1_2', 'c2_t1_2', 'c3_t1_2'],
                ['c1_t1_4', 'c2_t1_4', 'c3_t1_4']]
        res = [res1, res2]

        calculated = dc.c_data(self)
        for i, calc in enumerate(calculated):
            aae(calc, np.array(res[i]))

    def test_c_data_with_constants(self):
        self.add_constant = True
        res1 = [[1.0, 'c1_t0_0', 'c2_t0_0'], [1.0, 'c1_t0_1', 'c2_t0_1'],
                [1.0, 'c1_t0_2', 'c2_t0_2'], [1.0, 'c1_t0_4', 'c2_t0_4']]

        res2 = [[1.0, 'c1_t1_0', 'c2_t1_0', 'c3_t1_0'],
                [1.0, 'c1_t1_1', 'c2_t1_1', 'c3_t1_1'],
                [1.0, 'c1_t1_2', 'c2_t1_2', 'c3_t1_2'],
                [1.0, 'c1_t1_4', 'c2_t1_4', 'c3_t1_4']]
        res = [res1, res2]

        calculated = dc.c_data(self)
        for i, calc in enumerate(calculated):
            aae(calc, np.array(res[i], dtype=object))


class TestYData:
    def setup(self):
        self.periods = [0, 1, 2, 3]
        self.different_meas = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

        ind_tuples = []
        for t in self.periods:
            ind_tuples += [(t, d) for d in self.different_meas]
        ind_tuples.append((3, 'a'))

        index = pd.MultiIndex.from_tuples(
            ind_tuples, names=['period', 'variable'])

        dat = np.zeros((25, 1))
        df = DataFrame(data=dat, columns=['some_col'], index=index)
        self.update_info = df

        self.nupdates = 25
        self.nobs = 3

        self.obs_to_keep = np.array([True, True, False, True])

    def test_y_data_focus_on_rows(self):
        data = np.tile(np.arange(6), 16).reshape(16, 6)
        self.data = DataFrame(data=data, columns=self.different_meas)
        self.data['period'] = np.arange(4).repeat(4)
        self.data['a'] = 10

        res = np.vstack([np.arange(6).repeat(3).reshape(6, 3)] * 4)
        res = np.vstack([res, np.ones(3) * 10])

        aae(dc.y_data(self), res)

    def test_y_data_focus_on_columns(self):
        df = DataFrame(data=np.arange(4).repeat(4), columns=['period'])
        for var in self.different_meas + ['a']:
            df[var] = np.arange(16)
        self.data = df

        res = np.vstack(
            [np.array([[0, 1, 3]] * 6), np.array([[4, 5, 7]] * 6),
             np.array([[8, 9, 11]] * 6), np.array([[12, 13, 15]] * 7)])

        aae(dc.y_data(self), res)



if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
