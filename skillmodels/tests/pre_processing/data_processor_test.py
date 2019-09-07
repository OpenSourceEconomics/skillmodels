import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal as aae
from pandas import DataFrame

from skillmodels.pre_processing.data_processor import DataProcessor as dc
from skillmodels.pre_processing.data_processor import pre_process_data


def test_pre_process_data():
    df = pd.DataFrame(data=np.arange(10).reshape(10, 1), columns=["var"])
    df["period"] = [1, 2, 3, 2, 3, 4, 2, 4, 3, 1]
    df["id"] = [1, 1, 1, 3, 3, 3, 4, 4, 5, 5]
    df.set_index(["id", "period"], inplace=True)

    exp = pd.DataFrame()
    period = [0, 1, 2, 3] * 4
    id_ = np.arange(4).repeat(4)
    data = [
        0,
        1,
        2,
        np.nan,
        np.nan,
        3,
        4,
        5,
        np.nan,
        6,
        np.nan,
        7,
        9,
        np.nan,
        8,
        np.nan,
    ]
    data = np.column_stack([period, id_, data])
    exp = pd.DataFrame(data=data, columns=["__period__", "__id__", "var"])
    exp.set_index(["__id__", "__period__"], inplace=True)

    res = pre_process_data(df)

    assert res["var"].equals(exp["var"])


class TestCData:
    def setup(self):
        self.controls = [["c1", "c2"], ["c1", "c2", "c3"]]
        df = DataFrame(
            data=np.array([0] * 5 + [1] * 5).reshape(10, 1), columns=["__period__"]
        )
        df["c1"] = [f"c1_t0_{i}" for i in range(5)] + [f"c1_t1_{i}" for i in range(5)]

        df["c2"] = [f"c2_t0_{i}" for i in range(5)] + [f"c2_t1_{i}" for i in range(5)]

        df["c3"] = ["blubb"] * 5 + [f"c3_t1_{i}" for i in range(5)]

        self.data = df

        self.periods = [0, 1]

        self.obs_to_keep = np.array([True, True, True, False, True])

    def test_c_data_with_constants(self):
        res1 = [
            [1.0, "c1_t0_0", "c2_t0_0"],
            [1.0, "c1_t0_1", "c2_t0_1"],
            [1.0, "c1_t0_2", "c2_t0_2"],
            [1.0, "c1_t0_4", "c2_t0_4"],
        ]

        res2 = [
            [1.0, "c1_t1_0", "c2_t1_0", "c3_t1_0"],
            [1.0, "c1_t1_1", "c2_t1_1", "c3_t1_1"],
            [1.0, "c1_t1_2", "c2_t1_2", "c3_t1_2"],
            [1.0, "c1_t1_4", "c2_t1_4", "c3_t1_4"],
        ]
        res = [res1, res2]

        calculated = dc.c_data(self)
        for i, calc in enumerate(calculated):
            aae(calc, np.array(res[i], dtype=object))


class TestYData:
    def setup(self):
        self.periods = [0, 1, 2, 3]
        self.different_meas = ["m1", "m2", "m3", "m4", "m5", "m6"]

        ind_tuples = []
        for t in self.periods:
            ind_tuples += [(t, d) for d in self.different_meas]
        ind_tuples.append((3, "a"))

        index = pd.MultiIndex.from_tuples(ind_tuples, names=["__period__", "variable"])

        dat = np.zeros((25, 1))
        df = DataFrame(data=dat, columns=["some_col"], index=index)
        self.update_info = df

        self.nupdates = 25
        self.nobs = 3

        self.obs_to_keep = np.array([True, True, False, True])

    def test_y_data_focus_on_rows(self):
        data = np.tile(np.arange(6), 16).reshape(16, 6)
        self.data = DataFrame(data=data, columns=self.different_meas)
        self.data["__period__"] = np.arange(4).repeat(4)
        self.data["a"] = 10

        res = np.vstack([np.arange(6).repeat(3).reshape(6, 3)] * 4)
        res = np.vstack([res, np.ones(3) * 10])

        aae(dc.y_data(self), res)

    def test_y_data_focus_on_columns(self):
        df = DataFrame(data=np.arange(4).repeat(4), columns=["__period__"])
        for var in self.different_meas + ["a"]:
            df[var] = np.arange(16)
        self.data = df

        res = np.vstack(
            [
                np.array([[0, 1, 3]] * 6),
                np.array([[4, 5, 7]] * 6),
                np.array([[8, 9, 11]] * 6),
                np.array([[12, 13, 15]] * 7),
            ]
        )

        aae(dc.y_data(self), res)
