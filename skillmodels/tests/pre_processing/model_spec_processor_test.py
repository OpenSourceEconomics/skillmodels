from unittest.mock import Mock

import numpy as np
import pandas as pd
from nose.tools import assert_equal
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor


class TestTransitionEquationNames:
    def setup(self):
        self.factors = ("f1", "f2", "f3")
        names = ["linear", "ces", "ar1"]
        self._facinf = {
            factor: {"trans_eq": {"name": name}}
            for factor, name in zip(self.factors, names)
        }

    def test_transition_equation_names(self):
        ModelSpecProcessor._transition_equation_names(self)
        assert_equal(self.transition_names, ("linear", "ces", "ar1"))


class TestTransitionEquationIncludedFactors:
    def setup(self):
        self.factors = ("f1", "f2")
        self._facinf = {
            factor: {"trans_eq": {"included_factors": []}} for factor in self.factors
        }
        self._facinf["f1"]["trans_eq"]["included_factors"] = ["f2", "f1"]
        self._facinf["f2"]["trans_eq"]["included_factors"] = ["f2"]
        self.nfac = 2

    def test_transition_equation_included_factors(self):
        ModelSpecProcessor._transition_equation_included_factors(self)
        assert_equal(self.included_factors, (("f1", "f2"), ("f2",)))

    def test_transition_equation_included_factor_positions(self):
        ModelSpecProcessor._transition_equation_included_factors(self)
        exp_positions = [np.array([0, 1]), np.array([1])]
        for pos, exp_pos in zip(self.included_positions, exp_positions):
            assert_array_equal(pos, exp_pos)


class TestCleanMesaurementSpecifications:
    def setup(self):
        self.periods = (0, 1)
        inf = {"f1": {}, "f2": {}}
        inf["f1"]["measurements"] = [["m1", "m2", "m3", "m4"]] * 2
        inf["f2"]["measurements"] = [["m5", "m6", "m7", "m8"]] * 2
        self._facinf = inf
        self.factors = tuple(sorted(self._facinf.keys()))
        self.transition_names = ("log_ces", "blubb")
        cols = ["__period__"] + [f"m{i}" for i in range(1, 9)]
        dat = np.vstack([np.zeros(9), np.ones(9)])
        self.data = pd.DataFrame(columns=cols, data=dat)

    def test_check_measurements(self):
        res = {}
        res["f1"] = self._facinf["f1"]["measurements"]
        res["f2"] = self._facinf["f2"]["measurements"]
        ModelSpecProcessor._check_measurements(self)
        assert_equal(self.measurements, res)


class TestCleanControlSpecifications:
    def setup(self):
        self._timeinf = {"controls": [["c1", "c2"], ["c1", "c2"]]}
        self.periods = [0, 1]
        self.nperiods = 2
        self.model_name = "model"
        self.dataset_name = "data"
        cols = ["__period__", "c1", "c2", "m1", "m2"]
        dat = np.zeros((10, 5))
        dat[5:, 0] = 1
        self.data = DataFrame(data=dat, columns=cols)
        self.measurements = {"f1": [["m1", "m2"]] * 2, "f2": [["m1", "m2"]] * 2}
        self.factors = ["f1", "f2"]

    def test_clean_control_specs_nothing_to_clean(self):
        ModelSpecProcessor._clean_controls_specification(self)
        res = (("c1", "c2"), ("c1", "c2"))
        assert_equal(self.controls, res)


class TestCheckAndCleanNormalizations:
    def setup(self):
        self.periods = [0, 1]
        self._facinf = {"f1": {"measurements": [["m1", "m2", "m3", "m4"]] * 2}}
        self.f1_norm_list = [{"m1": 1}, {"m1": 1}]
        self.factors = sorted(self._facinf.keys())
        self.measurements = {"f1": [["m1", "m2", "m3", "m4"]] * 2}
        self.model_name = "model"
        self.dataset_name = "data"
        self.nperiods = len(self.periods)

    def test_check_normalizations_no_error_dictionaries(self):
        result = ModelSpecProcessor._check_and_clean_normalizations_list(
            self, "f1", self.f1_norm_list, "loadings"
        )
        assert_equal(result, [{"m1": 1}, {"m1": 1}])

    def test_check_normalizations_not_specified_error(self):
        f1_norm_list = [{"m10": 1}, {"m1": 1}]
        assert_raises(
            KeyError,
            ModelSpecProcessor._check_and_clean_normalizations_list,
            self,
            "f1",
            f1_norm_list,
            "loadings",
        )

    def test_check_normalizations_invalid_length_of_list(self):
        assert_raises(
            AssertionError,
            ModelSpecProcessor._check_and_clean_normalizations_list,
            self,
            "f1",
            self.f1_norm_list * 2,
            "loadings",
        )

    def test_check_normalizations_invalid_length_of_sublst(self):
        f1_norm_list = [["m1"], ["m1", 1]]
        assert_raises(
            AssertionError,
            ModelSpecProcessor._check_and_clean_normalizations_list,
            self,
            "f1",
            f1_norm_list,
            "loadings",
        )


class TestCheckAndFillNormalizationSpecifications:
    def setup(self):
        self._facinf = {
            "f1": {
                "normalizations": {
                    "loadings": [{"a": 1}, {"a": 1}, {"a": 1}],
                    "intercepts": [{"a": 1}, {"a": 1}, {"a": 1}],
                    "variances": [{"a": 1}, {"a": 1}, {"a": 1}],
                }
            },
            "f2": {},
        }

        self.nperiods = 3

        self.factors = sorted(self._facinf.keys())
        self._check_and_clean_normalizations_list = Mock(
            return_value=[{"a": 1}, {"a": 1}, {"a": 1}]
        )

    def test_check_and_fill_normalization_specifications(self):
        res = {
            "f1": {
                "loadings": [{"a": 1}, {"a": 1}, {"a": 1}],
                "intercepts": [{"a": 1}, {"a": 1}, {"a": 1}],
                "variances": [{"a": 1}, {"a": 1}, {"a": 1}],
            },
            "f2": {
                "loadings": [{}, {}, {}],
                "intercepts": [{}, {}, {}],
                "variances": [{}, {}, {}],
            },
        }

        ModelSpecProcessor._check_and_fill_normalization_specification(self)
        assert_equal(self.normalizations, res)


def factor_uinfo():

    dat = [
        [0, "m1", 1, 0, 0],
        [0, "m2", 1, 0, 0],
        [0, "m3", 0, 1, 0],
        [0, "m4", 0, 1, 0],
        [0, "m5", 0, 1, 1],
        [0, "m6", 0, 0, 1],
        [0, "a_f1", 1, 0, 0],
        [1, "m1", 1, 0, 0],
        [1, "m2", 1, 0, 0],
        [1, "m3", 0, 1, 0],
        [1, "m4", 0, 1, 0],
        [1, "m5", 0, 1, 1],
        [1, "m6", 0, 0, 1],
        [1, "a_f1", 1, 0, 0],
        [2, "m1", 1, 0, 0],
        [2, "m2", 1, 0, 0],
        [2, "m3", 0, 1, 0],
        [2, "m4", 0, 1, 0],
        [2, "m5", 0, 1, 1],
        [2, "m6", 0, 0, 1],
        [2, "a_f1", 1, 0, 0],
        [3, "m1", 1, 0, 0],
        [3, "m2", 1, 0, 0],
        [3, "m3", 0, 1, 0],
        [3, "m4", 0, 1, 0],
        [3, "m5", 0, 1, 1],
        [3, "m6", 0, 0, 1],
        [3, "a_f1", 1, 0, 0],
    ]
    cols = ["period", "variable", "f1", "f2", "f3"]
    df = DataFrame(data=dat, columns=cols)
    df.set_index(["period", "variable"], inplace=True)
    return df


class TestFactorUpdateInfo:
    def setup(self):
        self.periods = [0, 1, 2, 3]
        self.nperiods = len(self.periods)
        self.factors = ["f1", "f2", "f3"]
        self.nfac = len(self.factors)
        self.measurements = {
            "f1": [["m1", "m2"]] * 4,
            "f2": [["m3", "m4", "m5"]] * 4,
            "f3": [["m5", "m6"]] * 4,
        }
        self.anch_outcome = "a"
        self.anchored_factors = ["f1"]
        self.anchoring = True

    def test_factor_update_info(self):
        calc = ModelSpecProcessor._factor_update_info(self)
        exp = factor_uinfo()
        assert_frame_equal(calc, exp, check_dtype=False)


def purpose_uinfo():
    ind = factor_uinfo().index
    dat = (["measurement"] * 6 + ["anchoring"]) * 4
    return pd.Series(index=ind, data=dat, name="purpose")


class TestPurposeUpdateInfo:
    def setup(self):
        self.anchoring = True
        self.anchored_factors = ["f1"]
        self.anch_outcome = "a"
        self.nperiods = 4
        self._factor_update_info = Mock(return_value=factor_uinfo())
        self.periods = (0, 1, 2, 3)

    def test_update_info_purpose(self):
        calc = ModelSpecProcessor._purpose_update_info(self)
        exp = purpose_uinfo()
        assert calc.equals(exp)
