from nose.tools import assert_equal, assert_raises
import pandas as pd
from pandas import DataFrame
import numpy as np
from unittest.mock import Mock, call, patch  # noqa
from itertools import cycle
from numpy.testing import assert_array_equal as aae
from skillmodels.pre_processing.model_spec_processor import ModelSpecProcessor as msp
from pandas.testing import assert_frame_equal


class TestTransitionEquationNames:
    def setup(self):
        self.factors = ["f1", "f2", "f3"]
        names = ["linear", "ces", "ar1"]
        self._facinf = {
            factor: {"trans_eq": {"name": name}}
            for factor, name in zip(self.factors, names)
        }

    def test_transition_equation_names(self):
        msp._transition_equation_names(self)
        assert_equal(self.transition_names, ["linear", "ces", "ar1"])


class TestTransitionEquationIncludedFactors:
    def setup(self):
        self.factors = ["f1", "f2"]
        self._facinf = {
            factor: {"trans_eq": {"included_factors": []}} for factor in self.factors
        }
        self._facinf["f1"]["trans_eq"]["included_factors"] = ["f2", "f1"]
        self._facinf["f2"]["trans_eq"]["included_factors"] = ["f2"]
        self.nfac = 2

    def test_transition_equation_included_factors(self):
        msp._transition_equation_included_factors(self)
        assert_equal(self.included_factors, [["f1", "f2"], ["f2"]])

    def test_transition_equation_included_factor_positions(self):
        msp._transition_equation_included_factors(self)
        assert_equal(self.included_positions, [[0, 1], [1]])


class TestVariableCheckMethods:
    def setup(self):
        df1 = DataFrame(data=np.zeros((5, 2)), columns=["__period__", "var1"])
        df1.loc[1, "var1"] = 1
        df2 = DataFrame(data=np.ones((5, 2)), columns=["__period__", "var2"])
        df2.loc[1, "var2"] = 5
        self.data = pd.concat([df1, df2], axis=0, sort=True)
        self.missing_variables = "drop_variable"
        self.variables_without_variance = "drop_variable"
        self.model_name = "model"
        self.dataset_name = "dataset"

    def test_present_where_true(self):
        assert_equal(msp._present(self, "var1", 0), True)

    def test_present_where_false_no_raise(self):
        assert_equal(msp._present(self, "var1", 1), False)

    def test_present_where_false__raise(self):
        self.missing_variables = "raise_error"
        assert_raises(KeyError, msp._present, self, "var1", 1)

    def test_has_variance_where_true(self):
        assert_equal(msp._has_variance(self, "var1", 0), True)

    def test_has_variance_where_false_non_missing(self):
        self.data.loc[1, "var1"] = 0
        assert_equal(msp._has_variance(self, "var1", 0), False)

    def test_has_variance_where_false_missing(self):
        assert_equal(msp._has_variance(self, "var2", 0), False)


class TestCleanMesaurementSpecifications:
    def setup(self):
        self.periods = [0, 1]
        inf = {"f1": {}, "f2": {}}
        inf["f1"]["measurements"] = [["m1", "m2", "m3", "m4"]] * 2
        inf["f2"]["measurements"] = [["m5", "m6", "m7", "m8"]] * 2
        self._facinf = inf
        self.factors = sorted(list(self._facinf.keys()))
        self.transition_names = ["log_ces", "blubb"]

    def test_clean_measuremnt_specifications_nothing_to_clean(self):
        self._present = Mock(return_value=True)
        self._has_variance = Mock(return_value=True)
        res = {}
        res["f1"] = self._facinf["f1"]["measurements"]
        res["f2"] = self._facinf["f2"]["measurements"]
        msp._clean_measurement_specifications(self)
        assert_equal(self.measurements, res)

    def test_clean_measurement_specifications_half_of_variables_missing(self):
        self._present = Mock(side_effect=cycle([True, False]))
        self._has_variance = Mock(return_value=True)
        res = {}
        res["f1"] = [["m1", "m3"]] * 2
        res["f2"] = [["m5", "m7"]] * 2
        msp._clean_measurement_specifications(self)
        assert_equal(self.measurements, res)

    def test_clean_measurement_specs_half_of_variables_without_variance(self):
        self._present = Mock(return_value=True)
        self._has_variance = Mock(side_effecet=cycle([False, True]))
        res = {}
        res["f1"] = [["m2", "m4"]]
        res["f2"] = [["m6", "m8"]]


class TestCleanControlSpecifications:
    def setup(self):
        self._timeinf = {"controls": [["c1", "c2"], ["c1", "c2"]]}
        self.periods = [0, 1]
        self.nperiods = 2
        self._present = Mock(return_value=True)
        self._has_variance = Mock(return_value=True)
        self.controls_with_missings = "drop_variable"
        self.model_name = "model"
        self.dataset_name = "data"
        cols = ["__period__", "c1", "c2"]
        dat = np.zeros((10, 3))
        dat[5:, 0] = 1
        self.data = DataFrame(data=dat, columns=cols)

    def test_clean_control_specs_nothing_to_clean(self):
        msp._clean_controls_specification(self)
        res = [["c1", "c2"], ["c1", "c2"]]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_variable(self):
        self._present = Mock(side_effect=[True, False, True, True])
        msp._clean_controls_specification(self)
        res = [["c1"], ["c1", "c2"]]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_observations_drop_variable(self):
        self.data.loc[2, "c2"] = np.nan
        msp._clean_controls_specification(self)
        res = [["c1"], ["c1", "c2"]]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.ones(5, dtype=bool))

    def test_clean_control_specs_missing_observation_drop_observation(self):
        self.data.loc[2, "c2"] = np.nan
        self.controls_with_missings = "drop_observations"
        msp._clean_controls_specification(self)
        res = [["c1", "c2"], ["c1", "c2"]]
        assert_equal(self.controls, res)
        aae(self.obs_to_keep, np.array([True, True, False, True, True]))

    def test_clean_control_specs_missing_observations_error(self):
        self.data.loc[2, "c2"] = np.nan
        self.controls_with_missings = "raise_error"
        assert_raises(ValueError, msp._clean_controls_specification, self)


class TestCheckAndCleanNormalizations:
    def setup(self):
        self.periods = [0, 1]
        self._facinf = {"f1": {"measurements": [["m1", "m2", "m3", "m4"]] * 2}}
        self.f1_norm_list = [{"m1": 1}, {"m1": 1}]
        self.factors = sorted(list(self._facinf.keys()))
        self.measurements = {"f1": [["m1", "m2", "m3", "m4"]] * 2}
        self.model_name = "model"
        self.dataset_name = "data"
        self.nperiods = len(self.periods)

    def test_check_normalizations_lists(self):
        assert_raises(
            DeprecationWarning,
            msp._check_and_clean_normalizations_list,
            self,
            "f1",
            [["m1", 1], ["m1", 1]],
            "loadings",
        )

    def test_check_normalizations_no_error_dictionaries(self):
        result = msp._check_and_clean_normalizations_list(
            self, "f1", self.f1_norm_list, "loadings"
        )
        assert_equal(result, [{"m1": 1}, {"m1": 1}])

    def test_check_normalizations_not_specified_error(self):
        f1_norm_list = [{"m10": 1}, {"m1": 1}]
        assert_raises(
            KeyError,
            msp._check_and_clean_normalizations_list,
            self,
            "f1",
            f1_norm_list,
            "loadings",
        )

    def test_check_normalizations_dropped_error(self):
        self.measurements = {"f1": [["m2", "m3", "m4"]] * 2}
        assert_raises(
            KeyError,
            msp._check_and_clean_normalizations_list,
            self,
            "f1",
            self.f1_norm_list,
            "loadings",
        )

    def test_check_normalizations_invalid_length_of_list(self):
        assert_raises(
            AssertionError,
            msp._check_and_clean_normalizations_list,
            self,
            "f1",
            self.f1_norm_list * 2,
            "loadings",
        )

    def test_check_normalizations_invalid_length_of_sublst(self):
        f1_norm_list = [["m1"], ["m1", 1]]
        assert_raises(
            AssertionError,
            msp._check_and_clean_normalizations_list,
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

        self.factors = sorted(list(self._facinf.keys()))
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

        msp._check_and_fill_normalization_specification(self)
        assert_equal(self.normalizations, res)


def factor_uinfo():

    dat = [
        [0, "m1", 1, 0, 0],
        [0, "m2", 1, 0, 0],
        [0, "m3", 0, 1, 0],
        [0, "m4", 0, 1, 0],
        [0, "m5", 0, 1, 1],
        [0, "m6", 0, 0, 1],
        [1, "m1", 1, 0, 0],
        [1, "m2", 1, 0, 0],
        [1, "m3", 0, 1, 0],
        [1, "m4", 0, 1, 0],
        [1, "m5", 0, 1, 1],
        [1, "m6", 0, 0, 1],
        [2, "m1", 1, 0, 0],
        [2, "m2", 1, 0, 0],
        [2, "m3", 0, 1, 0],
        [2, "m4", 0, 1, 0],
        [2, "m5", 0, 1, 1],
        [2, "m6", 0, 0, 1],
        [3, "m1", 1, 0, 0],
        [3, "m2", 1, 0, 0],
        [3, "m3", 0, 1, 0],
        [3, "m4", 0, 1, 0],
        [3, "m5", 0, 1, 1],
        [3, "m6", 0, 0, 1],
        [3, "a", 1, 0, 1],
    ]
    cols = ["period", "variable", "f1", "f2", "f3"]
    df = DataFrame(data=dat, columns=cols)
    df.set_index(["period", "variable"], inplace=True)
    return df


def norm_uinfo():
    n = np.nan
    f = False
    t = True
    dat = [
        [0, "m1", 2.0, 0.0, 0.0, 0.0, 1.0, t, t, t],
        [0, "m2", 0, 0, 0, n, n, f, f, f],
        [0, "m3", 0, 1, 0, 1, n, t, t, f],
        [0, "m4", 0, 0, 0, n, n, f, f, f],
        [0, "m5", 0, 0, 4, 4, n, t, t, f],
        [0, "m6", 0, 0, 0, n, n, f, f, f],
        [1, "m1", 2, 0, 0, 0, 1, t, t, t],
        [1, "m2", 0, 0, 0, n, n, f, f, f],
        [1, "m3", 0, 1, 0, 1, n, t, t, f],
        [1, "m4", 0, 0, 0, n, n, f, f, f],
        [1, "m5", 0, 0, 4, 4, n, t, t, f],
        [1, "m6", 0, 0, 0, n, n, f, f, f],
        [2, "m1", 2, 0, 0, 0, 1, t, t, t],
        [2, "m2", 0, 0, 0, n, n, f, f, f],
        [2, "m3", 0, 1, 0, 1, n, t, t, f],
        [2, "m4", 0, 0, 0, n, n, f, f, f],
        [2, "m5", 0, 0, 4, 4, n, t, t, f],
        [2, "m6", 0, 0, 0, n, n, f, f, f],
        [3, "m1", 2, 0, 0, 0, 1, t, t, t],
        [3, "m2", 0, 0, 0, n, n, f, f, f],
        [3, "m3", 0, 0, 0, n, n, f, f, f],
        [3, "m4", 0, 0, 0, n, n, f, f, f],
        [3, "m5", 0, 5, 4, 4, n, t, t, f],
        [3, "m6", 0, 0, 0, n, n, f, f, f],
        [3, "a", 0, 0, 0, n, n, f, f, f],
    ]
    cols = [
        "period",
        "variable",
        "f1_loading_norm_value",
        "f2_loading_norm_value",
        "f3_loading_norm_value",
        "intercept_norm_value",
        "variance_norm_value",
        "has_normalized_loading",
        "has_normalized_intercept",
        "has_normalized_variance",
    ]

    df = DataFrame(data=dat, columns=cols)
    df.set_index(["period", "variable"], inplace=True)
    return df


def stage_uinfo():
    ind = factor_uinfo().index
    dat = [0] * 12 + [1] * 13
    return pd.Series(index=ind, data=dat, name="stage")


def purpose_uinfo():
    ind = factor_uinfo().index
    dat = ["measurement"] * 24 + ["anchoring"]
    return pd.Series(index=ind, data=dat, name="purpose")


def type_uinfo():
    ind = factor_uinfo().index
    dat = ["linear", "linear", "probit"] * 8 + ["linear"]
    return pd.Series(data=dat, index=ind, name="type")


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
        self.anchored_factors = ["f1", "f3"]
        self.anchoring = True

    def test_factor_update_info(self):
        calc = msp._factor_update_info(self)
        exp = factor_uinfo()
        assert_frame_equal(calc, exp, check_dtype=False)

    def test_that_anchoring_comes_last(self):
        calc = msp._factor_update_info(self)
        meas = list(calc.index.get_level_values("variable"))
        assert meas[-1] == self.anch_outcome


class TestNormalizationUpdateInfo:
    def setup(self):
        self._factor_update_info = Mock(return_value=factor_uinfo())
        self.factors = ["f1", "f2", "f3"]

        n = {"f1": {}, "f2": {}, "f3": {}}

        n["f1"]["loadings"] = [{"m1": 2}] * 4
        n["f2"]["loadings"] = [{"m3": 1}, {"m3": 1}, {"m3": 1}, {"m5": 5}]
        n["f3"]["loadings"] = [{"m5": 4}] * 4

        n["f1"]["intercepts"] = [{"m1": 0}] * 4
        # set normalization of m5 to 5 to check correct raising
        n["f2"]["intercepts"] = [{"m3": 1}, {"m3": 1}, {"m3": 1}, {"m5": 4}]
        n["f3"]["intercepts"] = [{"m5": 4}] * 4

        n["f1"]["variances"] = [{"m1": 1}, {"m1": 1}, {"m1": 1}, {"m1": 1}]
        n["f2"]["variances"] = [{}] * 4
        n["f3"]["variances"] = [{}] * 4

        self.normalizations = n

    def test_loading_normalizations(self):
        normcols = ["{}_loading_norm_value".format(f) for f in self.factors]
        calc = msp._normalization_update_info(self)[normcols]
        exp = norm_uinfo()[normcols]
        for col in calc.columns:
            assert calc[col].equals(exp[col])

    def test_invalid_intercept_normalizations(self):
        self.normalizations["f2"]["intercepts"][-1] = {"m5": 5}
        assert_raises(AssertionError, msp._normalization_update_info, self)

    def test_valid_intercept_normalizations(self):
        calc = msp._normalization_update_info(self)["intercept_norm_value"]
        exp = norm_uinfo()["intercept_norm_value"]
        assert calc.equals(exp)

    def test_variance_normalizations(self):
        calc = msp._normalization_update_info(self)["variance_norm_value"]
        exp = norm_uinfo()["variance_norm_value"]
        assert calc.equals(exp)

    def test_update_info_has_normalized_loading(self):
        calc = msp._normalization_update_info(self)["has_normalized_loading"]
        exp = norm_uinfo()["has_normalized_loading"]
        assert calc.equals(exp)

    def test_update_info_has_normalized_intercept(self):
        calc = msp._normalization_update_info(self)["has_normalized_intercept"]
        exp = norm_uinfo()["has_normalized_intercept"]
        assert calc.equals(exp)

    def test_update_info_has_normalized_variance(self):
        calc = msp._normalization_update_info(self)["has_normalized_variance"]
        exp = norm_uinfo()["has_normalized_variance"]
        assert calc.equals(exp)


class TestStageUpdateInfo:
    def setup(self):
        self._factor_update_info = Mock(return_value=factor_uinfo())
        self.stagemap = [0, 0, 1, 1]

    def test_update_info_stages(self):
        calc = msp._stage_udpate_info(self)
        exp = stage_uinfo()
        assert calc.equals(exp)


class TestPurposeUpdateInfo:
    def setup(self):
        self.anchoring = True
        self.anch_outcome = "a"
        self.nperiods = 4
        self._factor_update_info = Mock(return_value=factor_uinfo())

    def test_update_info_purpose(self):
        calc = msp._purpose_update_info(self)
        exp = purpose_uinfo()
        assert calc.equals(exp)


class TestNewTransitionCoeffs:
    def setup(self):
        self.stages = [0, 1, 2]
        self.nstages = len(self.stages)

        self.factors = ["f1", "f2", "f3", "f4", "f5"]
        self.nfac = len(self.factors)
        self.transition_names = ["constant", "ar1", "ces", "bla", "blubb"]
        arr = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=bool)
        self.enough_measurements_array = arr

    def test_new_transition_parameters(self):
        res = np.array([[-1, 1, 1, 1, 1], [-1, 0, 1, 1, 1], [-1, 0, 1, 1, 1]])
        aae(msp.new_trans_coeffs(self), res)
