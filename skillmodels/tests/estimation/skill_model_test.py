import json

import numpy as np
import pandas as pd
import pytest
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_equal as aae
from pandas import DataFrame
from pytest_mock import mocker  # noqa

from skillmodels import SkillModel


class TestDeltaRelatedMethods:
    def setup(self):
        self.periods = [0, 1, 2]
        self.controls = [["c1", "c2"], ["c1", "c2", "c3"], ["c3", "c4"]]
        self.factors = ["f1", "f2"]

        ind_tups = [
            (0, "m1"),
            (0, "m2"),
            (0, "m3"),
            (0, "m4"),
            (0, "m5"),
            (0, "m6"),
            (1, "m1"),
            (1, "m2"),
            (1, "m3"),
            (2, "m1"),
            (2, "m2"),
            (2, "m3"),
            (2, "m4"),
        ]

        self.update_info = DataFrame(index=pd.MultiIndex.from_tuples(ind_tups))

    def test_initial_delta_without_controls_besides_constant(self):
        self.controls = [[], [], []]
        expected = [np.zeros((6, 1)), np.zeros((3, 1)), np.zeros((4, 1))]
        calculated = SkillModel._initial_delta(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)

    def test_initial_delta_with_controls_and_constants(self):

        expected = [np.zeros((6, 3)), np.zeros((3, 4)), np.zeros((4, 3))]

        calculated = SkillModel._initial_delta(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)


def test_initial_h(mocker):  # noqa
    mocker.nfac = 5
    mocker.nupdates = 10
    calculated = SkillModel._initial_h(mocker)
    expected = np.zeros((10, 5))
    aae(calculated, expected)


def test_initial_r(mocker):  # noqa
    mocker.nupdates = 8
    calculated = SkillModel._initial_r(mocker)
    expected = np.zeros(8)
    aae(calculated, expected)


def test_initial_q(mocker):  # noqa
    mocker.nperiods = 5
    mocker.nfac = 3
    expected = np.zeros((4, 3, 3))
    calculated = SkillModel._initial_q(mocker)
    aae(calculated, expected)


def test_initial_x(mocker):  # noqa
    mocker.nobs = 10
    mocker.nmixtures = 2
    mocker.nfac = 3

    exp1 = np.zeros((10, 2, 3))
    exp2 = np.zeros((20, 3))

    calc1, calc2 = SkillModel._initial_x(mocker)

    aae(calc1, exp1)
    aae(calc2, exp2)

    calc1 += 1
    aae(calc2, np.ones((20, 3)))


def test_initial_w(mocker):  # noqa
    mocker.nobs = 10
    mocker.nmixtures = 3

    expected = np.ones((10, 3)) / 3

    calculated = SkillModel._initial_w(mocker)
    aae(calculated, expected)


@pytest.fixture
def p_mocker(mocker):  # noqa
    mocker.nobs = 10
    mocker.nmixtures = 2
    mocker.nfac = 3
    return mocker


def test_initial_p(p_mocker):
    expected = [np.zeros((10, 2, 4, 4)), np.zeros((20, 4, 4))]
    calculated = SkillModel._initial_p(p_mocker)
    for calc, exp in zip(calculated, expected):
        aae(calc, exp)

    # test that the second is pointing to the same data as the first.
    calc1, calc2 = calculated
    calc1 += 1
    aae(calc2, np.ones_like(calc2))


def test_initial_trans_coeffs(mocker):  # noqa
    mocker.factors = ["fac1", "fac2", "fac3"]
    mocker.transition_names = ["linear", "linear", "log_ces"]
    mocker.included_factors = [["fac1", "fac2"], ["fac2"], ["fac2", "fac3"]]
    mocker.nperiods = 5

    mock_linear = mocker.patch(
        "skillmodels.estimation.skill_model.tf.index_tuples_linear"
    )
    mock_linear.return_value = [0, 1, 2, 3]
    mock_log_ces = mocker.patch(
        "skillmodels.estimation.skill_model.tf.index_tuples_log_ces"
    )
    mock_log_ces.return_value = [0, 1, 2]

    expected = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 3))]

    calculated = SkillModel._initial_trans_coeffs(mocker)
    for calc, exp in zip(calculated, expected):
        aae(calc, exp)


class TestSigmaWeightsAndScalingFactor:
    def setup(self):
        self.nmixtures = 2
        self.nobs = 10
        self.nfac = 4
        self.sigma_points_scale = 1.5

        # these test results have been calculated with the sigma_point
        # function of the filterpy library
        with open(
            "skillmodels/tests/fast_routines/sigma_points_from_filterpy.json"
        ) as f:
            self.fixtures = json.load(f)

    def test_julier_sigma_weight_construction(self):
        expected_sws = self.fixtures["julier_wm"]
        aae(SkillModel.sigma_weights(self)[0], expected_sws)

    def test_julier_scaling_factor(self):
        expected_sf = 2.34520787991
        assert_almost_equal(SkillModel.sigma_scaling_factor(self), expected_sf)
