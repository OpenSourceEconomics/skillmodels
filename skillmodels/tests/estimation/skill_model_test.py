import json
from unittest.mock import Mock, call, patch
from pytest_mock import mocker
import pytest

import numpy as np
import pandas as pd
from nose.tools import assert_equal, assert_raises, assert_almost_equal
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal as aae
from pandas import DataFrame
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal

from skillmodels import SkillModel as smo


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
        calculated = smo._initial_delta(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)

    def test_initial_delta_with_controls_and_constants(self):

        expected = [np.zeros((6, 3)), np.zeros((3, 4)), np.zeros((4, 3))]

        calculated = smo._initial_delta(self)
        for calc, ex in zip(calculated, expected):
            aae(calc, ex)


def test_initial_h(mocker):
    mocker.nfac = 5
    mocker.nupdates = 10
    calculated = smo._initial_h(mocker)
    expected = np.zeros((10, 5))
    aae(calculated, expected)


def test_initial_r(mocker):
    mocker.nupdates = 8
    calculated = smo._initial_r(mocker)
    expected = np.zeros(8)
    aae(calculated, expected)


def test_initial_q(mocker):
    mocker.nperiods = 5
    mocker.nfac = 3
    expected = np.zeros((4, 3, 3))
    calculated = smo._initial_q(mocker)
    aae(calculated, expected)


def test_initial_x(mocker):
    mocker.nobs = 10
    mocker.nemf = 2
    mocker.nfac = 3

    exp1 = np.zeros((10, 2, 3))
    exp2 = np.zeros((20, 3))

    calc1, calc2 = smo._initial_x(mocker)

    aae(calc1, exp1)
    aae(calc2, exp2)

    # test that the second is pointing to the same data as the first.
    calc1 += 1
    aae(calc2, np.ones((20, 3)))


def test_initial_w(mocker):
    mocker.nobs = 10
    mocker.nemf = 3

    expected = np.ones((10, 3)) / 3

    calculated = smo._initial_w(mocker)
    aae(calculated, expected)


@pytest.fixture
def p_mocker(mocker):
    mocker.nobs = 10
    mocker.nemf = 2
    mocker.nfac = 3
    return mocker


def test_initial_p_square_root_filters(p_mocker):
    p_mocker.square_root_filters = True
    expected = [np.zeros((10, 2, 4, 4)), np.zeros((20, 4, 4))]
    calculated = smo._initial_p(p_mocker)
    for calc, exp in zip(calculated, expected):
        aae(calc, exp)

    # test that the second is pointing to the same data as the first.
    calc1, calc2 = calculated
    calc1 += 1
    aae(calc2, np.ones_like(calc2))


def test_initial_p_normal_filters(p_mocker):
    p_mocker.square_root_filters = False
    expected = [np.zeros((10, 2, 3, 3)), np.zeros((20, 3, 3))]
    calculated = smo._initial_p(p_mocker)
    for calc, exp in zip(calculated, expected):
        aae(calc, exp)

    # test that the second is pointing to the same data as the first.
    calc1, calc2 = calculated
    calc1 += 1
    aae(calc2, np.ones_like(calc2))


def test_initial_trans_coeffs(mocker):
    mocker.factors = ['fac1', 'fac2', 'fac3']
    mocker.transition_names = ['linear', 'linear', 'log_ces']
    mocker.included_factors = [['fac1', 'fac2'], ['fac2'], ['fac2', 'fac3']]
    mocker.nperiods = 5

    mock_linear = mocker.patch(
        'skillmodels.estimation.skill_model.tf.index_tuples_linear')
    mock_linear.return_value = [0, 1, 2, 3]
    mock_log_ces = mocker.patch(
        'skillmodels.estimation.skill_model.tf.index_tuples_log_ces')
    mock_log_ces.return_value = [0, 1, 2]

    expected = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 3))]

    calculated = smo._initial_trans_coeffs(mocker)
    for calc, exp in zip(calculated, expected):
        aae(calc, exp)


class TestSigmaWeightsAndScalingFactor:
    def setup(self):
        self.nemf = 2
        self.nobs = 10
        self.nfac = 4
        self.kappa = 1.5

        # these test results have been calculated with the sigma_point
        # function of the filterpy library
        with open(
            "skillmodels/tests/fast_routines/sigma_points_from_filterpy.json"
        ) as f:
            self.fixtures = json.load(f)

    def test_julier_sigma_weight_construction(self):
        expected_sws = self.fixtures["julier_wm"]
        aae(smo.sigma_weights(self)[0], expected_sws)

    def test_julier_scaling_factor(self):
        expected_sf = 2.34520787991
        assert_almost_equal(smo.sigma_scaling_factor(self), expected_sf)


class TestLikelihoodArgumentsDict:
    def setup(self):
        pass


class TestBSMethods:
    def setup(self):
        self.bootstrap_samples = [
            ["id_0", "id_1", "id_1"],
            ["id_0", "id_1", "id_0"],
            ["id_1", "id_0", "id_0"],
        ]
        self.bootstrap_nreps = 3
        self.model_name = "test_check_bs_sample"
        self.dataset_name = "test_data"
        self.periods = [0, 1, 2]
        self.data = pd.DataFrame(
            data=np.array(
                [
                    [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ]
            ).T,
            columns=["__period__", "arange"],
        )
        self.data["__id__"] = pd.Series(
            dtype="str",
            data=[
                "id_0",
                "id_0",
                "id_0",
                "id_1",
                "id_1",
                "id_1",
                "id_2",
                "id_2",
                "id_2",
            ],
        )  # noqa
        self.bootstrap_sample_size = 3
        self.nobs = 3
        self.bootstrap_nprocesses = 2

    def test_check_bs_samples_accepts_iterable(self):
        smo._check_bs_samples(self)

    def test_rejects_non_iterable(self):
        self.bootstrap_samples = 240
        assert_raises(AssertionError, smo._check_bs_samples, self)

    def test_raises_error_with_unknown_identifier(self):
        self.bootstrap_samples[2][0] = "a"
        assert_raises(AssertionError, smo._check_bs_samples, self)

    def test_only_bootstrap_samples_with_enough_samples(self):
        self.bootstrap_nreps = 10
        assert_raises(AssertionError, smo._check_bs_samples, self)

    def test_generate_bs_samples(self):
        np.random.seed(495)
        expected_samples = [
            ["id_1", "id_1", "id_1"],
            ["id_0", "id_2", "id_2"],
            ["id_2", "id_2", "id_1"],
        ]
        calc_samples = smo._generate_bs_samples(self)
        assert_equal(calc_samples, expected_samples)

    def test_select_bootstrap_data(self):
        expected_data = pd.DataFrame(
            data=np.array(
                [
                    [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0],
                ]
            ).T,
            columns=["__period__", "arange"],
        )
        expected_data["__id__"] = [
            "id_0",
            "id_0",
            "id_0",
            "id_1",
            "id_1",
            "id_1",
            "id_1",
            "id_1",
            "id_1",
        ]
        calc_data = smo._select_bootstrap_data(self, 0)
        assert_frame_equal(calc_data, expected_data)

    # define some mock functions. Mock objects don't work because they
    # cannot be pickled, which is required for Multiprocessing to work.
    def _bs_fit(self, rep, params):
        return rep * np.ones(3)

    def param_names(self, params_type):
        return ["p1", "p2", "p3"]

    def len_params(self, params_type):
        return 3

    def test_all_bootstrap_params(self):
        calc_params = smo.all_bootstrap_params(self, params=np.ones(3))
        expected_params = pd.DataFrame(
            data=[[0.0] * 3, [1.0] * 3, [2.0] * 3],
            index=["rep_0", "rep_1", "rep_2"],
            columns=["p1", "p2", "p3"],
        )
        assert_frame_equal(calc_params, expected_params)


class TestBootstrapParamsToConfInt:
    def setup(self):
        bs_data = np.zeros((100, 2))
        bs_data[:, 0] = np.arange(100)
        bs_data[:, 1] = np.arange(5).repeat(20)
        np.random.shuffle(bs_data)
        cols = ["p1", "p2"]
        ind = ["rep_{}".format(i) for i in range(100)]
        df = pd.DataFrame(data=bs_data, columns=cols, index=ind)

        self.stored_bootstrap_params = df

    def all_bootstrap_params(self, params):
        return self.stored_bootstrap_params

    def test_bootstrap_conf_int(self):
        expected_conf_int = pd.DataFrame(
            data=[[2.475, 96.525], [0, 4]],
            index=["p1", "p2"],
            columns=["lower", "upper"],
        )
        calc_conf_int = smo.bootstrap_conf_int(self, np.ones(3))
        aaae(calc_conf_int, expected_conf_int)


class TestBootstrapCovMatrix:
    def setup(self):
        np.random.seed(94355)
        expected_cov = np.array([[28 / 3, 17.0], [17.0, 31.0]])
        self.par_names = ["p1", "p2"]
        self.expected_cov = pd.DataFrame(
            data=expected_cov, columns=self.par_names, index=self.par_names
        )

        self.params = np.arange(2)
        self.model_name = "test_bootstrap_cov_matrix"
        self.dataset_name = "data_for_testing_covmatrix"

    def len_params(self, params_type):
        return 3

    def all_bootstrap_params(self, params):
        fake_bs_params = np.array([[1, 4], [3, 8], [7, 15]])
        fake_df = pd.DataFrame(
            data=fake_bs_params, columns=self.par_names, index=["rep1", "rep2", "rep3"]
        )
        return fake_df

    def test_bootstrap_cov_matrix(self):
        calc_cov = smo.bootstrap_cov_matrix(self, self.params)
        assert_frame_equal(calc_cov, self.expected_cov)


class TestBootstrapPValues:
    def setup(self):
        bs_params = pd.DataFrame(np.arange(10).reshape(5, 2), columns=["a", "b"])
        self.all_bootstrap_params = Mock(return_value=bs_params)

    def test_bootstrap_p_values(self):
        params = np.array([2, -9])
        expected_p_values = pd.Series([0.8333333, 0.333333], index=["a", "b"])
        calculated_p_values = smo.bootstrap_pvalues(self, params)
        assert_series_equal(expected_p_values, calculated_p_values)


class TestGenerateStartFactorsForMarginalEffects:
    def setup(self):
        sl = {}
        sl["X_zero"] = slice(0, 2)
        sl["P_zero"] = slice(2, 5)
        self.params_slices = Mock(return_value=sl)

        self.me_params = np.array([5, 10, 1, 0.1, 4])
        self.cholesky_of_P_zero = False

        self.estimator = "chs"
        self.nobs = 50000
        self.nemf = 1
        self.nfac = 2
        self.params_quants = ["X_zero"]
        self.exp_cov = np.array([[1, 0.1], [0.1, 4]])
        np.random.seed(5471)

    def test_generate_start_factors_mean_with_estimated_X_zero(self):
        exp_mean = np.array([5, 10])
        calc_mean = smo._generate_start_factors(self).mean(axis=0)
        aaae(calc_mean, exp_mean, decimal=2)

    def test_generate_start_factors_mean_with_normalized_X_zero(self):
        exp_mean = np.array([0, 0])
        self.params_quants = []
        calc_mean = smo._generate_start_factors(self).mean(axis=0)
        aaae(calc_mean, exp_mean, decimal=2)

    def test_generate_start_factors_cov_no_cholesky(self):
        self.nobs = 200000
        calc_factors = smo._generate_start_factors(self)
        df = pd.DataFrame(calc_factors)
        calc_cov = df.cov().to_numpy()
        aaae(calc_cov, self.exp_cov, decimal=2)

    def test_generate_start_factors_cov_cholesky(self):
        self.nobs = 200000
        self.me_params = np.array([5, 10, 1, 0.1, 1.99749844])
        self.cholesky_of_P_zero = True
        calc_factors = smo._generate_start_factors(self)
        df = pd.DataFrame(calc_factors)
        calc_cov = df.cov().to_numpy()
        aaae(calc_cov, self.exp_cov, decimal=2)


class TestGetEpsilon:
    def setup(self):
        self.nperiods = 5
        self.periods = range(self.nperiods)
        inter_facs = [
            np.ones((2, 2)),
            np.zeros((2, 2)),
            np.array([[1, 2], [3, 4]]),
            np.array([[-8, 0], [3, -4]]),
        ]
        self.factors = ["f1", "f2"]
        self.me_of = "f2"
        self._predict_final_factors = Mock(return_value=inter_facs)
        self._machine_accuracy = Mock(return_value=0.1)

    def test_get_epsilon_centered(self):
        exp_eps = np.array([0.4641588, 0.0464158, 1.85663533, 1.85663533])
        calc_eps = smo._get_epsilon(self, centered=True)
        aaae(calc_eps, exp_eps)

    def test_get_epsilon_uncentered(self):
        calc_eps = smo._get_epsilon(self, centered=False)
        exp_eps = np.array([0.3162277, 0.03162277, 1.26491106, 1.26491106])
        aaae(calc_eps, exp_eps)


class TestSelectFinalFactor:
    def setup(self):
        self.me_on = "f2"
        self.factors = ["f3", "f1", "f2"]
        self.final_factors = np.zeros((10, 3))
        self.final_factors[:, 2] = np.arange(10)

    def test_select_final_factor(self):
        calc = smo._select_final_factor(self, self.final_factors)
        aae(calc, np.arange(10))


class TestAnchorFinalFactorsAndAnchoringOutcome:
    def setup(self):
        self.final_factors = np.ones((10, 3))
        self.final_factors[:, 2] = 2
        self.anch_positions = [0, 2]
        self.al = np.array([0.4, 0, 0.6])
        self.ai = 2
        self.exp_anchored_factors = np.ones((10, 3))
        self.exp_anchored_factors[:, 0] *= 0.4
        self.exp_anchored_factors[:, 2] = 2
        self.exp_anchored_factors[:, 2] *= 0.6

    def test_anchor_final_factors_no_anchoring(self):
        # should pass and do nothing
        self.anchoring = False
        calc = smo._anchor_final_factors(self, self.final_factors, self.al)
        aae(calc, self.final_factors)

    def test_anchor_final_factors_with_linear_anchoring_integration(self):
        self.anchoring = True
        calc = smo._anchor_final_factors(self, self.final_factors, self.al)
        aae(calc, self.exp_anchored_factors)

    def test_anch_outcome_from_final_factors_with_linear_anchoring(self):
        self.anchoring = True
        self._anchor_final_factors = Mock(return_value=self.exp_anchored_factors)
        exp = np.ones(10) * 3.6
        calc = smo._anchoring_outcome_from_final_factors(
            self, self.final_factors, self.al, self.ai
        )
        aae(calc, exp)

    def test_anch_outcome_from_final_factors_no_anchoring(self):
        self.anchoring = False
        assert_raises(
            AssertionError,
            smo._anchoring_outcome_from_final_factors,
            self,
            self.final_factors,
            self.al,
            self.ai,
        )


def fake_tsp(
    stage,
    flat_sigma_points,
    transition_argument_dicts,
    transition_function_names,
    anchoring_type=None,
    anchoring_positions=None,
    anch_params=None,
    intercept=None,
):

    flat_sigma_points[:] *= 2


class TestPredictFinalFactors:
    def setup(self):
        self.change = np.array([1, 2])
        self.nperiods = 3
        self.lh_args = {
            "predict_args": {
                "transform_sigma_points_args": {
                    "transition_argument_dicts": {},
                    "transition_function_names": [],
                }
            },
            "parse_params_args": {},
        }
        self.me_at = np.ones((10, 2))
        self.me_of = "f1"
        self.factors = ["f1", "f2"]
        self.me_params = None
        self.stagemap = [0, 1, 1]

    def test_predict_final_factors_invalid_change(self):
        invalid_change = np.zeros(5)
        assert_raises(AssertionError, smo._predict_final_factors, self, invalid_change)

    @patch("skillmodels.estimation.skill_model.parse_params")
    @patch("skillmodels.estimation.skill_model.transform_sigma_points")
    def test_predict_ff_intermediate_false_mocked(self, mock_tsp, mock_pp):
        mock_tsp.side_effect = fake_tsp
        self.likelihood_arguments_dict = Mock(return_value=self.lh_args)
        exp = np.ones((10, 2)) * 4
        exp[:, 0] = 12
        calc = smo._predict_final_factors(self, self.change)
        aaae(calc, exp)

    @patch("skillmodels.estimation.skill_model.parse_params")
    @patch("skillmodels.estimation.skill_model.transform_sigma_points")
    def test_predict_ff_intermediate_true_mocked(self, mock_tsp, mock_pp):
        mock_tsp.side_effect = fake_tsp
        self.likelihood_arguments_dict = Mock(return_value=self.lh_args)
        exp1 = np.ones((10, 2))
        exp2 = np.ones((10, 2)) * 2
        exp2[:, 0] = 4
        exp = [exp1, exp2]
        calc = smo._predict_final_factors(self, self.change, True)
        for c, e in zip(calc, exp):
            aaae(c, e)

    @patch("skillmodels.estimation.skill_model.parse_params")
    @patch("skillmodels.estimation.skill_model.transform_sigma_points")
    def test_predict_ff_mocked_same_result_in_second(self, mock_tsp, mock_pp):
        # this test makes sure that y copy arrays where necessary
        mock_tsp.side_effect = fake_tsp
        self.likelihood_arguments_dict = Mock(return_value=self.lh_args)

        calc1 = smo._predict_final_factors(self, self.change)
        calc2 = smo._predict_final_factors(self, self.change)
        aaae(calc1, calc2)


def select_first(arr):
    return arr[:, 0]


def fake_anch(final_factors, anch_loadings):
    return final_factors * anch_loadings


def fake_anch_outcome(final_factors, anch_loadings, anch_intercept):
    return (final_factors * anch_loadings).sum(axis=1) + anch_intercept


class TestMarginalEffectOutcome:
    def setup(self):
        self.me_at = np.ones((10, 2))

        self._predict_final_factors = Mock(return_value=self.me_at)
        self.change = np.zeros(2)
        self._select_final_factor = Mock(side_effect=select_first)
        self.nfac = 2
        self.anch_positions = [0]
        self.anchored_factors = ["f1"]
        self.factors = ["f1", "f2"]

        sl = {"H": slice(0, 3), "deltas": [slice(3, 5)]}
        self.params_slices = Mock(return_value=sl)
        self.me_params = np.array([0, 0, 3, 0, 1])
        self._anchor_final_factors = Mock(side_effect=fake_anch)
        self._anchoring_outcome_from_final_factors = Mock(side_effect=fake_anch_outcome)

    def test_marginal_effect_outcome_no_anchoring(self):
        self.anchoring = False
        exp = np.ones((10))
        calc = smo._marginal_effect_outcome(self, self.change)
        aaae(calc, exp)

    def test_marginal_effect_outcome_with_anchoring(self):
        self.anchoring = True
        self.me_anchor_on = True
        self.me_on = "f1"
        exp = np.ones((10)) * 3
        calc = smo._marginal_effect_outcome(self, self.change)
        aaae(calc, exp)

    def test_marginal_effect_outcome_anch_outcome(self):
        self.anchoring = True
        self.me_anchor_on = True
        self.me_on = "anch_outcome"
        exp = np.ones((10)) * 4
        calc = smo._marginal_effect_outcome(self, self.change)
        aaae(calc, exp)
