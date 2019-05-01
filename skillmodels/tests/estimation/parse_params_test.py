from numpy.testing import assert_array_equal as aae
from unittest.mock import patch
import skillmodels.estimation.parse_params as pp
import numpy as np
import scipy.linalg as sl


class TestMapParamsToDeltasWithoutReplacments:
    def setup(self):
        self.params = np.arange(200)
        self.initial = [np.zeros((4, 3)), np.zeros((6, 2))]
        boo1 = np.ones((4, 3), dtype=bool)
        boo1[2, 0] = False

        boo2 = np.ones((6, 2), dtype=bool)
        boo2[0, 0] = False
        self.bools = [boo1, boo2]

        self.slices = [slice(10, 21), slice(21, 32)]

    def test_map_params_to_deltas_without_replacements(self):

        expected0 = np.array([[10, 11, 12], [13, 14, 15], [0, 16, 17], [18, 19, 20]])
        expected1 = np.array(
            [[0, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]
        )

        pp._map_params_to_deltas(self.params, self.initial, self.slices, self.bools)

        aae(self.initial[0], expected0)
        aae(self.initial[1], expected1)


class TestMapParamToDeltasWithReplacements:
    def setup(self):
        t = True
        f = False
        self.params = np.arange(200)
        self.initial = [np.zeros((4, 3)), np.zeros((6, 3))]
        boo1 = np.array([[f, t, t], [t, t, t], [f, t, t], [t, t, t]])
        boo2 = np.array(
            [[f, f, f], [f, f, f], [f, t, t], [t, t, t], [f, t, t], [t, t, t]]
        )
        self.bools = [boo1, boo2]
        self.slices = [slice(10, 20), slice(20, 30)]
        self.replacements = [[(1, 0), (0, 0)], [(1, 1), (0, 1)]]

    def test_map_params_to_deltas_with_replacements(self):
        expected0 = np.array([[0, 10, 11], [12, 13, 14], [0, 15, 16], [17, 18, 19]])
        expected1 = np.array(
            [
                [0, 10, 11],
                [12, 13, 14],
                [0, 20, 21],
                [22, 23, 24],
                [0, 25, 26],
                [27, 28, 29],
            ]
        )

        pp._map_params_to_deltas(
            self.params, self.initial, self.slices, self.bools, self.replacements
        )

        aae(self.initial[0], expected0)
        aae(self.initial[1], expected1)


class TestMapParamsToPsi:
    def setup(self):
        self.params = np.arange(100)
        self.slice = slice(10, 15)
        self.initial = np.zeros(10)
        self.boo = np.array([True, False] * 5)

    def test_map_params_to_psi(self):
        expected = np.array([10, 0, 11, 0, 12, 0, 13, 0, 14, 0])
        pp._map_params_to_psi(self.params, self.initial, self.slice, self.boo)
        aae(self.initial, expected)


class TestMapParamsToH:
    def setup(self):
        self.params = np.arange(100)

        self.initial = np.zeros((6, 3))
        self.initial[2, 1] = 3

        self.boo = np.zeros_like(self.initial, dtype=bool)
        self.boo[[0, 3], 0] = True
        self.boo[[1, 4], 1] = True
        self.boo[[2, 3, 4], 2] = True
        self.boo[5, :] = False

        self.slice = slice(10, 17)
        self.initial_copy = self.initial.copy()

    def test_map_params_to_H_without_psi_transformation(self):
        expected = np.array(
            [[10, 0, 0], [0, 11, 0], [0, 3, 12], [13, 0, 14], [0, 15, 16], [0, 0, 0]]
        )
        pp._map_params_to_H(
            params=self.params,
            initial=self.initial,
            params_slice=self.slice,
            boo=self.boo,
        )
        aae(self.initial, expected)

    def test_map_params_to_H_without_psi_but_with_replacements(self):
        expected = np.array(
            [[10, 0, 0], [0, 11, 0], [0, 3, 12], [13, 0, 14], [0, 15, 16], [13, 0, 14]]
        )

        replacements = [(5, 3)]
        pp._map_params_to_H(
            params=self.params,
            initial=self.initial,
            params_slice=self.slice,
            boo=self.boo,
            replacements=replacements,
        )
        aae(self.initial, expected)

    def test_map_params_to_H_with_psi_transformation(self):
        psi_boo = np.array([True, False, False, True, False, False])
        psi = np.array([1, 5, 8])
        arr1 = np.zeros((2, 1))

        expected = np.array(
            [
                [10, 50, 80],
                [0, 11, 0],
                [0, 3, 12],
                [13, 65, 118],
                [0, 15, 16],
                [0, 0, 0],
            ]
        )

        pp._map_params_to_H(
            params=self.params,
            initial=self.initial,
            params_slice=self.slice,
            boo=self.boo,
            psi_bool_for_H=psi_boo,
            psi=psi,
            arr1=arr1,
            endog_position=0,
            initial_copy=self.initial_copy,
        )
        aae(self.initial, expected)

    def test_map_params_to_H_with_psi_transformation_unclean_initial(self):
        self.initial[:] = 100
        self.test_map_params_to_H_with_psi_transformation()


class TestMapParamsToWZero:
    def setup(self):
        self.params = np.arange(100)
        self.slice = slice(15, 20)
        self.initial = np.zeros(5)

    def test_map_params_to_w_zero(self):
        pp._map_params_to_W_zero(self.params, self.initial, self.slice)
        aae(self.initial, np.array([15, 16, 17, 18, 19]))


class TestMapParamsToR:
    def setup(self):
        self.params = np.arange(100)
        self.slice = slice(15, 20)
        self.initial = np.zeros(10)
        self.boo = np.array([True, False] * 5)

    def test_map_params_to_r_without_replacements(self):
        self.square_root_filters = False
        pp._map_params_to_R(
            self.params, self.initial, self.slice, self.boo, self.square_root_filters
        )
        aae(self.initial, np.array([15, 0, 16, 0, 17, 0, 18, 0, 19, 0]))

    def test_map_params_to_r_square_root_filters_without_replacements(self):
        self.square_root_filters = True
        pp._map_params_to_R(
            self.params, self.initial, self.slice, self.boo, self.square_root_filters
        )
        aae(self.initial, np.sqrt([15, 0, 16, 0, 17, 0, 18, 0, 19, 0]))

    def test_map_params_to_r_with_replacements(self):
        self.square_root_filters = False
        self.boo = np.array([True, False] * 3 + [False] * 4)
        self.slice = slice(15, 18)
        self.replacements = [(6, 0), (8, 2)]
        expected = np.array([15, 0, 16, 0, 17, 0, 15, 0, 16, 0])
        pp._map_params_to_R(
            self.params,
            self.initial,
            self.slice,
            self.boo,
            self.square_root_filters,
            self.replacements,
        )
        aae(self.initial, expected)


class TestMapParamsToQ:
    def setup(self):
        self.initial = np.zeros((2, 3, 3))
        self.boo = np.zeros_like(self.initial, dtype=bool)
        self.boo[:] = np.eye(3, dtype=bool)
        self.boo[1, 0, 0] = False
        self.boo[1, 2, 2] = False

        self.replacements = [[(1, 0, 0), (0, 0, 0)]]

        self.params = np.arange(100)
        self.slice = slice(4, 8)

    def test_map_params_to_q_without_replacements(self):
        pp._map_params_to_Q(self.params, self.initial, self.slice, self.boo)
        expected = np.array(
            [[[4, 0, 0], [0, 5, 0], [0, 0, 6]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]]
        )
        aae(self.initial, expected)

    def test_map_params_to_q_with_replacements(self):
        pp._map_params_to_Q(
            self.params, self.initial, self.slice, self.boo, self.replacements
        )
        expected = np.array(
            [[[4, 0, 0], [0, 5, 0], [0, 0, 6]], [[4, 0, 0], [0, 7, 0], [0, 0, 0]]]
        )
        aae(self.initial, expected)


class TestMapParamsToXZero:
    def setup(self):
        self.nemf = 2
        self.nind = 10
        self.nfac = 3
        self.initial = np.zeros((self.nind, self.nemf, self.nfac))
        self.params = np.arange(100)
        self.slice = slice(10, 16)
        self.filler = np.zeros((self.nemf, self.nfac))
        self.replacements = [[(1, 0), (0, 0)]]

    def test_map_params_to_x_zero_with_replacement(self):
        exp = np.zeros((self.nind, self.nemf, self.nfac))
        exp[:, 0] = np.array([10, 11, 12])
        exp[:, 1] = np.array([23, 14, 15])
        pp._map_params_to_X_zero(
            self.params, self.initial, self.filler, self.slice, self.replacements
        )
        aae(self.initial, exp)

    def test_map_params_to_x_zero_without_replacement(self):
        exp = np.zeros((self.nind, self.nemf, self.nfac))
        exp[:, 0] = np.array([10, 11, 12])
        exp[:, 1] = np.array([13, 14, 15])
        pp._map_params_to_X_zero(self.params, self.initial, self.filler, self.slice)
        aae(self.initial, exp)

    def test_transform_params_for_X_zero_no_replacements_short_to_long(self):
        params_for_X_zero = self.params[self.slice]
        result = pp.transform_params_for_X_zero(
            params_for_X_zero, self.filler, "short_to_long"
        )
        aae(result, params_for_X_zero)

    def test_transform_params_for_X_zero_short_to_long(self):
        params_for_X_zero = self.params[self.slice]
        expected = np.array([10, 11, 12, 23, 14, 15])
        result = pp.transform_params_for_X_zero(
            params_for_X_zero, self.filler, "short_to_long", self.replacements
        )
        aae(result, expected)

    def test_transform_params_for_X_zero_long_to_short(self):
        params_for_X_zero = np.array([10, 11, 12, 23, 14, 15])
        expected = np.array([10, 11, 12, 13, 14, 15])
        result = pp.transform_params_for_X_zero(
            params_for_X_zero, self.filler, "long_to_short", self.replacements
        )
        aae(result, expected)


class TestMapParamsToPZero:
    def setup(self):
        self.nemf = 2
        self.nfac = 3
        self.nind = 10

        self.initial = np.zeros((self.nind, self.nemf, self.nfac, self.nfac))
        self.sqrt_initial = np.zeros(
            (self.nind, self.nemf, self.nfac + 1, self.nfac + 1)
        )
        self.params = np.arange(100)
        self.params[10:16] = np.array([10, 2, 1, 13, 3, 15])
        self.params[16:22] = np.array([16, 2, 1, 19, 4, 21])
        self.slice_2mat = slice(10, 22)
        self.slice_1mat = slice(10, 16)

        self.filler_1mat = np.zeros((1, self.nfac, self.nfac))
        self.filler_2mat = np.zeros((2, self.nfac, self.nfac))

        helper = np.zeros((self.nfac, self.nfac), dtype=bool)
        helper[np.triu_indices(self.nfac)] = True
        self.boo_1mat = np.zeros_like(self.filler_1mat, dtype=bool)
        self.boo_2mat = np.zeros_like(self.filler_2mat, dtype=bool)

        self.boo_1mat[:] = helper
        self.boo_2mat[:] = helper

        self.fill_one = np.array([[10, 2, 1], [0, 13, 3], [0, 0, 15]])
        self.fill_two = np.array([[16, 2, 1], [0, 19, 4], [0, 0, 21]])

    def test_map_params_to_p_zero_chol_to_chol_2_mat(self):
        self.params_type = "short"
        self.cholesky_of_P_zero = True
        self.square_root_filters = True

        exp = np.zeros_like(self.sqrt_initial)
        exp[:, 0, 1:, 1:] = self.fill_one
        exp[:, 1, 1:, 1:] = self.fill_two

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.sqrt_initial,
            self.slice_2mat,
            self.filler_2mat,
            self.boo_2mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.sqrt_initial, exp)

    def test_map_params_to_p_zero_chol_to_notchol_2_mat(self):
        self.params_type = "short"
        self.cholesky_of_P_zero = True
        self.square_root_filters = False

        exp = np.zeros_like(self.initial)
        exp[:, 0] = np.dot(self.fill_one.T, self.fill_one)
        exp[:, 1] = np.dot(self.fill_two.T, self.fill_two)

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.initial,
            self.slice_2mat,
            self.filler_2mat,
            self.boo_2mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.initial, exp)

    def test_map_params_to_p_zero_not_chol_to_not_chol_2_mat(self):
        self.params_type = "long"
        self.cholesky_of_P_zero = False
        self.square_root_filters = False

        exp = np.zeros_like(self.initial)
        exp[:, 0] = (
            self.fill_one + (self.fill_one - np.diag(np.diagonal(self.fill_one))).T
        )
        exp[:, 1] = (
            self.fill_two + (self.fill_two - np.diag(np.diagonal(self.fill_two))).T
        )

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.initial,
            self.slice_2mat,
            self.filler_2mat,
            self.boo_2mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.initial, exp)

    def test_map_params_to_p_zero_notchol_to_chol_2_mat(self):
        self.params_type = "long"
        self.cholesky_of_P_zero = False
        self.square_root_filters = True

        exp = np.zeros_like(self.sqrt_initial)
        exp[:, 0, 1:, 1:] = sl.cholesky(
            self.fill_one + (self.fill_one - np.diag(np.diagonal(self.fill_one))).T
        )

        exp[:, 1, 1:, 1:] = sl.cholesky(
            self.fill_two + (self.fill_two - np.diag(np.diagonal(self.fill_two))).T
        )

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.sqrt_initial,
            self.slice_2mat,
            self.filler_2mat,
            self.boo_2mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.sqrt_initial, exp)

    def test_map_params_to_p_zero_chol_to_chol_1_mat_restricted(self):
        self.params_type = "short"
        self.cholesky_of_P_zero = True
        self.square_root_filters = True

        exp = np.zeros_like(self.sqrt_initial)
        exp[:, :, 1:, 1:] = self.fill_one

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.sqrt_initial,
            self.slice_1mat,
            self.filler_1mat,
            self.boo_1mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.sqrt_initial, exp)

    def test_map_params_to_p_zero_chol_to_notchol_1_mat_restricted(self):
        self.params_type = "short"
        self.cholesky_of_P_zero = True
        self.square_root_filters = False

        exp = np.zeros_like(self.initial)
        exp[:] = np.dot(self.fill_one.T, self.fill_one)

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.initial,
            self.slice_1mat,
            self.filler_1mat,
            self.boo_1mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.initial, exp)

    def test_map_params_to_p_zero_notchol_to_chol_1_mat_restricted(self):
        self.params_type = "long"
        self.cholesky_of_P_zero = False
        self.square_root_filters = False

        exp = np.zeros_like(self.initial)
        exp[:] = self.fill_one + (self.fill_one - np.diag(np.diagonal(self.fill_one))).T

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.initial,
            self.slice_1mat,
            self.filler_1mat,
            self.boo_1mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.initial, exp)

    def test_map_params_to_p_zero_not_chol_to_not_chol_1_mat_restricted(self):
        self.params_type = "long"
        self.cholesky_of_P_zero = False
        self.square_root_filters = True

        exp = np.zeros_like(self.sqrt_initial)
        exp[:, :, 1:, 1:] = sl.cholesky(
            self.fill_one + (self.fill_one - np.diag(np.diagonal(self.fill_one))).T
        )

        pp._map_params_to_P_zero(
            self.params,
            self.params_type,
            self.sqrt_initial,
            self.slice_1mat,
            self.filler_1mat,
            self.boo_1mat,
            self.cholesky_of_P_zero,
            self.square_root_filters,
        )

        aae(self.sqrt_initial, exp)

    def test_transform_params_for_p_zero_no_transform_short_to_long(self):
        params_for_P_zero = self.params[self.slice_2mat]
        result = pp.transform_params_for_P_zero(
            params_for_P_zero, self.filler_2mat, self.boo_2mat, True, "short_to_long"
        )
        aae(result, params_for_P_zero)

    def test_transform_params_for_p_zero_short_to_long(self):
        params_for_P_zero = self.params[self.slice_2mat]
        result = pp.transform_params_for_P_zero(
            params_for_P_zero, self.filler_2mat, self.boo_2mat, False, "short_to_long"
        )

        expected = np.array([100, 20, 10, 173, 41, 235, 256, 32, 16, 365, 78, 458])

        aae(result, expected)

    def test_transform_params_for_p_zero_long_to_short(self):
        params_for_P_zero = expected = np.array(
            [100, 20, 10, 173, 41, 235, 256, 32, 16, 365, 78, 458]
        )
        result = pp.transform_params_for_P_zero(
            params_for_P_zero, self.filler_2mat, self.boo_2mat, False, "long_to_short"
        )

        expected = self.params[self.slice_2mat]
        aae(result, expected)


def fake_transform_func(coeffs, inlcluded_factors, direction, out):
    if direction == "short_to_long":
        out[:-1] = coeffs * 2
        out[-1] = 1
    else:
        out[:] = coeffs[:-1] / 2


def fake_map_func(
    params,
    initial,
    params_slice,
    transform_funcs=None,
    included_factors=None,
    direction="short_to_long",
):

    if direction == "short_to_long":
        initial[0][:] = np.arange(6).reshape(2, 3)
        initial[1][:] = np.arange(start=6, stop=16).reshape(2, 5)
    if direction == "long_to_short":
        initial[0][:] = np.arange(6).reshape(2, 3) / 2
        initial[1][:] = np.arange(start=6, stop=14).reshape(2, 4) / 2


class TestMapParamsToTransCoeffs:
    def setup(self):
        self.initial_no_transform = [np.zeros((2, 3)), np.zeros((2, 4))]
        self.initial_transform = [np.zeros((2, 3)), np.zeros((2, 5))]
        self.params = np.arange(100)
        self.short_slice = [
            [slice(10, 13), slice(13, 16)],
            [slice(16, 20), slice(20, 24)],
        ]
        self.long_slice = [
            [slice(10, 13), slice(13, 16)],
            [slice(16, 21), slice(21, 26)],
        ]
        self.transform_funcs = [None, "some_transform_func"]
        self.included = [[], ["f1", "f2"]]

    def test_map_params_to_trans_coeffs_without_transformation(self):
        exp1 = np.array([[10, 11, 12], [13, 14, 15]])
        exp2 = np.array([[16, 17, 18, 19], [20, 21, 22, 23]])

        pp._map_params_to_trans_coeffs(
            self.params, self.initial_no_transform, self.short_slice
        )

        aae(self.initial_no_transform[0], exp1)
        aae(self.initial_no_transform[1], exp2)

    @patch("skillmodels.estimation.parse_params.tf")
    def test_map_params_to_trans_coeffs_short_to_long(self, mock_tf):
        mock_tf.some_transform_func.side_effect = fake_transform_func
        exp1 = np.array([[10, 11, 12], [13, 14, 15]])
        exp2 = np.array([[32, 34, 36, 38, 1], [40, 42, 44, 46, 1]])

        pp._map_params_to_trans_coeffs(
            self.params,
            self.initial_transform,
            self.short_slice,
            self.transform_funcs,
            self.included,
        )

        aae(self.initial_transform[0], exp1)
        aae(self.initial_transform[1], exp2)

    @patch("skillmodels.estimation.parse_params.tf")
    def test_map_params_to_trans_coeffs_long_to_short(self, mock_tf):
        mock_tf.some_transform_func.side_effect = fake_transform_func
        exp1 = np.array([[10, 11, 12], [13, 14, 15]])
        exp2 = np.array([[8, 8.5, 9, 9.5], [10.5, 11, 11.5, 12]])
        pp._map_params_to_trans_coeffs(
            self.params,
            self.initial_no_transform,
            self.long_slice,
            self.transform_funcs,
            self.included,
            "long_to_short",
        )
        aae(self.initial_no_transform[0], exp1)
        aae(self.initial_no_transform[1], exp2)

    @patch("skillmodels.estimation.parse_params._map_params_to_trans_coeffs")
    def test_transform_params_for_trans_coeffs_short_to_long(self, mock):
        mock.side_effect = fake_map_func
        result = pp.transform_params_for_trans_coeffs(
            self.params,
            self.initial_transform,
            self.short_slice,
            self.transform_funcs,
            self.included,
            "short_to_long",
        )

        aae(result, np.arange(16))

    @patch("skillmodels.estimation.parse_params._map_params_to_trans_coeffs")
    def test_transform_params_for_trans_coeffs_long_to_short(self, mock):
        mock.side_effect = fake_map_func
        result = pp.transform_params_for_trans_coeffs(
            self.params,
            self.initial_no_transform,
            self.long_slice,
            self.transform_funcs,
            self.included,
            "long_to_short",
        )

        aae(result, np.arange(14) / 2)
