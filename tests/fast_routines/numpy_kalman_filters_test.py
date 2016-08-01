import skillmodels.fast_routines.numpy_kalman_filters as kf
from numpy.testing import assert_array_almost_equal as aaae
import numpy as np
from unittest.mock import patch
from numba import jit


@jit
def make_unique(qr_result_arr):
    long_side, m, n = qr_result_arr.shape
    for u in range(long_side):
        for j in range(n):
            if qr_result_arr[u, j, j] < 0:
                for k in range(n):
                    qr_result_arr[u, j, k] *= -1


class TestUnscentedPredict:
    def setup(self):
        nemf = 2
        nind = 3
        nsigma = 7
        nfac = 3
        self.stage = 1

        first = np.array([1.1, 1.2, 1.3])
        second = np.array([1.4, 1.5, 1.6])
        third = np.array([2.1, 2.2, 2.3])
        fourth = np.array([2.4, 2.5, 2.6])

        # these are sigma_points for the test with focus on columns
        sps1 = np.zeros((nemf, nind, nsigma, nfac))
        sps1[0, 0, :, :] = np.tile(first, nsigma).reshape(nsigma, nfac)
        sps1[0, 1, :, :] = np.tile(second, nsigma).reshape(nsigma, nfac)
        sps1[1, 0, :, :] = np.tile(third, nsigma).reshape(nsigma, nfac)
        sps1[1, 1, :, :] = np.tile(fourth, nsigma).reshape(nsigma, nfac)
        self.sps1 = sps1.reshape(nemf * nind, nsigma, nfac)
        self.flat_sps1 = sps1.reshape(nemf * nind * nsigma, nfac)

        expected_states1 = np.zeros((nemf, nind, nfac))
        expected_states1[0, 0, :] = first
        expected_states1[0, 1, :] = second
        expected_states1[1, 0, :] = third
        expected_states1[1, 1, :] = fourth
        self.expected_states1 = expected_states1.reshape(nemf * nind, nfac)

        # these are sigma_points for the test with focus on weighting
        sps2 = np.zeros((nemf, nind, nsigma, nfac))
        sps2[:, :, :, :] = \
            np.arange(nsigma).repeat(nfac).reshape(nsigma, nfac)
        self.sps2 = sps2.reshape(nemf * nind, nsigma, nfac)
        self.flat_sps2 = sps2.reshape(nemf * nind * nsigma, nfac)
        self.expected_states2 = np.ones((nemf * nind, nfac)) * 3

        # these are sigma_points for the test with focus on the covariances
        sps3 = np.zeros((nemf, nind, nsigma, nfac))
        sps3[:, :, 1, :] += 1
        sps3[:, :, 2, :] += 2
        sps3[:, :, 3, :] += 3
        sps3[:, :, 4, :] -= 1
        sps3[:, :, 5, :] -= 2
        sps3[:, :, 6, :] -= 3
        self.sps3 = sps3.reshape(nemf * nind, nsigma, nfac)
        self.flat_sps3 = sps3.reshape(nemf * nind * nsigma, nfac)

        self.exp_covs = np.zeros((nemf * nind, nfac, nfac))
        self.exp_covs[:] = np.array(
            [[4.75, 4.5, 4.5],
             [4.5, 4.75, 4.5],
             [4.5, 4.5, 4.75]])

        self.exp_cholcovs = np.zeros_like(self.exp_covs)
        self.exp_cholcovs[:] = np.array(
            [[2.23606798, 0.00000000, 0.00000000],
             [1.78885438, 1.34164079, 0.00000000],
             [1.78885438, 0.596284794, 1.20185043]]).T

        self.sws_m = np.ones(nsigma) / nsigma
        self.sws_c = self.sws_m

        q = np.eye(nfac)
        self.Q = np.zeros((2, nfac, nfac))
        self.Q[:] = q

        self.transform_sps_args = {}

        self.out_states = np.zeros((nemf * nind, nfac))
        self.out_sqrt_covs = np.zeros((nemf * nind, nfac + 1, nfac + 1))
        self.out_covs = self.out_sqrt_covs[:, 1:, 1:]

    to_patch = \
        'skillmodels.fast_routines.numpy_kalman_filters.transform_sigma_points'

    @patch(to_patch)
    def test_normal_unscented_predict_focus_on_colums(self, mock_transform):
        mock_transform.return_value = self.sps1
        kf.normal_unscented_predict(
            self.stage, self.sps1, self.flat_sps1, self.sws_m, self.sws_c,
            self.Q, self.transform_sps_args, self.out_states, self.out_covs)

        aaae(self.out_states, self.expected_states1)

    @patch(to_patch)
    def test_normal_unscented_predict_focus_on_weighting(self, mock_transform):
        mock_transform.return_value = self.sps2
        kf.normal_unscented_predict(
            self.stage, self.sps2, self.flat_sps2, self.sws_m, self.sws_c,
            self.Q, self.transform_sps_args, self.out_states, self.out_covs)

        aaae(self.out_states, self.expected_states2)

    @patch(to_patch)
    def test_normal_unscented_predict_focus_on_covs(self, mock_transform):
        mock_transform.return_value = self.sps3
        self.Q[:] = np.eye(3) * 0.25 + np.ones((3, 3)) * 0.5
        kf.normal_unscented_predict(
            self.stage, self.sps3, self.flat_sps3, self.sws_m, self.sws_c,
            self.Q, self.transform_sps_args, self.out_states, self.out_covs)

        aaae(self.out_covs, self.exp_covs)

    @patch(to_patch)
    def test_sqrt_unscented_predict_focus_on_colums(self, mock_transform):
        mock_transform.return_value = self.sps1
        kf.sqrt_unscented_predict(
            self.stage, self.sps1, self.flat_sps1, self.sws_m, self.sws_c,
            self.Q, self.transform_sps_args, self.out_states, self.out_sqrt_covs)

        aaae(self.out_states, self.expected_states1)

    @patch(to_patch)
    def test_sqrt_unscented_predict_focus_on_weighting(self, mock_transform):
        mock_transform.return_value = self.sps2
        kf.sqrt_unscented_predict(
            self.stage, self.sps2, self.flat_sps2, self.sws_m, self.sws_c,
            self.Q, self.transform_sps_args, self.out_states, self.out_sqrt_covs)

        aaae(self.out_states, self.expected_states2)

    @patch(to_patch)
    def test_sqrt_unscented_predict_focus_on_covs(self, mock_transform):
        mock_transform.return_value = self.sps3
        # self.q = np.eye(3) * 0.25 + np.ones((3, 3)) * 0.5
        kf.sqrt_unscented_predict(
            self.stage, self.sps3, self.flat_sps3, self.sws_m, self.sws_c,
            self.Q, self.transform_sps_args, self.out_states, self.out_sqrt_covs)
        make_unique(self.out_covs)
        aaae(self.out_covs, self.exp_cholcovs)


if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
