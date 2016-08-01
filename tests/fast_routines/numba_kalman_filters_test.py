import skillmodels.fast_routines.numba_kalman_filters as kf
from numpy.testing import assert_array_almost_equal as aaae
import numpy as np


def make_unique(qr_result_arr):
    long_side, m, n = qr_result_arr.shape
    for u in range(long_side):
        for j in range(n):
            if qr_result_arr[u, j, j] < 0:
                for k in range(n):
                    qr_result_arr[u, j, k] *= -1


class TestLinearUpdate:
    def setup(self):
        nemf = 2
        nind = 6
        nfac = 3

        self.states = np.ones((nind, nemf, nfac))
        self.states[:, 1, 2] *= 2

        self.covs = np.zeros((nind, nemf, nfac, nfac))
        self.covs[:] = np.ones((nfac, nfac)) * 0.1 + np.eye(nfac) * .6

        self.mcovs = np.zeros((nind, nemf, nfac + 1, nfac + 1))

        self.mcovs[:, :, 1:, 1:] = np.transpose(np.linalg.cholesky(self.covs),
                                                axes=(0, 1, 3, 2))

        self.weights = np.ones((nind, nemf))
        self.weights[:, 0] *= 0.4
        self.weights[:, 1] *= 0.6

        self.like_vec = np.ones(nind)
        self.y = np.array([3.5, 2.3, np.nan, 3.1, 4, np.nan])

        self.c = np.ones((nind, 2))

        self.delta = np.ones(2) / 2

        self.h = np.array([1, 1, 0.5])
        self.positions = np.array([0, 1, 2])

        self.sqrt_r = np.sqrt(np.array([0.3]))
        self.r = np.array([0.3])

        self.kf = np.zeros((nind, nfac))

        self.exp_states = np.array(
            [[[1.00000000, 1.00000000, 1.00000000],
              [0.81318681, 0.81318681, 1.87912088]],
             [[0.55164835, 0.55164835, 0.70989011],
              [0.36483516, 0.36483516, 1.58901099]],
             [[1.00000000, 1.00000000, 1.00000000],
              [1.00000000, 1.00000000, 2.00000000]],
             [[0.85054945, 0.85054945, 0.90329670],
              [0.66373626, 0.66373626, 1.78241758]],
             [[1.18681319, 1.18681319, 1.12087912],
              [1.00000000, 1.00000000, 2.00000000]],
             [[1.00000000, 1.00000000, 1.00000000],
              [1.00000000, 1.00000000, 2.00000000]]])

        self.exp_weights = np.array(
            [[0.41325632, 0.58674368],
             [0.47831766, 0.52168234],
             [0.40000000, 0.60000000],
             [0.43472272, 0.56527728],
             [0.38688853, 0.61311147],
             [0.40000000, 0.60000000]])

        self.expected_like_vec = np.array(
            [0.25601173, 0.16118176, 1., 0.23496064, 0.25883987, 1.])

        self.exp_covs = np.array(
            [[[[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]],
              [[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]]],
             [[[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]],
              [[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]]],
             [[[0.7, 0.1, 0.1],
               [0.1, 0.7, 0.1],
               [0.1, 0.1, 0.7]],
              [[0.7, 0.1, 0.1],
               [0.1, 0.7, 0.1],
               [0.1, 0.1, 0.7]]],
             [[[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]],
              [[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]]],
             [[[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]],
              [[0.38241758, -0.21758242, -0.10549451],
               [-0.21758242, 0.38241758, -0.10549451],
               [-0.10549451, -0.10549451, 0.56703297]]],
             [[[0.7, 0.1, 0.1],
               [0.1, 0.7, 0.1],
               [0.1, 0.1, 0.7]],
              [[0.7, 0.1, 0.1],
               [0.1, 0.7, 0.1],
               [0.1, 0.1, 0.7]]]])

        self.exp_cholcovs = np.transpose(np.linalg.cholesky(self.exp_covs),
                                         axes=(0, 1, 3, 2))

    def test_sqrt_state_update_with_nans(self):
        kf.sqrt_linear_update(
            self.states, self.mcovs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.sqrt_r, self.positions, self.weights)

        aaae(self.states, self.exp_states)

    def test_sqrt_cov_update_with_nans(self):
        kf.sqrt_linear_update(
            self.states, self.mcovs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.sqrt_r, self.positions, self.weights)
        cholcovs = self.mcovs[:, :, 1:, 1:]
        make_unique(cholcovs.reshape(12, 3, 3))
        aaae(cholcovs, self.exp_cholcovs)

    def test_sqrt_like_vec_update_with_nans(self):
        kf.sqrt_linear_update(
            self.states, self.mcovs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.sqrt_r, self.positions, self.weights)
        aaae(self.like_vec, self.expected_like_vec)

    def test_sqrt_weight_update_with_nans(self):
        kf.sqrt_linear_update(
            self.states, self.mcovs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.sqrt_r, self.positions, self.weights)
        aaae(self.weights, self.exp_weights)

    def test_normal_state_update_with_nans(self):
        kf.normal_linear_update(
            self.states, self.covs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.r, self.positions, self.weights, self.kf)

        aaae(self.states, self.exp_states)

    def test_normal_cov_update_with_nans(self):
        kf.normal_linear_update(
            self.states, self.covs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.r, self.positions, self.weights, self.kf)
        aaae(self.covs, self.exp_covs)

    def test_normal_like_vec_update_with_nans(self):
        kf.normal_linear_update(
            self.states, self.covs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.r, self.positions, self.weights, self.kf)
        aaae(self.like_vec, self.expected_like_vec)

    def test_normal_weight_update_with_nans(self):
        kf.normal_linear_update(
            self.states, self.covs, self.like_vec, self.y, self.c,
            self.delta, self.h, self.r, self.positions, self.weights, self.kf)
        aaae(self.weights, self.exp_weights)

if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
