import skillmodels.fast_routines.kalman_filters as kf
from numpy.testing import assert_array_almost_equal as aaae
import numpy as np
from unittest.mock import patch
import pytest



def make_unique(qr_result_arr):
    long_side, m, n = qr_result_arr.shape
    for u in range(long_side):
        for j in range(n):
            if qr_result_arr[u, j, j] < 0:
                for k in range(n):
                    qr_result_arr[u, j, k] *= -1


#test linear update with nans
@pytest.fixture
def setup_sqrt_state_with_nans():
    out = {}
            
    nemf, nind, nfac = 2, 6, 3
    states = np.ones((nind, nemf, nfac))
    states[:, 1, 2] *= 2
    out['state'] = states
    
    covs = np.zeros((nind, nemf, nfac, nfac))
    covs[:] = np.ones((nfac, nfac)) * 0.1 + np.eye(nfac) * .6
    mcovs = np.zeros((nind, nemf, nfac + 1, nfac + 1))
    mcovs[:, :, 1:, 1:] = np.transpose(np.linalg.cholesky(covs),
                                                axes=(0, 1, 3, 2))
    out['mcovs'] = mcovs
    
       
    out['like_vector'] = np.ones(nind)        
    
    out['y'] = np.array([3.5, 2.3, np.nan, 3.1, 4, np.nan])
    
    out['c'] = np.ones((nind, 2))
    
    out['delta'] = np.ones(2) / 2
    
    out['h'] = np.array([1, 1, 0.5])
    
    out['sqrt_r'] = np.sqrt(np.array([0.3]))
    
    out['positions'] = np.array([0, 1, 2])
    
    weights = np.ones((nind, nemf))
    weights[:, 0] *= 0.4
    weights[:, 1] *= 0.6
    out['weights'] = weights
    
    return out

@pytest.fixture
def expected_sqrt_state_with_nans():
    return np.array(
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

    
def test_sqrt_state_update_with_nans(setup_sqrt_state_with_nans,
                                     expected_sqrt_state_with_nans):
    kf.sqrt_linear_update(**setup_sqrt_state_with_nans)
    modified_states = setup_sqrt_state_with_nans['state']
    aaae(modified_states, expected_sqrt_state_with_nans)
    
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
        'skillmodels.fast_routines.kalman_filters.transform_sigma_points'

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

