from numpy.testing import assert_array_almost_equal as aaae
import numpy as np
from numpy.linalg import cholesky
from skillmodels.fast_routines.sigma_points import calculate_sigma_points
import json


class TestSigmaPointConstructionAgainstFilterpy:
    def setup(self):

        self.nemf = 2
        self.nind = 10
        self.nfac = 4
        self.nsigma = 9

        state1 = np.array([2, 3, 4, 5])
        state2 = np.array([[2, 1, 3, 0.5]])

        cov1 = np.array([[1, 0, 0.1, 0],
                        [0, 1.2, 0, 0.3],
                        [0.1, 0, 2.1, 0.2],
                        [0, 0.3, 0.2, 2.8]])
        cov2 = np.eye(4) + 0.2

        self.states = np.zeros((self.nemf, self.nind, self.nfac))
        self.states[:, :5, :] = state1
        self.states[:, 5:, :] = state2
        self.states = self.states.reshape(self.nemf * self.nind, self.nfac)

        self.covs = np.zeros((self.nemf, self.nind, self.nfac, self.nfac))
        self.covs[0, :, :, :] = cov1
        self.covs[1, :, :, :] = cov2

        self.lcovs_t = np.zeros((
            self.nemf, self.nind, self.nfac + 1, self.nfac + 1))

        self.lcovs_t[:, :, 1:, 1:] = np.transpose(
            cholesky(self.covs), axes=(0, 1, 3, 2))
        self.lcovs_t = self.lcovs_t.reshape(
            self.nemf * self.nind, self.nfac + 1, self.nfac + 1)
        self.out = np.zeros((self.nemf * self.nind, self.nsigma, self.nfac))

        # these test results have been calculated with the sigma_point
        # function of the filterpy library
        with open('tests/fast_routines/sigma_points_from_filterpy.json') as f:
            self.fixtures = json.load(f)

    def test_julier_sigma_point_construction(self):
        expected_sps = np.array(self.fixtures['julier_points']).reshape(
            self.nemf * self.nind, self.nsigma, self.nfac)
        calculate_sigma_points(states=self.states, flat_covs=self.lcovs_t,
                               scaling_factor=2.34520787991, out=self.out,
                               square_root_filters=True)
        aaae(self.out, expected_sps)

    def test_merwe_sigma_point_construction(self):
        expected_sps = np.array(self.fixtures['merwe_points']).reshape(
            self.nemf * self.nind, self.nsigma, self.nfac)
        calculate_sigma_points(states=self.states, flat_covs=self.lcovs_t,
                               scaling_factor=0.234520787991, out=self.out,
                               square_root_filters=True)
        aaae(self.out, expected_sps)

if __name__ == '__main__':
    from nose.core import runmodule
    runmodule()
