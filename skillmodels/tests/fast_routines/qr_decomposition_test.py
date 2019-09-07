import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.fast_routines.qr_decomposition import array_qr

# from numpy.core.umath_tests import matrix_multiply


def a_prime_a(a):
    return np.matmul(np.transpose(a, axes=(0, 2, 1)), a)


class TestRFromQR:
    def setup(self):
        pass

    def test_non_square_r_from_qr(self):
        self.some_array = np.random.randn(200, 7, 3)
        self.expected_prod = a_prime_a(self.some_array)
        prod = array_qr(self.some_array)
        aaae(a_prime_a(prod), self.expected_prod)

    def test_square_r_from_qr(self):
        self.some_array = np.random.randn(200, 12, 12)
        self.expected_prod = a_prime_a(self.some_array)
        prod = array_qr(self.some_array)
        aaae(a_prime_a(prod), self.expected_prod)
