from numpy.testing import assert_array_almost_equal as aaae
from numpy.core.umath_tests import matrix_multiply
import numpy as np
from skillmodels.fast_routines.qr_decomposition import numba_array_qr


def a_prime_a(a):
    return matrix_multiply(np.transpose(a, axes=(0, 2, 1)), a)


class TestRFromQR:
    def setup(self):
        pass

    def test_non_square_r_from_qr(self):
        self.some_array = np.random.randn(200, 7, 3)
        self.expected_prod = a_prime_a(self.some_array)
        prod = numba_array_qr(self.some_array)
        aaae(a_prime_a(prod), self.expected_prod)

    def test_square_r_from_qr(self):
        self.some_array = np.random.randn(200, 12, 12)
        self.expected_prod = a_prime_a(self.some_array)
        prod = numba_array_qr(self.some_array)
        aaae(a_prime_a(prod), self.expected_prod)
# def setup_square():
#     global some_array, expected_prod
#     some_array = np.random.randn(200, 8, 8)
#     expected_prod = a_prime_a(some_array)
#
#
# def setup_non_square():
#     global some_array, expected_prod
#     some_array = np.random.randn(100, 7, 3)
#     expected_prod = a_prime_a(some_array)
#
#
# @with_setup(setup_square)
# def test_square_numba_qr_decomposition_default():
#     prod = a_prime_a(numba_array_qr(some_array))
#     aaae(prod, expected_prod, decimal=12)
#
#
# @with_setup(setup_non_square)
# def test_non_square_numba_qr_decomposition_default():
#     prod = a_prime_a(numba_array_qr(some_array))
#     aaae(prod, expected_prod, decimal=12)


if __name__ == '__main__':
    np.random.seed(2310895471)
    from nose.core import runmodule
    runmodule()


