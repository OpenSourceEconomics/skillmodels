from numpy.testing import assert_array_almost_equal as aaae
from skillmodels.fast_routines.choldate import array_choldate
from numpy.core.umath_tests import matrix_multiply
import numpy as np
from numpy.linalg import cholesky


def setup_(weight):
    to_update = np.zeros((20, 4, 4))
    helper_bool = np.zeros((4, 4), dtype=bool)
    helper_bool[np.triu_indices(4)] = True

    for i in range(20):
        to_update[i][helper_bool] = np.random.randn(10) + 100

    pos_def_arr = matrix_multiply(np.transpose(to_update, axes=(0, 2, 1)),
                                  to_update)

    update_with = np.random.uniform(size=(20, 4))

    outer_prod = update_with.reshape(20, 4, 1) * \
        update_with.reshape(20, 1, 4)

    expected_result = np.transpose(
        cholesky(pos_def_arr + weight * outer_prod), axes=(0, 2, 1))

    return to_update, update_with, expected_result


def test_numba_choldate_with_positive_weight():
    w = 0.8
    tu, uw, er = setup_(weight=w)
    aaae(array_choldate(to_update=tu, update_with=uw, weight=w), er)


def test_numba_choldate_with_negative_weight():
    w = -0.1
    tu, uw, er = setup_(weight=w)
    aaae(array_choldate(to_update=tu, update_with=uw, weight=w), er)
