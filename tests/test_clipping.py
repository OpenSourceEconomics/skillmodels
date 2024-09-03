import jax.numpy as jnp
import numpy as np
from skillmodels.clipping import soft_clipping


def test_one_sided_soft_maximum():
    arr = jnp.array([-10.0, -5, -1, 1, 5, 10])
    lower_bound = -8
    lower_hardness = 3
    expected = []
    for x in arr:
        exp_part = jnp.exp(lower_hardness * x) + jnp.exp(lower_hardness * lower_bound)
        entry = jnp.log(exp_part) / lower_hardness
        expected.append(entry)

    res = soft_clipping(arr=arr, lower=lower_bound, lower_hardness=lower_hardness)
    # compare to calculation "by hand"
    np.testing.assert_allclose(res, np.array(expected))
    # compare that upper part is very close to true values
    np.testing.assert_allclose(res[1:], arr[1:], rtol=1e-05)


def test_one_sided_soft_minimum():
    arr = jnp.array([-10.0, -5, -1, 1, 5, 10])
    upper_bound = 8
    upper_hardness = 3

    expected = []
    for x in arr:
        # min(x, y) = -max(-x, -y)
        # min(3, 5) = 3 = -max(-3, -5) = -(-3)
        exp_part = jnp.exp(-upper_hardness * x) + jnp.exp(-upper_hardness * upper_bound)
        entry = -jnp.log(exp_part) / upper_hardness
        expected.append(entry)

    res = soft_clipping(arr=arr, upper=upper_bound, upper_hardness=upper_hardness)

    # compare to calculation "by hand"
    np.testing.assert_allclose(res, np.array(expected))

    # compare that the lower part is very close to true values
    np.testing.assert_allclose(res[:-1], arr[:-1], rtol=1e-05)
