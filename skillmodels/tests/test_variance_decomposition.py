import pandas as pd
import pytest
from jax import config
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.variance_decomposition import (
    create_dataset_with_variance_decomposition,
)

config.update("jax_enable_x64", True)

# ======================================================================================
# Variance decomposition
# ======================================================================================


@pytest.fixture
def setup_variance_decomposition():
    data1 = {
        "fac1": [0.1, 0.1, 0.1, 0.2],
        "fac2": [0.1] * 4,
        "fac3": [0.2, 0.2, 0.2, 0.4],
        "mixture": [0] * 4,
        "period": [0] * 4,
        "id": [0] * 4,
    }
    setup_filtered_states = pd.DataFrame(data1)

    value_loadings = [1, 0, 0] + [0, 0.1, 0] + [0, 0, 2]
    value_meas_sds = [0.05, 1.1, 0.1]
    iterables1 = [[0], ["y1", "y2", "y3"], ["fac1", "fac2", "fac3"]]
    index1 = pd.MultiIndex.from_product(iterables1, names=["period", "name1", "name2"])
    setup_loadings = pd.DataFrame(value_loadings, index=index1, columns=["value"])
    iterables2 = [[0], ["y1", "y2", "y3"]]
    index2 = pd.MultiIndex.from_product(iterables2, names=["period", "name1"])

    setup_meas = pd.DataFrame(value_meas_sds, index=index2, columns=["value"])
    setup_meas["name2"] = "-"
    setup_meas = setup_meas.reset_index()
    setup_meas = setup_meas.set_index(["period", "name1", "name2"])
    setup_params = pd.concat(
        [setup_loadings, setup_meas], keys=["loadings", "meas_sds"]
    )

    args = {"filtered_states": setup_filtered_states, "params": setup_params}
    return args


@pytest.fixture
def expected_variance_decomposition():
    value3 = [
        [1, 0.0025, 0.05, 0.5, 0.5],
        [0.1, 0, 1.1, 1, 0],
        [2, 0.01, 0.1, 0.2, 0.8],
    ]
    iterables3 = [(0, "y1", "fac1"), (0, "y2", "fac2"), (0, "y3", "fac3")]
    index3 = pd.MultiIndex.from_tuples(iterables3, names=("period", "name1", "name2"))
    expected_result = pd.DataFrame(
        value3,
        index=index3,
        columns=[
            "loadings",
            "variance of factor",
            "meas_sds",
            "fraction due to meas error",
            "fraction due to factor var",
        ],
    )
    return expected_result


def test_variance_decomposition(
    setup_variance_decomposition, expected_variance_decomposition
):
    aaae(
        create_dataset_with_variance_decomposition(**setup_variance_decomposition),
        expected_variance_decomposition,
    )
