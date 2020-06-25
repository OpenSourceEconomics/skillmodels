import io
import textwrap

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae

from skillmodels.process_data_jax import _generate_anchoring_variables_array
from skillmodels.process_data_jax import _generate_controls_array
from skillmodels.process_data_jax import _generate_measurements_array
from skillmodels.process_data_jax import _handle_controls_with_missings
from skillmodels.process_data_jax import _pre_process_data


def test_pre_process_data():
    df = pd.DataFrame(data=np.arange(10).reshape(10, 1), columns=["var"])
    df["period"] = [1, 2, 3, 2, 3, 4, 2, 4, 3, 1]
    df["id"] = [1, 1, 1, 3, 3, 3, 4, 4, 5, 5]
    df.set_index(["id", "period"], inplace=True)

    exp = pd.DataFrame()
    period = [0, 1, 2, 3] * 4
    id_ = np.arange(4).repeat(4)
    nan = np.nan
    data = [0, 1, 2, nan, nan, 3, 4, 5, nan, 6, nan, 7, 9, nan, 8, nan]
    data = np.column_stack([period, id_, data])
    exp = pd.DataFrame(data=data, columns=["__period__", "__id__", "var"])
    exp.set_index(["__id__", "__period__"], inplace=True)

    res = _pre_process_data(df)

    assert res["var"].equals(exp["var"])


def test_handle_controls_with_missings():
    controls = ["c1"]
    uinfo_ind_tups = [(0, "m1"), (0, "m2")]
    update_info = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_ind_tups))
    data = [[1, 1, 1], [np.nan, 1, 1], [np.nan, 1, np.nan], [np.nan, np.nan, np.nan]]
    df = pd.DataFrame(data=data, columns=["m1", "m2", "c1"])
    df["period"] = 0
    df["id"] = np.arange(4)
    df["__old_id__"] = df["id"]
    df["__old_period__"] = df["period"] + 1
    df.set_index(["id", "period"], inplace=True)

    with pytest.warns(UserWarning):
        calculated = _handle_controls_with_missings(df, controls, update_info)
    assert calculated.loc[(2, 0)].isnull().all()


def test_generate_measurements_array():
    uinfo_ind_tups = [(0, "m1"), (0, "m2"), (1, "m1"), (1, "m3")]
    update_info = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_ind_tups))

    csv = """
    id,period,m1,m2,m3
    0,0,1,2,3
    0,1,4,5,6
    1,0,7,8,9
    1,1,10,11,12
    """
    data = _read_csv_string(csv, ["id", "period"])

    expected = jnp.array([[1, 7], [2, 8], [4, 10], [6, 12.0]])

    calculated = _generate_measurements_array(data, update_info, 2)
    aae(calculated, expected)


def test_generate_controls_array():
    csv = """
    id,period,c1,c2
    0, 0, 1, 2
    0, 1, 3, 4
    1, 0, 5, 8
    1, 1, 7, 8
    """
    data = _read_csv_string(csv, ["id", "period"])

    labels = {"controls": ["c1", "c2"], "periods": [0, 1]}

    calculated = _generate_controls_array(data, labels, 2)
    expected = jnp.array([[[1, 2], [5, 8]], [[3, 4], [7, 8]]])
    aae(calculated, expected)


def test_generate_anchoring_variables_array():
    df = pd.DataFrame()
    df["id"] = [0, 1]
    df["period"] = 0
    df.set_index(["id", "period"], inplace=True)
    df["q1"] = [5, 6]

    labels = {"periods": [0], "factors": ["fac1", "fac2"]}
    anchoring_info = {"factors": ["fac1"], "outcome": "q1"}

    expected = jnp.array([[[5, 0], [6, 0]]])
    calculated = _generate_anchoring_variables_array(df, labels, anchoring_info, 2)
    aae(calculated, expected)


def _read_csv_string(string, index_cols):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)
