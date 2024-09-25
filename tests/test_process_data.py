import io
import textwrap

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae

from skillmodels.process_data import (
    _generate_controls_array,
    _generate_measurements_array,
    _generate_observed_factor_array,
    _handle_controls_with_missings,
    pre_process_data,
)
from skillmodels.process_model import _get_labels, get_dimensions, get_has_investments


def test_pre_process_data():
    df = pd.DataFrame(data=np.arange(20).reshape(2, 10).T, columns=["var", "inv"])
    df["period"] = [1, 2, 3, 2, 3, 4, 2, 4, 3, 1]
    df["id"] = [1, 1, 1, 3, 3, 3, 4, 4, 5, 5]
    df.set_index(["id", "period"], inplace=True)

    exp = pd.DataFrame()
    period = [0, 1, 2, 3] * 4
    id_ = np.arange(4).repeat(4)
    nan = np.nan
    data = {
        "var": [0, 1, 2, nan, nan, 3, 4, 5, nan, 6, nan, 7, 9, nan, 8, nan],
        "inv": [10, 11, 12, nan, nan, 13, 14, 15, nan, 16, nan, 17, 19, nan, 18, nan],
    }
    data = np.column_stack([period, id_, data["var"], data["inv"]])
    exp = pd.DataFrame(data=data, columns=["__period__", "__id__", "var", "inv"])
    exp.set_index(["__id__", "__period__"], inplace=True)

    res = pre_process_data(df, [0, 1, 2, 3])
    assert res[["var", "inv"]].equals(exp[["var", "inv"]])


def test_augment_data_for_investments():
    df = pd.DataFrame(data=np.arange(10).reshape(2, 5).T, columns=["var", "inv"])
    df["period"] = [1, 1, 2, 1, 2]
    df["id"] = [1, 3, 3, 5, 5]
    df.set_index(["id", "period"], inplace=True)

    model_dict = {
        "factors": {
            "fac1": {
                "measurements": ["var", "var"],
                "transition_function": "linear",
                "is_investment": False,
                "is_correction": False,
            },
            "fac2": {
                "measurements": ["inv", "inv"],
                "transition_function": "linear",
                "is_investment": True,
                "is_correction": False,
            },
        },
        "estimation_options": {},
    }

    has_investments = get_has_investments(model_dict["factors"])
    dims = get_dimensions(model_dict, has_investments)
    labels = _get_labels(
        model_dict=model_dict, has_investments=has_investments, dimensions=dims
    )
    pre_processed = pre_process_data(df, labels["periods_raw"])
    breakpoint()

    exp = pd.DataFrame(
        {
            "id": {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2},
            "period": {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1},
            "var": {0: 0.0, 1: nan, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0},
            "inv": {0: 5.0, 1: nan, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.0},
        }
    )
    period = [0, 1, 2, 3] * 8
    id_ = np.arange(4).repeat(4)
    nan = np.nan
    data = {
        "var": [0, 1, 2, nan, nan, 3, 4, 5, nan, 6, nan, 7, 9, nan, 8, nan],
        "inv": [10, 11, 12, nan, nan, 13, 14, 15, nan, 16, nan, 17, 19, nan, 18, nan],
    }
    data = np.column_stack([period, id_, data["var"], data["inv"]])
    exp = pd.DataFrame(data=data, columns=["__period__", "__id__", "var", "inv"])
    exp.set_index(["__id__", "__period__"], inplace=True)

    # breakpoint()
    assert ["var"].equals(exp["var"])


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
    assert calculated.loc[(2, 0)].isna().all()


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


def test_generate_observed_factor_array():
    csv = """
    id,period,v1,v2
    0, 0, 1, 2
    0, 1, 3, 4
    1, 0, 5, 8
    1, 1, 7, 8
    """
    data = _read_csv_string(csv, ["id", "period"])

    labels = {"observed_factors": ["v1", "v2"], "periods": [0, 1]}

    calculated = _generate_observed_factor_array(data, labels, 2)
    expected = jnp.array([[[1, 2], [5, 8]], [[3, 4], [7, 8]]])
    aae(calculated, expected)


def _read_csv_string(string, index_cols):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)
