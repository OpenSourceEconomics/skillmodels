import io
import textwrap

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae

from skillmodels.process_data import (
    _augment_data_for_investments,
    _generate_controls_array,
    _generate_measurements_array,
    _generate_observed_factor_array,
    _handle_controls_with_missings,
    pre_process_data,
)
from skillmodels.process_model import process_model


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
    df = pd.DataFrame(data=np.arange(15).reshape(3, 5).T, columns=["var", "inv", "of"])
    df["period"] = [1, 1, 2, 1, 2]
    df["id"] = [1, 3, 3, 5, 5]
    df.set_index(["id", "period"], inplace=True)

    model_dict = {
        "factors": {
            "fac1": {
                "measurements": [["var"], ["var"]],
                "transition_function": "linear",
                "is_investment": False,
                "is_correction": False,
            },
            "fac2": {
                "measurements": [["inv"], ["inv"]],
                "transition_function": "linear",
                "is_investment": True,
                "is_correction": False,
            },
        },
        "observed_factors": ["of"],
        "estimation_options": {},
    }

    model = process_model(model_dict)
    pre_processed_data = pre_process_data(df, model["labels"]["periods_raw"])
    pre_processed_data["constant"] = 1

    nan = np.nan
    exp = pd.DataFrame(
        {
            "id": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 2,
                9: 2,
                10: 2,
                11: 2,
            },
            "period": {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 0,
                5: 1,
                6: 2,
                7: 3,
                8: 0,
                9: 1,
                10: 2,
                11: 3,
            },
            "period_raw": {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 0,
                5: 0,
                6: 1,
                7: 1,
                8: 0,
                9: 0,
                10: 1,
                11: 1,
            },
            "constant": {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
                10: 1,
                11: 1,
            },
            "var": {
                0: 0.0,
                1: nan,
                2: nan,
                3: nan,
                4: 1.0,
                5: nan,
                6: 2.0,
                7: nan,
                8: 3.0,
                9: nan,
                10: 4.0,
                11: nan,
            },
            "inv": {
                0: nan,
                1: 5.0,
                2: nan,
                3: nan,
                4: nan,
                5: 6.0,
                6: nan,
                7: 7.0,
                8: nan,
                9: 8.0,
                10: nan,
                11: 9.0,
            },
            "of": {
                0: 10.0,
                1: 10.0,
                2: nan,
                3: nan,
                4: 11.0,
                5: 11.0,
                6: 12.0,
                7: 12.0,
                8: 13.0,
                9: 13.0,
                10: 14.0,
                11: 14.0,
            },
        }
    ).set_index(["id", "period"])

    res = _augment_data_for_investments(
        df=pre_processed_data,
        labels=model["labels"],
        update_info=model["update_info"],
    )
    cols = ["period_raw", "var", "inv", "constant", "of"]
    pd.testing.assert_frame_equal(res[cols], exp[cols])


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
