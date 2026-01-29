import io
import textwrap
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import model_spec_from_yaml_dict
from numpy.testing import assert_array_equal as aae

from skillmodels.config import TEST_DATA_DIR
from skillmodels.process_data import (
    _augment_data_for_endogenous_factors,
    _generate_controls_array,
    _generate_measurements_array,
    _generate_observed_factor_array,
    _handle_controls_with_missings,
    pre_process_data,
)
from skillmodels.process_model import process_model
from skillmodels.types import Labels


def test_pre_process_data() -> None:
    df = pd.DataFrame(data=np.arange(20).reshape(2, 10).T, columns=["var", "inv"])
    df["period"] = [1, 2, 3, 2, 3, 4, 2, 4, 3, 1]
    df["id"] = [1, 1, 1, 3, 3, 3, 4, 4, 5, 5]
    df = df.set_index(["id", "period"])

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
    exp = exp.set_index(["__id__", "__period__"])

    res = pre_process_data(df, [0, 1, 2, 3])
    assert res[["var", "inv"]].equals(exp[["var", "inv"]])


@pytest.fixture
def simplest_augmented():
    out = {}
    with (TEST_DATA_DIR / "simplest_augmented_model.yaml").open() as y:
        out["model"] = model_spec_from_yaml_dict(yaml.load(y, Loader=yaml.SafeLoader))
    _df = pd.DataFrame(data=np.arange(15).reshape(3, 5).T, columns=["var", "inv", "of"])
    _df["period"] = [1, 1, 2, 1, 2]
    _df["id"] = [1, 3, 3, 5, 5]
    out["data_input"] = _df.set_index(["id", "period"])
    out["data_exp"] = pd.read_csv(
        TEST_DATA_DIR / "simplest_augmented_data_expected.csv",
        index_col=["id", "aug_period"],
    )
    return out


def test_augment_data_for_endogenous_factors(simplest_augmented) -> None:
    processed_model = process_model(simplest_augmented["model"])
    pre_processed_data = pre_process_data(
        simplest_augmented["data_input"], processed_model.labels.periods
    )
    pre_processed_data["constant"] = 1
    res = _augment_data_for_endogenous_factors(
        df=pre_processed_data,
        labels=processed_model.labels,
        update_info=processed_model.update_info,
    )
    cols = ["var", "inv", "constant", "of"]
    pd.testing.assert_frame_equal(res[cols], simplest_augmented["data_exp"][cols])


def test_handle_controls_with_missings() -> None:
    controls = ("c1",)
    uinfo_ind_tups = [(0, "m1"), (0, "m2")]
    update_info = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_ind_tups))
    data = [[1, 1, 1], [np.nan, 1, 1], [np.nan, 1, np.nan], [np.nan, np.nan, np.nan]]
    df = pd.DataFrame(data=data, columns=["m1", "m2", "c1"])
    df["aug_period"] = 0
    df["id"] = np.arange(4)
    df["__old_id__"] = df["id"]
    df["__old_period__"] = df["aug_period"] + 1
    df = df.set_index(["id", "aug_period"])

    with pytest.warns(UserWarning):  # noqa: PT030
        calculated = _handle_controls_with_missings(df, controls, update_info)
    assert calculated.loc[(2, 0)].isna().all()  # ty: ignore[unresolved-attribute]


def test_generate_measurements_array() -> None:
    uinfo_ind_tups = [(0, "m1"), (0, "m2"), (1, "m1"), (1, "m3")]
    update_info = pd.DataFrame(index=pd.MultiIndex.from_tuples(uinfo_ind_tups))

    csv = """
    id,aug_period,m1,m2,m3
    0,0,1,2,3
    0,1,4,5,6
    1,0,7,8,9
    1,1,10,11,12
    """
    data = _read_csv_string(csv, ["id", "aug_period"])

    expected = jnp.array([[1, 7], [2, 8], [4, 10], [6, 12.0]])

    calculated = _generate_measurements_array(data, update_info, 2)
    aae(calculated, expected)


def test_generate_controls_array() -> None:
    csv = """
    id,aug_period,c1,c2
    0, 0, 1, 2
    0, 1, 3, 4
    1, 0, 5, 8
    1, 1, 7, 8
    """
    data = _read_csv_string(csv, ["id", "aug_period"])

    labels = Labels(
        latent_factors=(),
        observed_factors=(),
        controls=("c1", "c2"),
        periods=(0, 1),
        stagemap=(0, 0),
        stages=(0,),
        aug_periods=(0, 1),
        aug_periods_to_periods=MappingProxyType({0: 0, 1: 1}),
        aug_stagemap=(0, 0),
        aug_stages=(0,),
        aug_stages_to_stages=MappingProxyType({0: 0}),
    )

    calculated = _generate_controls_array(data, labels, 2)
    expected = jnp.array([[[1, 2], [5, 8]], [[3, 4], [7, 8]]])
    aae(calculated, expected)


def test_generate_observed_factor_array() -> None:
    csv = """
    id,aug_period,v1,v2
    0, 0, 1, 2
    0, 1, 3, 4
    1, 0, 5, 8
    1, 1, 7, 8
    """
    data = _read_csv_string(csv, ["id", "aug_period"])

    labels = Labels(
        latent_factors=(),
        observed_factors=("v1", "v2"),
        controls=("constant",),
        periods=(0, 1),
        stagemap=(0, 0),
        stages=(0,),
        aug_periods=(0, 1),
        aug_periods_to_periods=MappingProxyType({0: 0, 1: 1}),
        aug_stagemap=(0, 0),
        aug_stages=(0,),
        aug_stages_to_stages=MappingProxyType({0: 0}),
    )

    calculated = _generate_observed_factor_array(data, labels, 2)
    expected = jnp.array([[[1, 2], [5, 8]], [[3, 4], [7, 8]]])
    aae(calculated, expected)


def _read_csv_string(string, index_cols):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)
