"""Test parameter parsing with example model 2 from CHS2010.

Only test the create_parsing_info and parse_params jointly, to abstract from
implementation details.

"""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae

from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.model_spec import CorrectionSpec
from skillmodels.common.params_index import get_params_index
from skillmodels.common.parse_params import create_parsing_info, parse_params
from skillmodels.common.process_model import process_model
from skillmodels.common.types import Anchoring
from skillmodels.test_data.model2 import MODEL2


def test_parse_params_populates_first_stage_reserved_key() -> None:
    """The first-stage betas parse into a reserved trans_coeffs key.

    The contemporaneous first-stage equation's coefficients live under the
    `investment_eq` params category but are threaded into the transition-coeffs
    dict under a reserved `__first_stage_<inv>__` key, so the prediction DAG
    node reads them with no change to the predict signature.
    """
    fac3 = MODEL2.factors["fac3"]
    corr = CorrectionSpec(instruments=("inv_z",))
    new_fac3 = replace(fac3, is_endogenous=True, correction=corr)
    new_factors = dict(MODEL2.factors) | {"fac3": new_fac3}
    model = MODEL2._replace(factors=new_factors)._replace(stagemap=None)
    model = model._replace(observed_factors=("inv_z",))
    processed = process_model(model)

    p_index = get_params_index(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        transition_info=processed.transition_info,
        endogenous_factors_info=processed.endogenous_factors_info,
    )
    parsing_info = create_parsing_info(
        params_index=p_index,
        update_info=processed.update_info,
        labels=processed.labels,
        anchoring=processed.anchoring,
        has_endogenous_factors=True,
    )
    assert parsing_info.investment_factor == "fac3"

    params_vec = jnp.arange(len(p_index)).astype(float)
    _, _, _, parsed = parse_params(
        params_vec, parsing_info, processed.dimensions, processed.labels, n_obs=5
    )
    reserved = "__first_stage_fac3__"
    assert reserved in parsed.transition
    n_aug = len(processed.labels.aug_periods)
    # (aug_periods[:-2], state_predictors + instruments + constant) = (14, 4).
    assert parsed.transition[reserved].shape == (n_aug - 2, 4)
    # kappa reserved keys, one per target, with one cf column each.
    assert "__kappa_fac1__" in parsed.transition
    assert "__kappa_fac2__" in parsed.transition
    assert parsed.transition["__kappa_fac1__"].shape == (n_aug - 2, 1)


@pytest.fixture
def parsed_parameters():
    p_index = pd.read_csv(
        TEST_DATA_DIR / "model2_correct_params_index.csv",
        index_col=["category", "aug_period", "name1", "name2"],
    ).index

    processed = process_model(MODEL2)

    update_info = processed.update_info
    labels = processed.labels
    dimensions = processed.dimensions
    # this overwrites the anchoring setting from the model specification to get a
    # more meaningful test
    anchoring = Anchoring(
        anchoring=False,
        outcomes=MappingProxyType({}),
        factors=(),
        free_controls=True,
        free_constant=True,
        free_loadings=True,
        ignore_constant_when_anchoring=False,
    )

    parsing_info = create_parsing_info(
        params_index=p_index,  # ty: ignore[invalid-argument-type]
        update_info=update_info,
        labels=labels,
        anchoring=anchoring,
        has_endogenous_factors=False,
    )

    params_vec = jnp.arange(len(p_index))
    n_obs = 5

    states, upper_chols, log_weights, parsed_params = parse_params(
        params_vec, parsing_info, dimensions, labels, n_obs
    )

    return {
        "states": states,
        "upper_chols": upper_chols,
        "log_weights": log_weights,
        "parsed_params": parsed_params,
    }


def test_controls(parsed_parameters) -> None:
    expected = jnp.arange(118).reshape(59, 2)
    aae(parsed_parameters["parsed_params"].controls, expected)


def test_loadings(parsed_parameters) -> None:
    expected_values = jnp.arange(118, 177)
    calculated = parsed_parameters["parsed_params"].loadings
    calculated_values = calculated[calculated != 0]
    aae(expected_values, calculated_values)


def test_meas_sds(parsed_parameters) -> None:
    expected = jnp.arange(177, 236)
    aae(parsed_parameters["parsed_params"].meas_sds, expected)


def test_shock_sds(parsed_parameters) -> None:
    expected = jnp.arange(236, 257).reshape(7, 3)
    aae(parsed_parameters["parsed_params"].shock_sds, expected)


def test_initial_states(parsed_parameters) -> None:
    expected = jnp.arange(257, 260).reshape(1, 3).repeat(5, axis=0).reshape(5, 1, 3)
    aae(parsed_parameters["states"], expected)


def test_initial_upper_chols(parsed_parameters) -> None:
    expected = (
        jnp.array([[[261, 262, 264], [0, 263, 265], [0, 0, 266]]])
        .repeat(5, axis=0)
        .reshape(5, 1, 3, 3)
    )
    aae(parsed_parameters["upper_chols"], expected)


def test_transition_parameters(parsed_parameters) -> None:
    calculated = parsed_parameters["parsed_params"].transition

    aae(calculated["fac1"], jnp.arange(385, 413).reshape(7, 4) - 118)
    aae(calculated["fac2"], jnp.arange(413, 441).reshape(7, 4) - 118)
    aae(calculated["fac3"], jnp.zeros((7, 0)))

    assert isinstance(calculated, Mapping)


def test_anchoring_scaling_factors(parsed_parameters) -> None:
    calculated = parsed_parameters["parsed_params"].anchoring_scaling_factors
    expected = np.ones((8, 3))
    expected[:, 0] = jnp.array([127 + 7 * i for i in range(8)])
    aae(calculated, expected)


def test_anchoring_constants(parsed_parameters) -> None:
    calculated = parsed_parameters["parsed_params"].anchoring_constants
    expected = np.zeros((8, 3))
    expected[:, 0] = jnp.array([18 + i * 14 for i in range(8)])
    aae(calculated, expected)
