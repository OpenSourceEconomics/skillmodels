"""Test parameter parsing with example model 2 from CHS2010.

Only test the create_parsing_info and parse_params jointly, to abstract from
implementation details.

"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_array_equal as aae

from skillmodels.parse_params import create_parsing_info
from skillmodels.parse_params import parse_params
from skillmodels.process_model import process_model


@pytest.fixture
def parsed_parameters():
    test_dir = Path(__file__).parent.resolve()
    p_index = pd.read_csv(
        test_dir / "model2_correct_params_index.csv",
        index_col=["category", "period", "name1", "name2"],
    ).index

    with open(test_dir / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)

    processed = process_model(model_dict)

    update_info = processed["update_info"]
    labels = processed["labels"]
    dimensions = processed["dimensions"]
    # this overwrites the anchoring setting from the model specification to get a
    # more meaningful test
    anchoring = {"ignore_constant_when_anchoring": False}

    parsing_info = create_parsing_info(p_index, update_info, labels, anchoring)

    params_vec = jnp.arange(len(p_index))
    n_obs = 5

    parsed = parse_params(params_vec, parsing_info, dimensions, labels, n_obs)

    return dict(zip(["states", "upper_chols", "log_weights", "pardict"], parsed))


def test_controls(parsed_parameters):
    expected = jnp.arange(118).reshape(59, 2)
    aae(parsed_parameters["pardict"]["controls"], expected)


def test_loadings(parsed_parameters):
    expected_values = jnp.arange(118, 177)
    calculated = parsed_parameters["pardict"]["loadings"]
    calculated_values = calculated[calculated != 0]
    aae(expected_values, calculated_values)


def test_meas_sds(parsed_parameters):
    expected = jnp.arange(177, 236)
    aae(parsed_parameters["pardict"]["meas_sds"], expected)


def test_shock_sds(parsed_parameters):
    expected = jnp.arange(236, 257).reshape(7, 3)
    aae(parsed_parameters["pardict"]["shock_sds"], expected)


def test_initial_states(parsed_parameters):
    expected = jnp.arange(257, 260).reshape(1, 3).repeat(5, axis=0).reshape(5, 1, 3)
    aae(parsed_parameters["states"], expected)


def test_initial_upper_chols(parsed_parameters):
    expected = (
        jnp.array([[[261, 262, 264], [0, 263, 265], [0, 0, 266]]])
        .repeat(5, axis=0)
        .reshape(5, 1, 3, 3)
    )
    aae(parsed_parameters["upper_chols"], expected)


def test_transition_parameters(parsed_parameters):

    calculated = parsed_parameters["pardict"]["transition"]

    aae(calculated["fac1"], jnp.arange(385, 413).reshape(7, 4) - 118)
    aae(calculated["fac2"], jnp.arange(413, 441).reshape(7, 4) - 118)
    aae(calculated["fac3"], jnp.zeros((7, 0)))

    assert isinstance(calculated, dict)


def test_anchoring_scaling_factors(parsed_parameters):
    calculated = parsed_parameters["pardict"]["anchoring_scaling_factors"]
    expected = np.ones((8, 3))
    expected[:, 0] = jnp.array([127 + 7 * i for i in range(8)])
    aae(calculated, expected)


def test_anchoring_constants(parsed_parameters):
    calculated = parsed_parameters["pardict"]["anchoring_constants"]
    expected = np.zeros((8, 3))
    expected[:, 0] = jnp.array([18 + i * 14 for i in range(8)])
    aae(calculated, expected)
