"""Tests for filtered states."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import skillmodels.chs.maximization_inputs as mi
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.individual_states import (
    get_individual_states,
    get_individual_states_from_params,
)
from skillmodels.test_data.model2 import MODEL2, MODEL2_CHS_OPTIONS

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2():
    return MODEL2


@pytest.fixture
def model2_data():
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


def test_get_individual_states_from_params(model2, model2_data) -> None:
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    max_inputs = get_maximization_inputs(
        model2, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    params = params.loc[max_inputs["params_template"].index]

    calculated = get_individual_states_from_params(
        model_spec=model2,
        data=model2_data,
        params=params,
        chs_options=MODEL2_CHS_OPTIONS,
    )

    factors = ["fac1", "fac2", "fac3"]
    expected_ratios = [1.187757, 1, 1]
    for factor, expected_ratio in zip(factors, expected_ratios, strict=False):
        anch_ranges = calculated["anchored_states"]["state_ranges"][factor]
        unanch_ranges = calculated["unanchored_states"]["state_ranges"][factor]
        ratio = (anch_ranges / unanch_ranges).to_numpy()
        assert np.allclose(ratio, expected_ratio)


@pytest.fixture
def model2_params(model2, model2_data):
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])
    template = get_maximization_inputs(
        model2,
        model2_data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["params_template"]
    return params.loc[template.index]


def _fail(*args, **kwargs):
    raise AssertionError("start values must not be seeded")


def test_individual_states_from_params_never_seed_start_values(
    model2, model2_data, model2_params, monkeypatch
) -> None:
    """Filtering at given params skips AMN / Spearman start-value seeding."""
    monkeypatch.setattr(mi, "_estimate_amn_for_chs_seeding", _fail)
    monkeypatch.setattr(mi, "get_spearman_start_params", _fail)
    states = get_individual_states_from_params(
        model_spec=model2,
        data=model2_data,
        params=model2_params,
        chs_options=CHSEstimationOptions(start_params_strategy="amn"),
    )["unanchored_states"]["states"]
    assert len(states) == len(model2_data)


def test_individual_states_from_params_use_the_sigma_point_scale(
    model2, model2_data, model2_params
) -> None:
    """The filter runs with the likelihood settings of the given options."""

    def _states(scale):
        return get_individual_states_from_params(
            model_spec=model2,
            data=model2_data,
            params=model2_params,
            chs_options=CHSEstimationOptions(
                start_params_strategy="none", sigma_points_scale=scale
            ),
        )["unanchored_states"]["states"]["fac1"].to_numpy()

    assert not np.allclose(_states(2.0), _states(0.5))


def test_get_individual_states_requires_chs_options_for_chs_result(
    model2, model2_data, model2_params
) -> None:
    result = CHSEstimationResult(
        model_spec=model2,
        params=model2_params,
        success=True,
        loglikelihood=0.0,
        optimize_result=None,
    )
    with pytest.raises(TypeError, match="chs_options"):
        get_individual_states(data=model2_data, result=result)
