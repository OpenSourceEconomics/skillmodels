"""Tests for visualize transition equations."""

from pathlib import Path

import pandas as pd

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.visualize_transition_equations import (
    combine_transition_plots,
    get_transition_plots,
)
from skillmodels.test_data.model2 import MODEL2, MODEL2_CHS_OPTIONS

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def test_visualize_transition_equations_runs() -> None:
    model = MODEL2.with_added_observed_factors("ob1")

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])
    data["ob1"] = 0

    max_inputs = get_maximization_inputs(model, data, chs_options=MODEL2_CHS_OPTIONS)
    full_index = max_inputs["params_template"].index
    params = params.reindex(full_index)
    params["value"] = params["value"].fillna(0)
    subplots = get_transition_plots(
        model_spec=model,
        params=params,
        period=0,
        quantiles_of_other_factors=[0.1, 0.25, 0.5, 0.75, 0.9],
        data=data,
    )
    combine_transition_plots(subplots)
    subplots = get_transition_plots(
        model_spec=model,
        params=params,
        period=0,
        quantiles_of_other_factors=None,
        data=data,
    )
    combine_transition_plots(subplots)
