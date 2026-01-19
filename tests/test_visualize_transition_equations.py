from pathlib import Path

import pandas as pd
import yaml

from skillmodels.config import TEST_DATA_DIR
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.visualize_transition_equations import (
    combine_transition_plots,
    get_transition_plots,
)

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def test_visualize_transition_equations_runs() -> None:
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        model = yaml.load(y, Loader=yaml.SafeLoader)

    model["observed_factors"] = ["ob1"]

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])
    data["ob1"] = 0

    max_inputs = get_maximization_inputs(model, data)
    full_index = max_inputs["params_template"].index
    params = params.reindex(full_index)
    params["value"] = params["value"].fillna(0)
    subplots = get_transition_plots(
        model=model,
        params=params,
        period=0,
        quantiles_of_other_factors=[0.1, 0.25, 0.5, 0.75, 0.9],
        data=data,
    )
    combine_transition_plots(subplots)
    subplots = get_transition_plots(
        model=model,
        params=params,
        period=0,
        quantiles_of_other_factors=None,
        data=data,
    )
    combine_transition_plots(subplots)
