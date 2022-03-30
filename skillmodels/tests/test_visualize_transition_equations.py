from pathlib import Path

import pandas as pd
import yaml

from skillmodels.likelihood_function import get_maximization_inputs
from skillmodels.visualize_transition_equations import visualize_transition_equations


TEST_DIR = Path(__file__).parent.resolve()


def test_visualize_transition_equations_runs():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)

    model_dict["observed_factors"] = ["ob1"]

    params = pd.read_csv(TEST_DIR / "regression_vault" / f"one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DIR / "model2_simulated_data.dta")
    data.set_index(["caseid", "period"], inplace=True)
    print(data.columns)
    data["ob1"] = 0

    max_inputs = get_maximization_inputs(model_dict, data)
    full_index = max_inputs["params_template"].index
    params = params.reindex(full_index)
    params["value"] = params["value"].fillna(0)
    debug_loglike = max_inputs["debug_loglike"]
    debug_data = debug_loglike(params)
    filtered_states = debug_data["filtered_states"]

    visualize_transition_equations(
        model_dict=model_dict,
        params=params,
        states=filtered_states,
        period=0,
        quantiles_of_other_factors=[0.1, 0.25, 0.5, 0.75, 0.9],
        data=data,
    )
    visualize_transition_equations(
        model_dict=model_dict,
        params=params,
        states=filtered_states,
        period=0,
        quantiles_of_other_factors=None,
        data=data,
    )
