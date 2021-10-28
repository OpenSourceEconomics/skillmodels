from pathlib import Path

import pandas as pd
import yaml

from skillmodels.likelihood_function import get_maximization_inputs
from skillmodels.visualize_factor_distributions import (
    plot_pairwise_factor_distributions,
)


# importing the TEST_DIR from config does not work for test run in conda build
TEST_DIR = Path(__file__).parent.resolve()


def test_visualize_factor_distributions_runs():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)

    params = pd.read_csv(TEST_DIR / "regression_vault" / f"one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DIR / "model2_simulated_data.dta")
    data.set_index(["caseid", "period"], inplace=True)

    max_inputs = get_maximization_inputs(model_dict, data)
    debug_loglike = max_inputs["debug_loglike"]
    debug_data = debug_loglike(params)
    filtered_states = debug_data["filtered_states"]

    plot_pairwise_factor_distributions(
        states=filtered_states, model_dict=model_dict, add_3d_plots=True, period=1
    )
