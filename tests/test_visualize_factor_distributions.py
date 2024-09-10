from pathlib import Path

import pandas as pd
import yaml

from skillmodels.likelihood_function import get_maximization_inputs
from skillmodels.simulate_data import simulate_dataset
from skillmodels.visualize_factor_distributions import (
    bivariate_density_contours,
    bivariate_density_surfaces,
    combine_distribution_plots,
    univariate_densities,
)

# importing the TEST_DIR from config does not work for test run in conda build
TEST_DIR = Path(__file__).parent.resolve()


def test_visualize_factor_distributions_runs_with_filtered_states():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)

    params = pd.read_csv(TEST_DIR / "regression_vault" / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DIR / "model2_simulated_data.dta")
    data.set_index(["caseid", "period"], inplace=True)

    max_inputs = get_maximization_inputs(model_dict, data)
    params = params.loc[max_inputs["params_template"].index]
    kde = univariate_densities(
        data=data,
        model_dict=model_dict,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        model_dict=model_dict,
        params=params,
        period=1,
    )
    surfaces = bivariate_density_surfaces(
        data=data,
        model_dict=model_dict,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=surfaces,
    )


def test_visualize_factor_distributions_runs_with_simulated_states():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)

    data = pd.read_stata(TEST_DIR / "model2_simulated_data.dta")
    data.set_index(["caseid", "period"], inplace=True)

    params = pd.read_csv(TEST_DIR / "regression_vault" / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    max_inputs = get_maximization_inputs(model_dict, data)
    params = params.loc[max_inputs["params_template"].index]

    latent_data = simulate_dataset(model_dict, params, data=data, policies=None)[
        "unanchored_states"
    ]["states"]

    kde = univariate_densities(
        data=data,
        states=latent_data,
        model_dict=model_dict,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        states=latent_data,
        model_dict=model_dict,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )
