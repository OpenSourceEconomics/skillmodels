from pathlib import Path

import pandas as pd
import yaml

from skillmodels.config import TEST_DATA_DIR
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.simulate_data import simulate_dataset
from skillmodels.visualize_factor_distributions import (
    bivariate_density_contours,
    bivariate_density_surfaces,
    combine_distribution_plots,
    univariate_densities,
)

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def test_visualize_factor_distributions_runs_with_filtered_states() -> None:
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        model = yaml.load(y, Loader=yaml.SafeLoader)

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])

    max_inputs = get_maximization_inputs(model, data)
    params = params.loc[max_inputs["params_template"].index]
    kde = univariate_densities(
        data=data,
        model=model,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        model=model,
        params=params,
        period=1,
    )
    surfaces = bivariate_density_surfaces(
        data=data,
        model=model,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=surfaces,
    )


def test_visualize_factor_distributions_runs_with_simulated_states() -> None:
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        model = yaml.load(y, Loader=yaml.SafeLoader)

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    max_inputs = get_maximization_inputs(model, data)
    params = params.loc[max_inputs["params_template"].index]

    latent_data = simulate_dataset(model, params, data=data, policies=None)[
        "aug_unanchored_states"
    ]["states"]

    kde = univariate_densities(
        data=data,
        states=latent_data,
        model=model,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        states=latent_data,
        model=model,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )
