from pathlib import Path

import pandas as pd
import yaml
from conftest import model_spec_from_yaml_dict

from skillmodels.config import TEST_DATA_DIR
from skillmodels.filtered_states import get_filtered_states
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.process_model import process_model
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
        model = model_spec_from_yaml_dict(yaml.load(y, Loader=yaml.SafeLoader))

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])

    max_inputs = get_maximization_inputs(model, data)
    params = params.loc[max_inputs["params_template"].index]
    kde = univariate_densities(
        data=data,
        model_spec=model,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        model_spec=model,
        params=params,
        period=1,
    )
    surfaces = bivariate_density_surfaces(
        data=data,
        model_spec=model,
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
        model = model_spec_from_yaml_dict(yaml.load(y, Loader=yaml.SafeLoader))

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
        model_spec=model,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        states=latent_data,
        model_spec=model,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )


def test_visualize_factor_distributions_with_period_indexed_states() -> None:
    """Test visualization with states indexed by (id, period) without aug_period.

    This mimics the scenario where states come from a downstream task that has
    already mapped aug_period to period and dropped the aug_period column.
    """
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        model = model_spec_from_yaml_dict(yaml.load(y, Loader=yaml.SafeLoader))

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    max_inputs = get_maximization_inputs(model, data)
    params = params.loc[max_inputs["params_template"].index]

    # Get filtered states and convert to (id, period) index without aug_period
    filtered_states = get_filtered_states(model_spec=model, data=data, params=params)[
        "anchored_states"
    ]["states"]
    processed = process_model(model)

    # Add period column and drop aug_period
    # (mimics task_filtered_states_and_measurements)
    filtered_states["period"] = filtered_states["aug_period"].map(
        processed.labels.aug_periods_to_periods
    )
    filtered_states = filtered_states.drop(columns=["aug_period"]).set_index(
        ["id", "period"]
    )

    kde = univariate_densities(
        data=data,
        states=filtered_states,
        model_spec=model,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        states=filtered_states,
        model_spec=model,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )


def test_visualize_factor_distributions_with_both_aug_period_and_period() -> None:
    """Test visualization with states having both aug_period and period.

    This mimics the scenario where states have aug_period as a column and period
    in the index (or both as columns).
    """
    with (TEST_DATA_DIR / "model2.yaml").open() as y:
        model = model_spec_from_yaml_dict(yaml.load(y, Loader=yaml.SafeLoader))

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])

    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    max_inputs = get_maximization_inputs(model, data)
    params = params.loc[max_inputs["params_template"].index]

    # Get filtered states and add period while keeping aug_period
    filtered_states = get_filtered_states(model_spec=model, data=data, params=params)[
        "anchored_states"
    ]["states"]
    processed = process_model(model)

    # Add period column but keep aug_period (both are present)
    filtered_states["period"] = filtered_states["aug_period"].map(
        processed.labels.aug_periods_to_periods
    )
    filtered_states = filtered_states.set_index(["id", "period"])

    kde = univariate_densities(
        data=data,
        states=filtered_states,
        model_spec=model,
        params=params,
        period=1,
    )
    contours = bivariate_density_contours(
        data=data,
        states=filtered_states,
        model_spec=model,
        params=params,
        period=1,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )
