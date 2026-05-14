"""Tests for visualize factor distributions."""

from pathlib import Path

import pandas as pd

from skillmodels.chs.filtered_states import get_filtered_states
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.simulate_data import simulate_dataset
from skillmodels.common.visualize_factor_distributions import (
    bivariate_density_contours,
    bivariate_density_surfaces,
    combine_distribution_plots,
    univariate_densities,
)
from skillmodels.test_data.model2 import MODEL2, MODEL2_CHS_OPTIONS

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


def _load_model2_filtered() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (data, params, anchored_states) for MODEL2."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    params = params.set_index(["category", "period", "name1", "name2"])

    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    data = data.set_index(["caseid", "period"])

    max_inputs = get_maximization_inputs(MODEL2, data, chs_options=MODEL2_CHS_OPTIONS)
    params = params.loc[max_inputs["params_template"].index]

    states = get_filtered_states(model_spec=MODEL2, data=data, params=params)[
        "anchored_states"
    ]["states"]
    return data, params, states


def test_visualize_factor_distributions_runs_with_filtered_states() -> None:
    data, _params, states = _load_model2_filtered()
    kde = univariate_densities(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=states,
    )
    contours = bivariate_density_contours(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=states,
    )
    surfaces = bivariate_density_surfaces(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=states,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=surfaces,
    )


def test_visualize_factor_distributions_runs_with_simulated_states() -> None:
    data, params, _ = _load_model2_filtered()

    latent_data = simulate_dataset(MODEL2, params, data=data, policies=None)[
        "aug_unanchored_states"
    ]["states"]

    kde = univariate_densities(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=latent_data,
    )
    contours = bivariate_density_contours(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=latent_data,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )


def test_visualize_factor_distributions_with_period_indexed_states() -> None:
    """Visualisation with states indexed by (id, period) without aug_period.

    Mimics a downstream pipeline that has already mapped aug_period to
    period and dropped the aug_period column.
    """
    data, _params, states = _load_model2_filtered()
    states = states.set_index(["id", "period"])

    kde = univariate_densities(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=states,
    )
    contours = bivariate_density_contours(
        data=data,
        model_spec=MODEL2,
        period=1,
        filtered_states=states,
    )
    combine_distribution_plots(
        kde_plots=kde,
        contour_plots=contours,
        surface_plots=None,
    )
