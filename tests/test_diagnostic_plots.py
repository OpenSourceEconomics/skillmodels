"""Tests for diagnostic_plots module."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from skillmodels.diagnostic_plots import (
    plot_likelihood_contributions,
    plot_residual_boxplots,
)
from skillmodels.maximization_inputs import get_maximization_inputs

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2_diag_params(model2, model2_data):
    """Prepare params that match the expected index for diagnostic plots."""
    vault_params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    vault_params = vault_params.set_index(["category", "period", "name1", "name2"])
    max_inputs = get_maximization_inputs(model_spec=model2, data=model2_data)
    params = max_inputs["params_template"].copy()
    # Fill in values from vault params where indices match
    common_idx = params.index.intersection(vault_params.index)
    params.loc[common_idx, "value"] = vault_params.loc[common_idx, "value"]
    params["value"] = params["value"].fillna(0.0)
    return params


@pytest.mark.integration
def test_plot_residual_boxplots_single_period(
    model2, model2_data, model2_diag_params
) -> None:
    fig = plot_residual_boxplots(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=0,
    )
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Box) for t in fig.data)


@pytest.mark.integration
def test_plot_residual_boxplots_all_periods(
    model2, model2_data, model2_diag_params
) -> None:
    result = plot_residual_boxplots(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=None,
    )
    assert isinstance(result, dict)
    for fig in result.values():
        assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_plot_residual_boxplots_no_reference_line(
    model2, model2_data, model2_diag_params
) -> None:
    fig = plot_residual_boxplots(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=0,
        show_reference_line=False,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_plot_residual_boxplots_layout_kwargs(
    model2, model2_data, model2_diag_params
) -> None:
    fig = plot_residual_boxplots(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=0,
        layout_kwargs={"title": "Custom Title"},
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Custom Title"


@pytest.mark.integration
def test_plot_likelihood_contributions_single_period(
    model2, model2_data, model2_diag_params
) -> None:
    fig = plot_likelihood_contributions(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=0,
    )
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Box) for t in fig.data)


@pytest.mark.integration
def test_plot_likelihood_contributions_all_periods(
    model2, model2_data, model2_diag_params
) -> None:
    result = plot_likelihood_contributions(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=None,
    )
    assert isinstance(result, dict)
    for fig in result.values():
        assert isinstance(fig, go.Figure)


@pytest.mark.integration
def test_plot_likelihood_contributions_layout_kwargs(
    model2, model2_data, model2_diag_params
) -> None:
    fig = plot_likelihood_contributions(
        model_spec=model2,
        data=model2_data,
        params=model2_diag_params,
        period=0,
        layout_kwargs={"title": "Custom LL Title"},
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Custom LL Title"
