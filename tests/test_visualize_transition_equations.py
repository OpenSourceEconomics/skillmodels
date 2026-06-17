"""Tests for visualize transition equations."""

from pathlib import Path

import numpy as np
import pandas as pd

from skillmodels import CorrectionSpec, FactorSpec, ModelSpec, Normalizations
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.individual_states import get_individual_states_from_params
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

    states = get_individual_states_from_params(
        model_spec=model, data=data, params=params
    )["anchored_states"]["states"]

    subplots = get_transition_plots(
        model_spec=model,
        params=params,
        period=0,
        quantiles_of_other_factors=[0.1, 0.25, 0.5, 0.75, 0.9],
        filtered_states=states,
        data=data,
    )
    combine_transition_plots(subplots)

    subplots = get_transition_plots(
        model_spec=model,
        params=params,
        period=0,
        quantiles_of_other_factors=None,
        filtered_states=states,
        data=data,
    )
    combine_transition_plots(subplots)


def _correction_model() -> ModelSpec:
    """Small correction model: fac1/fac2 states, `inv` endogenous, instrument z1."""
    state_norm = Normalizations(
        loadings=({"y1": 1}, {"y1": 1}, {"y1": 1}), intercepts=({}, {}, {})
    )
    fac2_norm = Normalizations(
        loadings=({"w1": 1}, {"w1": 1}, {"w1": 1}), intercepts=({}, {}, {})
    )
    inv_norm = Normalizations(
        loadings=({"yi1": 1}, {"yi1": 1}, {}), intercepts=({}, {}, {})
    )
    factors = {
        "fac1": FactorSpec(
            measurements=(("y1", "y2", "y3"),) * 3,
            normalizations=state_norm,
            transition_function="linear",
        ),
        "fac2": FactorSpec(
            measurements=(("w1", "w2", "w3"),) * 3,
            normalizations=fac2_norm,
            transition_function="linear",
        ),
        "inv": FactorSpec(
            # Endogenous factor must NOT be measured in the last period.
            measurements=(("yi1", "yi2", "yi3"), ("yi1", "yi2", "yi3"), ()),
            normalizations=inv_norm,
            is_endogenous=True,
            transition_function="linear",
            correction=CorrectionSpec(
                state_predictors=("fac1", "fac2"),
                instruments=("z1",),
                targets=("fac1", "fac2"),
            ),
        ),
    }
    return ModelSpec(factors=factors, observed_factors=("z1",))


_CORRECTION_MEAS_COLS = ("y1", "y2", "y3", "w1", "w2", "w3", "yi1", "yi2", "yi3")


def _correction_panel(n_obs: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.MultiIndex.from_product(
        [np.arange(n_obs), [0, 1, 2]], names=["caseid", "period"]
    )
    panel = pd.DataFrame(index=index)
    panel["z1"] = rng.normal(size=len(index))
    for col in _CORRECTION_MEAS_COLS:
        panel[col] = rng.normal(size=len(index))
    return panel


def test_get_transition_plots_runs_for_correction_model() -> None:
    """Transition plots evaluate a correction target's full cf DAG.

    For a target factor the individual transition function is the grafted DAG
    that reads the reserved first-stage (`__first_stage_<inv>__`) and kappa
    (`__kappa_<target>__`) coefficient keys on top of its own production
    coefficients. The plotting helper must forward all transition keys, or
    evaluating the transition raises `KeyError` on the first-stage betas.
    """
    model = _correction_model()
    panel = _correction_panel()
    template = get_maximization_inputs(
        model,
        panel,
        chs_options=CHSEstimationOptions(start_params_strategy="spearman"),
    )["params_template"]
    assert not template["value"].isna().any()

    states = get_individual_states_from_params(
        model_spec=model, data=panel, params=template
    )["anchored_states"]["states"]

    plots = get_transition_plots(
        model_spec=model,
        params=template,
        period=0,
        quantiles_of_other_factors=None,
        filtered_states=states,
        data=panel,
    )
    # A target factor's plot exists -> its cf-corrected transition evaluated.
    assert ("inv", "fac1") in plots
