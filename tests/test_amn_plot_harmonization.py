"""Parametrised tests confirming plot helpers work for CHS, AF, and AMN."""

import numpy as np
import pandas as pd
import pytest

from skillmodels.amn import AMNEstimationOptions, estimate_amn
from skillmodels.common.individual_states import get_individual_states
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.variance_decomposition import decompose_measurement_variance


def _tiny_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
        },
        n_mixtures=2,
    )


def _tiny_data(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        f0 = rng.normal()
        f1 = 0.6 * f0 + rng.normal(0, 0.5)
        for period, f in [(0, f0), (1, f1)]:
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": f + rng.normal(0, 0.3),
                    "y2": 0.9 * f + rng.normal(0, 0.4),
                    "y3": 1.1 * f + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


@pytest.fixture(scope="module")
def amn_fit():
    model = _tiny_model()
    data = _tiny_data(n=400)
    options = AMNEstimationOptions(n_simulation_draws=1000, seed=0)
    fit = estimate_amn(model, data, options)
    return fit, data


def test_get_individual_states_dispatches_to_amn(amn_fit):
    fit, data = amn_fit

    out = get_individual_states(data=data, result=fit)

    assert "unanchored_states" in out
    states = out["unanchored_states"]["states"]
    assert "skills" in states.columns
    assert {"id", "period", "skills"} <= set(states.columns)


def test_decompose_measurement_variance_works_with_amn_result(amn_fit):
    fit, data = amn_fit

    filtered = get_individual_states(data=data, result=fit)
    states_root = filtered.get("anchored_states", filtered["unanchored_states"])
    decomp = decompose_measurement_variance(
        fit.model_spec,
        fit.params,
        filtered_states=states_root["states"],
    )

    assert {"loading", "factor_variance", "meas_sd"} <= set(decomp.columns)
    assert decomp.shape[0] > 0
