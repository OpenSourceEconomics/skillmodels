"""Parametrised tests confirming plot helpers work for CHS, AF, and AMN."""

import numpy as np
import pandas as pd
import pytest

from skillmodels import (
    AMNEstimationOptions,
    decompose_measurement_variance,
    estimate_amn,
)
from skillmodels.chs.filtered_states import get_filtered_states
from skillmodels.common.model_spec import (
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)


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
        estimation_options=EstimationOptions(
            robust_bounds=True, bounds_distance=0.001, n_mixtures=1
        ),
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
    options = AMNEstimationOptions(
        n_mixture_components=2, n_simulation_draws=1000, seed=0
    )
    fit = estimate_amn(model, data, options)
    return fit, data


def test_get_filtered_states_dispatches_to_amn(amn_fit):
    fit, data = amn_fit

    out = get_filtered_states(
        model_spec=fit.model_spec,
        data=data,
        params=fit.all_params,
        amn_result=fit,
    )

    assert "unanchored_states" in out
    states = out["unanchored_states"]["states"]
    assert "skills" in states.columns
    assert {"id", "period", "skills"} <= set(states.columns)


def test_get_filtered_states_rejects_both_af_and_amn_results(amn_fit):
    fit, data = amn_fit
    with pytest.raises(ValueError, match="only one of"):
        get_filtered_states(
            model_spec=fit.model_spec,
            data=data,
            params=fit.all_params,
            af_result=fit,
            amn_result=fit,
        )


def test_decompose_measurement_variance_works_with_amn_result(amn_fit):
    fit, data = amn_fit

    decomp = decompose_measurement_variance(
        fit.model_spec,
        fit.all_params,
        data,
        amn_result=fit,
    )

    assert {"loading", "factor_variance", "meas_sd"} <= set(decomp.columns)
    assert decomp.shape[0] > 0
