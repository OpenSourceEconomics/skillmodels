"""The CHS filter stops predicting a person after their last observed measurement.

Fixture: a two-factor translog panel over eight periods. Person 0 is last observed
in period 1 (death or attrition); person 1 has a gap in period 3 and is observed
again afterwards. With large squared transition terms, the predicted covariance of
person 0 explodes in the periods after the last observation.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.model_spec import FactorSpec, ModelSpec

jax.config.update("jax_enable_x64", True)

LATENT = ("fac1", "fac2")
N_PERIODS = 8
N_PERSONS = 6
LAST_OBSERVED_PERIOD = {0: 1}
GAP = (1, 3)
REFERENCE_FILE = (
    Path(__file__).parent / "regression_vault" / "frozen_after_last_observation.json"
)


def _model() -> ModelSpec:
    factors = {
        f: FactorSpec(
            measurements=((f"{f}_m1", f"{f}_m2"),) * N_PERIODS,
            transition_function="translog",
        )
        for f in LATENT
    }
    return ModelSpec(factors=factors)


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [range(N_PERSONS), range(N_PERIODS)], names=["id", "period"]
    )
    cols = [f"{f}_m{j}" for f in LATENT for j in (1, 2)]
    df = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    for person, period in LAST_OBSERVED_PERIOD.items():
        df.loc[(person, slice(period + 1, None)), :] = np.nan
    df.loc[GAP, :] = np.nan
    return df


@pytest.fixture(scope="module")
def inputs() -> dict:
    return get_maximization_inputs(
        _model(),
        _data(),
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )


def _params(template: pd.DataFrame, squares: float, seed: int | None) -> pd.DataFrame:
    """Fill the template; squared transition terms get the value `squares`.

    With a `seed`, every free value is perturbed by a uniform factor in [0.8, 1.2].
    """
    values = []
    for category, _, name1, name2 in template.index:
        if category == "initial_cholcovs":
            row, col = name2.split("-")
            val = 1.0 if row == col else 0.2
        elif category == "transition":
            if "**" in name2:
                val = squares
            elif "*" in name2:
                val = 0.1
            else:
                val = 1.1 if name2 == name1 else 0.0
        else:
            val = {
                "controls": 0.1,
                "loadings": 1.0,
                "meas_sds": 0.5,
                "shock_sds": 0.3,
                "initial_states": 0.0,
                "mixture_weights": 1.0,
            }[category]
        values.append(val)
    params = template.copy()
    params["value"] = values
    if seed is not None:
        rng = np.random.default_rng(seed)
        free = template["value"].isna().to_numpy()
        params.loc[free, "value"] *= rng.uniform(0.8, 1.2, size=free.sum())
    return params


def test_gradient_matches_finite_differences_with_exploding_post_exit_prediction(
    inputs: dict,
) -> None:
    params = _params(inputs["params_template"], squares=1.0, seed=None)
    _, grad = inputs["loglike_and_gradient"](params)
    free = inputs["params_template"]["value"].isna().to_numpy()
    step = 1e-6
    numerical = []
    for i in np.flatnonzero(free):
        up, down = params.copy(), params.copy()
        up.iloc[i, up.columns.get_loc("value")] += step
        down.iloc[i, down.columns.get_loc("value")] -= step
        numerical.append(
            (float(inputs["loglike"](up)) - float(inputs["loglike"](down))) / (2 * step)
        )
    assert_allclose(np.asarray(grad)[free], numerical, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_loglikeobs_equal_reference_values(inputs: dict, seed: int) -> None:
    params = _params(inputs["params_template"], squares=0.6, seed=seed)
    expected = json.loads(REFERENCE_FILE.read_text())["loglikeobs"][str(seed)]
    assert_allclose(np.asarray(inputs["loglikeobs"](params)), expected, rtol=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gradient_equals_reference_values_where_finite(inputs: dict, seed: int) -> None:
    params = _params(inputs["params_template"], squares=0.6, seed=seed)
    expected = json.loads(REFERENCE_FILE.read_text())["gradient"][str(seed)]
    _, grad = inputs["loglike_and_gradient"](params)
    assert_allclose(np.asarray(grad), expected, rtol=1e-10, atol=1e-12)


def test_filtered_states_are_nan_exactly_after_last_observation(inputs: dict) -> None:
    params = _params(inputs["params_template"], squares=0.6, seed=None)
    states = inputs["debug_loglike"](params)["filtered_states"]
    is_nan = states.set_index(["id", "aug_period"])[list(LATENT)].isna().all(axis=1)
    expected = pd.Series(
        [
            person in LAST_OBSERVED_PERIOD and period > LAST_OBSERVED_PERIOD[person]
            for person, period in is_nan.index
        ],
        index=is_nan.index,
    )
    pd.testing.assert_series_equal(is_nan, expected, check_names=False)
