"""End-to-end test that AF works for T = 5 periods.

The AF paper's iterative chain (Section 3) is described for general T,
but skillmodels' AF tests so far cover T = 3. This test runs the full
chain on a synthetic T=5 panel and confirms `estimate_af` produces
five per-period results with finite likelihoods and the expected
chain-link structure (k links after estimating period k).

Marked `end_to_end` so it does not run in the default test suite.
"""

import jax
import numpy as np
import pandas as pd
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.common.model_spec import (
    CHSEstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.params_index import get_params_index
from skillmodels.common.process_model import process_model

jax.config.update("jax_enable_x64", True)


def _build_t5_model() -> ModelSpec:
    """Two-factor T=5 model: linear `state`, linear `inv`, three measures each."""
    return ModelSpec(
        factors={
            "state": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 5,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 5,
                    intercepts=({"y1": 0},) * 5,
                ),
                transition_function="linear",
            ),
            "inv": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 5,
                normalizations=Normalizations(
                    loadings=({"z1": 1},) * 5,
                    intercepts=({"z1": 0},) * 5,
                ),
                transition_function="linear",
            ),
        },
        chs_estimation_options=CHSEstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )


def _truth_params_t5(model: ModelSpec) -> pd.DataFrame:
    """Build a truth params DataFrame for the T=5 model from the params index."""
    processed = process_model(model)
    p_index = get_params_index(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        transition_info=processed.transition_info,
        endogenous_factors_info=processed.endogenous_factors_info,
    )
    df = pd.DataFrame({"value": np.zeros(len(p_index))}, index=p_index)
    cat = df.index.get_level_values("category")
    df.loc[cat == "loadings", "value"] = 1.0
    df.loc[cat == "meas_sds", "value"] = 0.3
    df.loc[cat == "shock_sds", "value"] = 0.4
    df.loc[cat == "mixture_weights", "value"] = 1.0
    for aug_period in range(4):
        for factor, other in (("state", "inv"), ("inv", "state")):
            for regressor, val in (
                (factor, 0.7),
                (other, 0.2),
                ("constant", 0.1),
            ):
                loc = ("transition", aug_period, factor, regressor)
                if loc in df.index:
                    df.loc[loc, "value"] = val
    cholcov_diag_mask = pd.Series(
        [
            idx[0] == "initial_cholcovs"
            and "-" in idx[3]
            and idx[3].split("-")[0] == idx[3].split("-")[1]
            for idx in df.index
        ],
        index=df.index,
    )
    df.loc[cholcov_diag_mask, "value"] = 1.0
    return df


def _simulate_synthetic_t5(
    model: ModelSpec,
    params: pd.DataFrame,
    n_obs: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate (states + measurements) directly for the T=5 model."""
    n_periods = 5
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 1.0, size=(n_obs, 2))  # (state_t, inv_t)
    state_history = [state.copy()]

    def _val(loc: tuple) -> float:
        return float(params.loc[loc, "value"])

    for t in range(1, n_periods):
        prev = state_history[-1]
        new_state = np.zeros_like(prev)
        for f, idx in (("state", 0), ("inv", 1)):
            other_idx = 1 - idx
            other = "inv" if f == "state" else "state"
            a = _val(("transition", t - 1, f, f))
            b = _val(("transition", t - 1, f, other))
            c = _val(("transition", t - 1, f, "constant"))
            sigma = _val(("shock_sds", t - 1, f, "-"))
            new_state[:, idx] = (
                a * prev[:, idx]
                + b * prev[:, other_idx]
                + c
                + sigma * rng.normal(size=n_obs)
            )
        state_history.append(new_state)

    records: list[dict] = []
    for obs_id in range(n_obs):
        for t in range(n_periods):
            row: dict[str, float | int] = {"caseid": obs_id, "period": t}
            st = state_history[t][obs_id]
            for f, idx in (("state", 0), ("inv", 1)):
                meas_prefix = "y" if f == "state" else "z"
                for k in (1, 2, 3):
                    meas_name = f"{meas_prefix}{k}"
                    lam = _val(("loadings", t, meas_name, f))
                    sigma_eps = _val(("meas_sds", t, meas_name, "-"))
                    row[meas_name] = float(lam * st[idx] + sigma_eps * rng.normal())
            records.append(row)
    return pd.DataFrame.from_records(records).set_index(["caseid", "period"])


@pytest.mark.end_to_end
def test_af_chain_runs_for_t5() -> None:
    """`estimate_af` runs the full T=5 chain and produces finite per-period llik."""
    model = _build_t5_model()
    params = _truth_params_t5(model)
    data = _simulate_synthetic_t5(model, params, n_obs=200, seed=20260510)

    af_options = AFEstimationOptions(
        n_halton_points=20,
        n_halton_points_shock=10,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    result = estimate_af(model_spec=model, data=data, af_options=af_options)

    assert len(result.period_results) == 5, (
        f"Expected 5 per-period results for T=5; got {len(result.period_results)}"
    )
    for pr in result.period_results:
        assert np.isfinite(pr.loglikelihood), (
            f"period {pr.period}: non-finite loglikelihood {pr.loglikelihood}"
        )
    assert len(result.conditional_distributions) == 5
    # Each period after 0 carries one chain link per prior transition.
    for t, cd in enumerate(result.conditional_distributions):
        assert len(cd.chain_links) == max(t, 0)
