"""Cross-period equality propagation in `estimate_af`.

skane-struct-bw and similar applications impose equality constraints
across aug-periods (e.g., shock_sds, transition coefficients constant
within a stage, loadings/meas_sds constant across periods). AF's
sequential MLE estimates each period independently and would silently
violate those constraints; the new `constraints=` kwarg on
`estimate_af` propagates equality groups by pinning every member of a
group to whichever member is estimated first.

These tests exercise the propagation directly via the helpers and
end-to-end via a small synthetic T=3 fit.
"""

import functools

import jax
import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.af.estimate import (
    _extract_equality_groups,
    _propagate_equality_groups,
)
from skillmodels.af.types import AFPeriodResult
from skillmodels.common.constraints import select_by_loc
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.params_index import get_params_index
from skillmodels.common.process_model import process_model

jax.config.update("jax_enable_x64", True)


def _equality_constraint(loc: pd.MultiIndex) -> om.EqualityConstraint:
    return om.EqualityConstraint(
        selector=functools.partial(select_by_loc, loc=loc),
    )


def test_extract_equality_groups_returns_only_equality_constraints() -> None:
    loc = pd.MultiIndex.from_tuples(
        [("transition", 0, "fac1", "fac1"), ("transition", 1, "fac1", "fac1")],
        names=["category", "period", "name1", "name2"],
    )
    constraints: list[om.constraints.Constraint] = [
        _equality_constraint(loc),
        om.FixedConstraint(selector=functools.partial(select_by_loc, loc=loc)),
    ]
    groups = _extract_equality_groups(constraints)
    assert len(groups) == 1
    assert groups[0].equals(loc)


def test_extract_equality_groups_handles_empty_input() -> None:
    assert _extract_equality_groups(None) == []
    assert _extract_equality_groups([]) == []


def test_propagate_equality_groups_pins_other_periods() -> None:
    period_0 = AFPeriodResult(
        period=0,
        params=pd.DataFrame(
            {"value": [0.42]},
            index=pd.MultiIndex.from_tuples(
                [("shock_sds", 0, "skills", "-")],
                names=["category", "period", "name1", "name2"],
            ),
        ),
        loglikelihood=-1.0,
        success=True,
        optimize_result=None,
    )
    group = pd.MultiIndex.from_tuples(
        [
            ("shock_sds", 0, "skills", "-"),
            ("shock_sds", 1, "skills", "-"),
            ("shock_sds", 2, "skills", "-"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    fixed_params = _propagate_equality_groups(
        period_results=[period_0],
        fixed_params=None,
        equality_groups=[group],
    )
    assert fixed_params is not None
    assert ("shock_sds", 1, "skills", "-") in fixed_params.index
    assert ("shock_sds", 2, "skills", "-") in fixed_params.index
    assert fixed_params.loc[("shock_sds", 1, "skills", "-"), "value"] == 0.42
    assert fixed_params.loc[("shock_sds", 2, "skills", "-"), "value"] == 0.42


def test_propagate_equality_groups_respects_existing_pins() -> None:
    period_0 = AFPeriodResult(
        period=0,
        params=pd.DataFrame(
            {"value": [0.42]},
            index=pd.MultiIndex.from_tuples(
                [("shock_sds", 0, "skills", "-")],
                names=["category", "period", "name1", "name2"],
            ),
        ),
        loglikelihood=-1.0,
        success=True,
        optimize_result=None,
    )
    fixed_params_initial = pd.DataFrame(
        {"value": [0.99]},
        index=pd.MultiIndex.from_tuples(
            [("shock_sds", 1, "skills", "-")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    group = pd.MultiIndex.from_tuples(
        [("shock_sds", 0, "skills", "-"), ("shock_sds", 1, "skills", "-")],
        names=["category", "period", "name1", "name2"],
    )
    out = _propagate_equality_groups(
        period_results=[period_0],
        fixed_params=fixed_params_initial,
        equality_groups=[group],
    )
    assert out is not None
    assert out.loc[("shock_sds", 1, "skills", "-"), "value"] == 0.99


def _build_t3_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "state": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 3,
                    intercepts=({"y1": 0},) * 3,
                ),
                transition_function="linear",
            ),
        },
    )


def _truth_params(model: ModelSpec) -> pd.DataFrame:
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
    for aug in range(2):
        df.loc[("transition", aug, "state", "state"), "value"] = 0.8
        df.loc[("transition", aug, "state", "constant"), "value"] = 0.0
    diag_mask = pd.Series(
        [
            idx[0] == "initial_cholcovs"
            and "-" in idx[3]
            and idx[3].split("-")[0] == idx[3].split("-")[1]
            for idx in df.index
        ],
        index=df.index,
    )
    df.loc[diag_mask, "value"] = 1.0
    return df


def _simulate_t3(
    model: ModelSpec, params: pd.DataFrame, n_obs: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = [rng.normal(0.0, 1.0, size=n_obs)]

    def _val(loc: tuple) -> float:
        return float(params.loc[loc, "value"])

    for t in range(1, 3):
        a = _val(("transition", t - 1, "state", "state"))
        c = _val(("transition", t - 1, "state", "constant"))
        sigma = _val(("shock_sds", t - 1, "state", "-"))
        states.append(a * states[-1] + c + sigma * rng.normal(size=n_obs))
    rows: list[dict] = []
    for obs_id in range(n_obs):
        for t in range(3):
            row: dict[str, float | int] = {"caseid": obs_id, "period": t}
            for k in (1, 2, 3):
                meas = f"y{k}"
                lam = _val(("loadings", t, meas, "state"))
                eps = _val(("meas_sds", t, meas, "-"))
                row[meas] = lam * states[t][obs_id] + eps * rng.normal()
            rows.append(row)
    return pd.DataFrame.from_records(rows).set_index(["caseid", "period"])


@pytest.mark.end_to_end
def test_estimate_af_enforces_equality_across_periods() -> None:
    """Pinning shock_sds equal across periods makes the chain return one value."""
    model = _build_t3_model()
    params = _truth_params(model)
    data = _simulate_t3(model, params, n_obs=300, seed=20260510)

    af_options = AFEstimationOptions(
        n_halton_points=20,
        n_halton_points_shock=10,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    eq_loc = pd.MultiIndex.from_tuples(
        [
            ("shock_sds", 0, "state", "-"),
            ("shock_sds", 1, "state", "-"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    constraints: list[om.constraints.Constraint] = [_equality_constraint(eq_loc)]

    result = estimate_af(
        model_spec=model,
        data=data,
        af_options=af_options,
        constraints=constraints,
    )

    def _val(period_idx: int, loc: tuple) -> float:
        return float(result.period_results[period_idx].params.loc[loc, "value"])

    period1_sd = _val(1, ("shock_sds", 0, "state", "-"))
    period2_sd = _val(2, ("shock_sds", 1, "state", "-"))
    assert period1_sd == pytest.approx(period2_sd, rel=1e-9)
