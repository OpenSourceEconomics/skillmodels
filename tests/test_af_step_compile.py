"""Tests for AF step param-index compilation and the cumulative param registry.

A mixed-calendar target block (destination skills at `d`, source investment at `s`)
must still be emitted in the flat parser's global category order -- all controls, then
all loadings, then all measurement SDs -- while each row keeps its true `param_period`.
Importance terms read fixed values for period-0 static factors (MC/MN) that the
immediately-previous step result does not contain, so a cumulative `HistoricalParams`
registry keyed by the full MultiIndex is required.
"""

import pandas as pd

from skillmodels.af.step_layout import (
    AFFactorInfo,
    AFFactorRole,
    HistoricalParams,
    compile_af_step_layouts,
    compile_target_measurement_index,
)


def _cnlsy_factor_infos() -> tuple[AFFactorInfo, ...]:
    return (
        AFFactorInfo("skills", AFFactorRole.DYNAMIC, (("sk_a", "sk_b"),) * 3),
        AFFactorInfo("MC", AFFactorRole.STATIC_PERSISTENT, (("mc_1", "mc_2"), (), ())),
        AFFactorInfo("MN", AFFactorRole.STATIC_PERSISTENT, (("mn_1",), (), ())),
        AFFactorInfo(
            "investment",
            AFFactorRole.ENDOGENOUS,
            (("inv_a", "inv_b"), ("inv_a", "inv_b"), ()),
        ),
    )


def _params_df(rows: list[tuple[str, int, str, str, float]]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3]) for r in rows],
        names=["category", "period", "name1", "name2"],
    )
    return pd.DataFrame({"value": [r[4] for r in rows]}, index=idx)


def test_target_measurement_index_uses_global_category_order() -> None:
    layout = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)[0]
    index = compile_target_measurement_index(layout, controls=("constant",))
    categories = [tup[0] for tup in index]
    # 4 target measures (2 skills + 2 inv): all controls, then loadings, then sds.
    assert categories == ["controls"] * 4 + ["loadings"] * 4 + ["meas_sds"] * 4


def test_target_measurement_index_keeps_true_param_periods() -> None:
    layout = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)[0]
    index = set(compile_target_measurement_index(layout, controls=("constant",)))
    # destination skills indexed at period 1; source investment stays at period 0.
    assert ("loadings", 1, "sk_a", "skills") in index
    assert ("loadings", 0, "inv_a", "investment") in index
    assert ("controls", 0, "inv_a", "constant") in index
    assert ("meas_sds", 1, "sk_b", "-") in index


def test_historical_params_reaches_period0_static_factor() -> None:
    initial = _params_df(
        [
            ("loadings", 0, "mc_1", "MC", 1.3),
            ("loadings", 0, "sk_a", "skills", 1.0),
        ]
    )
    step01 = _params_df(
        [
            ("loadings", 1, "sk_a", "skills", 0.9),
            ("loadings", 0, "inv_a", "investment", 0.7),
        ]
    )
    hist = HistoricalParams.from_param_frames([initial, step01])
    # MC_0 is reachable from the cumulative registry...
    assert hist.value("loadings", 0, "mc_1", "MC") == 1.3
    # ...but NOT from the immediately-previous step result alone (the AF1 defect).
    assert ("loadings", 0, "mc_1", "MC") not in step01.index


def test_historical_params_rejects_duplicate_index() -> None:
    initial = _params_df([("loadings", 0, "sk_a", "skills", 1.0)])
    dup = _params_df([("loadings", 0, "sk_a", "skills", 2.0)])
    try:
        HistoricalParams.from_param_frames([initial, dup])
    except ValueError:
        return
    msg = "Expected duplicate-index ValueError"
    raise AssertionError(msg)
