"""Tests for assembling an AF step's target + importance arrays from a layout.

This is the heart of the calendar fix: for step `s -> d`, the target block must source
destination skills from period `d` and source investment from period `s` (so age-7
indicators measure I_0), and the importance block must include every static-persistent
factor's period-0 measurement rows at *every* step (the dropped-MC/MN fix), with their
fixed params pulled from the cumulative `HistoricalParams` (period-0 result), not the
immediately-previous step.
"""

import numpy as np
import pandas as pd

from skillmodels.af.step_assembly import assemble_step_arrays
from skillmodels.af.step_layout import (
    AFFactorInfo,
    AFFactorRole,
    HistoricalParams,
    compile_af_step_layouts,
)


def _layouts():
    return compile_af_step_layouts(
        (
            AFFactorInfo("skills", AFFactorRole.DYNAMIC, (("sk",), ("sk",), ("sk",))),
            AFFactorInfo("MC", AFFactorRole.STATIC_PERSISTENT, (("mc",), (), ())),
            AFFactorInfo(
                "investment", AFFactorRole.ENDOGENOUS, (("inv",), ("inv",), ())
            ),
        ),
        n_periods=3,
    )


def _frames():
    # age-7 investment lives at period 0; age-9 at period 1. MC measured at period 0.
    return {
        0: pd.DataFrame(
            {"sk": [1.0, 2.0], "mc": [7.0, 8.0], "inv": [0.1, 0.2]}, index=[1, 2]
        ),
        1: pd.DataFrame({"sk": [3.0, 4.0], "inv": [0.3, 0.4]}, index=[1, 2]),
        2: pd.DataFrame({"sk": [5.0, 6.0]}, index=[1, 2]),
    }


def _historical():
    rows = [
        ("loadings", 0, "mc", "MC", 1.5),
        ("controls", 0, "mc", "constant", 0.0),
        ("meas_sds", 0, "mc", "-", 0.4),
        ("loadings", 0, "sk", "skills", 1.0),
        ("controls", 0, "sk", "constant", 0.0),
        ("meas_sds", 0, "sk", "-", 0.5),
        ("loadings", 1, "sk", "skills", 1.0),
        ("controls", 1, "sk", "constant", 0.0),
        ("meas_sds", 1, "sk", "-", 0.5),
    ]
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3]) for r in rows],
        names=["category", "period", "name1", "name2"],
    )
    return HistoricalParams(pd.DataFrame({"value": [r[4] for r in rows]}, index=idx))


def test_step_0_to_1_target_sources_investment_from_period0() -> None:
    arrays = assemble_step_arrays(
        _layouts()[0],
        _frames(),
        ("skills", "MC", "investment"),
        _historical(),
        ("constant",),
    )
    # Target columns: skills@1 (=[3,4]) and investment@0 (age-7, =[0.1,0.2]).
    cols = {name: arrays.target_measurements[:, j] for j, name in arrays.target_order}
    np.testing.assert_array_equal(cols[("sk", 1)], [3.0, 4.0])
    np.testing.assert_array_equal(cols[("inv", 0)], [0.1, 0.2])


def test_step_1_to_2_importance_includes_static_mc_from_period0() -> None:
    arrays = assemble_step_arrays(
        _layouts()[1],
        _frames(),
        ("skills", "MC", "investment"),
        _historical(),
        ("constant",),
    )
    imp = {name: arrays.importance_measurements[:, j] for j, name in arrays.imp_order}
    # F3: MC_0 data is present in the 1->2 importance block (and source skills@1).
    np.testing.assert_array_equal(imp[("mc", 0)], [7.0, 8.0])
    np.testing.assert_array_equal(imp[("sk", 1)], [3.0, 4.0])
    # MC's fixed loading/SD come from the period-0 history.
    mc_row = [n for _, n in arrays.imp_order].index(("mc", 0))
    assert arrays.importance_loadings_flat[mc_row] == 1.5
    assert arrays.importance_meas_sds[mc_row] == 0.4


def _frames_with_control():
    # A non-constant measurement control `x` whose value differs by period, so a row
    # sourced from the wrong period is detectable.
    return {
        0: pd.DataFrame(
            {"sk": [1.0, 2.0], "mc": [7.0, 8.0], "inv": [0.1, 0.2], "x": [10.0, 20.0]},
            index=[1, 2],
        ),
        1: pd.DataFrame(
            {"sk": [3.0, 4.0], "inv": [0.3, 0.4], "x": [100.0, 200.0]}, index=[1, 2]
        ),
        2: pd.DataFrame({"sk": [5.0, 6.0], "x": [1000.0, 2000.0]}, index=[1, 2]),
    }


def _historical_with_control():
    rows = [
        ("loadings", 0, "mc", "MC", 1.5),
        ("controls", 0, "mc", "x", 0.5),
        ("meas_sds", 0, "mc", "-", 0.4),
        ("loadings", 1, "sk", "skills", 1.0),
        ("controls", 1, "sk", "x", 2.0),
        ("meas_sds", 1, "sk", "-", 0.5),
    ]
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2], r[3]) for r in rows],
        names=["category", "period", "name1", "name2"],
    )
    return HistoricalParams(pd.DataFrame({"value": [r[4] for r in rows]}, index=idx))


def test_importance_control_contrib_sources_each_row_from_its_control_period() -> None:
    # 1->2 importance block: MC@0 (control_period 0) + source skills@1 (period 1).
    # Each row's control contribution must use that row's OWN period's control data and
    # fixed params, not one shared source-period matrix.
    arrays = assemble_step_arrays(
        _layouts()[1],
        _frames_with_control(),
        ("skills", "MC", "investment"),
        _historical_with_control(),
        ("x",),
    )
    rows = [n for _, n in arrays.imp_order]
    mc_row = rows.index(("mc", 0))
    sk_row = rows.index(("sk", 1))
    # MC_0: x@period0 ([10,20]) * 0.5; skills_1: x@period1 ([100,200]) * 2.0.
    np.testing.assert_allclose(
        arrays.importance_control_contrib[:, mc_row], [5.0, 10.0]
    )
    np.testing.assert_allclose(
        arrays.importance_control_contrib[:, sk_row], [200.0, 400.0]
    )


def _frames_with_string_ids():
    # Same data as `_frames`, but individuals keyed by non-integer (string) IDs.
    return {
        0: pd.DataFrame(
            {"sk": [1.0, 2.0], "mc": [7.0, 8.0], "inv": [0.1, 0.2]}, index=["a", "b"]
        ),
        1: pd.DataFrame({"sk": [3.0, 4.0], "inv": [0.3, 0.4]}, index=["a", "b"]),
        2: pd.DataFrame({"sk": [5.0, 6.0]}, index=["a", "b"]),
    }


def test_assemble_preserves_non_integer_ids() -> None:
    # The adapter's contract is ID-indexed alignment, not numeric IDs: string/UUID
    # individual identifiers must align rather than be coerced or rejected.
    arrays = assemble_step_arrays(
        _layouts()[0],
        _frames_with_string_ids(),
        ("skills", "MC", "investment"),
        _historical(),
        ("constant",),
    )
    assert list(arrays.ids) == ["a", "b"]
    cols = {name: arrays.target_measurements[:, j] for j, name in arrays.target_order}
    np.testing.assert_array_equal(cols[("inv", 0)], [0.1, 0.2])


def test_target_controls_source_each_row_from_its_control_period() -> None:
    # 1->2 target block: skills@2 (control_period 2) + source investment@1 (period 1).
    # The per-row control-data tensor must read each row's own period's control values.
    arrays = assemble_step_arrays(
        _layouts()[1],
        _frames_with_control(),
        ("skills", "MC", "investment"),
        _historical_with_control(),
        ("x",),
    )
    order = [n for _, n in arrays.target_order]
    sk_col = order.index(("sk", 2))
    inv_col = order.index(("inv", 1))
    # target_controls has shape (n_ids, n_target, n_controls); single control `x`.
    # skills@2 reads x from period 2 ([1000,2000]); investment@1 reads x from period 1
    # ([100,200]) -- each row sourced from its own control_period.
    np.testing.assert_allclose(arrays.target_controls[:, sk_col, 0], [1000.0, 2000.0])
    np.testing.assert_allclose(arrays.target_controls[:, inv_col, 0], [100.0, 200.0])
