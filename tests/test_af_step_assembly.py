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
