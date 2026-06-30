"""Tests for ID-aligned assembly of an AF step's measurement arrays.

A mixed-calendar step block sources columns from different calendar periods
(destination skills at `d`, source investment at `s`). Those per-period frames may have
different individual orders and samples, so the assembler must join on individual ID
rather than concatenate positionally (the AF2 defect), and source each term's column
from its own `data_period`.
"""

import numpy as np
import pandas as pd

from skillmodels.af.step_data import (
    build_block_loading_mask,
    build_step_measurement_array,
)
from skillmodels.af.step_layout import (
    AFFactorInfo,
    AFFactorRole,
    compile_af_step_layouts,
)


def _step_0_to_1_targets():
    layout = compile_af_step_layouts(
        (
            AFFactorInfo("skills", AFFactorRole.DYNAMIC, (("sk_a",), ("sk_a",))),
            AFFactorInfo("investment", AFFactorRole.ENDOGENOUS, (("inv_a",), ())),
        ),
        n_periods=2,
    )[0]
    return layout.target_terms()


def test_step_measurement_array_preserves_non_integer_ids() -> None:
    # The assembler joins on individual ID, which must hold for string/UUID IDs, not
    # only CNLSY-style numeric case IDs.
    frame0 = pd.DataFrame({"sk_a": [10.0, 11.0], "inv_a": [1.0, 2.0]}, index=["b", "a"])
    frame1 = pd.DataFrame({"sk_a": [100.0, 200.0]}, index=["a", "b"])
    ids, values, term_order = build_step_measurement_array(
        _step_0_to_1_targets(), {0: frame0, 1: frame1}
    )
    assert ids.tolist() == ["a", "b"]
    inv_col = term_order.index(("inv_a", 0))
    # investment from period 0, ID-aligned: a->2, b->1.
    np.testing.assert_array_equal(values[:, inv_col], [2.0, 1.0])


def test_step_measurement_array_sources_each_term_from_its_data_period() -> None:
    # Period frames deliberately have DIFFERENT individual orders.
    frame0 = pd.DataFrame(
        {"sk_a": [10.0, 11.0, 12.0], "inv_a": [1.0, 2.0, 3.0]}, index=[2, 1, 3]
    )
    frame1 = pd.DataFrame({"sk_a": [100.0, 200.0, 300.0]}, index=[3, 1, 2])
    ids, values, term_order = build_step_measurement_array(
        _step_0_to_1_targets(), {0: frame0, 1: frame1}
    )

    assert ids.tolist() == [1, 2, 3]
    # skills from period 1 (destination), ID-aligned: id1->200, id2->300, id3->100.
    sk_col = term_order.index(("sk_a", 1))
    np.testing.assert_array_equal(values[:, sk_col], [200.0, 300.0, 100.0])
    # investment from period 0 (source), ID-aligned: id1->2, id2->1, id3->3.
    inv_col = term_order.index(("inv_a", 0))
    np.testing.assert_array_equal(values[:, inv_col], [2.0, 1.0, 3.0])


def test_step_measurement_array_uses_id_intersection_under_attrition() -> None:
    # Individual 3 is missing from period 1; the step sample is the intersection.
    frame0 = pd.DataFrame(
        {"sk_a": [10.0, 11.0, 12.0], "inv_a": [1.0, 2.0, 3.0]}, index=[1, 2, 3]
    )
    frame1 = pd.DataFrame({"sk_a": [100.0, 200.0]}, index=[1, 2])
    ids, values, term_order = build_step_measurement_array(
        _step_0_to_1_targets(), {0: frame0, 1: frame1}
    )
    assert ids.tolist() == [1, 2]
    inv_col = term_order.index(("inv_a", 0))
    np.testing.assert_array_equal(values[:, inv_col], [1.0, 2.0])


def test_step_measurement_array_does_not_mispair_positionally() -> None:
    # If assembled positionally, id-2's investment (period 0, row 0 = 99) would be
    # paired with id-2's skill (period 1, row 0 = 100), which is wrong.
    frame0 = pd.DataFrame({"sk_a": [0.0, 0.0], "inv_a": [99.0, 7.0]}, index=[2, 1])
    frame1 = pd.DataFrame({"sk_a": [100.0, 500.0]}, index=[1, 2])
    ids, values, term_order = build_step_measurement_array(
        _step_0_to_1_targets(), {0: frame0, 1: frame1}
    )
    inv_col = term_order.index(("inv_a", 0))
    sk_col = term_order.index(("sk_a", 1))
    # id 1: inv=7 (period0), skill=100 (period1); id 2: inv=99, skill=500.
    assert ids.tolist() == [1, 2]
    np.testing.assert_array_equal(values[:, inv_col], [7.0, 99.0])
    np.testing.assert_array_equal(values[:, sk_col], [100.0, 500.0])


def test_block_loading_mask_marks_each_term_factor() -> None:
    targets = _step_0_to_1_targets()  # sk_a -> skills, inv_a -> investment
    # Latent order is [skills, investment]; the integrand dots loadings against it.
    mask = build_block_loading_mask(targets, latent_factors=("skills", "investment"))
    term_order = [(t.measurement, t.eval_node.value) for t in targets]
    sk_row = term_order.index(("sk_a", "theta_dest"))
    inv_row = term_order.index(("inv_a", "inv_src"))
    # skills row loads only on the skills column; investment row only on investment.
    np.testing.assert_array_equal(mask[sk_row], [True, False])
    np.testing.assert_array_equal(mask[inv_row], [False, True])
