"""Tests for `skillmodels.common.selector` and the projection helper.

`select_by_loc`, `align_index_names`, `collect_fixed_locs`, and
`project_to_probability_constraints` are the small, pure helpers that
glue user-supplied `fixed_params` / `start_params` to optimagic's
constraint pipeline. Each helper has a specific shape contract that
matters in only one or two call sites; these tests pin those
contracts so future refactors don't silently reintroduce the bugs
they were written to fix.
"""

import functools

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.common.constraints import (
    FixedConstraintWithValue,
    collect_fixed_locs,
    project_to_probability_constraints,
)
from skillmodels.common.selector import align_index_names, select_by_loc

# --- select_by_loc ---------------------------------------------------------


def _params_with_bounds() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            ("loadings", 0, "m1", "fac1"),
            ("loadings", 0, "m2", "fac1"),
            ("loadings", 0, "m3", "fac1"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "lower_bound": [-np.inf, 0.1, -np.inf],
            "upper_bound": [np.inf, np.inf, np.inf],
        },
        index=idx,
    )


def test_select_by_loc_single_tuple_returns_scalar_value() -> None:
    """A single-tuple `loc` must return just the value, not the row Series.

    `params.loc[single_tuple]` gives a row Series indexed by column
    names; if optimagic's pytree walk sees it, the `±inf` bounds get
    cast to int64 sentinels and `_fail_if_duplicates` raises
    `IndexError`. The projection to the `value` cell prevents that.
    """
    params = _params_with_bounds()
    out = select_by_loc(params, ("loadings", 0, "m1", "fac1"))
    assert out == pytest.approx(1.0)
    assert not isinstance(out, pd.Series)


def test_select_by_loc_list_of_tuples_returns_value_series() -> None:
    """A list-of-tuples `loc` projects the result DataFrame to the value column."""
    params = _params_with_bounds()
    locs = [
        ("loadings", 0, "m1", "fac1"),
        ("loadings", 0, "m2", "fac1"),
    ]
    out = select_by_loc(params, locs)
    assert isinstance(out, pd.Series)
    assert out.tolist() == [1.0, 2.0]
    assert "lower_bound" not in (getattr(out, "name", None) or "")


def test_select_by_loc_no_value_column_returns_unchanged() -> None:
    """When the params frame has no `value` column, no projection happens."""
    idx = pd.MultiIndex.from_tuples(
        [("loadings", 0, "m1", "fac1")],
        names=["category", "period", "name1", "name2"],
    )
    params = pd.DataFrame({"other": [9.0]}, index=idx)
    out = select_by_loc(params, ("loadings", 0, "m1", "fac1"))
    assert "other" in out.index
    assert out["other"] == pytest.approx(9.0)


# --- align_index_names -----------------------------------------------------


def test_align_index_names_renames_period_to_aug_period() -> None:
    """Renaming preserves the underlying tuples bit-for-bit."""
    idx = pd.MultiIndex.from_tuples(
        [("loadings", 1, "m1", "fac1")],
        names=["category", "period", "name1", "name2"],
    )
    df = pd.DataFrame({"value": [0.7]}, index=idx)
    out = align_index_names(
        df, target_names=["category", "aug_period", "name1", "name2"]
    )
    assert list(out.index.names) == ["category", "aug_period", "name1", "name2"]
    assert list(out.index[0]) == ["loadings", 1, "m1", "fac1"]
    assert out.loc[("loadings", 1, "m1", "fac1"), "value"] == pytest.approx(0.7)


def test_align_index_names_passes_through_when_already_matching() -> None:
    """When the names already match, `align_index_names` returns the input as-is."""
    idx = pd.MultiIndex.from_tuples(
        [("loadings", 0, "m1", "fac1")],
        names=["category", "aug_period", "name1", "name2"],
    )
    df = pd.DataFrame({"value": [0.5]}, index=idx)
    out = align_index_names(
        df, target_names=["category", "aug_period", "name1", "name2"]
    )
    assert out is df


# --- collect_fixed_locs ----------------------------------------------------


def test_collect_fixed_locs_unpacks_single_tuple() -> None:
    constraints = [
        FixedConstraintWithValue(loc=("loadings", 0, "m1", "fac1"), value=1.0),
    ]
    out = collect_fixed_locs(constraints)
    assert out == {("loadings", 0, "m1", "fac1")}


def test_collect_fixed_locs_unpacks_tuple_of_tuples() -> None:
    """Anchoring constraints pack many locs into a single `tuple(loc_tuples)`."""
    inner = (
        ("controls", 0, "Q1_fac1", "constant"),
        ("controls", 1, "Q1_fac1", "constant"),
    )
    constraints = [FixedConstraintWithValue(loc=inner, value=0.0)]
    out = collect_fixed_locs(constraints)
    assert out == set(inner)


def test_collect_fixed_locs_unpacks_pd_multiindex() -> None:
    """`FixedConstraintWithValue.loc` permits `pd.MultiIndex` per its annotation."""
    mi = pd.MultiIndex.from_tuples(
        [
            ("loadings", 0, "m1", "fac1"),
            ("loadings", 1, "m1", "fac1"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    constraints = [FixedConstraintWithValue(loc=mi, value=1.0)]
    out = collect_fixed_locs(constraints)
    assert out == {
        ("loadings", 0, "m1", "fac1"),
        ("loadings", 1, "m1", "fac1"),
    }


def test_collect_fixed_locs_skips_non_fixed_constraints() -> None:
    """Equality / probability constraints don't carry parameter values."""
    eq = om.EqualityConstraint(
        selector=functools.partial(select_by_loc, loc=("transition", 0, "a", "b")),
    )
    out = collect_fixed_locs([eq])
    assert out == set()


# --- project_to_probability_constraints ------------------------------------


def _gamma_constraint(period: int) -> om.ProbabilityConstraint:
    locs = [
        ("transition", period, "skills", "skills"),
        ("transition", period, "skills", "MC"),
        ("transition", period, "skills", "investment"),
    ]
    return om.ProbabilityConstraint(
        selector=functools.partial(select_by_loc, loc=locs),
    )


def _three_gamma_template(period: int = 0, values=(0.5, 0.5, 0.5)) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            ("transition", period, "skills", "skills"),
            ("transition", period, "skills", "MC"),
            ("transition", period, "skills", "investment"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    return pd.DataFrame({"value": list(values)}, index=idx)


def test_project_renormalises_free_entries_to_one_when_none_pinned() -> None:
    template = _three_gamma_template(values=(0.5, 0.5, 0.5))
    out = project_to_probability_constraints(
        params_template=template, constraints=[_gamma_constraint(period=0)]
    )
    assert out["value"].tolist() == pytest.approx([1.0 / 3, 1.0 / 3, 1.0 / 3])


def test_project_leaves_pinned_entries_alone_and_rescales_free() -> None:
    """Partial-pin path: pinned entries stay, free entries sum to 1 - pinned_total."""
    template = _three_gamma_template(values=(0.7, 0.0, 0.7))
    pinned_loc = ("transition", 0, "skills", "MC")
    constraints = [
        _gamma_constraint(period=0),
        FixedConstraintWithValue(loc=pinned_loc, value=0.0),
    ]
    # Caller (`enforce_fixed_constraints`) is what writes 0.0 into the
    # template at the pinned cell; the projection should preserve it.
    out = project_to_probability_constraints(
        params_template=template, constraints=constraints
    )
    assert out.loc[pinned_loc, "value"] == pytest.approx(0.0)
    free = out["value"].drop(index=pinned_loc)
    assert float(free.sum()) == pytest.approx(1.0)
    # Free entries were both 0.7; they should be rescaled by 1 / 1.4 = 5/7.
    assert free.iloc[0] == pytest.approx(0.5)
    assert free.iloc[1] == pytest.approx(0.5)


def test_project_respects_non_zero_pinned_value() -> None:
    """When a gamma is pinned to a value in (0, 1), free entries fill the residual."""
    template = _three_gamma_template(values=(0.5, 0.2, 0.5))
    pinned_loc = ("transition", 0, "skills", "MC")
    constraints = [
        _gamma_constraint(period=0),
        FixedConstraintWithValue(loc=pinned_loc, value=0.2),
    ]
    out = project_to_probability_constraints(
        params_template=template, constraints=constraints
    )
    assert out.loc[pinned_loc, "value"] == pytest.approx(0.2)
    free = out["value"].drop(index=pinned_loc)
    assert float(free.sum()) == pytest.approx(0.8)


def test_project_skips_when_already_on_simplex() -> None:
    template = _three_gamma_template(values=(0.2, 0.3, 0.5))
    out = project_to_probability_constraints(
        params_template=template, constraints=[_gamma_constraint(period=0)]
    )
    # No mutation; same DataFrame object returned.
    assert out is template


def test_project_skips_when_free_entries_are_zero() -> None:
    """Degenerate case: all-zero free entries. Caller has to fix it."""
    template = _three_gamma_template(values=(0.0, 0.0, 0.0))
    out = project_to_probability_constraints(
        params_template=template, constraints=[_gamma_constraint(period=0)]
    )
    assert out["value"].tolist() == [0.0, 0.0, 0.0]
