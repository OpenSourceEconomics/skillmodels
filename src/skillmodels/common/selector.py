"""Selector helpers for optimagic constraint plumbing.

Both `skillmodels.common.constraints` and
`skillmodels.common.transition_functions` build optimagic constraint
selectors of the form `functools.partial(select_by_loc, loc=...)`.
Hosting `select_by_loc` here breaks the previous circular-import
workaround (`constraints` -> `transition_functions` -> a hand-copy
of `select_by_loc`) by giving both call sites a single dependency
that pulls nothing else from `skillmodels`.

`align_index_names` lives here because it is also a selector-side
concern: users supply `fixed_params` / `start_params` keyed by the
public-facing `period` level name, while internal frames are keyed
by `aug_period`. The two-line rename lets downstream `MultiIndex`
set operations keep their level names.
"""

from collections.abc import Hashable, Sequence
from typing import Any

import pandas as pd


def select_by_loc(params: Any, loc: Any) -> Any:  # noqa: ANN401
    """Select parameters by location, restricted to the `value` column.

    optimagic's pytree machinery flattens whatever the selector
    returns. A bare `params.loc[single_tuple]` is a row `Series`
    whose index is the column names (`value`, `lower_bound`,
    `upper_bound`); flattening that yields all three values, and the
    bounds' `±inf` collapse to int64 sentinels inside
    `_fail_if_duplicates`. Project down to the `value` column so the
    selector returns exactly the parameter values regardless of
    whether bounds columns are present.
    """
    selected = params.loc[loc]
    if isinstance(selected, pd.Series) and "value" in selected.index:
        return selected["value"]
    if isinstance(selected, pd.DataFrame) and "value" in selected.columns:
        return selected["value"]
    return selected


def align_index_names(
    overrides: pd.DataFrame, target_names: Sequence[Hashable]
) -> pd.DataFrame:
    """Return `overrides` with its MultiIndex level names matched to `target_names`.

    `MultiIndex.union` silently strips any level whose name differs
    across the two operands, collapsing the result to anonymous
    levels and breaking `params.loc[...]` for callers downstream.
    Re-stamping the overrides' level names keeps the underlying
    tuples bit-identical (`set_names` is metadata-only) while making
    the union name-preserving. Used wherever user-supplied
    `fixed_params` / `start_params` (typically keyed by `period`)
    meet an internal params frame keyed by `aug_period`.
    """
    if list(overrides.index.names) == list(target_names):
        return overrides
    new_index = overrides.index.set_names(list(target_names))
    out = overrides.copy()
    out.index = new_index
    return out
