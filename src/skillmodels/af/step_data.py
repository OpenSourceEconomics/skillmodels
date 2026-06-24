"""ID-aligned assembly of an AF step's mixed-calendar measurement arrays.

A compiled `AFStepLayout` draws measurement columns from different calendar periods
(destination skills at `d`, source investment at `s`, static factors at 0). Their
per-period frames can differ in individual order and sample, so columns must be joined
on individual ID -- never concatenated positionally -- and each column sourced from its
term's own `data_period`.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from skillmodels.af.step_layout import AFMeasurementTerm


def build_step_measurement_array(
    terms: Sequence[AFMeasurementTerm],
    frames_by_period: Mapping[int, pd.DataFrame],
) -> tuple[NDArray[np.int64], NDArray[np.float64], list[tuple[str, int]]]:
    """Assemble one ID-aligned measurement value array for a set of step terms.

    Each term's column is read from `frames_by_period[term.data_period]` at the term's
    measurement variable. The step sample is the intersection of individual IDs across
    every sourced period, sorted ascending, so columns from different periods are paired
    by ID rather than by row position.

    Args:
        terms: The measurement terms (e.g. a layout's target or importance terms).
        frames_by_period: Calendar period -> frame indexed by individual ID with
            measurement-variable columns.

    Return:
        `(ids, values, term_order)` where `ids` is the sorted common ID vector, `values`
        is `(n_ids, n_terms)` with `values[:, j]` the `j`-th term's column, and
        `term_order` lists `(measurement, data_period)` per column.

    """
    if not terms:
        ids_empty = np.array([], dtype=np.int64)
        return ids_empty, np.zeros((0, 0), dtype=np.float64), []

    needed_periods = {term.data_period for term in terms}
    common_ids: set | None = None
    for period in needed_periods:
        period_ids = set(frames_by_period[period].index)
        common_ids = period_ids if common_ids is None else (common_ids & period_ids)
    ids = np.array(sorted(common_ids or set()), dtype=np.int64)

    columns = []
    term_order: list[tuple[str, int]] = []
    for term in terms:
        frame = frames_by_period[term.data_period]
        column = frame.loc[ids, term.measurement].to_numpy(dtype=np.float64)
        columns.append(column)
        term_order.append((term.measurement, term.data_period))

    values = (
        np.column_stack(columns)
        if columns
        else np.zeros((len(ids), 0), dtype=np.float64)
    )
    return ids, values, term_order
