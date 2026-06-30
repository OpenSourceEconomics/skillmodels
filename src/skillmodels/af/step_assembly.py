"""Assemble an AF transition step's target + importance arrays from a compiled layout.

Combines the layout (which says *which* measurement belongs in each block and from which
period) with the per-period ID-indexed data frames and the cumulative `HistoricalParams`
to produce the arrays the integrand consumes:

- the free **target** block (destination skills + source investment) -- data + loading
  mask;
- the fixed **importance** block (source skills + every static-persistent factor's
  period-0 rows) -- data, loading mask, and fixed loadings/controls/SDs read from
  history by each row's true `param_period`.

All columns are sourced on one common individual-ID set (the intersection across every
period the step touches), so target and importance rows refer to the same individuals.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from skillmodels.af.step_data import build_block_loading_mask
from skillmodels.af.step_layout import (
    AFMeasurementTerm,
    AFStepLayout,
    HistoricalParams,
)


@dataclass(frozen=True)
class AFStepArrays:
    """Target + importance measurement arrays for one AF transition step."""

    ids: np.ndarray
    """The step's common individual-ID vector (sorted intersection)."""
    target_measurements: np.ndarray
    """`(n_ids, n_target)` free target measurement values."""
    target_loading_mask: np.ndarray
    """`(n_target, n_latent)` target loading mask."""
    target_order: list[tuple[int, tuple[str, int]]]
    """`(column, (measurement, data_period))` for each target column."""
    importance_measurements: np.ndarray
    """`(n_ids, n_importance)` fixed importance measurement values."""
    importance_loading_mask: np.ndarray
    """`(n_importance, n_latent)` importance loading mask."""
    imp_order: list[tuple[int, tuple[str, int]]]
    """`(column, (measurement, data_period))` for each importance column."""
    importance_loadings_flat: np.ndarray
    """Fixed importance loadings, packed in loading-mask order, from history."""
    importance_control_params: np.ndarray
    """`(n_importance, n_controls)` fixed importance control params, from history."""
    importance_meas_sds: np.ndarray
    """`(n_importance,)` fixed importance measurement SDs, from history."""
    target_controls: np.ndarray
    """`(n_ids, n_target, n_controls)` per-row target control data, each row sourced
    from its term's own `control_period` (the free target control coefficients are
    estimated, so only the data -- not the contribution -- can be precompiled)."""
    importance_control_contrib: np.ndarray
    """`(n_ids, n_importance)` fixed importance control contribution, each row's control
    data (at its `control_period`) dotted with its fixed control params from history."""


def assemble_step_arrays(
    layout: AFStepLayout,
    frames_by_period: Mapping[int, pd.DataFrame],
    latent_factors: Sequence[str],
    historical: HistoricalParams,
    controls: tuple[str, ...],
) -> AFStepArrays:
    """Assemble the target + importance arrays for one transition step.

    Args:
        layout: The compiled step layout.
        frames_by_period: Calendar period -> frame indexed by individual ID.
        latent_factors: Latent-factor ordering the integrand dots loadings against.
        historical: Cumulative parameter registry for fixed importance params.
        controls: Control variable names (for importance control params).

    Return:
        An `AFStepArrays` with target/importance data, masks, and fixed params.

    """
    # @pro: calendar split for step s->d. The FREE target block = {destination skills
    # on theta_d, source investment I_s on period s}; the FIXED importance block =
    # {source skills on theta_s, every static-persistent factor's period-0 rows}. The
    # importance loadings/controls/SDs are read from `historical` by each row's true
    # param_period (period-0 for static factors), so the carry-over density is fixed at
    # its originally estimated value. Confirm target/importance membership and the
    # history sourcing are the right ones for the MATLAB sequential likelihood.
    targets = layout.target_terms()
    importances = layout.importance_terms()

    needed = {term.data_period for term in (*targets, *importances)}
    common: set | None = None
    for period in needed:
        period_ids = set(frames_by_period[period].index)
        common = period_ids if common is None else (common & period_ids)
    # Preserve the individual-ID values as-is (object dtype) rather than coercing to
    # int64: the adapter's contract is stable ID alignment, which must hold for string,
    # UUID, or other non-integer identifiers, not just CNLSY-style numeric case IDs.
    ids = np.array(sorted(common or set()), dtype=object)

    target_order, target_values = _source_block(targets, frames_by_period, ids)
    imp_order, imp_values = _source_block(importances, frames_by_period, ids)
    target_mask = build_block_loading_mask(targets, latent_factors)
    imp_mask = build_block_loading_mask(importances, latent_factors)

    loadings_flat: list[float] = []
    for row, term in enumerate(importances):
        for factor_idx, factor in enumerate(latent_factors):
            if imp_mask[row, factor_idx]:
                loadings_flat.append(
                    historical.value(
                        "loadings", term.param_period, term.measurement, factor
                    )
                )
    control_params = np.array(
        [
            [
                historical.value(
                    "controls", term.control_period, term.measurement, ctrl
                )
                for ctrl in controls
            ]
            for term in importances
        ],
        dtype=np.float64,
    ).reshape(len(importances), len(controls))
    meas_sds = np.array(
        [
            historical.value("meas_sds", term.param_period, term.measurement, "-")
            for term in importances
        ],
        dtype=np.float64,
    )

    target_controls = _source_control_tensor(targets, frames_by_period, ids, controls)
    imp_controls = _source_control_tensor(importances, frames_by_period, ids, controls)
    # Importance control params are fixed (from history), so the contribution can be
    # precompiled per row at its own control_period: (n_ids, n_imp, n_ctrl) . (n_imp,
    # n_ctrl) -> (n_ids, n_imp). Target control params are estimated, so only the data
    # tensor is precompiled; the kernel forms its contribution each iteration.
    importance_control_contrib = np.einsum(
        "imc,mc->im", imp_controls, control_params, optimize=False
    )

    return AFStepArrays(
        ids=ids,
        target_measurements=target_values,
        target_loading_mask=target_mask,
        target_order=list(enumerate(target_order)),
        importance_measurements=imp_values,
        importance_loading_mask=imp_mask,
        imp_order=list(enumerate(imp_order)),
        importance_loadings_flat=np.array(loadings_flat, dtype=np.float64),
        importance_control_params=control_params,
        importance_meas_sds=meas_sds,
        target_controls=target_controls,
        importance_control_contrib=importance_control_contrib,
    )


def _source_block(
    terms: Sequence[AFMeasurementTerm],
    frames_by_period: Mapping[int, pd.DataFrame],
    ids: np.ndarray,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Source a block's columns on a shared ID set, each from its term's data period."""
    order = [(term.measurement, term.data_period) for term in terms]
    if not terms:
        return order, np.zeros((len(ids), 0), dtype=np.float64)
    columns = [
        frames_by_period[term.data_period]
        .loc[ids, term.measurement]
        .to_numpy(dtype=np.float64)
        for term in terms
    ]
    return order, np.column_stack(columns)


def _source_control_tensor(
    terms: Sequence[AFMeasurementTerm],
    frames_by_period: Mapping[int, pd.DataFrame],
    ids: np.ndarray,
    controls: tuple[str, ...],
) -> np.ndarray:
    """Source each term's control values on the shared IDs from its `control_period`.

    Returns shape `(n_ids, n_terms, n_controls)`. A `"constant"` control is a column of
    ones; a named control is read from that term's `control_period` frame; an absent
    control is zero. Reading each row at its own `control_period` is what keeps a
    mixed-calendar block (e.g. destination skills at `d` with source investment at `s`)
    from evaluating every row against one shared period's controls.
    """
    tensor = np.zeros((len(ids), len(terms), len(controls)), dtype=np.float64)
    for row, term in enumerate(terms):
        frame = frames_by_period[term.control_period]
        for col, ctrl in enumerate(controls):
            if ctrl == "constant":
                tensor[:, row, col] = 1.0
            elif ctrl in frame.columns:
                tensor[:, row, col] = frame.loc[ids, ctrl].to_numpy(dtype=np.float64)
    return tensor
