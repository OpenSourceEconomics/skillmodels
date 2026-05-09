"""Stage-1 measurement-system estimation for AF (factor-analysis pre-step).

Estimate the measurement system parameters (loadings, intercepts,
sigma_meas) period-by-period and factor-by-factor from cross-covariance
moments of multi-indicator measurements (standard
Spearman / multi-indicator factor-analysis identification), and pack the
result into a `fixed_params`-shaped DataFrame so the AF Stage-2
optimizer can hold those values fixed.

This eliminates the sigma_inv / sigma_meas constant-Var(I_meas) ridge
that causes ~40% sigma_inv_0 boundary collapse on translog-style DGPs:
once sigma_meas is pinned, sigma_inv is identified by the marginal
Var(I_meas) directly. See the obsidian note
``af-sigma-inv-identification-analysis-2026-05-08.md`` for the
theoretical background.

Standard-error caveat: Stage 2's existing sandwich treats the Stage-1
outputs as known and therefore under-states variance for any Stage-2
parameter that covaries with sigma_meas (notably sigma_inv, sigma_shock,
mixture covariance). Users wanting fully-correct SEs should run a
parametric bootstrap until a Murphy-Topel correction lands.
"""

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from skillmodels.af.moment_init import spearman_factor_moments
from skillmodels.af.params import (
    get_measurements_per_factor,
    get_normalizations_for_period,
)
from skillmodels.model_spec import ModelSpec


def estimate_measurement_system(  # noqa: C901
    model_spec: ModelSpec,
    data: pd.DataFrame,
    *,
    user_fixed_params: pd.DataFrame | None = None,
    min_n_per_factor: int = 50,
) -> pd.DataFrame:
    """Estimate the AF measurement system via Spearman cross-covariances.

    For each calendar period and each latent factor with at least two
    measurements, run Spearman moment estimation on the cross-covariance
    matrix of that factor's measurements (after residualizing on
    controls). Pack the recovered loadings and sigma_meas into a
    `fixed_params`-shaped DataFrame that the AF Stage-2 optimizer can hold
    fixed.

    Args:
        model_spec: Model specification.
        data: Long-format DataFrame indexed by ``(id, period)``.
        user_fixed_params: Existing user-supplied fixed_params. Indices
            present here are not overwritten by Stage-1 outputs.
        min_n_per_factor: Minimum complete-case sample size per
            (factor, period). Skipped with a warning below this threshold.

    Return:
        DataFrame with the standard 4-level MultiIndex
        ``(category, period, name1, name2)`` and a single ``value`` column,
        restricted to ``loadings`` and ``meas_sds`` rows (controls are not
        produced; the AF optimizer keeps fitting those).

    """
    period_col = str(data.index.names[1])
    user_indices = (
        set(user_fixed_params.index) if user_fixed_params is not None else set()
    )
    n_periods = _max_period(model_spec) + 1

    rows: list[tuple[tuple[str, int, str, str], float]] = []

    for period in range(n_periods):
        measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
        if not measurements_pt:
            continue

        norms = get_normalizations_for_period(model_spec.factors, period=period)
        loading_norms = norms.get("loadings", {})

        period_mask = data.index.get_level_values(period_col) == period
        period_df = data.loc[period_mask]

        for factor, factor_meas in measurements_pt.items():
            if len(factor_meas) < 2:
                _warn_skip(
                    factor=factor,
                    period=period,
                    reason="fewer than two measurements (Spearman not identified)",
                )
                continue

            meas_cols = [m for m in factor_meas if m in period_df.columns]
            if len(meas_cols) < 2:
                _warn_skip(
                    factor=factor,
                    period=period,
                    reason="measurement columns missing from data",
                )
                continue

            measurements_arr = period_df[meas_cols].to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            n_complete = int(np.all(np.isfinite(measurements_arr), axis=1).sum())
            if n_complete < min_n_per_factor:
                _warn_skip(
                    factor=factor,
                    period=period,
                    reason=(
                        f"only {n_complete} complete cases; below "
                        f"min_n_per_factor={min_n_per_factor}"
                    ),
                )
                continue

            anchor_local, anchor_loading = _resolve_anchor(
                meas_cols=meas_cols,
                factor=factor,
                loading_norms=loading_norms,
            )

            result = spearman_factor_moments(
                measurements_arr,
                anchor_idx=anchor_local,
                anchor_loading=anchor_loading,
            )
            if not result.valid:
                _warn_skip(
                    factor=factor,
                    period=period,
                    reason="Spearman returned valid=False (degenerate cov)",
                )
                continue

            for local_idx, meas_name in enumerate(meas_cols):
                load_loc = ("loadings", period, meas_name, factor)
                if load_loc not in user_indices:
                    rows.append((load_loc, float(result.loadings[local_idx])))
                sd_loc = ("meas_sds", period, meas_name, "-")
                if sd_loc not in user_indices:
                    rows.append((sd_loc, float(result.meas_sds[local_idx])))

    if not rows:
        return pd.DataFrame(
            {"value": []},
            index=pd.MultiIndex.from_tuples(
                [], names=["category", "period", "name1", "name2"]
            ),
        )

    # Deduplicate any rows that may have been written twice (e.g. a
    # measurement loading on multiple factors). Last-write wins; in
    # practice the code path above writes each loading at most once per
    # (factor, measurement) pair so this is a defensive cleanup.
    deduped: dict[tuple[str, int, str, str], float] = dict(rows)

    index = pd.MultiIndex.from_tuples(
        list(deduped.keys()), names=["category", "period", "name1", "name2"]
    )
    return pd.DataFrame({"value": list(deduped.values())}, index=index)


def merge_with_user_fixed_params(
    user_fixed: pd.DataFrame | None,
    stage1: pd.DataFrame,
) -> pd.DataFrame:
    """Merge user `fixed_params` with Stage-1 outputs.

    User-pinned entries always win (Stage-1 only contributes rows whose
    indices are NOT already in `user_fixed`).
    """
    if user_fixed is None or len(user_fixed) == 0:
        return stage1
    if len(stage1) == 0:
        return user_fixed
    new_only = stage1.loc[stage1.index.difference(user_fixed.index)]
    return pd.concat([user_fixed, new_only])


def _max_period(model_spec: ModelSpec) -> int:
    """Return the maximum user period index used by any factor's measurements."""
    max_t = -1
    for spec in model_spec.factors.values():
        if not spec.measurements:
            continue
        for t, meas_at_t in enumerate(spec.measurements):
            if meas_at_t:
                max_t = max(max_t, t)
    return max_t


def _resolve_anchor(
    *,
    meas_cols: Iterable[str],
    factor: str,
    loading_norms: dict[tuple[str, str], float],
) -> tuple[int, float]:
    """Pick the anchor index + loading from user normalizations."""
    for local_idx, meas_name in enumerate(meas_cols):
        if (meas_name, factor) in loading_norms:
            return local_idx, float(loading_norms[(meas_name, factor)])
    return 0, 1.0


def _warn_skip(*, factor: str, period: int, reason: str) -> None:
    msg = (
        f"Stage-1 measurement-system estimation skipped factor "
        f"{factor!r} at period {period}: {reason}. The AF Stage-2 "
        f"optimizer will fit those parameters with the standard "
        f"initialization."
    )
    warnings.warn(msg, stacklevel=2)
