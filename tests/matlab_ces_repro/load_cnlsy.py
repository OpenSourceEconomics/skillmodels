"""Load and preprocess the CNLSY MATLAB input data for AF reproduction.

Mirrors the column construction and per-period standardisation in
`AF_Application_One_Normal_CES.m` lines 30-53. The resulting long-format
DataFrame feeds directly into `estimate_af`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Column groups (MATLAB lines 30-42).
_MC_COLS: tuple[str, ...] = (
    "asvab2",
    "asvab3",
    "asvab4",
    "asvab5",
    "asvab6",
    "asvab8",
)
_MN_NEG_COLS: tuple[str, ...] = ("se1", "se2", "se4", "se6")
_MN_POS_COLS: tuple[str, ...] = ("se3", "se5", "se8", "se9", "se10")
_MN_ROTTER_COLS: tuple[str, ...] = ("rotter1", "rotter2", "rotter3", "rotter4")
_SKILL_COLS_BY_WAVE: tuple[tuple[str, ...], ...] = (
    ("math7", "recog7", "comp7"),
    ("math9", "recog9", "comp9"),
    ("math11", "recog11", "comp11"),
)
_INV_COLS_BY_WAVE: tuple[tuple[str, ...], ...] = (
    ("often_mom_reads7", "often_museum7", "often_praised7"),
    ("often_mom_reads9", "often_museum9", "often_praised9"),
)
_INCOME_COLS_BY_WAVE: tuple[str, ...] = ("faminc7", "faminc9")

# Measurement names used in the skillmodels ModelSpec (period-independent).
MC_MEASURES: tuple[str, ...] = tuple(f"mc_{i + 1}" for i in range(len(_MC_COLS)))
MN_MEASURES: tuple[str, ...] = ("mn_neg", "mn_pos", "mn_rotter")
SKILL_MEASURES: tuple[str, ...] = ("skill_math", "skill_recog", "skill_comp")
INV_MEASURES: tuple[str, ...] = ("inv_reads", "inv_museum", "inv_praised")
INCOME_MEASURE: str = "log_income_observed"


def _standardise(values: np.ndarray) -> np.ndarray:
    """Z-score columns of a 2D array (mean 0, sd 1 per column)."""
    mean = np.nanmean(values, axis=0, keepdims=True)
    sd = np.nanstd(values, axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (values - mean) / sd


def load_measurements(path: Path) -> pd.DataFrame:
    """Load CNLSY measurements into long format and standardise per period.

    The MATLAB code standardises each measurement block separately:
    - ``Z_MC`` is standardised across the whole sample (time-invariant).
    - ``Z_MN`` is standardised across the whole sample (time-invariant).
    - ``Z_skills`` and ``Z_inv`` are standardised within each period.

    Args:
        path: Path to ``complete_7_9_11.xls``.

    Return:
        Long-format ``pd.DataFrame`` indexed by ``(caseid, period)`` with
        columns for every measurement used in the estimation. Time-invariant
        blocks (``mc_*``, ``mn_*``) are written only in period 0 and filled
        with NaN in later periods so the measurement system does not double
        count them. Investment measurements appear in periods 0 and 1 only.
    """
    raw = pd.read_excel(path)

    n_periods = len(_SKILL_COLS_BY_WAVE)

    caseid = np.asarray(raw["child_id_nlsy"].to_numpy())

    # MC: 6 asvab measures, standardised once across the sample.
    mc = _standardise(raw[list(_MC_COLS)].to_numpy(dtype=np.float64))

    # MN: three aggregated measures (means of neg / pos / rotter items).
    mn_raw = np.column_stack(
        [
            raw[list(_MN_NEG_COLS)].to_numpy(dtype=np.float64).mean(axis=1),
            raw[list(_MN_POS_COLS)].to_numpy(dtype=np.float64).mean(axis=1),
            raw[list(_MN_ROTTER_COLS)].to_numpy(dtype=np.float64).mean(axis=1),
        ]
    )
    mn = _standardise(mn_raw)

    # Skills: per-period standardisation.
    skills_by_period: list[np.ndarray] = []
    for cols in _SKILL_COLS_BY_WAVE:
        skills_by_period.append(
            _standardise(raw[list(cols)].to_numpy(dtype=np.float64))
        )

    # Investment: per-period standardisation (only periods 0 and 1).
    inv_by_period: list[np.ndarray] = []
    for cols in _INV_COLS_BY_WAVE:
        inv_by_period.append(_standardise(raw[list(cols)].to_numpy(dtype=np.float64)))

    # Log income (already log-transformed in the source; no standardisation).
    income_by_period: list[np.ndarray] = [
        raw[col].to_numpy(dtype=np.float64) for col in _INCOME_COLS_BY_WAVE
    ]

    rows = _assemble_rows(
        caseid=caseid,
        n_periods=n_periods,
        skills_by_period=skills_by_period,
        mc=mc,
        mn=mn,
        inv_by_period=inv_by_period,
        income_by_period=income_by_period,
    )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def _assemble_rows(
    *,
    caseid: np.ndarray,
    n_periods: int,
    skills_by_period: list[np.ndarray],
    mc: np.ndarray,
    mn: np.ndarray,
    inv_by_period: list[np.ndarray],
    income_by_period: list[np.ndarray],
) -> list[dict[str, float | int]]:
    """Assemble the long-format row dictionaries for ``load_measurements``."""
    rows: list[dict[str, float | int]] = []
    for i in range(len(caseid)):
        for t in range(n_periods):
            row: dict[str, float | int] = {
                "caseid": int(caseid[i]),
                "period": t,
            }
            _fill_skills(row, i, t, skills_by_period)
            _fill_static(row, i, t, mc, mn)
            _fill_investment(row, i, t, inv_by_period)
            _fill_income(row, i, t, income_by_period)
            rows.append(row)
    return rows


def _fill_skills(
    row: dict[str, float | int],
    i: int,
    t: int,
    skills_by_period: list[np.ndarray],
) -> None:
    for j, name in enumerate(SKILL_MEASURES):
        row[name] = float(skills_by_period[t][i, j])


def _fill_static(
    row: dict[str, float | int],
    i: int,
    t: int,
    mc: np.ndarray,
    mn: np.ndarray,
) -> None:
    if t == 0:
        for j, name in enumerate(MC_MEASURES):
            row[name] = float(mc[i, j])
        for j, name in enumerate(MN_MEASURES):
            row[name] = float(mn[i, j])
    else:
        for name in (*MC_MEASURES, *MN_MEASURES):
            row[name] = float("nan")


def _fill_investment(
    row: dict[str, float | int],
    i: int,
    t: int,
    inv_by_period: list[np.ndarray],
) -> None:
    if t < len(_INV_COLS_BY_WAVE):
        for j, name in enumerate(INV_MEASURES):
            row[name] = float(inv_by_period[t][i, j])
    else:
        for name in INV_MEASURES:
            row[name] = float("nan")


def _fill_income(
    row: dict[str, float | int],
    i: int,
    t: int,
    income_by_period: list[np.ndarray],
) -> None:
    if t < len(_INCOME_COLS_BY_WAVE):
        row[INCOME_MEASURE] = float(income_by_period[t][i])
    else:
        row[INCOME_MEASURE] = float("nan")
