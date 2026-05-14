"""Generic posterior-state range utilities, estimator-agnostic.

`create_state_ranges` reduces a `(obs x period x factor)` DataFrame of
filtered or simulated latent-factor values to a per-factor, per-period
range (min/max or symmetric quantile bounds). The function is purely
DataFrame-level and does not depend on which estimator produced the
input — any caller that can hand it a DataFrame with a "period" (or
"aug_period") column and one column per factor can use it.

Historically this lived under `skillmodels.chs.process_debug_data` but
the implementation never depended on CHS-specific machinery; AF and
AMN consumers (`posterior_states.py` in both subpackages) already
imported it across the subpackage boundary, which motivated the move.
"""

import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import DIAGNOSTICS_CONF


@beartype(conf=DIAGNOSTICS_CONF)
def create_state_ranges(
    filtered_states: pd.DataFrame,
    factors: tuple[str, ...] | list[str],
    quantile_cutoff: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute minimum and maximum state values for each factor by period.

    Args:
        filtered_states: DataFrame with filtered states. Must have a "period"
            column (or "aug_period" — that one wins if both are present).
        factors: List of factor names to compute ranges for.
        quantile_cutoff: If provided, use quantiles instead of min/max. The
            cutoff is applied symmetrically: the minimum is the
            `quantile_cutoff` quantile and the maximum is the
            `1 - quantile_cutoff` quantile. For example,
            `quantile_cutoff=0.01` uses the 1st and 99th percentiles.

    Return:
        Dictionary mapping factor names to DataFrames with "minimum" and
        "maximum" columns, indexed by period.

    """
    ranges: dict[str, pd.DataFrame] = {}
    period_col = "aug_period" if "aug_period" in filtered_states.columns else "period"

    if quantile_cutoff is not None:
        if not 0 < quantile_cutoff < 0.5:
            raise ValueError("quantile_cutoff must be between 0 and 0.5 (exclusive)")
        minima = filtered_states.groupby(period_col).quantile(quantile_cutoff)
        maxima = filtered_states.groupby(period_col).quantile(1 - quantile_cutoff)
    else:
        minima = filtered_states.groupby(period_col).min()
        maxima = filtered_states.groupby(period_col).max()

    for factor in factors:
        df = pd.concat([minima[factor], maxima[factor]], axis=1)
        df.columns = pd.Index(["minimum", "maximum"])
        ranges[factor] = df
    return ranges
