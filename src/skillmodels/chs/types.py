"""Frozen result dataclass for the CHS Kalman-MLE estimator."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from skillmodels.common.model_spec import ModelSpec


@dataclass(frozen=True)
class CHSEstimationResult:
    """Complete result from CHS Kalman-MLE estimation.

    Conforms to `skillmodels.common.estimation.CommonEstimationResult`:
    `loglikelihood` is always populated (CHS maximises a likelihood) and
    `md_criterion` is always `None`.
    """

    model_spec: ModelSpec
    """The ModelSpec used for estimation."""

    params: pd.DataFrame
    """Estimated parameters with the standard 4-level MultiIndex
    (category, period, name1, name2) and a `"value"` column."""

    success: bool
    """Whether the optimagic maximisation converged."""

    loglikelihood: float
    """Maximised log-likelihood at the optimum."""

    optimize_result: Any
    """Raw optimagic `OptimizeResult` for full diagnostics."""

    md_criterion: float | None = None
    """Always `None` for CHS; present to satisfy the common result Protocol."""
