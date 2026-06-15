"""Cross-estimator structural Protocols.

The three estimators (CHS, AF, AMN) each return a rich, concrete result
dataclass with estimator-specific extras (`period_results`, `stages`,
`optimize_result`, ...). `CommonEstimationResult` and
`CommonEstimationOptions` capture the *minimal* surface every consumer can
rely on regardless of which estimator produced the object, so harmonised
post-estimation code (state extraction, plotting, tabulation) can be written
once. They are `runtime_checkable` Protocols, not base classes: the concrete
dataclasses conform structurally and a single conformance test guards the
contract.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from skillmodels.common.model_spec import ModelSpec


@runtime_checkable
class CommonEstimationResult(Protocol):
    """Minimal result surface shared by every estimator's result object.

    Invariant: **exactly one** of `loglikelihood` / `md_criterion` is
    non-`None`. Likelihood-based estimators (CHS, AF) fill `loglikelihood`
    and leave `md_criterion` `None`; the minimum-distance estimator (AMN)
    does the reverse. This lets a caller branch on *which* objective was
    optimised without importing the concrete result types.
    """

    model_spec: ModelSpec
    """The `ModelSpec` that was estimated."""

    params: pd.DataFrame
    """Estimated parameters with the standard 4-level MultiIndex
    (category, period, name1, name2) and a `"value"` column."""

    success: bool
    """Whether the estimator's optimisation converged."""

    loglikelihood: float | None
    """Maximised log-likelihood (CHS, AF); `None` for AMN."""

    md_criterion: float | None
    """Minimum-distance criterion at the optimum (AMN); `None` for CHS/AF."""


@runtime_checkable
class CommonEstimationOptions(Protocol):
    """Minimal options surface shared by every estimator's options object.

    Every estimator ultimately drives `optimagic`; these two fields are the
    universal optimiser knobs. Estimator-specific tuning (Halton counts, EM
    restarts, start-value strategy, ...) lives on the concrete options
    dataclasses and is intentionally absent here.
    """

    optimizer_algorithm: str
    """`optimagic` algorithm name passed to `minimize`/`maximize`."""

    optimizer_options: Mapping[str, Any]
    """Extra keyword arguments forwarded to the `optimagic` call."""
