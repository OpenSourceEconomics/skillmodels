"""Backward-compat re-export of `skillmodels.moment_init`.

The Spearman / OLS moment helpers moved to the top-level
`skillmodels.moment_init` so the CHS estimator can share them. This
shim keeps existing `from skillmodels.af.moment_init import ...`
imports working.
"""

from skillmodels.moment_init import (
    SpearmanResult,
    derive_unexplained_sd,
    seed_beta_from_ols,
    spearman_factor_moments,
)

__all__ = [
    "SpearmanResult",
    "derive_unexplained_sd",
    "seed_beta_from_ols",
    "spearman_factor_moments",
]
