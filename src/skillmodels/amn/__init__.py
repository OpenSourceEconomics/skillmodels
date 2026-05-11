"""AMN-flavoured moment estimators, used as start values across estimators.

The Spearman cross-covariance (`spearman_factor_moments`) and the
Bartlett-score OLS (`seed_beta_from_ols`) -- the building blocks of
Attanasio-Meghir-Nix (2020) -- are not exposed as a final estimator
(`estimate_amn` was removed because the Bartlett-OLS step is biased on
translog cross-products). They live on as the **start-value generator**
that both CHS and AF consume: `get_moment_based_start_params` seeds
every free parameter from data moments before the full MLE runs.

Public API:

* `spearman_factor_moments`, `derive_unexplained_sd`,
  `seed_beta_from_ols`, `SpearmanResult` -- the underlying estimators.
* `get_moment_based_start_params` -- fills a CHS params template from
  data moments.
* `pool_equality_groups` -- pools moment-init seeds across equality
  groups (e.g. time-invariant loadings).
"""

from skillmodels.amn.moments import (
    SpearmanResult,
    derive_unexplained_sd,
    seed_beta_from_ols,
    spearman_factor_moments,
)
from skillmodels.amn.start_values import (
    get_moment_based_start_params,
    pool_equality_groups,
)

__all__ = [
    "SpearmanResult",
    "derive_unexplained_sd",
    "get_moment_based_start_params",
    "pool_equality_groups",
    "seed_beta_from_ols",
    "spearman_factor_moments",
]
