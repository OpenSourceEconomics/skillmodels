"""AMN (Attanasio-Meghir-Nix 2020) point-estimate estimator.

The AMN method estimates skill-production parameters in two stages:

1. **Spearman moments** identify the measurement system (loadings,
   meas SDs).
2. **OLS on Bartlett-scored factor proxies**, with an
   errors-in-variables (EIV) correction that subtracts the known
   measurement-error covariance from `X'X/n` before inverting,
   recovers transition coefficients.

The result is a final point estimate, not a starting value. AMN is
much cheaper than CHS or AF: closed-form per equation, no
nonlinear optimisation. The trade-off is that the EIV correction
only handles linear regressors cleanly; translog cross-products
(`x * y`) keep the naive OLS coefficient and are therefore biased
toward zero.
"""

from skillmodels.amn.estimate import estimate_amn
from skillmodels.amn.types import AMNEstimationOptions, AMNEstimationResult

__all__ = ["AMNEstimationOptions", "AMNEstimationResult", "estimate_amn"]
