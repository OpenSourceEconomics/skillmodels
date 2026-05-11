"""Frozen dataclass definitions for the AMN estimator."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AMNEstimationOptions:
    """Configuration options for the AMN (Attanasio-Meghir-Nix 2020) estimator."""

    use_bias_correction: bool = True
    """Apply the errors-in-variables correction to OLS coefficients.

    Without correction, OLS on noisy Bartlett-proxy regressors is
    attenuated (biased toward zero) by approximately
    `Var(F) / Var(F_proxy) = Var(F) / (Var(F) + sigma_eta^2)`. With
    correction, the standard EIV adjustment subtracts the known
    measurement-error covariance matrix from `X'X/n` before
    inverting: `beta_corrected = ((X'X/n) - Sigma_eta)^(-1) (X'y/n)`.
    Sigma_eta is diagonal for Bartlett proxies of different factors
    or periods (measurement noises are independent); for translog
    cross-product regressors (`x * y`) the correction is **not**
    applied because the noise structure of a product is non-standard.
    """

    sd_floor: float = 1e-3
    """Floor on returned SDs for numerical stability."""

    fail_below_min_singular_value: float = 1e-9
    """Threshold below which an OLS or bias-corrected design is
    declared rank-deficient; the relevant transition equation falls
    back to NaN coefficients in that case."""


@dataclass(frozen=True)
class AMNEstimationResult:
    """Result of an AMN run.

    The `params` DataFrame matches the standard skillmodels params
    MultiIndex `(category, period, name1, name2)`, so the result can
    be passed straight to `simulate_dataset`, `get_filtered_states`,
    or used as start values for `estimate_af` / `estimate_ml`.
    """

    params: pd.DataFrame
    """Estimated parameter values. Free entries hold AMN point
    estimates; user-fixed entries hold their pinned values."""

    measurement_system: pd.DataFrame
    """Spearman-estimated loadings + meas_sds + intercepts, packed
    into the standard params index. Same shape as what
    `skillmodels.af.measurement_first_stage.estimate_measurement_system`
    returns."""

    factor_proxies: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    """Bartlett-scored factor proxy per `(aug_period, factor)`,
    shape `(n_obs,)`. Used internally and exposed for inspection /
    follow-up regressions."""

    proxy_meas_err_var: dict[tuple[int, str], float] = field(default_factory=dict)
    """Per-proxy measurement-error variance:
    `1 / sum_k (lambda_k^2 / sigma_k^2)`. Drives the EIV bias
    correction."""

    n_obs: int = 0
    """Number of observations used in the OLS regressions."""

    regression_diagnostics: dict[tuple[int, str], dict] = field(default_factory=dict)
    """Per-equation diagnostics: `n_used`, `min_singular_value`,
    `r_squared`, `shock_sd`. Indexed by `(aug_period, dependent_factor)`."""
