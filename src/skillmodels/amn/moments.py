"""Spearman / multi-indicator moment estimators for starting values.

Pure NumPy helpers used to seed optimizer starting values from data
moments instead of static defaults (sigma_inv = 0.5 etc.). They derive
loadings, measurement-error SDs, and latent-factor variances from the
cross-covariance structure of multi-indicator measurements — the
standard Spearman / factor-analysis identification.

Used by both the AF estimator (chain-wide moment seeds in
`af.initial_period` / `af.transition_period`) and the CHS estimator
(via `skillmodels.amn.start_values.get_spearman_start_params`).

This module is called once before optimization (no JAX dependency) and
exposes single-pass, robust estimators with floor clamps for numerical
edge cases.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpearmanResult:
    """Single-factor Spearman moment estimates from cross-covariances."""

    loadings: np.ndarray
    """Recovered loadings, shape ``(n_meas,)``. The anchor entry equals 1.0
    by construction (or the user-provided anchor value)."""

    meas_sds: np.ndarray
    """Recovered measurement-error SDs, shape ``(n_meas,)``."""

    latent_var: float
    """Recovered latent-factor variance Var(F)."""

    valid: bool
    """False when identification fails (anchor uncorrelated with all other
    measurements, or fewer than two measurements available)."""


def spearman_factor_moments(
    measurements: np.ndarray,
    *,
    anchor_idx: int = 0,
    anchor_loading: float = 1.0,
    sd_floor: float = 1e-3,
    var_floor: float = 1e-6,
) -> SpearmanResult:
    """Recover loadings, sigma_meas, Var(F) from multi-indicator covariances.

    For a single latent factor F observed via ``measurements[:, k] = λ_k F +
    ε_k`` (after residualizing out controls), the off-diagonal covariances
    identify the loadings up to scale and the diagonal residual variances
    give sigma_meas². Anchor measurement ``anchor_idx`` is normalized so its
    loading equals ``anchor_loading``.

    Algorithm (pairwise complete cases):

    * ``S = pairwise_cov(measurements)``.
    * Pool ``Var(F)`` via robust median across triples ``S[a,j] S[a,k] /
      S[j,k]`` for ``j ≠ k ≠ a``.
    * ``λ_k = S[a, k] / Var(F)`` for ``k ≠ a`` (then rescaled so anchor
      matches ``anchor_loading``).
    * ``sigma_meas_k² = max(S[k, k] - λ_k² Var(F), sd_floor²)``.

    If the anchor's covariances with all other measurements are below
    numerical noise, rotate to a different anchor and retry. If all
    candidates fail, return ``valid=False``.

    Args:
        measurements: Shape ``(n_obs, n_meas)``. NaN values are handled via
            pairwise-complete cases.
        anchor_idx: Index of the anchor measurement. Loadings are reported
            on a scale where ``loadings[anchor_idx] == anchor_loading``.
        anchor_loading: Pinned anchor loading (typically 1.0 from a
            normalization).
        sd_floor: Minimum returned measurement SD to avoid zero / negative
            estimates from sample noise.
        var_floor: Minimum returned latent variance.

    Return:
        `SpearmanResult` with recovered loadings, sigma_meas, latent_var, and a
        `valid` flag.

    """
    arr = np.asarray(measurements, dtype=float)
    if arr.ndim != 2:
        msg = f"measurements must be 2D; got shape {arr.shape}"
        raise ValueError(msg)
    n_meas = arr.shape[1]
    if n_meas < 2:
        return SpearmanResult(
            loadings=np.full(n_meas, anchor_loading),
            meas_sds=np.full(n_meas, sd_floor),
            latent_var=var_floor,
            valid=False,
        )

    s = _pairwise_cov(arr)

    # Try the requested anchor first; rotate through other candidates if
    # it has no usable cross-covariances.
    anchor_order = [anchor_idx, *(k for k in range(n_meas) if k != anchor_idx)]
    for candidate in anchor_order:
        result = _spearman_with_anchor(
            s,
            anchor=candidate,
            anchor_loading=anchor_loading,
            target_anchor=anchor_idx,
            sd_floor=sd_floor,
            var_floor=var_floor,
        )
        if result is not None:
            return result

    return SpearmanResult(
        loadings=np.full(n_meas, anchor_loading),
        meas_sds=np.full(n_meas, sd_floor),
        latent_var=var_floor,
        valid=False,
    )


def derive_unexplained_sd(
    latent_var: float,
    beta: np.ndarray,
    prev_state_cov: np.ndarray,
    *,
    sd_floor: float = 1e-3,
) -> float:
    """Return the residual SD of a regression with explained variance β'Σβ.

    Given a regression ``F = β'·prev_state + ε`` where ``Var(prev_state) =
    Σ`` and ``Var(F) = latent_var``, the residual variance is ``Var(ε) =
    Var(F) - β'Σβ``. Clamped at ``sd_floor`` to avoid NaN when sample noise
    pushes ``β'Σβ`` above ``Var(F)``.

    Used to seed sigma_shock (production shock SD) and sigma_inv (investment shock
    SD) from the latent factor variance plus the regression coefficients.

    Args:
        latent_var: Marginal variance of the dependent factor.
        beta: Regression coefficients, shape ``(n_state,)``.
        prev_state_cov: Covariance matrix of the regressors, shape
            ``(n_state, n_state)``.
        sd_floor: Minimum returned SD.

    Return:
        ``sqrt(max(latent_var - β'Σβ, sd_floor²))``.

    """
    beta = np.asarray(beta, dtype=float).ravel()
    cov = np.asarray(prev_state_cov, dtype=float)
    explained = float(beta @ cov @ beta)
    residual_var = max(float(latent_var) - explained, sd_floor**2)
    return float(np.sqrt(residual_var))


def seed_beta_from_ols(
    response: np.ndarray,
    regressors: np.ndarray,
) -> np.ndarray:
    """OLS coefficient estimate for seeding inv-equation β.

    Pure-numpy OLS of ``response`` (n_obs,) on ``regressors`` (n_obs,
    n_features). Drops rows with any NaN. Returns zeros when the design
    is rank-deficient.

    Args:
        response: Shape ``(n_obs,)``.
        regressors: Shape ``(n_obs, n_features)``.

    Return:
        β estimate, shape ``(n_features,)``. Zero vector if the design is
        rank-deficient or the sample is too small.

    """
    y = np.asarray(response, dtype=float).ravel()
    x = np.asarray(regressors, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n_features = x.shape[1]
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if mask.sum() <= n_features:
        return np.zeros(n_features)
    try:
        coef, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(n_features)
    if not np.all(np.isfinite(coef)):
        return np.zeros(n_features)
    return coef


def _pairwise_cov(arr: np.ndarray) -> np.ndarray:
    """Compute pairwise-complete sample covariance matrix.

    Each entry ``S[i, j]`` is the sample covariance over rows where both
    columns ``i`` and ``j`` are finite. Diagonal entries are sample
    variances over rows where the column is finite.
    """
    n_meas = arr.shape[1]
    s = np.zeros((n_meas, n_meas))
    finite = np.isfinite(arr)
    for i in range(n_meas):
        for j in range(i, n_meas):
            mask = finite[:, i] & finite[:, j]
            if mask.sum() < 2:
                s[i, j] = s[j, i] = 0.0
                continue
            xi = arr[mask, i]
            xj = arr[mask, j]
            mi = xi.mean()
            mj = xj.mean()
            cov = float(((xi - mi) * (xj - mj)).sum() / (mask.sum() - 1))
            s[i, j] = s[j, i] = cov
    return s


def _spearman_with_anchor(  # noqa: C901, PLR0912
    s: np.ndarray,
    *,
    anchor: int,
    anchor_loading: float,
    target_anchor: int,
    sd_floor: float,
    var_floor: float,
) -> SpearmanResult | None:
    """Spearman estimates with a specified anchor; ``None`` if degenerate."""
    n_meas = s.shape[0]
    diag = np.maximum(np.diag(s), sd_floor**2)
    sds = np.sqrt(diag)
    cov_threshold = 1e-3 * sds[anchor] * sds

    # The anchor must covary meaningfully with at least one other column.
    cross = np.array(
        [
            (k, abs(s[anchor, k]))
            for k in range(n_meas)
            if k != anchor and abs(s[anchor, k]) > cov_threshold[k]
        ]
    )
    if cross.size == 0:
        return None

    # Pool Var(F) via the median of triples S[a,j] S[a,k] / S[j,k] for
    # j, k != a, j != k, with S[j,k] above noise.
    triples = []
    for j in range(n_meas):
        if j == anchor or abs(s[anchor, j]) <= cov_threshold[j]:
            continue
        for k in range(j + 1, n_meas):
            if k == anchor or abs(s[anchor, k]) <= cov_threshold[k]:
                continue
            cross_threshold = 1e-3 * sds[j] * sds[k]
            if abs(s[j, k]) <= cross_threshold:
                continue
            triples.append(s[anchor, j] * s[anchor, k] / s[j, k])

    if not triples:
        # Only one measurement covaries with the anchor — Var(F) is
        # under-identified. Fall back to S[anchor, k] / S[k, k] times
        # diagonal (rough), then clamp.
        partner_idx = int(cross[np.argmax(cross[:, 1]), 0])
        latent_var_raw = abs(s[anchor, partner_idx])
    else:
        latent_var_raw = float(np.median(triples))

    latent_var = max(latent_var_raw, var_floor)

    raw_loadings = np.zeros(n_meas)
    raw_loadings[anchor] = 1.0
    for k in range(n_meas):
        if k == anchor:
            continue
        raw_loadings[k] = s[anchor, k] / latent_var

    # Rescale so the user-supplied target anchor reports ``anchor_loading``.
    # If we rotated to a different anchor candidate, the recovered scale
    # must be re-anchored on ``target_anchor``.
    if target_anchor != anchor:
        if abs(raw_loadings[target_anchor]) <= 1e-12:
            return None
        scale = anchor_loading / raw_loadings[target_anchor]
    else:
        scale = anchor_loading
    loadings = raw_loadings * scale
    # Var(F) absorbs the inverse square of the rescale.
    latent_var = latent_var / (scale**2)
    latent_var = max(latent_var, var_floor)

    meas_var = np.maximum(diag - loadings**2 * latent_var, sd_floor**2)
    meas_sds = np.sqrt(meas_var)

    return SpearmanResult(
        loadings=loadings,
        meas_sds=meas_sds,
        latent_var=latent_var,
        valid=True,
    )
