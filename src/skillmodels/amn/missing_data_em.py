"""Gaussian-mixture EM that marginalises over missing entries (AMN Stage 1).

`sklearn`'s `GaussianMixture` is complete-case only: it drops every row with a
missing entry. On an unbalanced panel (e.g. age-binned waves with attrition) the
augmented measure vector can have *zero* rows observed in every column, so the
complete-case fit is infeasible. This module fits the same mixture by the
standard missing-data EM (Ghahramani & Jordan 1994; Hunt & Jorgensen 2003):

- the E-step scores each observation on its *observed* sub-vector -- the Gaussian
  marginal obtained by dropping the missing dimensions;
- the M-step fills each missing entry with its per-component conditional
  expectation `E[x_m | x_o]` and adds the conditional-covariance correction
  `Cov[x_m | x_o]` to the missing-missing block of the scatter.

Valid under MAR missingness. Rows are grouped by missing pattern so the
observed-block factorisation is shared within a pattern.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp


@dataclass(frozen=True)
class MissingDataMixtureFit:
    """Fitted mixture parameters from the missing-data EM."""

    weights: np.ndarray
    """Mixture weights, shape `(n_components,)`."""

    means: np.ndarray
    """Per-component means, shape `(n_components, n_dim)`."""

    covariances: np.ndarray
    """Per-component covariances, shape `(n_components, n_dim, n_dim)`."""

    loglikelihood: float
    """Observed-data log-likelihood at the returned parameters."""

    n_iter: int
    """EM iterations run for the best restart."""

    converged: bool
    """Whether the best restart hit the tolerance before `max_iter`."""


def _group_by_pattern(obs: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """Map each observed-column tuple to the row indices sharing that pattern."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for row, mask in enumerate(obs):
        key = tuple(int(j) for j in np.flatnonzero(mask))
        groups.setdefault(key, []).append(row)
    return {key: np.array(rows) for key, rows in groups.items()}


def _log_marginal_density(
    x_obs: np.ndarray, mean_obs: np.ndarray, cov_obs: np.ndarray
) -> np.ndarray:
    """Log N(x_obs | mean_obs, cov_obs) for a batch; 0 when there is no observed dim."""
    n_rows, n_obs = x_obs.shape
    if n_obs == 0:
        return np.zeros(n_rows)
    chol = np.linalg.cholesky(cov_obs)
    diff = (x_obs - mean_obs).T
    sol = solve_triangular(chol, diff, lower=True)
    maha = np.sum(sol**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (n_obs * np.log(2.0 * np.pi) + logdet + maha)


def _conditional_moments(
    x_obs: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    observed: list[int],
    missing: list[int],
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(E[x_m | x_o], Cov[x_m | x_o])` for one component and pattern."""
    cov_mm = cov[np.ix_(missing, missing)]
    if not observed:
        cond_mean = np.broadcast_to(mean[missing], (x_obs.shape[0], len(missing)))
        return np.array(cond_mean), cov_mm
    cov_oo = cov[np.ix_(observed, observed)] + reg_covar * np.eye(len(observed))
    cov_mo = cov[np.ix_(missing, observed)]
    beta = cho_solve(cho_factor(cov_oo, lower=True), cov_mo.T).T  # (|m|, |o|)
    diff = x_obs - mean[observed]
    cond_mean = mean[missing] + diff @ beta.T
    cond_cov = cov_mm - beta @ cov[np.ix_(observed, missing)]
    return cond_mean, cond_cov


def _initialise(
    x: np.ndarray, obs: np.ndarray, n_components: int, reg_covar: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warm-start from a mixture fitted on the mean-imputed data."""
    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    col_mean = np.where(
        obs.any(axis=0), np.nanmean(np.where(obs, x, np.nan), axis=0), 0.0
    )
    imputed = np.where(obs, x, col_mean)
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=1,
        reg_covar=reg_covar,
        random_state=seed,
    ).fit(imputed)
    return gm.weights_.copy(), gm.means_.copy(), gm.covariances_.copy()


def _e_step(
    x: np.ndarray,
    patterns: dict[tuple[int, ...], np.ndarray],
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    *,
    n_components: int,
    reg_covar: float,
) -> tuple[np.ndarray, float]:
    """Return responsibilities and the observed-data log-likelihood."""
    log_prob = np.empty((x.shape[0], n_components))
    for observed, rows in patterns.items():
        obs_cols = list(observed)
        x_obs = x[rows][:, obs_cols]
        for k in range(n_components):
            cov_oo = covs[k][np.ix_(obs_cols, obs_cols)] + reg_covar * np.eye(
                len(obs_cols)
            )
            log_prob[rows, k] = _log_marginal_density(x_obs, means[k, obs_cols], cov_oo)
    weighted = log_prob + np.log(weights)
    log_norm = logsumexp(weighted, axis=1, keepdims=True)
    resp = np.exp(weighted - log_norm)
    return resp, float(log_norm.sum())


def _m_step(
    x: np.ndarray,
    patterns: dict[tuple[int, ...], np.ndarray],
    resp: np.ndarray,
    means_old: np.ndarray,
    covs_old: np.ndarray,
    *,
    n_components: int,
    n_dim: int,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update weights, means and covariances by the imputed-moment M-step."""
    nk = resp.sum(axis=0) + 1e-12
    weights = nk / x.shape[0]
    sum_x = np.zeros((n_components, n_dim))
    imputed: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
    cond_cov: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
    for observed, rows in patterns.items():
        obs_cols = list(observed)
        mis_cols = [j for j in range(n_dim) if j not in observed]
        x_obs = x[rows][:, obs_cols]
        for k in range(n_components):
            x_hat = np.empty((len(rows), n_dim))
            if obs_cols:
                x_hat[:, obs_cols] = x_obs
            correction = np.zeros((n_dim, n_dim))
            if mis_cols:
                cond_mean, block = _conditional_moments(
                    x_obs, means_old[k], covs_old[k], obs_cols, mis_cols, reg_covar
                )
                x_hat[:, mis_cols] = cond_mean
                correction[np.ix_(mis_cols, mis_cols)] = block
            imputed[observed, k] = x_hat
            cond_cov[observed, k] = correction
            sum_x[k] += resp[rows, k] @ x_hat
    means = sum_x / nk[:, None]

    covs = np.zeros((n_components, n_dim, n_dim))
    for observed, rows in patterns.items():
        for k in range(n_components):
            diff = imputed[observed, k] - means[k]
            r = resp[rows, k]
            covs[k] += (diff * r[:, None]).T @ diff
            covs[k] += r.sum() * cond_cov[observed, k]
    covs = covs / nk[:, None, None] + reg_covar * np.eye(n_dim)
    return weights, means, covs


def _em_iterations(
    x: np.ndarray,
    patterns: dict[tuple[int, ...], np.ndarray],
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    *,
    n_components: int,
    n_dim: int,
    max_iter: int,
    tol: float,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Run EM from a warm start; return fitted params and diagnostics."""
    prev_ll = -np.inf
    loglik = -np.inf
    converged = False
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        resp, loglik = _e_step(
            x,
            patterns,
            weights,
            means,
            covs,
            n_components=n_components,
            reg_covar=reg_covar,
        )
        weights, means, covs = _m_step(
            x,
            patterns,
            resp,
            means,
            covs,
            n_components=n_components,
            n_dim=n_dim,
            reg_covar=reg_covar,
        )
        if abs(loglik - prev_ll) <= tol * (1.0 + abs(loglik)):
            converged = True
            break
        prev_ll = loglik

    return weights, means, covs, loglik, n_iter, converged


def fit_gaussian_mixture_missing(
    augmented: np.ndarray,
    *,
    n_components: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    n_init: int = 5,
    reg_covar: float = 1e-6,
    seed: int = 0,
) -> MissingDataMixtureFit:
    """Fit a Gaussian mixture by missing-data EM, keeping the best of `n_init`.

    Args:
        augmented: `(n_obs, n_dim)` data with `NaN` for missing entries.
        n_components: Number of mixture components.
        max_iter: Maximum EM iterations per restart.
        tol: Relative log-likelihood tolerance for convergence.
        n_init: Number of warm-started restarts; the best fit is kept.
        reg_covar: Diagonal ridge for numerical stability.
        seed: RNG seed for the restarts.

    Return:
        `MissingDataMixtureFit` with the highest-likelihood restart.

    """
    x = np.asarray(augmented, dtype=float)
    if x.ndim != 2:
        msg = "augmented must be a 2D array."
        raise ValueError(msg)
    obs = ~np.isnan(x)
    if not obs.any():
        msg = "augmented has no observed entries; cannot fit mixture."
        raise ValueError(msg)

    patterns = _group_by_pattern(obs)
    n_dim = x.shape[1]
    rng = np.random.default_rng(seed)
    best: MissingDataMixtureFit | None = None
    for _ in range(n_init):
        init_seed = int(rng.integers(0, 2**31 - 1))
        weights, means, covs = _initialise(x, obs, n_components, reg_covar, init_seed)
        weights, means, covs, loglik, n_iter, converged = _em_iterations(
            x,
            patterns,
            weights,
            means,
            covs,
            n_components=n_components,
            n_dim=n_dim,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
        )
        if best is None or loglik > best.loglikelihood:
            best = MissingDataMixtureFit(
                weights=weights,
                means=means,
                covariances=covs,
                loglikelihood=loglik,
                n_iter=n_iter,
                converged=converged,
            )
    if best is None:
        msg = "n_init must be at least 1."
        raise ValueError(msg)
    return best
