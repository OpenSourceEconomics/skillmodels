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

Valid under MAR missingness.

The EM is vectorised with the **masked-covariance** identity so it runs as dense
batched linear algebra on the GPU rather than a Python loop over missing
patterns (prohibitive when nearly every individual has a distinct pattern). For
a row with observed mask `M` (diagonal, 1 = observed), define
`G = M Sigma M + (I - M)`. Then `G` is block-diagonal `[[Sigma_oo, 0], [0, I]]`,
so a single full-dimension Cholesky yields the observed-block log-density,
log-determinant, conditional mean and the masked inverse `M G^-1 M` (whose
non-zero block is `Sigma_oo^-1`) all at once -- uniformly across rows regardless
of which entries are missing. Rows are processed in fixed-size padded chunks so
only `(chunk, d, d)` arrays are ever materialised.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


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


def _chunk_size_for(n_dim: int) -> int:
    """Rows per chunk keeping the `(chunk, d, d)` working set near 0.5 GB (f64)."""
    return int(max(32, min(1024, 6_000_000 // max(n_dim * n_dim, 1))))


def _component_stats(
    x: jax.Array,
    mask: jax.Array,
    mean: jax.Array,
    cov: jax.Array,
    reg_covar: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-row log-density, imputed mean and masked inverse for one component."""
    n_dim = x.shape[1]
    eye = jnp.eye(n_dim)
    cov_reg = cov + reg_covar * eye
    mm = mask[:, :, None] * mask[:, None, :]
    # G = M (cov + reg I) M + (I - M):  block-diag [[cov_oo + reg, 0], [0, I]].
    g = mm * cov_reg[None] + (1.0 - mask)[:, :, None] * eye[None]
    chol = jnp.linalg.cholesky(g)
    diff = mask * (x - mean)
    z = jax.scipy.linalg.cho_solve((chol, True), diff)  # G^-1 (masked diff)
    maha = jnp.sum(diff * z, axis=1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=1)
    n_obs = jnp.sum(mask, axis=1)
    log_density = -0.5 * (n_obs * jnp.log(2.0 * jnp.pi) + logdet + maha)
    x_hat = mean + z @ cov  # conditional mean fills the missing block
    ginv = jax.scipy.linalg.cho_solve(
        (chol, True), jnp.broadcast_to(eye, (x.shape[0], n_dim, n_dim))
    )
    masked_ginv = mm * ginv  # M G^-1 M: observed-block is Sigma_oo^-1
    return log_density, x_hat, masked_ginv


def _chunk_accumulate(
    x: jax.Array,
    mask: jax.Array,
    valid: jax.Array,
    log_weights: jax.Array,
    means: jax.Array,
    covs: jax.Array,
    reg_covar: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return this chunk's contribution to the EM sufficient statistics.

    `valid` is 1 for real rows and 0 for padding; padded rows contribute zero.
    """
    log_density, x_hat, masked_ginv = jax.vmap(
        _component_stats, in_axes=(None, None, 0, 0, None)
    )(x, mask, means, covs, reg_covar)  # (K, B), (K, B, d), (K, B, d, d)

    weighted = log_density.T + log_weights  # (B, K)
    log_norm = jax.scipy.special.logsumexp(weighted, axis=1)  # (B,)
    resp = jnp.exp(weighted - log_norm[:, None]) * valid[:, None]  # (B, K)
    loglik = jnp.sum(log_norm * valid)
    nk = jnp.sum(resp, axis=0)  # (K,)
    sum_x = jnp.einsum("bk,kbd->kd", resp, x_hat)  # (K, d)
    scatter = jnp.einsum("bk,kbd,kbe->kde", resp, x_hat, x_hat)  # (K, d, d)
    corr = jnp.einsum("bk,kbde->kde", resp, masked_ginv)  # (K, d, d)
    return loglik, nk, sum_x, scatter, corr


# Jitted at runtime (not via a typed decorator) so the type checker sees the
# plain tuple-returning signature above; `reg_covar` is a compile-time constant.
_chunk_accumulate_jit = jax.jit(_chunk_accumulate, static_argnames=("reg_covar",))


def _run_em(
    x_filled: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Run masked-covariance EM from a warm start; return params and diagnostics."""
    n_obs, n_dim = x_filled.shape
    chunk = _chunk_size_for(n_dim)
    n_padded = int(np.ceil(n_obs / chunk) * chunk)
    pad = n_padded - n_obs
    xj = jnp.asarray(np.vstack([x_filled, np.zeros((pad, n_dim))]))
    mj = jnp.asarray(np.vstack([mask, np.zeros((pad, n_dim))]))
    valid = jnp.asarray(np.concatenate([np.ones(n_obs), np.zeros(pad)]))
    eye = jnp.eye(n_dim)

    weights_j = jnp.asarray(weights)
    means_j = jnp.asarray(means)
    covs_j = jnp.asarray(covs)
    prev_ll = -np.inf
    loglik = -np.inf
    converged = False
    n_iter = 0
    n_components = means.shape[0]
    for _ in range(max_iter):
        n_iter += 1
        log_weights = jnp.log(weights_j)
        loglik_t = jnp.array(0.0)
        nk = jnp.zeros(n_components)
        sum_x = jnp.zeros((n_components, n_dim))
        scatter = jnp.zeros((n_components, n_dim, n_dim))
        corr = jnp.zeros((n_components, n_dim, n_dim))
        for start in range(0, n_padded, chunk):
            sl = slice(start, start + chunk)
            ll_c, nk_c, sx_c, sc_c, cr_c = _chunk_accumulate_jit(
                xj[sl],
                mj[sl],
                valid[sl],
                log_weights,
                means_j,
                covs_j,
                reg_covar=reg_covar,
            )
            loglik_t = loglik_t + ll_c
            nk = nk + nk_c
            sum_x = sum_x + sx_c
            scatter = scatter + sc_c
            corr = corr + cr_c
        loglik = float(loglik_t)

        nk_safe = nk + 1e-12
        weights_j = nk_safe / n_obs
        means_j = sum_x / nk_safe[:, None]
        # cov_k = [scatter - Nk mu mu' + Nk cov - cov T cov] / Nk + reg I
        outer = means_j[:, :, None] * means_j[:, None, :]
        cov_t_cov = jnp.einsum("kde,kef,kfg->kdg", covs_j, corr, covs_j)
        covs_j = (
            scatter
            - nk_safe[:, None, None] * outer
            + nk_safe[:, None, None] * covs_j
            - cov_t_cov
        ) / nk_safe[:, None, None] + reg_covar * eye[None]

        if abs(loglik - prev_ll) <= tol * (1.0 + abs(loglik)):
            converged = True
            break
        prev_ll = loglik

    return (
        np.asarray(weights_j),
        np.asarray(means_j),
        np.asarray(covs_j),
        loglik,
        n_iter,
        converged,
    )


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

    x_filled = np.where(obs, x, 0.0)
    mask = obs.astype(float)
    rng = np.random.default_rng(seed)
    best: MissingDataMixtureFit | None = None
    for _ in range(n_init):
        init_seed = int(rng.integers(0, 2**31 - 1))
        weights, means, covs = _initialise(x, obs, n_components, reg_covar, init_seed)
        weights, means, covs, loglik, n_iter, converged = _run_em(
            x_filled,
            mask,
            weights,
            means,
            covs,
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
