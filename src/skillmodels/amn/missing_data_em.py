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

import warnings
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

    cross_covariance_identified: bool
    """Whether the column co-observation graph is connected. When `False`, some
    blocks of dimensions are never observed together, so their cross-covariances
    are not pinned by the data (only the means and within-block covariances are);
    the EM still converges, but the returned cross-block covariances are
    arbitrary and should not be trusted."""


def _chunk_size_for(n_dim: int) -> int:
    """Rows per chunk keeping the `(chunk, d, d)` working set near 0.5 GB (f64)."""
    return int(max(32, min(1024, 6_000_000 // max(n_dim * n_dim, 1))))


def _co_observation_connected(obs: np.ndarray) -> bool:
    """Whether the column co-observation graph is connected.

    Columns `i` and `j` are linked when at least one row observes both. If the
    graph splits into separate components, no observation ties those blocks
    together, so the missing-data EM cannot identify the cross-block covariances
    (only the within-block blocks and all means are identified under MAR).
    """
    n_dim = obs.shape[1]
    if n_dim <= 1:
        return True
    co_observed = (obs.T.astype(np.int64) @ obs.astype(np.int64)) > 0
    seen = np.zeros(n_dim, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        neighbours = np.nonzero(co_observed[node] & ~seen)[0]
        for nb in neighbours:
            seen[nb] = True
            stack.append(int(nb))
    return bool(seen.all())


def _component_stats(
    x: jax.Array,
    mask: jax.Array,
    mean: jax.Array,
    cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-row log-density, imputed mean and masked inverse for one component.

    `cov` already carries the ridge added by the M-step, so it is used as-is.
    Observed entries are treated as exact (no extra measurement noise), which is
    what makes the observed-block density and the conditional moments the *exact*
    EM identities rather than a noisy approximation.
    """
    n_dim = x.shape[1]
    eye = jnp.eye(n_dim)
    mm = mask[:, :, None] * mask[:, None, :]
    # G = M cov M + (I - M):  block-diagonal [[cov_oo, 0], [0, I]].
    g = mm * cov[None] + (1.0 - mask)[:, :, None] * eye[None]
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return this chunk's contribution to the EM sufficient statistics.

    `valid` is 1 for real rows and 0 for padding; padded rows contribute zero.
    """
    log_density, x_hat, masked_ginv = jax.vmap(
        _component_stats, in_axes=(None, None, 0, 0)
    )(x, mask, means, covs)  # (K, B), (K, B, d), (K, B, d, d)

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
# plain tuple-returning signature above.
_chunk_accumulate_jit = jax.jit(_chunk_accumulate)


def _accumulate_sufficient_stats(
    xj: jax.Array,
    mj: jax.Array,
    valid: jax.Array,
    log_weights: jax.Array,
    means: jax.Array,
    covs: jax.Array,
    *,
    n_padded: int,
    chunk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sum the chunked EM sufficient statistics over the whole padded sample.

    Returns `(loglik, nk, sum_x, scatter, corr)` evaluated at the supplied
    `(log_weights, means, covs)`. Padding rows contribute zero via `valid`.
    """
    n_components, n_dim = means.shape
    loglik = jnp.array(0.0)
    nk = jnp.zeros(n_components)
    sum_x = jnp.zeros((n_components, n_dim))
    scatter = jnp.zeros((n_components, n_dim, n_dim))
    corr = jnp.zeros((n_components, n_dim, n_dim))
    for start in range(0, n_padded, chunk):
        sl = slice(start, start + chunk)
        ll_c, nk_c, sx_c, sc_c, cr_c = _chunk_accumulate_jit(
            xj[sl], mj[sl], valid[sl], log_weights, means, covs
        )
        loglik = loglik + ll_c
        nk = nk + nk_c
        sum_x = sum_x + sx_c
        scatter = scatter + sc_c
        corr = corr + cr_c
    return loglik, nk, sum_x, scatter, corr


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
    for _ in range(max_iter):
        n_iter += 1
        loglik_t, nk, sum_x, scatter, corr = _accumulate_sufficient_stats(
            xj,
            mj,
            valid,
            jnp.log(weights_j),
            means_j,
            covs_j,
            n_padded=n_padded,
            chunk=chunk,
        )
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

    # Score the *returned* (post-M-step) parameters, so the reported loglik and
    # the cross-restart ranking match what is returned rather than the pre-M-step
    # parameters scored inside the final iteration.
    final_loglik = _accumulate_sufficient_stats(
        xj,
        mj,
        valid,
        jnp.log(weights_j),
        means_j,
        covs_j,
        n_padded=n_padded,
        chunk=chunk,
    )[0]
    loglik = float(final_loglik)

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

    # Per-column observed mean, robust to all-missing columns (mean -> 0) without
    # tripping numpy's "Mean of empty slice" warning from nanmean.
    counts = obs.sum(axis=0)
    sums = np.where(obs, x, 0.0).sum(axis=0)
    col_mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
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
    # Identification diagnostics. A column observed in no row, or a co-observation
    # graph that splits into blocks, leaves some means/covariances unidentified.
    # These are *not* fatal: the EM still fits the identified part, and the
    # unidentified entries fall back to a neutral (ridge) seed -- which matters
    # because a never-observed column can be a transient artefact of subsampling
    # rows for a seed rather than a genuinely absent measurement. We flag it so a
    # caller never mistakes ordinary convergence for a fully identified fit.
    never_observed = np.nonzero(~obs.any(axis=0))[0]
    cross_covariance_identified = _co_observation_connected(obs)
    if never_observed.size:
        warnings.warn(
            f"Missing-data mixture EM: columns {never_observed.tolist()} are "
            "never observed in any row; their means and (co)variances are "
            "unidentified and are seeded only at the ridge default. Drop these "
            "measurements if this is not a transient row-subsampling artefact.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif not cross_covariance_identified:
        warnings.warn(
            "Missing-data mixture EM: the column co-observation graph is "
            "disconnected -- no individual is observed on measurements from "
            "different blocks (e.g. periods that no one spans). Means and "
            "within-block covariances are identified, but the cross-block "
            "covariances are not pinned by the data and should not be trusted.",
            RuntimeWarning,
            stacklevel=2,
        )

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
                cross_covariance_identified=cross_covariance_identified,
            )
    if best is None:
        msg = "n_init must be at least 1."
        raise ValueError(msg)
    return best
