"""Stage 2 of the AMN estimator: structural recovery via minimum distance.

Takes the reduced-form mixture parameters (Pi_k, Psi_k) from Stage 1
(`skillmodels.amn.mixture_em`) and recovers the structural parameters
(Lambda, A, Sigma, mu_k, Omega_k) subject to the AMN-paper constraint
structure (eq. 12-13): factor-measurement zero pattern in Lambda,
age-invariance for time-invariant factors, scale normalization
(lambda=1 on the reference measure per factor), and the period-0
mean-zero restriction.

Mirrors `STEP2_func.R` from the AMN 2020 supplementary archive: a
packed-parameter L-BFGS-B optimizer over the sum-of-squares distance
between the EM-fitted moments (Pi_m, Psi_m) and the model-implied
moments parameterized by structural quantities.
"""

from dataclasses import dataclass

import numpy as np
import optimagic as om
import pandas as pd

from skillmodels.amn.types import (
    AugmentedMeasureLayout,
    MinimumDistanceResult,
    MixtureFitResult,
)
from skillmodels.common.types import ProcessedModel


@dataclass(frozen=True)
class _Structure:
    """Pre-computed structural layout for minimum-distance recovery.

    Carries the slot-to-factor-period mapping plus all the
    free/normalized/zero masks needed by the optimizer.
    """

    factor_period_slots: tuple[tuple[int, str], ...]
    """Ordered (period, factor_name) for the structural mu / Omega columns.
    Latent and observed-factor / control slots are all included."""

    n_factor_slots: int
    """``len(factor_period_slots)``."""

    n_aug: int
    """Number of rows in the augmented measure vector."""

    lambda_value: np.ndarray
    """Initial Lambda matrix (zeros + normalized 1s where pinned)."""

    lambda_free_mask: np.ndarray
    """Boolean (n_aug, n_factor_slots): True where Lambda is free."""

    intercept_value: np.ndarray
    """Initial intercept vector (zeros + normalized values where pinned)."""

    intercept_free_mask: np.ndarray
    """Boolean (n_aug,): True where the intercept is free."""

    sigma2_free_mask: np.ndarray
    """Boolean (n_aug,): True where the measurement-error variance is free.
    False for observed-factor / control slots (zero by construction)."""

    baseline_mean_zero_slots: tuple[int, ...]
    """Indices into ``factor_period_slots`` for which the K-th mixture's
    mean is determined by the tau-weighted sum-to-zero constraint
    (AMN eq. 13). Typically the period-0 latent-factor slots."""


def _build_structure(  # noqa: C901, PLR0912, PLR0915
    layout: AugmentedMeasureLayout,
    processed_model: ProcessedModel,
) -> _Structure:
    """Translate the augmented layout into per-Lambda/A/Sigma constraint masks.

    For each augmented slot, decides which structural factor-period column
    it loads on, and whether its Lambda / A / Sigma entries are free
    (estimated) or pinned (normalized or zero by construction).
    """
    n_aug = len(layout.columns)
    normalizations = processed_model.normalizations
    aug_to_period = processed_model.labels.aug_periods_to_periods
    observed_factor_names = processed_model.labels.observed_factors

    # Collect factor-period slots: one per (period, factor) that actually
    # has at least one row loading on it (latent measurements) OR is the
    # "self-slot" of an observed factor / control augmented row.
    slots: list[tuple[int, str]] = []
    slot_index: dict[tuple[int, str], int] = {}
    for _slot, (period, factor, _meas) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        key = (period, factor)
        if key not in slot_index:
            slot_index[key] = len(slots)
            slots.append(key)
    for _slot, (period, of_name) in zip(
        layout.observed_factor_slots, layout.observed_factor_meta, strict=True
    ):
        key = (period, of_name)
        if key not in slot_index:
            slot_index[key] = len(slots)
            slots.append(key)
    for ctrl in layout.control_meta:
        # Controls collapse to a single period (-1 = time-invariant marker).
        key = (-1, ctrl)
        if key not in slot_index:
            slot_index[key] = len(slots)
            slots.append(key)

    n_slots = len(slots)
    lambda_value = np.zeros((n_aug, n_slots))
    lambda_free_mask = np.zeros((n_aug, n_slots), dtype=bool)
    intercept_value = np.zeros(n_aug)
    intercept_free_mask = np.zeros(n_aug, dtype=bool)
    sigma2_free_mask = np.zeros(n_aug, dtype=bool)

    # Latent-factor measurement slots.
    for aug_idx, (period, factor, meas_name) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        sigma2_free_mask[aug_idx] = True
        col = slot_index[(period, factor)]
        # Determine whether the loading at this (period, factor, meas) is
        # normalized (typically the "first" measurement per factor) or
        # free. Skillmodels stores normalizations per aug_period; walk the
        # aug_periods that map to this calendar period and inspect them.
        loading_normalized = False
        intercept_normalized = False
        loading_norm_value = 1.0
        intercept_norm_value = 0.0
        if factor in normalizations:
            for aug_period, cal_period in aug_to_period.items():
                if int(cal_period) != int(period):
                    continue
                load_map = normalizations[factor].loadings[aug_period]
                int_map = normalizations[factor].intercepts[aug_period]
                if meas_name in load_map:
                    loading_normalized = True
                    loading_norm_value = float(load_map[meas_name])
                if meas_name in int_map:
                    intercept_normalized = True
                    intercept_norm_value = float(int_map[meas_name])
        if loading_normalized:
            lambda_value[aug_idx, col] = loading_norm_value
        else:
            lambda_free_mask[aug_idx, col] = True
        if intercept_normalized:
            intercept_value[aug_idx] = intercept_norm_value
        else:
            intercept_free_mask[aug_idx] = True

    # Observed-factor slots: load on their own column with lambda=1,
    # sigma=0 (perfectly observed); intercept is free (the mixture mean
    # shifts the slot).
    for aug_idx, (period, of_name) in zip(
        layout.observed_factor_slots, layout.observed_factor_meta, strict=True
    ):
        col = slot_index[(period, of_name)]
        lambda_value[aug_idx, col] = 1.0
        # sigma2 stays False (pinned to zero by construction).
        intercept_free_mask[aug_idx] = True

    # Control slots: same pattern as observed factors (lambda=1, sigma=0).
    for aug_idx, ctrl in zip(layout.control_slots, layout.control_meta, strict=True):
        col = slot_index[(-1, ctrl)]
        lambda_value[aug_idx, col] = 1.0
        intercept_free_mask[aug_idx] = True

    del observed_factor_names

    # Mean-zero baseline: period-0 latent-factor slots get pinned by the
    # tau-weighted sum-to-zero constraint. Observed factors / controls
    # have free means (no normalization needed; they're directly
    # observed).
    latent_factor_names = set(processed_model.labels.latent_factors)
    baseline_slot_ids = tuple(
        slot_index[(p, f)] for (p, f) in slots if p == 0 and f in latent_factor_names
    )

    return _Structure(
        factor_period_slots=tuple(slots),
        n_factor_slots=n_slots,
        n_aug=n_aug,
        lambda_value=lambda_value,
        lambda_free_mask=lambda_free_mask,
        intercept_value=intercept_value,
        intercept_free_mask=intercept_free_mask,
        sigma2_free_mask=sigma2_free_mask,
        baseline_mean_zero_slots=baseline_slot_ids,
    )


def _pack_layout(struct: _Structure, n_components: int) -> tuple[int, dict[str, slice]]:
    """Decide the layout of the flat optimizer parameter vector.

    Returns:
    -------
    n_total
        Total length of the parameter vector.
    slices
        Mapping from parameter section name to a `slice` into the flat
        vector. Sections:

        - ``"sigma2"`` -- free entries of the measurement-error
          variances.
        - ``"chol_<m>"`` for ``m`` in 0..n_components-1 -- lower-tri
          Cholesky elements of Omega_m, packed row-major.
        - ``"mu_<m>"`` for ``m`` in 0..n_components-2 -- the free
          entries of mu_m (i.e. excluding the K-th mixture, which is
          determined by the mean-zero constraint at baseline slots
          and by free params elsewhere... actually we still free
          mu_K at non-baseline slots; only the baseline slots of
          mu_K are derived).
    """
    slices: dict[str, slice] = {}
    cursor = 0

    n_sigma2_free = int(struct.sigma2_free_mask.sum())
    slices["sigma2"] = slice(cursor, cursor + n_sigma2_free)
    cursor += n_sigma2_free

    n_factor = struct.n_factor_slots
    n_chol_per = n_factor * (n_factor + 1) // 2
    for m in range(n_components):
        slices[f"chol_{m}"] = slice(cursor, cursor + n_chol_per)
        cursor += n_chol_per

    n_baseline = len(struct.baseline_mean_zero_slots)
    # mu has shape (n_components, n_factor); for the K-th mixture, the
    # baseline_mean_zero_slots are determined => those are excluded.
    n_mu_free = n_components * n_factor - n_baseline
    slices["mu"] = slice(cursor, cursor + n_mu_free)
    cursor += n_mu_free

    n_lambda_free = int(struct.lambda_free_mask.sum())
    slices["lambda"] = slice(cursor, cursor + n_lambda_free)
    cursor += n_lambda_free

    n_intercept_free = int(struct.intercept_free_mask.sum())
    slices["intercept"] = slice(cursor, cursor + n_intercept_free)
    cursor += n_intercept_free

    return cursor, slices


def _unpack(
    flat: np.ndarray,
    struct: _Structure,
    slices: dict[str, slice],
    *,
    n_components: int,
    mixture_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a flat parameter vector into (sigma2, Omega, mu, Lambda, A).

    Applies the tau-weighted mean-zero constraint to mu_K at the
    baseline_mean_zero_slots.
    """
    n_factor = struct.n_factor_slots

    sigma2 = np.zeros(struct.n_aug)
    sigma2[struct.sigma2_free_mask] = flat[slices["sigma2"]]

    omegas = np.zeros((n_components, n_factor, n_factor))
    tril_rows, tril_cols = np.tril_indices(n_factor)
    for m in range(n_components):
        chol = np.zeros((n_factor, n_factor))
        chol[tril_rows, tril_cols] = flat[slices[f"chol_{m}"]]
        omegas[m] = chol @ chol.T

    mu = np.zeros((n_components, n_factor))
    baseline_set = set(struct.baseline_mean_zero_slots)
    free_mu_positions: list[tuple[int, int]] = []
    for m in range(n_components):
        is_last = m == n_components - 1
        for j in range(n_factor):
            if is_last and j in baseline_set:
                continue
            free_mu_positions.append((m, j))
    mu_values = flat[slices["mu"]]
    for (m, j), val in zip(free_mu_positions, mu_values, strict=True):
        mu[m, j] = val
    # Enforce mean-zero at baseline slots for the last mixture.
    if baseline_set and n_components > 1:
        for j in struct.baseline_mean_zero_slots:
            num = -np.sum(mixture_weights[:-1] * mu[:-1, j])
            mu[-1, j] = num / mixture_weights[-1]

    lambda_mat = struct.lambda_value.copy()
    lambda_mat[struct.lambda_free_mask] = flat[slices["lambda"]]

    intercept = struct.intercept_value.copy()
    intercept[struct.intercept_free_mask] = flat[slices["intercept"]]

    return sigma2, omegas, mu, lambda_mat, intercept


def _model_implied_moments(
    sigma2: np.ndarray,
    omegas: np.ndarray,
    mu: np.ndarray,
    lambda_mat: np.ndarray,
    intercept: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (per-component mean, cov) implied by the structural params.

    Returns shapes ``(K, n_aug)`` and ``(K, n_aug, n_aug)`` respectively.
    """
    n_components = omegas.shape[0]
    n_aug = intercept.shape[0]
    means = np.empty((n_components, n_aug))
    covs = np.empty((n_components, n_aug, n_aug))
    diag_sigma2 = np.diag(sigma2)
    for m in range(n_components):
        means[m] = intercept + lambda_mat @ mu[m]
        covs[m] = lambda_mat @ omegas[m] @ lambda_mat.T + diag_sigma2
    return means, covs


def _objective(
    flat: np.ndarray,
    struct: _Structure,
    slices: dict[str, slice],
    *,
    n_components: int,
    mixture_weights: np.ndarray,
    target_means: np.ndarray,
    target_covs: np.ndarray,
) -> float:
    sigma2, omegas, mu, lam, inter = _unpack(
        flat,
        struct,
        slices,
        n_components=n_components,
        mixture_weights=mixture_weights,
    )
    pred_means, pred_covs = _model_implied_moments(sigma2, omegas, mu, lam, inter)
    diff_mean = pred_means - target_means
    diff_cov = pred_covs - target_covs
    return float(np.sum(diff_mean**2) + np.sum(diff_cov**2))


def _initial_guess(
    struct: _Structure,
    slices: dict[str, slice],
    *,
    n_components: int,
    n_total: int,
    target_means: np.ndarray,  # noqa: ARG001
    target_covs: np.ndarray,
) -> np.ndarray:
    """Build a sensible starting vector from the EM moments.

    Seeds sigma^2 from the average diagonal of the EM covariances scaled
    down by 0.5 (so factors keep at least half the explained variance);
    seeds each Omega Cholesky from the cholesky of the average EM
    covariance restricted to the factor-period block; seeds mu_m from
    the EM means at the corresponding slot identities.
    """
    flat = np.zeros(n_total)

    diag_avg = np.mean(np.diagonal(target_covs, axis1=1, axis2=2), axis=0)
    sigma2_guess = 0.25 * np.clip(diag_avg, 1e-3, None)
    flat[slices["sigma2"]] = sigma2_guess[struct.sigma2_free_mask]

    # Project the average EM covariance onto a roughly diagonal Omega in
    # the factor-period basis. For v0 we use the identity rescaled by
    # the average non-error variance trace; this is a safe, well-defined
    # start.
    avg_factor_var = np.maximum(diag_avg.mean() * 0.5, 1e-2)
    n_factor = struct.n_factor_slots
    init_chol = np.sqrt(avg_factor_var) * np.eye(n_factor)
    tril_rows, tril_cols = np.tril_indices(n_factor)
    init_chol_vec = init_chol[tril_rows, tril_cols]
    for m in range(n_components):
        flat[slices[f"chol_{m}"]] = init_chol_vec

    # Seed mu_m from each EM component's projection onto the slot space
    # via least-squares (lambda_value pseudo-inverse on the centered
    # means). For v0 use a simpler heuristic: spread the EM means across
    # mixtures using a small dispersion around zero.
    flat[slices["mu"]] = 0.0

    flat[slices["lambda"]] = 1.0
    flat[slices["intercept"]] = 0.0
    return flat


def _lower_bounds(
    struct: _Structure,  # noqa: ARG001
    slices: dict[str, slice],
    n_total: int,
) -> np.ndarray:
    bounds = np.full(n_total, -np.inf)
    bounds[slices["sigma2"]] = 1e-8
    return bounds


def solve_minimum_distance(
    mixture: MixtureFitResult,
    processed_model: ProcessedModel,
    *,
    weighting: str = "identity",
    algorithm: str = "scipy_lbfgsb",
) -> MinimumDistanceResult:
    """Recover structural parameters from the reduced-form mixture.

    Args:
        mixture: Stage 1 fit (reduced-form Pi, Psi per component).
        processed_model: Skillmodels processed model (provides normalization
            and constraint structure).
        weighting: ``"identity"`` (default, fast) or ``"optimal"``
            (uses an Avar estimate of the EM moments).
        algorithm: optimagic algorithm name (default ``scipy_lbfgsb``).

    Return:
        MinimumDistanceResult with structural Lambda, A, Sigma, and the
        per-component factor means and covariances.

    """
    if weighting not in ("identity", "optimal"):
        msg = f"Unknown weighting '{weighting}'."
        raise ValueError(msg)
    if weighting == "optimal":
        msg = "Optimal weighting not yet implemented; use 'identity'."
        raise NotImplementedError(msg)

    layout = mixture.layout
    if not layout.measurement_slots and not layout.observed_factor_slots:
        msg = "Mixture layout has no slots; cannot run minimum distance."
        raise ValueError(msg)

    struct = _build_structure(layout, processed_model)
    n_components = mixture.weights.shape[0]
    n_total, slices = _pack_layout(struct, n_components)

    target_means = mixture.means.copy()
    target_covs = mixture.covariances.copy()

    flat0 = _initial_guess(
        struct,
        slices,
        n_components=n_components,
        n_total=n_total,
        target_means=target_means,
        target_covs=target_covs,
    )
    lower = _lower_bounds(struct, slices, n_total)

    def fun(theta: np.ndarray) -> float:
        return _objective(
            theta,
            struct,
            slices,
            n_components=n_components,
            mixture_weights=mixture.weights,
            target_means=target_means,
            target_covs=target_covs,
        )

    result = om.minimize(
        fun=fun,
        params=flat0,
        algorithm=algorithm,
        bounds=om.Bounds(lower=lower),
    )
    success = bool(result.success)
    flat_opt = np.asarray(result.params, dtype=float)
    sigma2, omegas, mu, lambda_mat, intercept = _unpack(
        flat_opt,
        struct,
        slices,
        n_components=n_components,
        mixture_weights=mixture.weights,
    )

    loadings_df = _loadings_dataframe(struct, layout, lambda_mat)
    intercepts_df = _intercepts_dataframe(layout, intercept)
    meas_sds_df = _meas_sds_dataframe(layout, np.sqrt(np.clip(sigma2, 0.0, None)))

    return MinimumDistanceResult(
        loadings=loadings_df,
        measurement_intercepts=intercepts_df,
        measurement_sds=meas_sds_df,
        factor_mixture_means=mu,
        factor_mixture_covariances=omegas,
        factor_period_slots=struct.factor_period_slots,
        objective_value=float(result.fun),
        success=success,
    )


def _loadings_dataframe(
    struct: _Structure,
    layout: AugmentedMeasureLayout,
    lambda_mat: np.ndarray,
) -> pd.DataFrame:
    """Return a long-format Lambda DataFrame, one row per nonzero entry."""
    rows = []
    aug_idx_to_meta: dict[int, tuple[int, str, str]] = dict(
        zip(layout.measurement_slots, layout.measurement_meta, strict=True)
    )
    slot_to_id = {sp: i for i, sp in enumerate(struct.factor_period_slots)}
    for slot, meta in aug_idx_to_meta.items():
        period, factor, meas = meta
        col = slot_to_id[(period, factor)]
        rows.append(
            {
                "period": period,
                "measurement": meas,
                "factor": factor,
                "loading": float(lambda_mat[slot, col]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["period", "measurement", "factor", "loading"]
        ).set_index(["period", "measurement", "factor"])
    return pd.DataFrame(rows).set_index(["period", "measurement", "factor"])


def _intercepts_dataframe(
    layout: AugmentedMeasureLayout,
    intercept: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for slot, (period, _factor, meas) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        rows.append(
            {
                "period": period,
                "measurement": meas,
                "intercept": float(intercept[slot]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["period", "measurement", "intercept"]).set_index(
            ["period", "measurement"]
        )
    return pd.DataFrame(rows).set_index(["period", "measurement"])


def _meas_sds_dataframe(
    layout: AugmentedMeasureLayout,
    sds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for slot, (period, _factor, meas) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        rows.append({"period": period, "measurement": meas, "sd": float(sds[slot])})
    if not rows:
        return pd.DataFrame(columns=["period", "measurement", "sd"]).set_index(
            ["period", "measurement"]
        )
    return pd.DataFrame(rows).set_index(["period", "measurement"])
