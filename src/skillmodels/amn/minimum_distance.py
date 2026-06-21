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

# The JAX objective below uses `.at[idx].set(...)` functional array updates, which
# ruff's pandas-vet rule misreads as pandas `.at` scalar access. This module has
# no pandas `.at` usage, so the rule is disabled file-wide.
# ruff: noqa: PD008

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd

from skillmodels.amn.types import (
    AugmentedMeasureLayout,
    MinimumDistanceResult,
    MixtureFitResult,
)
from skillmodels.common.types import ProcessedModel

_CES_TRANSITION_NAMES = frozenset(
    {"log_ces", "log_ces_with_constant", "log_ces_general"}
)


def _validate_ces_stage2_anchors(
    processed_model: ProcessedModel,
    layout: AugmentedMeasureLayout,
    *,
    allow_overnormalization: bool,
) -> None:
    """Check the per-period CES anchors of AMN's Stage-2 transformed factors.

    This is NOT a primitive Freyberger-minimal normalization check (audit F1).
    For the restricted CES with psi=1, Freyberger requires only ONE primitive
    scale anchor (e.g. lambda_theta,0,1=1); the later skill and investment
    loadings are then identified through the CES restrictions, so pinning one
    loading per factor-period over-restricts the primitive model. AMN instead
    works in *transformed* (tilde) factor coordinates in Stages 1-2, where one
    anchor per factor-period is the correct scale normalization of those
    internal coordinates. This guard enforces exactly that internal anchoring;
    it does not certify the returned loadings as primitive estimates.

    Restricted `log_ces` / `log_ces_with_constant` are rejected upstream
    (`estimate_amn` standalone guard) because the primitive scale-recovery step
    is not implemented, so in practice this runs for `log_ces_general` (which
    can express the transformed CES) and for the CHS/AF seeding path. One anchor
    per factor-period pins the transformed scale; more than one over-normalizes
    those internal coordinates (raise unless `allow_overnormalization`); zero
    leaves the transformed scale unidentified.

    Known gap (P4): the check examines only factors whose own transition is CES,
    so a normalized investment loading is not flagged when investment has a
    linear transition.
    """
    labels = processed_model.labels
    transition_info = processed_model.transition_info
    if transition_info is None:
        return
    func_names = transition_info.function_names
    normalizations = processed_model.normalizations
    aug_to_period = labels.aug_periods_to_periods
    for factor in labels.latent_factors:
        if func_names.get(factor) not in _CES_TRANSITION_NAMES:
            continue
        norm = normalizations.get(factor)
        for period in labels.periods:
            has_meas = any(
                p == int(period) and f == factor
                for (p, f, _m) in layout.measurement_meta
            )
            if not has_meas:
                continue
            aug_periods = [a for a, p in aug_to_period.items() if int(p) == int(period)]
            n_norm = 0
            if norm is not None:
                for a in aug_periods:
                    n_norm += len(norm.loadings[a])
            if n_norm == 0:
                msg = (
                    f"CES factor '{factor}' has no loading normalization in "
                    f"period {period}; the Stage-2 transformed-factor scale is "
                    "unidentified. Pin exactly one measurement loading per "
                    "period."
                )
                raise ValueError(msg)
            if n_norm > 1 and not allow_overnormalization:
                msg = (
                    f"CES factor '{factor}' has {n_norm} loading normalizations "
                    f"in period {period}, but one anchor per factor-period "
                    "already pins the Stage-2 transformed-factor scale. Extra "
                    "anchors over-normalize those internal coordinates. Pin "
                    "exactly one loading per period, or pass "
                    "allow_ces_overnormalization=True for a deliberate "
                    "fixed-loadings analysis."
                )
                raise ValueError(msg)


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
    # sigma=0 (perfectly observed); intercept pinned to zero so the
    # factor mean carries the observed level.
    for aug_idx, (period, of_name) in zip(
        layout.observed_factor_slots, layout.observed_factor_meta, strict=True
    ):
        col = slot_index[(period, of_name)]
        lambda_value[aug_idx, col] = 1.0
        # sigma2 stays False (pinned to zero by construction).
        # Intercept pinned to zero: with lambda=1 and sigma2=0 the slot's
        # reduced-form level Pi_k = intercept + mu_k is otherwise split
        # arbitrarily between the (free) intercept and the (free) factor
        # mean mu_k. Pin intercept=0 so the factor mean carries the full
        # observed level -- the level Stage 3 draws and the posterior uses.
        intercept_value[aug_idx] = 0.0

    # Control slots: same pattern as observed factors (lambda=1, sigma=0).
    for aug_idx, ctrl in zip(layout.control_slots, layout.control_meta, strict=True):
        col = slot_index[(-1, ctrl)]
        lambda_value[aug_idx, col] = 1.0
        # Intercept pinned to zero (see observed-factor block): the
        # control's factor-mean slot carries its full observed level.
        intercept_value[aug_idx] = 0.0

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
    # Identity-metric minimum-distance criterion (AMN 2020 step 2 default).
    # NOTE: this is an UNWEIGHTED sum of squares over per-component means and
    # the FULL covariance matrices. Off-diagonal covariance moments are thus
    # implicitly weighted twice (the matrices are symmetric) and every mixture
    # component is weighted equally irrespective of its weight tau_k. This is
    # consistent under correct specification and full identification (the
    # criterion is minimised at zero moment discrepancy) but is not the
    # efficient / optimal-weighted or tau-weighted MD criterion and selects a
    # different pseudo-true value under misspecification. A vech-packed and/or
    # tau-/Avar-weighted variant is intentionally NOT applied here to preserve
    # the existing estimand; it should be added as a separate opt-in weighting.
    return float(np.sum(diff_mean**2) + np.sum(diff_cov**2))


def _make_objective_and_grad(
    struct: _Structure,
    slices: dict[str, slice],
    *,
    n_components: int,
    mixture_weights: np.ndarray,
    target_means: np.ndarray,
    target_covs: np.ndarray,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build jitted (value, gradient) of the identity-metric MD criterion.

    The criterion is identical to `_objective`, but written in JAX so the
    optimizer receives an *exact* analytical gradient (one backward pass) rather
    than a finite-difference gradient that costs `n_params` objective
    evaluations. With a large factor-period block the parameter vector runs to
    thousands of entries, so the finite-difference cost per L-BFGS-B step is the
    difference between seconds and hours.

    Every scatter target (free-entry indices, the lower-triangular Cholesky
    pattern, the baseline mean-zero constraint) is a static function of the model
    structure, so the only traced input is the flat parameter vector.
    """
    n_factor = struct.n_factor_slots
    n_aug = struct.n_aug
    # Flat (single-axis) scatter indices throughout: a 2-axis `.at[rows, cols]`
    # reads to ruff as a pandas `.at` scalar access (PD008); linear indices keep
    # the functional update unambiguous and equally differentiable.
    sigma2_idx = jnp.asarray(np.nonzero(struct.sigma2_free_mask)[0])
    chol_slices = [slices[f"chol_{m}"] for m in range(n_components)]
    tr_rows, tr_cols = np.tril_indices(n_factor)
    tril_lin = jnp.asarray(tr_rows * n_factor + tr_cols)

    baseline_set = set(struct.baseline_mean_zero_slots)
    free_mu_positions = [
        m * n_factor + j
        for m in range(n_components)
        for j in range(n_factor)
        if not (m == n_components - 1 and j in baseline_set)
    ]
    mu_lin = jnp.asarray(free_mu_positions)
    baseline_cols = jnp.asarray(list(struct.baseline_mean_zero_slots), dtype=int)
    has_baseline = bool(struct.baseline_mean_zero_slots) and n_components > 1

    lam_r, lam_c = np.nonzero(struct.lambda_free_mask)
    lam_lin = jnp.asarray(lam_r * n_factor + lam_c)
    inter_idx = jnp.asarray(np.nonzero(struct.intercept_free_mask)[0])

    lambda_flat = jnp.asarray(struct.lambda_value).reshape(-1)
    intercept_value = jnp.asarray(struct.intercept_value)
    weights_j = jnp.asarray(mixture_weights)
    tmeans = jnp.asarray(target_means)
    tcovs = jnp.asarray(target_covs)

    s_sig, s_mu = slices["sigma2"], slices["mu"]
    s_lam, s_int = slices["lambda"], slices["intercept"]

    def _value(flat: jax.Array) -> jax.Array:
        sigma2 = jnp.zeros(n_aug).at[sigma2_idx].set(flat[s_sig.start : s_sig.stop])

        omega_list = []
        for sl in chol_slices:
            chol = jnp.zeros(n_factor * n_factor).at[tril_lin].set(flat[sl])
            chol = chol.reshape(n_factor, n_factor)
            omega_list.append(chol @ chol.T)
        omegas = jnp.stack(omega_list)

        mu_flat = jnp.zeros(n_components * n_factor)
        mu_flat = mu_flat.at[mu_lin].set(flat[s_mu.start : s_mu.stop])
        if has_baseline:
            mu_grid = mu_flat.reshape(n_components, n_factor)
            num = -(weights_j[:-1] @ mu_grid[:-1][:, baseline_cols])
            base_lin = (n_components - 1) * n_factor + baseline_cols
            mu_flat = mu_flat.at[base_lin].set(num / weights_j[-1])
        mu = mu_flat.reshape(n_components, n_factor)

        lam = lambda_flat.at[lam_lin].set(flat[s_lam.start : s_lam.stop])
        lam = lam.reshape(n_aug, n_factor)
        inter = intercept_value.at[inter_idx].set(flat[s_int.start : s_int.stop])

        means = inter[None, :] + mu @ lam.T  # (K, n_aug)
        covs = jnp.einsum("af,kfg,bg->kab", lam, omegas, lam) + jnp.diag(sigma2)[None]
        return jnp.sum((means - tmeans) ** 2) + jnp.sum((covs - tcovs) ** 2)

    value_jit = jax.jit(_value)
    grad_jit = jax.jit(jax.grad(_value))

    def value_fn(flat: np.ndarray) -> float:
        return float(value_jit(jnp.asarray(flat, dtype=float)))

    def grad_fn(flat: np.ndarray) -> np.ndarray:
        return np.asarray(grad_jit(jnp.asarray(flat, dtype=float)), dtype=float)

    return value_fn, grad_fn


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


# Parameter categories whose pins solve_minimum_distance can hold: the
# structural measurement system fit in Stage 2. Each maps to a free entry of
# the packed optimiser vector (loadings -> Lambda, controls -> intercepts,
# meas_sds -> sqrt of the measurement variances).
_STAGE2_FIXED_CATEGORIES = frozenset({"loadings", "controls", "meas_sds"})


def _free_flat_index(
    rank: int,
    slice_start: int,
    label: object,
    what: str,
    *,
    is_free: bool,
) -> int:
    """Return the packed-vector index of a free structural entry, or raise.

    ``is_free`` / ``rank`` come from the entry's free-mask and its cumulative
    rank among free entries; ``slice_start`` is the start of the owning vector
    section. A non-free entry is normalized or pinned by the model spec and so
    cannot be pinned again.
    """
    if not is_free:
        msg = (
            f"Cannot pin {what} {label!r}: it is normalized or fixed by the model "
            "spec, so it is not a free Stage-2 parameter."
        )
        raise ValueError(msg)
    return slice_start + int(rank)


def _stage2_fixed_indices(
    fixed_params: pd.DataFrame | None,
    struct: _Structure,
    layout: AugmentedMeasureLayout,
    slices: dict[str, slice],
) -> list[tuple[int, float]]:
    """Map Stage-2 ``fixed_params`` rows to ``(flat_index, value)`` pairs.

    Handles the structural measurement categories: ``loadings`` (a free entry
    of the Lambda matrix), ``controls`` (a free measurement intercept), and
    ``meas_sds`` (a free measurement-error SD; stored internally as the
    variance, so the squared value is pinned). Rows of other categories are
    ignored -- they belong to other stages. Raises if a row targets a
    measurement that does not exist or a parameter the model spec already
    normalizes/pins (hence not a free Stage-2 parameter).
    """
    if fixed_params is None or fixed_params.empty:
        return []

    meas_slot = {
        (int(period), meas): aug
        for aug, (period, _factor, meas) in zip(
            layout.measurement_slots, layout.measurement_meta, strict=True
        )
    }
    slot_col = {pf: i for i, pf in enumerate(struct.factor_period_slots)}
    n_slots = struct.n_factor_slots
    lambda_flat_mask = struct.lambda_free_mask.ravel()
    lambda_rank = np.cumsum(lambda_flat_mask) - 1
    intercept_rank = np.cumsum(struct.intercept_free_mask) - 1
    sigma2_rank = np.cumsum(struct.sigma2_free_mask) - 1

    out: list[tuple[int, float]] = []
    for label, value in fixed_params["value"].items():
        category, period, name1, name2 = label  # ty: ignore[not-iterable]
        if category not in _STAGE2_FIXED_CATEGORIES:
            continue
        period = int(period)
        meas = str(name1)
        aug = meas_slot.get((period, meas))
        if aug is None:
            msg = (
                f"fixed_params row {label!r} targets measurement '{meas}' at "
                f"period {period}, which is not in the model's measurement system."
            )
            raise ValueError(msg)
        if category == "loadings":
            factor = str(name2)
            col = slot_col.get((period, factor))
            if col is None:
                msg = (
                    f"fixed_params row {label!r} targets factor '{factor}', which "
                    f"has no structural slot at period {period}."
                )
                raise ValueError(msg)
            pos = aug * n_slots + col
            flat_i = _free_flat_index(
                int(lambda_rank[pos]),
                slices["lambda"].start,
                label,
                "loading",
                is_free=bool(lambda_flat_mask[pos]),
            )
            out.append((flat_i, float(value)))
        elif category == "controls":
            flat_i = _free_flat_index(
                int(intercept_rank[aug]),
                slices["intercept"].start,
                label,
                "intercept",
                is_free=bool(struct.intercept_free_mask[aug]),
            )
            out.append((flat_i, float(value)))
        else:  # meas_sds: stored as a variance, so pin the squared value.
            flat_i = _free_flat_index(
                int(sigma2_rank[aug]),
                slices["sigma2"].start,
                label,
                "measurement SD",
                is_free=bool(struct.sigma2_free_mask[aug]),
            )
            out.append((flat_i, float(value) ** 2))
    return out


def solve_minimum_distance(
    mixture: MixtureFitResult,
    processed_model: ProcessedModel,
    *,
    weighting: str = "identity",
    algorithm: str = "scipy_lbfgsb",
    allow_overnormalization: bool = False,
    algo_options: Mapping[str, Any] | None = None,
    fixed_params: pd.DataFrame | None = None,
) -> MinimumDistanceResult:
    """Recover structural parameters from the reduced-form mixture.

    Args:
        mixture: Stage 1 fit (reduced-form Pi, Psi per component).
        processed_model: Skillmodels processed model (provides normalization
            and constraint structure).
        weighting: ``"identity"`` (default; the AMN paper's choice). This is
            an unweighted identity-metric criterion over per-component means
            and the full covariance matrices, so off-diagonal moments are
            implicitly counted twice and components are weighted equally
            regardless of tau_k. ``"optimal"`` is reserved for a future
            Avar-weighted criterion and currently raises
            ``NotImplementedError``.
        algorithm: optimagic algorithm name (default ``scipy_lbfgsb``).
        allow_overnormalization: Opt out of the CES minimal-normalization
            guard. When True, extra normalized CES loadings are treated as a
            deliberate fixed-loadings analysis instead of an error.
        algo_options: Optional optimagic ``algo_options`` for the L-BFGS-B
            solver (e.g. ``{"stopping_maxiter": 500}``). CHS seeding caps the
            iterations here so a rough structural seed stays fast on a large
            factor-period block; standalone estimation leaves it unbounded.
        fixed_params: optional params frame whose structural measurement rows
            (``loadings`` / ``controls`` / ``meas_sds``) pin the corresponding
            free entries of the packed optimiser vector. The pinned entries are
            seeded to their values and held there via an optimagic
            ``FixedConstraint`` so the remaining structural parameters are fit
            conditional on the pins. Rows of other categories are ignored.

    Return:
        MinimumDistanceResult with structural Lambda, A, Sigma, and the
        per-component factor means and covariances.

    """
    if weighting not in ("identity", "optimal"):
        msg = f"Unknown weighting '{weighting}'."
        raise ValueError(msg)
    if weighting == "optimal":
        msg = (
            "weighting='optimal' is documented but not yet implemented; "
            "only weighting='identity' is currently supported."
        )
        raise NotImplementedError(msg)

    layout = mixture.layout
    if not layout.measurement_slots and not layout.observed_factor_slots:
        msg = "Mixture layout has no slots; cannot run minimum distance."
        raise ValueError(msg)

    _validate_ces_stage2_anchors(
        processed_model, layout, allow_overnormalization=allow_overnormalization
    )

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

    # Pass an exact JAX gradient: with a large factor-period block the parameter
    # vector has thousands of entries, so a finite-difference gradient would cost
    # thousands of dense objective evaluations per L-BFGS-B step (hours at panel
    # scale). The analytical gradient is one backward pass.
    fun, jac = _make_objective_and_grad(
        struct,
        slices,
        n_components=n_components,
        mixture_weights=mixture.weights,
        target_means=target_means,
        target_covs=target_covs,
    )

    fixed_idx = _stage2_fixed_indices(fixed_params, struct, layout, slices)
    constraints: list[om.constraints.Constraint] | None = None
    if fixed_idx:
        for flat_i, value in fixed_idx:
            flat0[flat_i] = value
        fixed_positions = np.array([i for i, _ in fixed_idx], dtype=int)
        constraints = [om.FixedConstraint(selector=lambda x, p=fixed_positions: x[p])]

    result = om.minimize(
        fun=fun,
        jac=jac,
        params=flat0,
        algorithm=algorithm,
        bounds=om.Bounds(lower=lower),
        constraints=constraints,
        algo_options=dict(algo_options) if algo_options else None,
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
