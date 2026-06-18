"""Stage 1 of the AMN estimator: mixture-of-normals EM on augmented measurements.

Fits

    F_{M,X} = sum_k tau_k * Normal(Pi_k, Psi_k)

to the joint vector of (factor measurements, observed factor values,
controls) across all periods. Matches AMN 2020 equations (11)-(14).

The fitted mixture is the reduced-form input to Stage 2's structural
minimum-distance recovery (`skillmodels.amn.minimum_distance`).
"""

import warnings
from collections.abc import Mapping
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from skillmodels.amn.missing_data_em import fit_gaussian_mixture_missing
from skillmodels.amn.types import AugmentedMeasureLayout, MixtureFitResult
from skillmodels.common.types import ProcessedModel


def build_augmented_measure_layout(
    processed_model: ProcessedModel,
) -> AugmentedMeasureLayout:
    """Compute the column layout of the augmented measure vector.

    The augmented vector concatenates, in order:

    1. Factor measurements at each period (one slot per `(period,
       measurement)` row of `processed_model.update_info`).
    2. Observed factor values at each period (one slot per `(period,
       observed_factor)` pair).
    3. Controls at the first period (treated as time-invariant; one slot
       per non-constant control).

    Slots 2 and 3 are treated as zero-measurement-error observations
    with loading 1 in the AMN measurement-system mapping (paper p. 2522:
    "we set the corresponding standard deviation in Sigma to zero and
    the corresponding factor loading to one").

    Args:
        processed_model: The output of `common.process_model.process_model`.

    Return:
        AugmentedMeasureLayout with slot metadata for downstream Stage 2
        bookkeeping.

    """
    update_info = processed_model.update_info
    periods = processed_model.labels.periods
    aug_to_period = processed_model.labels.aug_periods_to_periods
    observed_factors = processed_model.labels.observed_factors
    controls = tuple(c for c in processed_model.labels.controls if c != "constant")

    columns: list[str] = []
    measurement_slots: list[int] = []
    measurement_meta: list[tuple[int, str, str]] = []
    observed_factor_slots: list[int] = []
    observed_factor_meta: list[tuple[int, str]] = []
    control_slots: list[int] = []

    # Walk update_info rows in canonical (aug_period, measurement) order.
    # Each row is one measurement update; map aug_period -> calendar period
    # via labels.aug_periods_to_periods so the layout metadata is in
    # AMN-paper terms (calendar period).
    factor_columns = [c for c in update_info.columns if c != "purpose"]
    for index, row in update_info.iterrows():
        aug_period, meas_name = index  # ty: ignore[not-iterable]
        purpose = row.get("purpose", "measurement")
        if purpose != "measurement":
            continue
        loadings = row[factor_columns].astype(bool)
        if not loadings.any():
            continue
        factor = next(f for f in factor_columns if loadings[f])
        period = int(aug_to_period[int(aug_period)])
        slot = len(columns)
        columns.append(f"meas[{period}|{factor}|{meas_name}]")
        measurement_slots.append(slot)
        measurement_meta.append((period, str(factor), str(meas_name)))

    for period in periods:
        for of in observed_factors:
            slot = len(columns)
            columns.append(f"obs_factor[{period}|{of}]")
            observed_factor_slots.append(slot)
            observed_factor_meta.append((int(period), str(of)))

    for ctrl in controls:
        slot = len(columns)
        columns.append(f"control[{ctrl}]")
        control_slots.append(slot)

    return AugmentedMeasureLayout(
        columns=tuple(columns),
        measurement_slots=tuple(measurement_slots),
        observed_factor_slots=tuple(observed_factor_slots),
        control_slots=tuple(control_slots),
        measurement_meta=tuple(measurement_meta),
        observed_factor_meta=tuple(observed_factor_meta),
        control_meta=tuple(controls),
    )


def _build_period_views(
    data: pd.DataFrame,
    periods: tuple[int, ...],
    period_level: str,
    caseids: pd.Index,
) -> dict[int, pd.DataFrame]:
    """Return one (n_obs, n_cols) DataFrame per period, reindexed by caseids."""
    period_views: dict[int, pd.DataFrame] = {}
    for period in periods:
        sub = data.xs(period, level=period_level, drop_level=True)
        if isinstance(sub, pd.Series):
            sub = sub.to_frame()
        sub = sub.reindex(caseids)
        period_views[int(period)] = sub
    return period_views


def _fill_controls(
    out: np.ndarray,
    period_views: dict[int, pd.DataFrame],
    layout: AugmentedMeasureLayout,
    periods: tuple[int, ...],
) -> None:
    """Fill control slots from the first period each control is observed in."""
    for slot, ctrl in zip(layout.control_slots, layout.control_meta, strict=True):
        for period in periods:
            sub = period_views[int(period)]
            if ctrl not in sub.columns:
                continue
            col = sub[ctrl].to_numpy()
            mask = np.isnan(out[:, slot])
            out[mask, slot] = col[mask]


def build_augmented_measure_matrix(
    data: pd.DataFrame,
    processed_model: ProcessedModel,
    layout: AugmentedMeasureLayout,
) -> np.ndarray:
    """Stack each child's augmented measure vector into an ``(n_obs, n_aug)`` matrix.

    Reshapes the long-format `data` into one row per individual (caseid),
    pulling the right column for each layout slot from the corresponding
    period.

    Args:
        data: Panel dataset in long format with MultiIndex
            ``(caseid, period)``.
        processed_model: Output of `process_model.process_model`.
        layout: Slot layout for the augmented vector.

    Return:
        ``(n_obs, n_aug)`` numpy array. Missing values are NaN.

    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels < 2:
        msg = "data must have a 2-level MultiIndex (caseid, period)."
        raise ValueError(msg)
    period_level = str(data.index.names[1])
    case_level = str(data.index.names[0])

    caseids = data.index.get_level_values(case_level).unique()
    n_obs = len(caseids)
    n_aug = len(layout.columns)
    out = np.full((n_obs, n_aug), np.nan)

    periods = processed_model.labels.periods
    period_views = _build_period_views(data, periods, period_level, caseids)

    for slot, (period, _factor, meas_name) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        sub = period_views[period]
        if meas_name in sub.columns:
            out[:, slot] = sub[meas_name].to_numpy()

    for slot, (period, of_name) in zip(
        layout.observed_factor_slots, layout.observed_factor_meta, strict=True
    ):
        sub = period_views[period]
        if of_name in sub.columns:
            out[:, slot] = sub[of_name].to_numpy()

    if layout.control_slots:
        _fill_controls(out, period_views, layout, periods)

    return out


def _normalization_measurement_slots(
    layout: AugmentedMeasureLayout,
    processed_model: ProcessedModel,
) -> set[int]:
    """Return measurement slots whose loading is normalized (must not be dropped).

    Mirrors the normalization detection in
    `minimum_distance._build_structure`: a `(period, factor, measurement)`
    slot is protected if `measurement` carries a fixed loading in any
    `aug_period` mapping to that calendar period.
    """
    normalizations = processed_model.normalizations
    aug_to_period = processed_model.labels.aug_periods_to_periods
    protected: set[int] = set()
    for slot, (period, factor, meas_name) in zip(
        layout.measurement_slots, layout.measurement_meta, strict=True
    ):
        if factor not in normalizations:
            continue
        for aug_period, cal_period in aug_to_period.items():
            if int(cal_period) != int(period):
                continue
            if meas_name in normalizations[factor].loadings[aug_period]:
                protected.add(slot)
                break
    return protected


def _subset_layout(
    layout: AugmentedMeasureLayout,
    keep_mask: np.ndarray,
) -> AugmentedMeasureLayout:
    """Drop the columns where `keep_mask` is False and re-index every slot.

    Slot indices are absolute positions in the augmented vector, so removing
    columns shifts the survivors; `old -> new` is the running count of kept
    columns. Only measurement slots are ever dropped, so the observed-factor
    and control slots survive and are merely renumbered.
    """
    old_to_new = np.cumsum(keep_mask) - 1

    def _remap[T](
        slots: tuple[int, ...], metas: tuple[T, ...]
    ) -> tuple[tuple[int, ...], tuple[T, ...]]:
        kept = [
            (int(old_to_new[slot]), meta)
            for slot, meta in zip(slots, metas, strict=True)
            if keep_mask[slot]
        ]
        if not kept:
            return (), ()
        new_slots, new_metas = zip(*kept, strict=True)
        return tuple(new_slots), tuple(new_metas)

    meas_slots, meas_meta = _remap(layout.measurement_slots, layout.measurement_meta)
    obs_slots, obs_meta = _remap(
        layout.observed_factor_slots, layout.observed_factor_meta
    )
    ctrl_slots, ctrl_meta = _remap(layout.control_slots, layout.control_meta)
    columns = tuple(
        c for c, keep in zip(layout.columns, keep_mask, strict=True) if keep
    )
    return AugmentedMeasureLayout(
        columns=columns,
        measurement_slots=meas_slots,
        observed_factor_slots=obs_slots,
        control_slots=ctrl_slots,
        measurement_meta=meas_meta,
        observed_factor_meta=obs_meta,
        control_meta=ctrl_meta,
    )


def _n_complete(augmented: np.ndarray, keep_mask: np.ndarray) -> int:
    """Count rows with no missing value among the kept columns."""
    return int((~np.isnan(augmented[:, keep_mask]).any(axis=1)).sum())


def reduce_to_seedable_measurements(
    layout: AugmentedMeasureLayout,
    augmented: np.ndarray,
    processed_model: ProcessedModel,
    *,
    n_components: int,
    min_complete_cases: int = 50,
) -> tuple[AugmentedMeasureLayout, np.ndarray, tuple[tuple[int, str, str], ...]]:
    """Drop high-missing measurements so Stage 1 can seed on complete cases.

    Stage 1's mixture EM is complete-case only: a row survives only if every
    augmented column is observed. Rotating-subsample measurements (missing for
    most person-waves) -- and, in an unbalanced panel, ordinary measurements that
    are simply absent in some waves -- can drive the complete-case count to zero.

    When the full vector has fewer complete cases than `n_components`, greedily
    drop the highest-missing non-normalization measurement as long as the drop
    adds complete cases, until at least `min_complete_cases` remain (or no more
    droppable measurement helps). Each factor's normalization measurement is never
    dropped. The dropped measurements are simply not AMN-seeded; their params fall
    back to the neutral/Spearman seeding defaults. If even dropping every droppable
    measurement leaves the subset infeasible, the inputs are returned as far as
    reduced and `fit_mixture_em` raises its informative complete-case error.

    Returns the (possibly unchanged) layout and augmented matrix plus the meta of
    the dropped measurements. When the full vector already has enough complete
    cases the inputs are returned untouched, so healthy models are unaffected.

    Args:
        layout: Augmented-measure layout to (possibly) reduce.
        augmented: `(n_obs, n_aug)` augmented matrix aligned with `layout`.
        processed_model: Processed model, for normalization detection.
        n_components: Mixture components; the complete-case feasibility floor.
        min_complete_cases: Greedy drops stop once this many complete cases
            remain (capped below by `n_components`).

    Return:
        `(reduced_layout, reduced_augmented, dropped_measurement_meta)`.

    """
    n_aug = augmented.shape[1]
    full_mask = np.ones(n_aug, dtype=bool)
    if _n_complete(augmented, full_mask) >= n_components:
        return layout, augmented, ()

    protected = _normalization_measurement_slots(layout, processed_model)
    n_rows = max(augmented.shape[0], 1)
    missing_rate = np.isnan(augmented).sum(axis=0) / n_rows
    # Highest-missing non-normalization measurements first; fully observed ones
    # (missing rate 0) never block and are never dropped.
    droppable = sorted(
        (
            slot
            for slot in layout.measurement_slots
            if slot not in protected and missing_rate[slot] > 0.0
        ),
        key=lambda slot: missing_rate[slot],
        reverse=True,
    )

    target = max(n_components, min_complete_cases)
    keep_mask = full_mask.copy()
    dropped: list[int] = []
    for slot in droppable:
        if _n_complete(augmented, keep_mask) >= target:
            break
        keep_mask[slot] = False
        dropped.append(slot)

    if not dropped or _n_complete(augmented, keep_mask) < n_components:
        # Either nothing is droppable, or even dropping every droppable
        # measurement leaves the subset infeasible (e.g. cross-period attrition,
        # or a normalization is the blocker). Return the inputs untouched so
        # fit_mixture_em raises its informative complete-case error on the full
        # set rather than on a uselessly thinned one.
        return layout, augmented, ()

    drop_set = set(dropped)
    dropped_meta = tuple(
        meta
        for slot, meta in zip(
            layout.measurement_slots, layout.measurement_meta, strict=True
        )
        if slot in drop_set
    )
    n_after = _n_complete(augmented, keep_mask)
    pretty = ", ".join(f"{p}|{f}|{m}" for p, f, m in dropped_meta)
    warnings.warn(
        f"AMN Stage 1 seeding: too few complete-case rows over the full "
        f"measurement set (< {n_components} mixture components). Greedily dropped "
        f"{len(dropped_meta)} high-missing measurement(s) and seeded the mixture "
        f"on the always-observed subset ({n_after} complete cases): {pretty}. "
        f"Their loadings/SDs fall back to the start-value defaults.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _subset_layout(layout, keep_mask), augmented[:, keep_mask], dropped_meta


def _fit_complete_case(
    augmented: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    tol: float,
    n_init: int,
    reg_covar: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Fit `sklearn`'s GaussianMixture on listwise-complete rows."""
    complete_mask = ~np.isnan(augmented).any(axis=1)
    n_complete = int(complete_mask.sum())
    if n_complete < n_components:
        msg = (
            f"Only {n_complete} complete-case rows available for "
            f"{n_components}-component mixture."
        )
        raise ValueError(msg)
    n_total = int(augmented.shape[0])
    n_dropped = n_total - n_complete
    if n_dropped > 0:
        msg = (
            f"AMN Stage 1 mixture EM is complete-case only: dropped "
            f"{n_dropped}/{n_total} rows with missing augmented "
            f"measurements before fitting. The recovered reduced-form "
            f"mixture targets the population distribution only under a "
            f"complete-data or MCAR assumption; under an unbalanced panel "
            f"or non-MCAR missingness the estimates may be biased."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    fit_data = augmented[complete_mask]
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        reg_covar=reg_covar,
        init_params="kmeans",
        random_state=seed,
    )
    gm.fit(fit_data)
    return (
        np.asarray(gm.weights_, dtype=float),
        np.asarray(gm.means_, dtype=float),
        np.asarray(gm.covariances_, dtype=float),
        float(gm.score(fit_data) * n_complete),
        int(gm.n_iter_),
        bool(gm.converged_),
    )


def fit_mixture_em(
    augmented: np.ndarray,
    *,
    n_components: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    n_init: int = 5,
    reg_covar: float = 1e-6,
    seed: int = 0,
    layout: AugmentedMeasureLayout | None = None,
    init_params: Mapping[str, np.ndarray] | None = None,
    method: str = "complete_case",
) -> MixtureFitResult:
    """Fit a Gaussian mixture to the augmented measure matrix via EM.

    With `method="complete_case"` (the default) this fits
    `sklearn.mixture.GaussianMixture` on the listwise-complete rows -- valid
    only under a complete-data or MCAR assumption, and raising when fewer than
    `n_components` complete rows remain. With `method="missing_data"` it uses
    `missing_data_em.fit_gaussian_mixture_missing`, which marginalises over each
    row's missing entries and so handles unbalanced panels with no complete
    cases at all.

    Args:
        augmented: ``(n_obs, n_aug)`` augmented measure matrix from
            `build_augmented_measure_matrix`.
        n_components: Number of mixture components K.
        max_iter: Maximum EM iterations per restart.
        tol: Log-likelihood convergence tolerance.
        n_init: Number of EM restarts; the best fit is kept.
        reg_covar: Diagonal ridge added to each component covariance for
            numerical stability.
        seed: RNG seed.
        layout: Slot layout to embed in the result (carried through to
            Stage 2).
        init_params: Optional warm-start values. Currently unused — kept
            for forward-compatibility with a custom Spearman-seeded init
            once Stage 1 results from the moment-init pipeline become
            available as warm starts.
        method: ``"complete_case"`` or ``"missing_data"`` (see above).

    Return:
        MixtureFitResult holding the fitted weights, means, covariances
        and convergence diagnostics.

    """
    del init_params  # reserved for follow-up
    if augmented.ndim != 2:
        msg = "augmented must be a 2D array."
        raise ValueError(msg)
    if augmented.shape[0] == 0:
        msg = "augmented has zero rows; cannot fit mixture."
        raise ValueError(msg)

    if method == "missing_data":
        fit = fit_gaussian_mixture_missing(
            augmented,
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            reg_covar=reg_covar,
            seed=seed,
        )
        weights, means, covs = fit.weights, fit.means, fit.covariances
        loglik, n_iter, converged = fit.loglikelihood, fit.n_iter, fit.converged
    elif method == "complete_case":
        weights, means, covs, loglik, n_iter, converged = _fit_complete_case(
            augmented,
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            reg_covar=reg_covar,
            seed=seed,
        )
    else:
        msg = f"Unknown mixture EM method {method!r}."
        raise ValueError(msg)

    if layout is None:
        # Caller didn't supply a layout; synthesize a minimal one purely
        # from column indices so downstream code that doesn't need slot
        # metadata still works.
        n_aug = augmented.shape[1]
        layout = AugmentedMeasureLayout(
            columns=tuple(f"col[{i}]" for i in range(n_aug)),
            measurement_slots=tuple(range(n_aug)),
            observed_factor_slots=(),
            control_slots=(),
            measurement_meta=(),
            observed_factor_meta=(),
            control_meta=(),
        )

    return MixtureFitResult(
        weights=weights,
        means=means,
        covariances=covs,
        loglikelihood=loglik,
        n_iter=n_iter,
        converged=converged,
        layout=layout,
    )


def _all_slot_ids(layout: AugmentedMeasureLayout) -> tuple[int, ...]:
    """Return the union of all slot id tuples in canonical order."""
    return tuple(
        chain(
            layout.measurement_slots,
            layout.observed_factor_slots,
            layout.control_slots,
        )
    )
