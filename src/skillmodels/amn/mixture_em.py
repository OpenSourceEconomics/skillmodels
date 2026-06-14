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
) -> MixtureFitResult:
    """Fit a Gaussian mixture to the augmented measure matrix via EM.

    Uses `sklearn.mixture.GaussianMixture` under the hood with k-means
    initialization and multiple restarts.

    Scope: this estimator is COMPLETE-CASE ONLY. Rows containing any NaN
    in the augmented measure vector are dropped before fitting (listwise
    deletion). The fitted mixture therefore targets the population
    reduced-form distribution F_{M,X} (and hence the downstream Stage 2
    `Pi_k`, `Psi_k` and all structural parameters) only under a complete-
    data or MCAR (missing-completely-at-random) assumption. Under an
    unbalanced panel or MAR/MNAR missingness the target shifts and the
    recovered parameters can be biased. A `RuntimeWarning` is emitted
    whenever any rows are dropped. A future revision will integrate over
    missing dimensions in the E-step (observed-data EM) to relax this.

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
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
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
        weights=np.asarray(gm.weights_, dtype=float),
        means=np.asarray(gm.means_, dtype=float),
        covariances=np.asarray(gm.covariances_, dtype=float),
        loglikelihood=float(gm.score(fit_data) * n_complete),
        n_iter=int(gm.n_iter_),
        converged=bool(gm.converged_),
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
