"""Moment-based starting values for the CHS estimator.

Replaces the legacy `0.5` / `1.0` / `0.0` constant fills with
data-derived seeds. Two-stage hybrid:

1. **Spearman cross-covariance moments** identify the measurement
   system (loadings + measurement-error SDs + latent factor SDs)
   per period.
2. **OLS on Bartlett-scored factor proxies** identifies transition
   coefficients and the residual SD of the production shock —
   the AMN (Attanasio-Meghir-Nix 2020) flavour the AF paper §7
   recommends as starting values, just bootstrapped from the
   Spearman estimates rather than from a separate AMN run.

Together these give a data-derived seed for every category that has
moment-based identification. Categories Spearman + Bartlett-OLS
cannot identify (mixture weights, initial means, controls) fall
back to neutral defaults — these affect convergence speed only,
not identification.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import optimagic as om
import pandas as pd

from skillmodels.amn.moments import (
    SpearmanResult,
    seed_beta_from_ols,
    spearman_factor_moments,
)
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_data import process_data
from skillmodels.common.process_model import process_model
from skillmodels.common.types import Normalizations, ProcessedModel


def get_spearman_start_params(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params_template: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of `params_template` with moment-based seed values.

    Walks the params index and fills each row using:

    * `loadings`, `meas_sds`: per-period Spearman moments on the
      single-factor measurements of each latent factor.
    * `initial_cholcovs`: diagonal entries set to `sqrt(latent_var)`
      from the period-0 Spearman result; off-diagonals 0.
    * `initial_states`: 0 (location is unidentified from cross-covs).
    * `mixture_weights`: uniform `1 / n_mixtures`.
    * `controls`: 0.
    * `shock_sds`: 0.5.
    * `transition`: 0.5.

    Rows where `lower_bound == upper_bound` (user normalizations,
    fixed_params pins, model-implied fixes) are left untouched.

    Args:
        model_spec: Model specification.
        data: Long-format panel with the same `(id, period)` MultiIndex
            consumed by `get_maximization_inputs`.
        params_template: The params DataFrame returned by
            `get_maximization_inputs(...)["params_template"]` — it
            already has the right MultiIndex, bounds, and pinned
            values.

    Return:
        Copy of `params_template` with the `value` column populated.

    """
    processed_model = process_model(model_spec)
    processed_data = process_data(
        df=data,
        has_endogenous_factors=processed_model.endogenous_factors_info.has_endogenous_factors,
        labels=processed_model.labels,
        update_info=processed_model.update_info,
        anchoring_info=processed_model.anchoring,
        purpose="estimation",
    )
    measurements = np.asarray(processed_data["measurements"])
    update_info = processed_model.update_info
    latent_factors = processed_model.labels.latent_factors
    n_mixtures = processed_model.dimensions.n_mixtures
    loading_norms = _collect_loading_norms(processed_model.normalizations)
    aug_periods = processed_model.labels.aug_periods

    out = params_template.copy()
    # `free` here means "this entry still needs a value" — i.e. it has
    # not been pinned by `enforce_fixed_constraints` or by the caller.
    # We use NaN-detection instead of `lower_bound != upper_bound` because
    # `enforce_fixed_constraints` only writes `value` and leaves bounds
    # untouched; bound-equality alone would misclassify fixed entries.
    free = out["value"].isna()

    _apply_neutral_defaults(out, free, n_mixtures=n_mixtures)

    update_info_periods = set(update_info.index.get_level_values("aug_period"))
    spearman_per_period: dict[tuple[int, str], SpearmanResult] = {}
    for aug_period in aug_periods:
        if aug_period not in update_info_periods:
            continue
        period_meas_index = _measurement_row_index(update_info, aug_period)
        for factor in latent_factors:
            factor_meas = _single_factor_measurements(
                update_info,
                aug_period=aug_period,
                factor=factor,
                all_factors=latent_factors,
            )
            if len(factor_meas) < 2:
                continue
            cols = [period_meas_index[m] for m in factor_meas]
            sub = measurements[cols, :].T  # (n_obs, n_meas)
            anchor_local, anchor_loading = _pick_anchor(
                factor_meas=factor_meas, factor=factor, loading_norms=loading_norms
            )
            result = spearman_factor_moments(
                sub, anchor_idx=anchor_local, anchor_loading=anchor_loading
            )
            if not result.valid:
                continue
            spearman_per_period[(aug_period, factor)] = result
            _override_loadings_meas_sds(
                out,
                free,
                aug_period=aug_period,
                factor=factor,
                factor_meas=factor_meas,
                result=result,
            )

    _override_initial_cholcovs(
        out,
        free,
        spearman_per_period=spearman_per_period,
        latent_factors=latent_factors,
        n_mixtures=n_mixtures,
    )

    _override_transition_via_ols(
        out,
        free,
        processed_model=processed_model,
        measurements=measurements,
        spearman_per_period=spearman_per_period,
        observed_factors=np.asarray(processed_data["observed_factors"]),
    )

    _pool_within_stage_equality(
        out,
        free=free,
        processed_model=processed_model,
    )

    return out


def get_amn_start_params(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params_template: pd.DataFrame,
    amn_params: pd.DataFrame,
) -> pd.DataFrame:
    """Seed start values from AMN estimates, pooling stage-tied params.

    Fills via `get_spearman_start_params` (covering entries AMN does not
    produce — e.g. mixture weights and initial Cholesky diagonals), overlays
    the AMN estimates onto the common free entries, then re-pools the
    `transition` / `shock_sds` seeds within each stage. The re-pool is
    essential: AMN estimates per aug_period, so its raw overlay violates the
    within-stage `PairwiseEqualityConstraint`s that `optimagic` checks at the
    start point (`get_spearman_start_params` pools them, but the AMN overlay
    re-breaks the ties).

    Args:
        model_spec: Model specification.
        data: Long-format panel with the `(id, period)` MultiIndex consumed by
            `get_maximization_inputs`.
        params_template: Template from `get_maximization_inputs`, with pinned
            entries already filled.
        amn_params: `AMNEstimationResult.params` to overlay onto the free
            common entries.

    Return:
        Copy of `params_template` with seeded, stage-pooled `value`s.

    """
    pre_pinned = params_template["value"].notna()
    out = get_spearman_start_params(
        model_spec=model_spec, data=data, params_template=params_template
    )
    common = amn_params.index.intersection(out.index)
    free_common = common[~pre_pinned.reindex(common, fill_value=False)]
    out.loc[free_common, "value"] = amn_params.loc[free_common, "value"]
    _pool_within_stage_equality(
        out, free=~pre_pinned, processed_model=process_model(model_spec)
    )
    return out


def pool_equality_groups(  # noqa: C901
    params: pd.DataFrame,
    constraints: list[om.constraints.Constraint],
    *,
    keep_pinned_values: pd.Series | None = None,
) -> pd.DataFrame:
    """Pool param values within each `om.EqualityConstraint` group.

    For each `om.EqualityConstraint` whose selector is the standard
    `select_by_loc(loc=multi_index)` form, replace the values of all
    members of the group with a single shared value so the equality
    constraint holds at the start values. If a member is flagged as
    "pinned" (via `keep_pinned_values=True` for that loc), the pinned
    value is used for the whole group; otherwise the group is averaged.

    Use after moment-based starting values: Spearman seeds each period
    independently, which violates user equality constraints across
    periods (e.g., loadings or meas_sds constant across periods).
    Calling this with the user constraint list restores the equalities
    while keeping the data-derived information (now pooled).

    Args:
        params: Params DataFrame with a `"value"` column and the
            standard 4-level MultiIndex.
        constraints: List of optimagic Constraint objects. Only
            `om.EqualityConstraint` entries with a `select_by_loc`
            partial as `selector` are honoured.
        keep_pinned_values: Optional boolean Series indexed like
            `params`. Entries where this is True keep their value;
            the pooling logic copies that value to every other member
            of the same equality group.

    Return:
        Modified copy of `params`.
    """
    out = params.copy()
    for c in constraints:
        if not isinstance(c, om.EqualityConstraint):
            continue
        selector = c.selector
        keywords = getattr(selector, "keywords", None)
        if not keywords or "loc" not in keywords:
            continue
        loc = keywords["loc"]
        if not isinstance(loc, pd.MultiIndex) or len(loc) <= 1:
            continue
        members = [m for m in loc if m in out.index]
        if len(members) <= 1:
            continue
        if keep_pinned_values is not None:
            pinned = [
                float(out.loc[m, "value"])
                for m in members
                if bool(keep_pinned_values.loc[m]) and pd.notna(out.loc[m, "value"])
            ]
        else:
            pinned = []
        if pinned:
            target = pinned[0]
        else:
            raw = [
                float(out.loc[m, "value"])
                for m in members
                if pd.notna(out.loc[m, "value"])
            ]
            if not raw:
                continue
            target = float(np.mean(raw))
        for m in members:
            if keep_pinned_values is None or not bool(keep_pinned_values.loc[m]):
                out.loc[m, "value"] = target
    return out


def _apply_neutral_defaults(
    params: pd.DataFrame,
    free: pd.Series,
    *,
    n_mixtures: int,
) -> None:
    cat = params.index.get_level_values("category")
    params.loc[free & (cat == "controls"), "value"] = 0.0
    params.loc[free & (cat == "loadings"), "value"] = 1.0
    params.loc[free & (cat == "meas_sds"), "value"] = 0.5
    params.loc[free & (cat == "shock_sds"), "value"] = 0.5
    params.loc[free & (cat == "initial_states"), "value"] = 0.0
    params.loc[free & (cat == "mixture_weights"), "value"] = 1.0 / max(n_mixtures, 1)
    params.loc[free & (cat == "initial_cholcovs"), "value"] = 0.0
    params.loc[free & (cat == "transition"), "value"] = 0.5
    # Control-function categories (present only under a CorrectionSpec). Seed
    # small/neutral: no first-stage relationship and no correction initially.
    params.loc[free & (cat == "investment_eq"), "value"] = 0.0
    params.loc[free & (cat == "kappa"), "value"] = 0.0
    diag_values = pd.Series(
        [_is_cholcov_diag(idx) for idx in params.index],
        index=params.index,
    )
    diag_mask = free & (cat == "initial_cholcovs") & diag_values
    params.loc[diag_mask, "value"] = 1.0


def _is_cholcov_diag(idx: tuple) -> bool:
    if idx[0] != "initial_cholcovs":
        return False
    name2 = idx[3]
    if "-" not in name2:
        return False
    a, b = name2.split("-", 1)
    return a == b


def _measurement_row_index(
    update_info: pd.DataFrame, aug_period: int
) -> dict[str, int]:
    out: dict[str, int] = {}
    for flat_idx, (a_period, meas) in enumerate(update_info.index):
        if a_period == aug_period:
            out[meas] = flat_idx
    return out


def _single_factor_measurements(
    update_info: pd.DataFrame,
    *,
    aug_period: int,
    factor: str,
    all_factors: Iterable[str],
) -> tuple[str, ...]:
    """Return measurements at `aug_period` that load only on `factor`."""
    period_rows = update_info.xs(aug_period, level="aug_period")
    measurement_rows = period_rows.loc[period_rows["purpose"] == "measurement"]
    out: list[str] = []
    factors = list(all_factors)
    for meas, row in measurement_rows.iterrows():
        if not bool(row[factor]):
            continue
        if any(bool(row[f]) for f in factors if f != factor):
            continue
        out.append(str(meas))
    return tuple(out)


def _collect_loading_norms(
    normalizations: Mapping[str, Normalizations],
) -> dict[tuple[str, str], float]:
    """Flatten per-factor loading normalizations into a (meas, factor) → value dict."""
    out: dict[tuple[str, str], float] = {}
    for factor, norms in normalizations.items():
        loadings_per_period = norms.loadings
        for period_norms in loadings_per_period:
            for meas, value in period_norms.items():
                out[(meas, factor)] = float(value)
    return out


def _pick_anchor(
    *,
    factor_meas: tuple[str, ...],
    factor: str,
    loading_norms: dict[tuple[str, str], float],
) -> tuple[int, float]:
    for local_idx, meas in enumerate(factor_meas):
        if (meas, factor) in loading_norms:
            return local_idx, loading_norms[(meas, factor)]
    return 0, 1.0


def _override_loadings_meas_sds(
    params: pd.DataFrame,
    free: pd.Series,
    *,
    aug_period: int,
    factor: str,
    factor_meas: tuple[str, ...],
    result: SpearmanResult,
) -> None:
    for local_idx, meas in enumerate(factor_meas):
        loc_load = ("loadings", aug_period, meas, factor)
        if loc_load in params.index and free.loc[loc_load]:
            params.loc[loc_load, "value"] = float(result.loadings[local_idx])
        loc_sd = ("meas_sds", aug_period, meas, "-")
        if loc_sd in params.index and free.loc[loc_sd]:
            params.loc[loc_sd, "value"] = float(result.meas_sds[local_idx])


def _override_initial_cholcovs(
    params: pd.DataFrame,
    free: pd.Series,
    *,
    spearman_per_period: dict[tuple[int, str], SpearmanResult],
    latent_factors: tuple[str, ...],
    n_mixtures: int,
) -> None:
    for factor in latent_factors:
        result = spearman_per_period.get((0, factor))
        if result is None:
            continue
        sd_factor = float(np.sqrt(max(result.latent_var, 1e-12)))
        for comp in range(n_mixtures):
            loc = (
                "initial_cholcovs",
                0,
                f"mixture_{comp}",
                f"{factor}-{factor}",
            )
            if loc in params.index and free.loc[loc]:
                params.loc[loc, "value"] = sd_factor


def _pool_within_stage_equality(  # noqa: C901, PLR0912
    params: pd.DataFrame,
    *,
    free: pd.Series,
    processed_model: ProcessedModel,
) -> None:
    """Pool `transition` and `shock_sds` seeds within each stage.

    The `_get_stage_constraints` machinery imposes pairwise equality
    constraints across aug_periods belonging to the same stage. Our
    OLS-based seeds produce period-specific values; this post-processing
    pools them into a single stage value so the constraints hold at
    the start values. Pinned entries (set by `enforce_fixed_constraints`
    before the moment-based fill) take precedence — if any member of
    the equality group is pinned, the whole group uses that pinned
    value; otherwise the group is averaged.
    """
    stagemap = processed_model.labels.aug_stagemap
    stages: dict[int, list[int]] = {}
    for aug_period, stage in enumerate(stagemap):
        stages.setdefault(stage, []).append(aug_period)

    for stage_periods in stages.values():
        if len(stage_periods) <= 1:
            continue
        for category in ("transition", "shock_sds"):
            try:
                cat_slice = params.loc[category]
            except KeyError:
                continue
            existing_periods = set(cat_slice.index.get_level_values(0))
            shared = [p for p in stage_periods if p in existing_periods]
            if len(shared) <= 1:
                continue
            sub_index = cat_slice.loc[shared[0]].index
            for inner_loc in sub_index:
                full_locs = [
                    (category, p, *inner_loc)
                    for p in shared
                    if (category, p, *inner_loc) in params.index
                ]
                if len(full_locs) <= 1:
                    continue
                pinned_values = [
                    float(params.loc[loc, "value"])
                    for loc in full_locs
                    if not bool(free.loc[loc]) and pd.notna(params.loc[loc, "value"])
                ]
                if pinned_values:
                    target = pinned_values[0]
                else:
                    raw_values = [
                        float(params.loc[loc, "value"])
                        for loc in full_locs
                        if pd.notna(params.loc[loc, "value"])
                    ]
                    if not raw_values:
                        continue
                    target = float(np.mean(raw_values))
                for loc in full_locs:
                    if free.loc[loc]:
                        params.loc[loc, "value"] = target


def _bartlett_score(
    measurements: np.ndarray,
    cols: list[int],
    loadings: np.ndarray,
    meas_sds: np.ndarray,
) -> np.ndarray:
    r"""Bartlett factor-score estimator from per-indicator measurements.

    Returns the inverse-noise-weighted single-factor proxy
    :math:`\hat F = \sum_k w_k Z_k / \sum_k w_k \lambda_k`
    with :math:`w_k = \lambda_k / \sigma_k^2`, over rows where all
    `cols` are finite. Rows with any NaN get NaN proxy.
    """
    sub = measurements[cols, :].T  # (n_obs, n_meas)
    weights = loadings / np.maximum(meas_sds**2, 1e-12)
    denom = float(np.sum(weights * loadings))
    if denom < 1e-9:
        return np.full(sub.shape[0], np.nan)
    score = (sub * weights).sum(axis=1) / denom
    mask = np.all(np.isfinite(sub), axis=1)
    score[~mask] = np.nan
    return score


def _override_transition_via_ols(  # noqa: C901, PLR0912, PLR0915
    params: pd.DataFrame,
    free: pd.Series,
    *,
    processed_model: ProcessedModel,
    measurements: np.ndarray,
    spearman_per_period: dict[tuple[int, str], SpearmanResult],
    observed_factors: np.ndarray,
) -> None:
    """Seed transition coefficients + shock_sds via OLS on Bartlett scores.

    For each transition equation that maps state factors at one
    aug-period to a factor at the next aug-period with measurements,
    run OLS of the target Bartlett score on regressors derived from
    the source aug-period's Bartlett scores + observed factors.
    Coefficients are written into the matching `transition` rows;
    the residual SD is written to the matching `shock_sds` row.

    Currently implemented for `linear` and `translog` transition
    functions. Other transition functions keep the constant-default
    seeds set in `_apply_neutral_defaults`.
    """
    update_info = processed_model.update_info
    update_info_periods = list(update_info.index.get_level_values("aug_period"))
    aug_periods = processed_model.labels.aug_periods
    latent_factors = processed_model.labels.latent_factors
    observed_factor_names = processed_model.labels.observed_factors
    transition_info = processed_model.transition_info

    bartlett_proxies: dict[tuple[int, str], np.ndarray] = {}
    for (aug_period, factor), result in spearman_per_period.items():
        period_meas_index = _measurement_row_index(update_info, aug_period)
        factor_meas = _single_factor_measurements(
            update_info,
            aug_period=aug_period,
            factor=factor,
            all_factors=latent_factors,
        )
        cols = [period_meas_index[m] for m in factor_meas]
        proxy = _bartlett_score(
            measurements,
            cols,
            result.loadings,
            result.meas_sds,
        )
        bartlett_proxies[(aug_period, factor)] = proxy

    n_obs = measurements.shape[1] if measurements.ndim == 2 else 0
    n_calendar_periods = processed_model.dimensions.n_periods

    for src_idx, src_aug in enumerate(aug_periods[:-1]):
        tgt_aug = aug_periods[src_idx + 1]
        if tgt_aug not in update_info_periods:
            continue
        cal_idx_src = _aug_to_calendar_idx(
            processed_model,
            src_aug,
            n_calendar_periods,
        )
        if cal_idx_src is None:
            continue
        if observed_factors.ndim == 3:
            obs_at_src = observed_factors[cal_idx_src]
        else:
            obs_at_src = np.zeros((n_obs, 0))

        for factor in latent_factors:
            func_name = transition_info.function_names.get(factor)
            if func_name not in ("linear", "translog"):
                continue
            if (tgt_aug, factor) not in bartlett_proxies:
                continue
            target = bartlett_proxies[(tgt_aug, factor)]

            source_factor_proxies: dict[str, np.ndarray] = {}
            for src_factor in latent_factors:
                if (src_aug, src_factor) in bartlett_proxies:
                    source_factor_proxies[src_factor] = bartlett_proxies[
                        (src_aug, src_factor)
                    ]
            if factor not in source_factor_proxies:
                # Need at least the dependent factor's source proxy
                # for the regression to be meaningful.
                continue

            param_names = transition_info.param_names[factor]
            design, regressor_to_col = _build_design_for_transition(
                func_name=func_name,
                param_names=param_names,
                latent_factors=latent_factors,
                source_factor_proxies=source_factor_proxies,
                observed_factor_names=observed_factor_names,
                observed_factor_data=obs_at_src,
            )
            if design is None:
                continue
            mask = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
            if mask.sum() <= design.shape[1] + 1:
                continue
            beta = seed_beta_from_ols(target[mask], design[mask])
            if not np.all(np.isfinite(beta)):
                continue
            for regressor, col_idx in regressor_to_col.items():
                loc = ("transition", src_aug, factor, regressor)
                if loc in params.index and free.loc[loc]:
                    params.loc[loc, "value"] = float(beta[col_idx])

            # Residual SD → shock_sds[src_aug][factor].
            residual = target[mask] - design[mask] @ beta
            tgt_result = spearman_per_period.get((tgt_aug, factor))
            if tgt_result is None:
                continue
            # Bartlett-score residual variance includes
            # shock_var + (Bartlett-score-noise) ≈ shock_var + 1/Σ w·λ.
            score_noise_var = 1.0 / max(
                np.sum(
                    tgt_result.loadings**2 / np.maximum(tgt_result.meas_sds**2, 1e-12),
                ),
                1e-9,
            )
            raw_var = float(np.var(residual, ddof=1))
            shock_var = max(raw_var - score_noise_var, 1e-6)
            shock_sd = float(np.sqrt(shock_var))
            loc_sd = ("shock_sds", src_aug, factor, "-")
            if loc_sd in params.index and free.loc[loc_sd]:
                params.loc[loc_sd, "value"] = shock_sd


def _aug_to_calendar_idx(
    processed_model: ProcessedModel,
    aug_period: int,
    n_calendar_periods: int,
) -> int | None:
    """Map an aug-period to the calendar period of `observed_factors`.

    `processed_data["observed_factors"]` has shape
    `(n_periods, n_obs, n_observed_factors)`; this returns the
    calendar period index for the given aug-period, or `None` if it
    falls outside the calendar range.
    """
    mapping = processed_model.labels.aug_periods_to_periods
    cal = mapping.get(aug_period)
    if cal is None:
        return None
    if 0 <= int(cal) < n_calendar_periods:
        return int(cal)
    return None


def _build_design_for_transition(  # noqa: C901
    *,
    func_name: str,  # noqa: ARG001
    param_names: tuple[str, ...],
    latent_factors: tuple[str, ...],  # noqa: ARG001
    source_factor_proxies: dict[str, np.ndarray],
    observed_factor_names: tuple[str, ...],
    observed_factor_data: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, int]]:
    """Build the OLS design matrix matching `param_names`.

    Returns `(design, regressor_to_col)` where `regressor_to_col` maps
    each handled regressor name to its column index in `design`.
    Regressors that cannot be built from the available proxies are
    omitted (the corresponding transition coefficient stays at the
    constant-default seed).
    """
    n_obs = next(iter(source_factor_proxies.values())).shape[0]
    columns: list[np.ndarray] = []
    regressor_to_col: dict[str, int] = {}

    def _proxy_for(name: str) -> np.ndarray | None:
        if name in source_factor_proxies:
            return source_factor_proxies[name]
        if name in observed_factor_names:
            idx = observed_factor_names.index(name)
            if observed_factor_data.shape[1] > idx:
                return observed_factor_data[:, idx]
        return None

    for regressor in param_names:
        if regressor == "constant":
            columns.append(np.ones(n_obs))
            regressor_to_col[regressor] = len(columns) - 1
        elif " ** 2" in regressor:
            name = regressor.replace(" ** 2", "").strip()
            proxy = _proxy_for(name)
            if proxy is not None:
                columns.append(proxy * proxy)
                regressor_to_col[regressor] = len(columns) - 1
        elif " * " in regressor:
            a, b = (s.strip() for s in regressor.split(" * "))
            pa, pb = _proxy_for(a), _proxy_for(b)
            if pa is not None and pb is not None:
                columns.append(pa * pb)
                regressor_to_col[regressor] = len(columns) - 1
        else:
            proxy = _proxy_for(regressor)
            if proxy is not None:
                columns.append(proxy)
                regressor_to_col[regressor] = len(columns) - 1

    if not columns:
        return None, {}
    design = np.column_stack(columns)
    return design, regressor_to_col
