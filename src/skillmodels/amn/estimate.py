"""AMN (Attanasio-Meghir-Nix 2020) point-estimate estimator.

Three-step procedure:

1. **Measurement system via Spearman cross-covariances.** Reuses
   `skillmodels.af.measurement_first_stage.estimate_measurement_system`
   to recover per-period loadings and measurement-error SDs.
2. **Bartlett factor proxies.** For each `(period, factor)` build an
   inverse-noise-weighted proxy
   ``F_hat_{i,t} = sum_k (lambda_k / sigma_k^2) Z_{i,k,t}
                 / sum_k (lambda_k^2 / sigma_k^2)``.
   The proxy has measurement-error variance
   ``sigma_eta^2 = 1 / sum_k (lambda_k^2 / sigma_k^2)``.
3. **OLS with errors-in-variables (EIV) correction.** For each
   transition equation (next-period factor proxy regressed on
   current-period proxies plus observed factors), run

       beta_corrected = ((X'X / n) - Sigma_eta)^(-1) (X'y / n)

   where `Sigma_eta` is the diagonal cov matrix of the regressors'
   measurement noise. The EIV correction is applied to linear
   regressors only; product regressors (e.g., `skills * investment`
   in translog) keep the naive OLS coefficient because the noise
   structure of a product of proxies is non-standard. The shock SD
   is recovered from the OLS residual variance minus the dependent
   proxy's measurement-error variance.

The result is a point estimate, not a starting value. Compare to
`estimate_af` (joint Halton MLE) or `get_maximization_inputs` →
`estimate_ml` (CHS Kalman MLE). AMN is far cheaper but biased on
nonlinear transition coefficients (translog cross-terms) because the
EIV correction does not extend to them.
"""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from skillmodels.af.measurement_first_stage import estimate_measurement_system
from skillmodels.amn.types import AMNEstimationOptions, AMNEstimationResult
from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.model_spec import ModelSpec
from skillmodels.process_data import process_data
from skillmodels.process_model import process_model
from skillmodels.types import EstimationOptions, ProcessedModel


def _options_with_strategy_none(model_spec: ModelSpec) -> EstimationOptions:
    base = model_spec.estimation_options
    if base is None:
        return EstimationOptions(start_params_strategy="none")
    return replace(base, start_params_strategy="none")


def estimate_amn(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    amn_options: AMNEstimationOptions | None = None,
    fixed_params: pd.DataFrame | None = None,
) -> AMNEstimationResult:
    """Estimate a latent factor model via Attanasio-Meghir-Nix (2020).

    Args:
        model_spec: Standard skillmodels `ModelSpec`.
        data: Long-format panel.
        amn_options: AMN-specific configuration. Defaults to
            `AMNEstimationOptions(use_bias_correction=True)`.
        fixed_params: Optional user-supplied pins. Overlapping rows
            are written into the returned `params` after the AMN
            point estimates.

    Return:
        `AMNEstimationResult` carrying point estimates packed into a
        skillmodels-shaped params DataFrame.

    """
    if amn_options is None:
        amn_options = AMNEstimationOptions()

    processed_model = process_model(model_spec)
    processed_data = process_data(
        df=data,
        has_endogenous_factors=processed_model.endogenous_factors_info.has_endogenous_factors,
        labels=processed_model.labels,
        update_info=processed_model.update_info,
        anchoring_info=processed_model.anchoring,
        purpose="estimation",
    )

    measurement_system = estimate_measurement_system(
        model_spec=model_spec,
        data=data,
        user_fixed_params=fixed_params,
    )

    measurements = np.asarray(processed_data["measurements"])
    observed_factor_data = np.asarray(processed_data["observed_factors"])

    proxies, proxy_var = _build_factor_proxies(
        measurements=measurements,
        measurement_system=measurement_system,
        processed_model=processed_model,
    )

    # Build a NaN template so AMN's point estimates land cleanly; the
    # moment-init template (default of `get_maximization_inputs`) would
    # pre-fill values, and we'd then need to distinguish "AMN estimated
    # this" from "moment-init seeded this".
    template_spec = model_spec.with_estimation_options(
        _options_with_strategy_none(model_spec)
    )
    template = get_maximization_inputs(
        model_spec=template_spec, data=data, fixed_params=fixed_params
    )["params_template"]
    out = template.copy()
    out = _write_measurement_system(out, measurement_system)

    out, diagnostics = _fit_transition_equations(
        out=out,
        processed_model=processed_model,
        proxies=proxies,
        proxy_var=proxy_var,
        observed_factor_data=observed_factor_data,
        amn_options=amn_options,
    )

    out = _apply_neutral_defaults(out, processed_model)
    if fixed_params is not None:
        for loc in fixed_params.index:
            if loc in out.index:
                out.loc[loc, "value"] = float(fixed_params.loc[loc, "value"])

    return AMNEstimationResult(
        params=out,
        measurement_system=measurement_system,
        factor_proxies=proxies,
        proxy_meas_err_var=proxy_var,
        n_obs=int(measurements.shape[1]) if measurements.ndim == 2 else 0,
        regression_diagnostics=diagnostics,
    )


def _fit_transition_equations(
    *,
    out: pd.DataFrame,
    processed_model: ProcessedModel,
    proxies: dict[tuple[int, str], np.ndarray],
    proxy_var: dict[tuple[int, str], float],
    observed_factor_data: np.ndarray,
    amn_options: AMNEstimationOptions,
) -> tuple[pd.DataFrame, dict[tuple[int, str], dict]]:
    """Run AMN regressions for every transition equation and write results."""
    diagnostics: dict[tuple[int, str], dict] = {}
    aug_periods = processed_model.labels.aug_periods
    latent_factors = processed_model.labels.latent_factors

    for src_idx, src_aug in enumerate(aug_periods[:-1]):
        tgt_aug = aug_periods[src_idx + 1]
        cal_src = _aug_to_calendar(processed_model, src_aug)
        for factor in latent_factors:
            func_name = processed_model.transition_info.function_names.get(factor)
            if func_name not in ("linear", "translog"):
                continue
            if (tgt_aug, factor) not in proxies:
                continue

            beta, beta_meta = _run_amn_regression(
                src_aug=src_aug,
                tgt_aug=tgt_aug,
                factor=factor,
                processed_model=processed_model,
                proxies=proxies,
                proxy_var=proxy_var,
                observed_factor_data=observed_factor_data,
                cal_src=cal_src,
                amn_options=amn_options,
            )
            if beta is None:
                continue
            _write_transition_estimates(
                out=out,
                src_aug=src_aug,
                factor=factor,
                beta=beta,
                shock_sd=beta_meta.get("shock_sd"),
            )
            diagnostics[(src_aug, factor)] = beta_meta

    return out, diagnostics


def _write_transition_estimates(
    *,
    out: pd.DataFrame,
    src_aug: int,
    factor: str,
    beta: Mapping[str, float],
    shock_sd: float | None,
) -> None:
    """Write per-equation AMN estimates back into the params template."""
    for regressor, value in beta.items():
        loc = ("transition", src_aug, factor, regressor)
        if loc in out.index and pd.isna(out.loc[loc, "value"]):
            out.loc[loc, "value"] = float(value)
    if shock_sd is not None:
        loc_sd = ("shock_sds", src_aug, factor, "-")
        if loc_sd in out.index and pd.isna(out.loc[loc_sd, "value"]):
            out.loc[loc_sd, "value"] = float(shock_sd)


def _build_factor_proxies(
    *,
    measurements: np.ndarray,
    measurement_system: pd.DataFrame,
    processed_model: ProcessedModel,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[tuple[int, str], float]]:
    """Build Bartlett-scored factor proxies for every `(aug_period, factor)`.

    Returns the proxy array (shape `n_obs`) and its measurement-error
    variance `sigma_eta^2 = 1 / sum_k (lambda_k^2 / sigma_k^2)`.
    """
    update_info = processed_model.update_info
    latent_factors = processed_model.labels.latent_factors
    aug_periods = processed_model.labels.aug_periods

    proxies: dict[tuple[int, str], np.ndarray] = {}
    proxy_var: dict[tuple[int, str], float] = {}

    update_info_periods = set(update_info.index.get_level_values("aug_period"))

    for aug_period in aug_periods:
        if aug_period not in update_info_periods:
            continue
        period_rows = update_info.xs(aug_period, level="aug_period")
        measurement_rows = period_rows.loc[period_rows["purpose"] == "measurement"]
        for factor in latent_factors:
            factor_meas = tuple(
                str(m)
                for m, row in measurement_rows.iterrows()
                if bool(row[factor])
                and not any(bool(row[f]) for f in latent_factors if f != factor)
            )
            if len(factor_meas) < 2:
                continue
            cols = []
            loadings = []
            sigmas = []
            for m in factor_meas:
                cols.append(_row_index(update_info, aug_period, m))
                loc_load = ("loadings", aug_period, m, factor)
                loc_sd = ("meas_sds", aug_period, m, "-")
                if loc_load not in measurement_system.index:
                    break
                loadings.append(float(measurement_system.loc[loc_load, "value"]))  # ty: ignore[invalid-argument-type]
                sigmas.append(float(measurement_system.loc[loc_sd, "value"]))  # ty: ignore[invalid-argument-type]
            if len(cols) != len(factor_meas):
                continue
            lam = np.asarray(loadings, dtype=float)
            sig = np.maximum(np.asarray(sigmas, dtype=float), 1e-6)
            weights_unnorm = lam / sig**2
            denom = float(np.sum(weights_unnorm * lam))
            if denom < 1e-9:
                continue
            sub = measurements[cols, :].T  # (n_obs, n_meas)
            mask = np.all(np.isfinite(sub), axis=1)
            proxy = np.full(sub.shape[0], np.nan)
            proxy[mask] = (sub[mask] * weights_unnorm).sum(axis=1) / denom
            proxies[(aug_period, factor)] = proxy
            proxy_var[(aug_period, factor)] = 1.0 / denom

    return proxies, proxy_var


def _run_amn_regression(  # noqa: C901, PLR0912, PLR0915
    *,
    src_aug: int,
    tgt_aug: int,
    factor: str,
    processed_model: ProcessedModel,
    proxies: dict[tuple[int, str], np.ndarray],
    proxy_var: dict[tuple[int, str], float],
    observed_factor_data: np.ndarray,
    cal_src: int | None,
    amn_options: AMNEstimationOptions,
) -> tuple[dict[str, float] | None, dict]:
    """Run the EIV-corrected OLS for one transition equation."""
    param_names = processed_model.transition_info.param_names[factor]
    observed_factor_names = processed_model.labels.observed_factors

    target = proxies[(tgt_aug, factor)]

    obs_at_src = (
        observed_factor_data[cal_src]
        if cal_src is not None
        and observed_factor_data.ndim == 3
        and cal_src < observed_factor_data.shape[0]
        else np.zeros((target.shape[0], 0))
    )

    columns: list[np.ndarray] = []
    column_names: list[str] = []
    column_eiv_var: list[float] = []  # diagonal entries of Sigma_eta
    is_product: list[bool] = []

    def _proxy_for(name: str) -> tuple[np.ndarray | None, float]:
        if (src_aug, name) in proxies:
            return proxies[(src_aug, name)], proxy_var[(src_aug, name)]
        if name in observed_factor_names:
            idx = observed_factor_names.index(name)
            if obs_at_src.shape[1] > idx:
                return obs_at_src[:, idx], 0.0
        return None, 0.0

    for regressor in param_names:
        if regressor == "constant":
            columns.append(np.ones_like(target))
            column_names.append(regressor)
            column_eiv_var.append(0.0)
            is_product.append(False)
        elif " ** 2" in regressor:
            name = regressor.replace(" ** 2", "").strip()
            proxy, _ = _proxy_for(name)
            if proxy is None:
                continue
            columns.append(proxy * proxy)
            column_names.append(regressor)
            column_eiv_var.append(0.0)
            is_product.append(True)
        elif " * " in regressor:
            a, b = (s.strip() for s in regressor.split(" * "))
            pa, _ = _proxy_for(a)
            pb, _ = _proxy_for(b)
            if pa is None or pb is None:
                continue
            columns.append(pa * pb)
            column_names.append(regressor)
            column_eiv_var.append(0.0)
            is_product.append(True)
        else:
            proxy, var = _proxy_for(regressor)
            if proxy is None:
                continue
            columns.append(proxy)
            column_names.append(regressor)
            column_eiv_var.append(var)
            is_product.append(False)

    if not columns:
        return None, {"n_used": 0}

    design = np.column_stack(columns)
    mask = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    n_used = int(mask.sum())
    if n_used <= design.shape[1] + 1:
        return None, {"n_used": n_used}

    x = design[mask]
    y = target[mask]
    n = float(n_used)
    xtx_over_n = (x.T @ x) / n
    xty_over_n = (x.T @ y) / n

    sigma_eta = np.zeros_like(xtx_over_n)
    if amn_options.use_bias_correction:
        for i, var in enumerate(column_eiv_var):
            if not is_product[i]:
                sigma_eta[i, i] = float(var)

    adjusted = xtx_over_n - sigma_eta
    try:
        sv = np.linalg.svd(adjusted, compute_uv=False)
        min_sv = float(sv.min()) if sv.size else 0.0
    except np.linalg.LinAlgError:
        min_sv = 0.0

    if min_sv < amn_options.fail_below_min_singular_value:
        return None, {"n_used": n_used, "min_singular_value": min_sv}

    try:
        beta_vec = np.linalg.solve(adjusted, xty_over_n)
    except np.linalg.LinAlgError:
        return None, {"n_used": n_used, "min_singular_value": min_sv}

    residual = y - x @ beta_vec
    residual_var = float(np.var(residual, ddof=max(design.shape[1], 1)))
    target_eiv_var = float(proxy_var.get((tgt_aug, factor), 0.0))
    shock_var = max(residual_var - target_eiv_var, amn_options.sd_floor**2)
    shock_sd = float(np.sqrt(shock_var))

    beta = dict(zip(column_names, beta_vec.tolist(), strict=True))
    diagnostics = {
        "n_used": n_used,
        "min_singular_value": min_sv,
        "residual_var": residual_var,
        "shock_sd": shock_sd,
        "target_eiv_var": target_eiv_var,
    }
    return beta, diagnostics


def _row_index(update_info: pd.DataFrame, aug_period: int, meas: str) -> int:
    """Flat-row index of `(aug_period, meas)` in `update_info`."""
    for flat_idx, (a_period, m) in enumerate(update_info.index):
        if a_period == aug_period and m == meas:
            return flat_idx
    msg = f"Measurement {meas!r} not found at aug_period {aug_period}"
    raise KeyError(msg)


def _aug_to_calendar(processed_model: ProcessedModel, aug_period: int) -> int | None:
    mapping: Mapping[int, int] = processed_model.labels.aug_periods_to_periods
    cal = mapping.get(aug_period)
    if cal is None:
        return None
    return int(cal)


def _write_measurement_system(
    params: pd.DataFrame, measurement_system: pd.DataFrame
) -> pd.DataFrame:
    """Copy loading + meas_sds + intercept entries into params."""
    out = params.copy()
    for loc in measurement_system.index:
        if loc not in out.index:
            continue
        out.loc[loc, "value"] = float(measurement_system.loc[loc, "value"])
    return out


def _apply_neutral_defaults(
    params: pd.DataFrame, processed_model: ProcessedModel
) -> pd.DataFrame:
    """Fill remaining NaN rows with sensible defaults for downstream consumers.

    AMN does not estimate initial-distribution or mixture parameters;
    those fall back to 0 / uniform mixture / unit cov diagonals.
    """
    out = params.copy()
    n_mixtures = processed_model.dimensions.n_mixtures
    cat = out.index.get_level_values("category")
    na = out["value"].isna()
    out.loc[na & (cat == "controls"), "value"] = 0.0
    out.loc[na & (cat == "loadings"), "value"] = 1.0
    out.loc[na & (cat == "meas_sds"), "value"] = 0.5
    out.loc[na & (cat == "shock_sds"), "value"] = 0.5
    out.loc[na & (cat == "initial_states"), "value"] = 0.0
    out.loc[na & (cat == "mixture_weights"), "value"] = 1.0 / max(n_mixtures, 1)
    out.loc[na & (cat == "initial_cholcovs"), "value"] = 0.0
    out.loc[na & (cat == "transition"), "value"] = 0.0
    diag_mask = pd.Series(
        [
            idx[0] == "initial_cholcovs"
            and "-" in idx[3]
            and idx[3].split("-")[0] == idx[3].split("-")[1]
            for idx in out.index
        ],
        index=out.index,
    )
    out.loc[out["value"].isna() & diag_mask, "value"] = 1.0
    return out
