"""Stage 3 of the AMN estimator: simulate latent factors and regress.

Draws a synthetic latent-factor panel from the structural mixture
fitted in Stage 2 and recovers the per-period transition / investment
parameters by least-squares regression (linear for linear transitions
and the investment equation; Levenberg-Marquardt NLS for `log_ces` and
`log_ces_with_constant`).

Mirrors the Stage 3 logic in
`Monte Carlo Simulations/master_approx_simulationces2periodrho_5.R`.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from skillmodels.amn.types import (
    MinimumDistanceResult,
    ProductionFitResult,
)
from skillmodels.common.types import ProcessedModel


def _draw_factor_panel(
    structural: MinimumDistanceResult,
    mixture_weights: np.ndarray,
    *,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Sample ``n_draws`` rows from the K-component Gaussian mixture.

    Returns a DataFrame with one column per ``(period, factor)`` slot.
    """
    rng = np.random.default_rng(seed)
    means = structural.factor_mixture_means
    covs = structural.factor_mixture_covariances
    n_components, n_factor = means.shape

    counts = np.floor(n_draws * mixture_weights).astype(int)
    deficit = n_draws - counts.sum()
    if deficit > 0:
        order = np.argsort(-(n_draws * mixture_weights - counts))
        for idx in order[:deficit]:
            counts[idx] += 1

    chunks = []
    for k in range(n_components):
        if counts[k] == 0:
            continue
        cov = covs[k]
        cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(n_factor)
        samples = rng.multivariate_normal(means[k], cov, size=counts[k])
        chunks.append(samples)
    panel = np.vstack(chunks)
    rng.shuffle(panel)

    columns = [f"f[{t}|{f}]" for t, f in structural.factor_period_slots]
    return pd.DataFrame(panel, columns=columns)


def _slot_column(period: int, factor: str) -> str:
    return f"f[{period}|{factor}]"


def _fit_linear(
    y: np.ndarray,
    x_design: np.ndarray,
    regressor_names: list[str],
) -> tuple[dict[str, float], float]:
    """OLS regression with an intercept (added as the last column).

    Returns:
        ``(params_by_name, residual_sd)`` with `constant` included as
        the trailing parameter.

    """
    n = x_design.shape[0]
    full_design = np.column_stack([x_design, np.ones(n)])
    coefs, *_ = np.linalg.lstsq(full_design, y, rcond=None)
    resid = y - full_design @ coefs
    sd = float(np.sqrt(np.mean(resid**2)))
    out = dict(zip([*regressor_names, "constant"], coefs.tolist(), strict=True))
    return out, sd


def _fit_log_ces(
    y: np.ndarray,
    x_design: np.ndarray,
    regressor_names: list[str],
    *,
    with_constant: bool,
) -> tuple[dict[str, float], float]:
    """Fit log_ces (or log_ces_with_constant) via Levenberg-Marquardt.

    Parametrises ``y = delta + (1/rho) * log(sum_i gamma_i * exp(X_i * rho))``
    with gammas constrained to the simplex via softmax. When
    ``with_constant=False``, the additive ``delta`` is held at 0.
    """
    n_reg = len(regressor_names)
    eps = 1e-12

    def residuals(theta: np.ndarray) -> np.ndarray:
        logits = np.concatenate([theta[: n_reg - 1], [0.0]])
        gammas = np.exp(logits - logits.max())
        gammas = gammas / gammas.sum()
        rho = theta[n_reg - 1]
        constant = theta[n_reg] if with_constant else 0.0
        exponents = x_design * rho
        max_exp = np.max(exponents, axis=1, keepdims=True)
        shifted = np.exp(exponents - max_exp)
        log_inside = np.log(np.clip((gammas * shifted).sum(axis=1), eps, None))
        pred = constant + (max_exp[:, 0] + log_inside) / rho
        return pred - y

    n_unknowns = n_reg + (1 if with_constant else 0)
    theta0 = np.zeros(n_unknowns)
    theta0[n_reg - 1] = 0.5
    result = least_squares(residuals, theta0, method="lm", max_nfev=2000)
    theta = result.x
    logits = np.concatenate([theta[: n_reg - 1], [0.0]])
    gammas = np.exp(logits - logits.max())
    gammas = gammas / gammas.sum()
    rho = float(theta[n_reg - 1])
    constant = float(theta[n_reg]) if with_constant else 0.0
    resid = residuals(theta)
    sd = float(np.sqrt(np.mean(resid**2)))

    out: dict[str, float] = dict(zip(regressor_names, gammas.tolist(), strict=True))
    out["phi"] = rho
    if with_constant:
        out["constant"] = constant
    return out, sd


def _fit_transition(
    transition_name: str,
    y: np.ndarray,
    x_design: np.ndarray,
    regressor_names: list[str],
) -> tuple[dict[str, float], float]:
    if transition_name == "linear":
        return _fit_linear(y, x_design, regressor_names)
    if transition_name == "log_ces":
        return _fit_log_ces(y, x_design, regressor_names, with_constant=False)
    if transition_name == "log_ces_with_constant":
        return _fit_log_ces(y, x_design, regressor_names, with_constant=True)
    msg = (
        f"AMN Stage 3 does not yet support transition function "
        f"'{transition_name}'. Supported: linear, log_ces, "
        f"log_ces_with_constant."
    )
    raise NotImplementedError(msg)


def _factors_at_period(processed_model: ProcessedModel) -> tuple[str, ...]:
    """Latent + observed factor names (used as transition regressors)."""
    return (
        *processed_model.labels.latent_factors,
        *processed_model.labels.observed_factors,
    )


def simulate_and_regress(  # noqa: C901
    structural: MinimumDistanceResult,
    processed_model: ProcessedModel,
    mixture_weights: np.ndarray,
    *,
    n_draws: int = 100_000,
    seed: int = 0,
    investment_endogeneity: bool = True,
) -> ProductionFitResult:
    """Simulate the joint latent-factor distribution and run Stage-3 regressions.

    Args:
        structural: Stage 2 output (structural mixture, loadings, etc.).
        processed_model: Skillmodels processed model.
        mixture_weights: Per-component mixture weights from Stage 1.
        n_draws: Synthetic-panel size.
        seed: RNG seed.
        investment_endogeneity: Reserved for future control-function
            extension; currently the investment equation is fit with
            plain OLS regardless.

    Return:
        ProductionFitResult with production-function and investment-equation
        parameter DataFrames.

    """
    del investment_endogeneity  # placeholder; control function is v2

    panel = _draw_factor_panel(structural, mixture_weights, n_draws=n_draws, seed=seed)

    periods = processed_model.labels.periods
    endog_info = processed_model.endogenous_factors_info
    transition_info = processed_model.transition_info
    factor_to_function_name = (
        dict(transition_info.function_names) if transition_info is not None else {}
    )

    transition_rows: list[tuple[str, int, str, str, float]] = []
    investment_rows: list[tuple[str, int, str, str, float]] = []

    for t_idx in range(len(periods) - 1):
        t = int(periods[t_idx])
        t_next = int(periods[t_idx + 1])
        factor_names = _factors_at_period(processed_model)
        regressor_cols = [_slot_column(t, f) for f in factor_names]
        present_pairs = [
            (f, c)
            for f, c in zip(factor_names, regressor_cols, strict=True)
            if c in panel.columns
        ]
        if not present_pairs:
            continue
        present_factor_names = [f for f, _ in present_pairs]
        x_design = panel[[c for _, c in present_pairs]].to_numpy()

        for factor in processed_model.labels.latent_factors:
            is_endog = (
                factor in endog_info.factor_info
                and endog_info.factor_info[factor].is_endogenous
            )
            target_col = _slot_column(t_next, factor)
            if target_col not in panel.columns:
                continue
            y = panel[target_col].to_numpy()
            trans_name = factor_to_function_name.get(factor, "linear")
            if trans_name == "constant":
                continue
            if is_endog:
                params, sd = _fit_linear(y, x_design, present_factor_names)
                for regname, value in params.items():
                    investment_rows.append(
                        ("investment_eq", t, factor, regname, float(value))
                    )
                investment_rows.append(("investment_sds", t, factor, "-", sd))
            else:
                params, sd = _fit_transition(
                    trans_name, y, x_design, present_factor_names
                )
                for regname, value in params.items():
                    transition_rows.append(
                        ("transition", t, factor, regname, float(value))
                    )
                transition_rows.append(("shock_sds", t, factor, "-", sd))

    def _rows_to_df(
        rows: list[tuple[str, int, str, str, float]],
    ) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                {"value": []},
                index=pd.MultiIndex.from_tuples(
                    [], names=["category", "aug_period", "name1", "name2"]
                ),
            )
        index = pd.MultiIndex.from_tuples(
            [(c, p, n1, n2) for c, p, n1, n2, _ in rows],
            names=["category", "aug_period", "name1", "name2"],
        )
        values = [v for *_, v in rows]
        return pd.DataFrame({"value": values}, index=index)

    return ProductionFitResult(
        production_params=_rows_to_df(transition_rows),
        investment_params=_rows_to_df(investment_rows),
        n_draws=n_draws,
        seed=seed,
    )
