"""Stage 3 of the AMN estimator: simulate latent factors and regress.

Draws a synthetic latent-factor panel from the structural mixture
fitted in Stage 2 and recovers the per-period transition / investment
parameters by least-squares regression.

Specialised fitters: closed-form OLS for `linear`; softmax-constrained
Levenberg-Marquardt for `log_ces` and `log_ces_with_constant` (keeps
gammas on the simplex). Everything else (translog, robust_translog,
linear_and_squares, log_ces_general, and any user
`@register_params`-decorated transition) goes through a generic NLS
path that calls the transition function directly via `jax.vmap`. This
mirrors the per-factor NLS in
`Monte Carlo Simulations/master_approx_simulationces2periodrho_5.R`
but generalises beyond the paper's CES-only case.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from skillmodels.amn.types import (
    MinimumDistanceResult,
    ProductionFitResult,
)
from skillmodels.common.model_spec import ModelSpec
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


def _fit_investment_residual(
    y_invest: np.ndarray,
    x_determinants: np.ndarray,
    determinant_names: list[str],
) -> tuple[dict[str, float], float, np.ndarray]:
    """First-stage investment equation for the control-function correction.

    OLS of the (latent) log-investment factor on its contemporaneous
    determinants -- the production-input state factors theta_t and the
    excluded observed instruments Y_t (income/prices) -- with an intercept:

        ln I_t = beta_0 + beta_theta . theta_t + beta_Y . Y_t + eta_{I,t}.

    The in-sample OLS residual `eta_{I,t} = ln I_t - E[ln I_t | theta_t, Y_t]`
    is the control function added to the production regression (AMN 2020
    eq. 7-8 / AF Sec. 3.5, Assumption 1(g)). Returns the coefficients
    (keyed by ``determinant_names + ["constant"]``), the residual SD
    (= SD(eta_{I,t})), and the residual vector of shape ``(n_draws,)``.
    """
    n = x_determinants.shape[0]
    full_design = np.column_stack([x_determinants, np.ones(n)])
    coefs, *_ = np.linalg.lstsq(full_design, y_invest, rcond=None)
    resid = y_invest - full_design @ coefs
    sd = float(np.sqrt(np.mean(resid**2)))
    out = dict(zip([*determinant_names, "constant"], coefs.tolist(), strict=True))
    return out, sd, resid


def _fit_log_ces(
    y: np.ndarray,
    x_design: np.ndarray,
    regressor_names: list[str],
    *,
    with_constant: bool,
    cf: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """Fit log_ces (or log_ces_with_constant) via Levenberg-Marquardt.

    Parametrises ``y = delta + (1/rho) * log(sum_i gamma_i * exp(X_i * rho))``
    with gammas constrained to the simplex via softmax. When
    ``with_constant=False``, the additive ``delta`` is held at 0.

    If ``cf`` is given (the investment control-function residual), an extra
    additive linear term ``kappa * cf`` is fitted OUTSIDE the CES aggregator
    (AMN's ``+ alpha*cf``), with ``kappa`` (= the period-specific endogeneity
    coefficient) estimated jointly with the CES parameters and returned under
    the ``"cf"`` key.
    """
    n_reg = len(regressor_names)
    eps = 1e-12
    has_cf = cf is not None
    kappa_idx = n_reg + (1 if with_constant else 0)

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
        if has_cf:
            pred = pred + theta[kappa_idx] * cf
        return pred - y

    n_unknowns = n_reg + (1 if with_constant else 0) + (1 if has_cf else 0)
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
    if has_cf:
        out["cf"] = float(theta[kappa_idx])
    return out, sd


def _make_user_transition_callable(
    user_func: Callable,
    factor_names: tuple[str, ...],
    param_names: tuple[str, ...],
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Wrap a `@register_params`-decorated user function as `(states, params)`.

    Mirrors `skillmodels.af.transition_period._wrap_registered_transition_function`
    so Stage 3 can pass user transitions through `jax.vmap` for NLS.
    """
    sig = inspect.signature(user_func)
    arg_names = [name for name in sig.parameters if name != "params"]
    arg_positions = tuple(factor_names.index(name) for name in arg_names)

    def wrapped(states: jnp.ndarray, params_vec: jnp.ndarray) -> jnp.ndarray:
        kwargs: dict[str, jnp.ndarray | dict[str, jnp.ndarray]] = {
            name: states[pos]
            for name, pos in zip(arg_names, arg_positions, strict=True)
        }
        kwargs["params"] = dict(zip(param_names, params_vec, strict=True))
        return user_func(**kwargs)

    return wrapped


def _resolve_transition_callable(
    transition_name: str,
    factor: str,
    processed_model: ProcessedModel,
    model_spec: ModelSpec,
) -> tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], tuple[str, ...]]:
    """Return a ``(states, params) -> scalar`` callable plus param names.

    For built-in transitions this is the function imported from
    `skillmodels.common.transition_functions`; for user functions it is
    `_make_user_transition_callable(...)` applied to the raw callable on
    the model spec.
    """
    from skillmodels.common import transition_functions as tf  # noqa: PLC0415

    builtin_names = {
        "linear",
        "translog",
        "robust_translog",
        "linear_and_squares",
        "log_ces",
        "log_ces_with_constant",
        "log_ces_general",
    }
    factor_names = (
        *processed_model.labels.latent_factors,
        *processed_model.labels.observed_factors,
    )
    transition_info = processed_model.transition_info
    if transition_info is None:
        msg = "ProcessedModel has no transition_info; cannot run Stage 3."
        raise ValueError(msg)
    param_names = tuple(transition_info.param_names[factor])

    if transition_name in builtin_names:
        func = getattr(tf, transition_name)
        return func, param_names

    factor_spec = model_spec.factors.get(factor)
    if factor_spec is None:
        msg = (
            f"Cannot resolve transition callable for factor '{factor}' "
            f"(transition='{transition_name}'). Factor not found on "
            "model_spec.factors."
        )
        raise KeyError(msg)
    raw = factor_spec.transition_function
    if not callable(raw):
        msg = (
            f"Factor '{factor}' has transition_function={raw!r} which is "
            "neither a built-in name nor a callable."
        )
        raise TypeError(msg)
    wrapped = _make_user_transition_callable(raw, factor_names, param_names)
    return wrapped, param_names


def _seed_generic_nls_theta0(
    param_names: tuple[str, ...],
    init_overrides: dict[str, float],
    *,
    n_unknowns: int,
) -> np.ndarray:
    """Seed the NLS start vector for `_fit_generic_nls`.

    Applies `init_overrides`, then seeds elasticity-style params ("phi",
    "rho", "sigma") at 0.5 so CES / general-CES log expressions do not
    divide by zero, and gives the remaining (simplex-style "gamma") params a
    uniform initial share when the function looks CES-shaped. The trailing
    cf-coefficient slot (if `n_unknowns > len(param_names)`) stays at zero.
    """
    theta0 = np.zeros(n_unknowns)
    for name, val in init_overrides.items():
        if name in param_names:
            theta0[param_names.index(name)] = val
    for j, name in enumerate(param_names):
        if name in {"phi", "rho", "sigma"} and name not in init_overrides:
            theta0[j] = 0.5
    has_elasticity = any(n in {"phi", "rho", "sigma"} for n in param_names)
    if has_elasticity:
        share_candidates = [
            j
            for j, n in enumerate(param_names)
            if n not in {"phi", "rho", "sigma", "constant"}
        ]
        if share_candidates:
            theta0[share_candidates] = 1.0 / len(share_candidates)
    return theta0


def _fit_generic_nls(
    transition_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    param_names: tuple[str, ...],
    y: np.ndarray,
    states_panel: np.ndarray,
    *,
    init_overrides: dict[str, float] | None = None,
    cf: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """Generic Levenberg-Marquardt NLS via `jax.vmap` over the panel.

    Works for any `(states, params) -> scalar` callable, including
    translog, robust_translog, linear_and_squares, log_ces_general, and
    user-registered transitions.

    Args:
        transition_func: callable taking a 1D state vector and a 1D
            param vector and returning a scalar.
        param_names: names of the parameters in the order accepted by
            `transition_func`.
        y: target vector, shape ``(n_obs,)``.
        states_panel: state matrix, shape ``(n_obs, n_state_features)``.
        init_overrides: optional ``{name: value}`` to seed specific
            parameters before NLS. Useful for setting `phi != 0` on
            log_ces-family functions.
        cf: optional investment control-function residual, shape
            ``(n_obs,)``. When given, an extra additive linear term
            ``kappa * cf`` is fitted OUTSIDE the transition aggregator
            (AMN's ``+ alpha*cf``); ``kappa`` is appended as an unknown
            (init 0) and returned under the ``"cf"`` key. ``states_panel``
            stays the production states (cf is a separate, additive term).

    """
    init_overrides = init_overrides or {}
    has_cf = cf is not None
    kappa_idx = len(param_names)

    @jax.jit
    def predict_batch(theta: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(transition_func, in_axes=(0, None))(states, theta)

    states_jnp = jnp.asarray(states_panel)

    def residuals(theta_np: np.ndarray) -> np.ndarray:
        trans_theta = jnp.asarray(theta_np[: len(param_names)])
        preds = np.asarray(predict_batch(trans_theta, states_jnp))
        if has_cf:
            preds = preds + theta_np[kappa_idx] * cf
        return preds - y

    n_unknowns = len(param_names) + (1 if has_cf else 0)
    theta0 = _seed_generic_nls_theta0(
        param_names, init_overrides, n_unknowns=n_unknowns
    )

    result = least_squares(residuals, theta0, method="lm", max_nfev=5000)
    theta = result.x
    resid = residuals(theta)
    sd = float(np.sqrt(np.mean(resid**2)))
    out = dict(
        zip(param_names, [float(v) for v in theta[: len(param_names)]], strict=True)
    )
    if has_cf:
        out["cf"] = float(theta[kappa_idx])
    return out, sd


def _fit_transition(
    transition_name: str,
    factor: str,
    processed_model: ProcessedModel,
    model_spec: ModelSpec,
    y: np.ndarray,
    x_design: np.ndarray,
    regressor_names: list[str],
    *,
    cf: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """Dispatch to the right per-transition fitter.

    `linear` and `log_ces`-family functions get specialised fitters for
    speed / simplex constraints; everything else (translog,
    robust_translog, linear_and_squares, log_ces_general, user) falls
    through to a generic `jax.vmap`-based NLS.

    When ``cf`` is given (the investment control-function residual), an
    extra additive linear term ``kappa * cf`` is fitted jointly with the
    transition parameters and returned under the ``"cf"`` key. For
    `linear` the residual is just an extra design column; for the
    `log_ces`-family and the generic NLS path it is added OUTSIDE the
    aggregator (AMN 2020 eq. 7-8 / AF Sec. 3.5).
    """
    if transition_name == "linear":
        if cf is not None:
            x_design = np.column_stack([x_design, cf])
            regressor_names = [*regressor_names, "cf"]
        return _fit_linear(y, x_design, regressor_names)
    if transition_name == "log_ces":
        return _fit_log_ces(y, x_design, regressor_names, with_constant=False, cf=cf)
    if transition_name == "log_ces_with_constant":
        return _fit_log_ces(y, x_design, regressor_names, with_constant=True, cf=cf)

    func, param_names = _resolve_transition_callable(
        transition_name, factor, processed_model, model_spec
    )
    return _fit_generic_nls(func, param_names, y, x_design, cf=cf)


def _factors_at_period(processed_model: ProcessedModel) -> tuple[str, ...]:
    """Latent + observed factor names (used as transition regressors)."""
    return (
        *processed_model.labels.latent_factors,
        *processed_model.labels.observed_factors,
    )


def _fit_first_stage_investment(
    panel: pd.DataFrame,
    period: int,
    *,
    present_investment: list[str],
    latent_present_names: list[str],
    present_observed: list[str],
) -> tuple[dict[str, np.ndarray], list[tuple[str, int, str, str, float]]]:
    """Fit the contemporaneous investment equation(s) (AMN eq. 7).

    For each present investment factor, regress its (latent) log-investment
    value on the present production-input state factors plus the excluded
    observed instruments, and keep the in-sample residual as the control
    function for the production step (AMN 2020 eq. 7-8 / AF Sec. 3.5).

    Args:
        panel: Simulated latent-factor panel.
        period: Calendar period `t` (the `aug_period` index emitted on rows).
        present_investment: Investment factors present at `period`.
        latent_present_names: Latent factors present at `period` (state +
            investment), in design order.
        present_observed: Observed factors present at `period` (the excluded
            instruments).

    Return:
        Tuple `(cf_by_factor, investment_rows)` where `cf_by_factor` maps each
        investment factor to its residual control-function vector and
        `investment_rows` are the `investment_eq` / `investment_sds` param
        rows.

    Raises:
        NotImplementedError: If more than one investment factor is present
            (the control-function choice is ambiguous).
        ValueError: If no observed instrument is present (the residual would
            be collinear with the production inputs).

    """
    if len(present_investment) > 1:
        msg = (
            "AMN investment_endogeneity with more than one present investment "
            f"factor at period {period} is unsupported: the control-function "
            "choice for the state-factor production regressions is ambiguous."
        )
        raise NotImplementedError(msg)

    cf_by_factor: dict[str, np.ndarray] = {}
    investment_rows: list[tuple[str, int, str, str, float]] = []
    for inv_factor in present_investment:
        if not present_observed:
            msg = (
                "AMN investment_endogeneity requires at least one present "
                f"observed factor at period {period} to identify the "
                f"control-function coefficient for '{inv_factor}': without an "
                "excluded instrument the residual eta_{I,t} is collinear with "
                "the production inputs (theta_t, I_t). Add an observed factor "
                "(income/prices) or set investment_endogeneity=False."
            )
            raise ValueError(msg)
        state_determinants = [f for f in latent_present_names if f != inv_factor]
        determinant_names = [*state_determinants, *present_observed]
        y_invest = panel[_slot_column(period, inv_factor)].to_numpy()
        x_determinants = panel[
            [_slot_column(period, f) for f in determinant_names]
        ].to_numpy()
        coefs, inv_sd, residual = _fit_investment_residual(
            y_invest, x_determinants, determinant_names
        )
        cf_by_factor[inv_factor] = residual
        for name, value in coefs.items():
            investment_rows.append(
                ("investment_eq", period, inv_factor, name, float(value))
            )
        investment_rows.append(("investment_sds", period, inv_factor, "-", inv_sd))
    return cf_by_factor, investment_rows


@dataclass(frozen=True)
class _ProductionContext:
    """Per-period design context for the Stage-3 production regressions."""

    x_design: np.ndarray
    """All-present-factor design (non-control-function path)."""
    present_factor_names: list[str]
    """Names matching `x_design`'s columns."""
    x_design_latent: np.ndarray
    """Latent-only design (control-function path: observed factors excluded)."""
    latent_present_names: list[str]
    """Names matching `x_design_latent`'s columns."""
    cf_by_factor: dict[str, np.ndarray]
    """Investment factor -> control-function residual for this period."""


def _fit_period_production(
    panel: pd.DataFrame,
    period: int,
    next_period: int,
    processed_model: ProcessedModel,
    model_spec: ModelSpec,
    *,
    factor_to_function_name: dict[str, str],
    investment_factors: list[str],
    context: _ProductionContext,
    run_cf: bool,
) -> list[tuple[str, int, str, str, float]]:
    """Run the production regressions for every latent outcome at `period`.

    Under `run_cf` the production inputs are the present latent factors only
    (observed factors are the excluded instruments), and the control-function
    residual is injected as a `kappa*cf` covariate into *state* outcomes only
    -- not the investment factor's own transition.
    Without `run_cf` the regressors are all present factors (legacy behaviour).

    Return:
        Transition / shock_sd parameter rows for this period.

    """
    if run_cf:
        fit_x_design = context.x_design_latent
        fit_names = context.latent_present_names
    else:
        fit_x_design = context.x_design
        fit_names = context.present_factor_names

    transition_rows: list[tuple[str, int, str, str, float]] = []
    for factor in processed_model.labels.latent_factors:
        target_col = _slot_column(next_period, factor)
        if target_col not in panel.columns:
            continue
        trans_name = factor_to_function_name.get(factor, "linear")
        if trans_name == "constant":
            continue
        y = panel[target_col].to_numpy()

        inject_cf = run_cf and factor not in investment_factors
        cf: np.ndarray | None = None
        if inject_cf and context.cf_by_factor:
            cf = next(iter(context.cf_by_factor.values()))

        params, sd = _fit_transition(
            trans_name,
            factor,
            processed_model,
            model_spec,
            y,
            fit_x_design,
            fit_names,
            cf=cf,
        )
        for regname, value in params.items():
            transition_rows.append(
                ("transition", period, factor, regname, float(value))
            )
        transition_rows.append(("shock_sds", period, factor, "-", sd))
    return transition_rows


def simulate_and_regress(
    structural: MinimumDistanceResult,
    processed_model: ProcessedModel,
    model_spec: ModelSpec,
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
        model_spec: Original model spec; used to look up raw transition
            callables for user-registered `@register_params` functions.
        mixture_weights: Per-component mixture weights from Stage 1.
        n_draws: Synthetic-panel size.
        seed: RNG seed.
        investment_endogeneity: Whether to apply the AMN eq.-7-8 investment
            control-function correction (AF Sec. 3.5). When True AND the
            model has endogenous (investment) factors, a contemporaneous
            first-stage investment equation `ln I_t = b0 + b_theta.theta_t +
            b_Y.Y_t + eta_{I,t}` is fitted per investment factor (determinants
            = the present production-state latents plus the excluded observed
            instruments Y_t), and its in-sample residual `eta_{I,t}` is added
            as an extra additive `kappa*cf` covariate to each *state* outcome
            factor's production regression. Under this path the production
            inputs are the present LATENT factors only (observed factors are
            the excluded instruments). For models without endogenous factors
            this flag is a no-op. Identification requires at least one present
            observed factor (else `cf` is collinear with the production
            inputs); otherwise a `ValueError` is raised.

    Return:
        ProductionFitResult with production-function and investment-equation
        parameter DataFrames.

    """
    endog_info = processed_model.endogenous_factors_info
    run_cf = investment_endogeneity and endog_info.has_endogenous_factors

    control_function = endog_info.control_function
    if run_cf and control_function is None:
        msg = (
            "investment_endogeneity=True requires a CorrectionSpec declaring the "
            "control function on the endogenous investment factor."
        )
        raise ValueError(msg)
    investment_factors = (
        [control_function.investment_factor] if control_function is not None else []
    )

    panel = _draw_factor_panel(structural, mixture_weights, n_draws=n_draws, seed=seed)

    periods = processed_model.labels.periods
    transition_info = processed_model.transition_info
    factor_to_function_name = (
        dict(transition_info.function_names) if transition_info is not None else {}
    )

    latent_factor_set = set(processed_model.labels.latent_factors)
    observed_factor_set = set(processed_model.labels.observed_factors)

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

        # Latent-only production design (used under the control-function
        # path: observed factors are the excluded instruments).
        latent_present_pairs = [
            (f, c) for f, c in present_pairs if f in latent_factor_set
        ]
        latent_present_names = [f for f, _ in latent_present_pairs]
        x_design_latent = (
            panel[[c for _, c in latent_present_pairs]].to_numpy()
            if latent_present_pairs
            else x_design
        )

        # First stage: fit the contemporaneous investment equation(s) and
        # cache the control-function residual(s) for the production step.
        cf_by_factor: dict[str, np.ndarray] = {}
        if run_cf:
            cf_by_factor, period_investment_rows = _fit_first_stage_investment(
                panel,
                t,
                present_investment=[
                    f for f in present_factor_names if f in investment_factors
                ],
                latent_present_names=latent_present_names,
                present_observed=[
                    f for f in present_factor_names if f in observed_factor_set
                ],
            )
            investment_rows.extend(period_investment_rows)

        context = _ProductionContext(
            x_design=x_design,
            present_factor_names=present_factor_names,
            x_design_latent=x_design_latent,
            latent_present_names=latent_present_names,
            cf_by_factor=cf_by_factor,
        )
        transition_rows.extend(
            _fit_period_production(
                panel,
                t,
                t_next,
                processed_model,
                model_spec,
                factor_to_function_name=factor_to_function_name,
                investment_factors=investment_factors,
                context=context,
                run_cf=run_cf,
            )
        )

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
