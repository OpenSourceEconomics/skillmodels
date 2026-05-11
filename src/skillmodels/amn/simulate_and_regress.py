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


def _fit_generic_nls(
    transition_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    param_names: tuple[str, ...],
    y: np.ndarray,
    states_panel: np.ndarray,
    *,
    init_overrides: dict[str, float] | None = None,
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

    """
    init_overrides = init_overrides or {}

    @jax.jit
    def predict_batch(theta: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(transition_func, in_axes=(0, None))(states, theta)

    states_jnp = jnp.asarray(states_panel)

    def residuals(theta_np: np.ndarray) -> np.ndarray:
        preds = predict_batch(jnp.asarray(theta_np), states_jnp)
        return np.asarray(preds) - y

    theta0 = np.zeros(len(param_names))
    for name, val in init_overrides.items():
        if name in param_names:
            theta0[param_names.index(name)] = val
    # phi-style elasticity defaults: any "phi", "rho", "sigma" param
    # that doesn't otherwise have an override gets seeded at 0.5 so the
    # CES / general-CES log expressions don't divide by zero.
    for j, name in enumerate(param_names):
        if name in {"phi", "rho", "sigma"} and name not in init_overrides:
            theta0[j] = 0.5
    # Simplex-style "gammas" (anything listed as a factor name in the
    # param list) get a uniform initial share if the function looks
    # CES-shaped (has a "phi"-like param).
    has_elasticity = any(n in {"phi", "rho", "sigma"} for n in param_names)
    if has_elasticity:
        share_candidates = [
            j
            for j, n in enumerate(param_names)
            if n not in {"phi", "rho", "sigma", "constant"}
        ]
        if share_candidates:
            theta0[share_candidates] = 1.0 / len(share_candidates)

    result = least_squares(residuals, theta0, method="lm", max_nfev=5000)
    theta = result.x
    resid = residuals(theta)
    sd = float(np.sqrt(np.mean(resid**2)))
    out = dict(zip(param_names, [float(v) for v in theta], strict=True))
    return out, sd


def _fit_transition(
    transition_name: str,
    factor: str,
    processed_model: ProcessedModel,
    model_spec: ModelSpec,
    y: np.ndarray,
    x_design: np.ndarray,
    regressor_names: list[str],
) -> tuple[dict[str, float], float]:
    """Dispatch to the right per-transition fitter.

    `linear` and `log_ces`-family functions get specialised fitters for
    speed / simplex constraints; everything else (translog,
    robust_translog, linear_and_squares, log_ces_general, user) falls
    through to a generic `jax.vmap`-based NLS.
    """
    if transition_name == "linear":
        return _fit_linear(y, x_design, regressor_names)
    if transition_name == "log_ces":
        return _fit_log_ces(y, x_design, regressor_names, with_constant=False)
    if transition_name == "log_ces_with_constant":
        return _fit_log_ces(y, x_design, regressor_names, with_constant=True)

    func, param_names = _resolve_transition_callable(
        transition_name, factor, processed_model, model_spec
    )
    return _fit_generic_nls(func, param_names, y, x_design)


def _factors_at_period(processed_model: ProcessedModel) -> tuple[str, ...]:
    """Latent + observed factor names (used as transition regressors)."""
    return (
        *processed_model.labels.latent_factors,
        *processed_model.labels.observed_factors,
    )


def simulate_and_regress(  # noqa: C901
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
                    trans_name,
                    factor,
                    processed_model,
                    model_spec,
                    y,
                    x_design,
                    present_factor_names,
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
