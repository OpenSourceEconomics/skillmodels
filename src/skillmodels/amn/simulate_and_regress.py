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
from collections.abc import Callable, Mapping
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
    fixed: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    """OLS regression with an intercept (added as the last column).

    When ``fixed`` pins some coefficients to values, those columns are
    partialled out -- ``fixed_value * column`` is moved to the LHS and the
    column dropped -- so the remaining free coefficients are fit conditional
    on the pins. The pinned coefficients are reported at their fixed values and
    the residual SD reflects the full (free + pinned) prediction.

    Returns:
        ``(params_by_name, residual_sd)`` with `constant` included as
        the trailing parameter.

    """
    fixed = fixed or {}
    n = x_design.shape[0]
    full_design = np.column_stack([x_design, np.ones(n)])
    names = [*regressor_names, "constant"]

    y_adj = y.astype(float).copy()
    free_idx: list[int] = []
    out: dict[str, float] = {}
    for j, name in enumerate(names):
        if name in fixed:
            out[name] = float(fixed[name])
            y_adj = y_adj - out[name] * full_design[:, j]
        else:
            free_idx.append(j)
    if free_idx:
        coefs, *_ = np.linalg.lstsq(full_design[:, free_idx], y_adj, rcond=None)
        for k, j in enumerate(free_idx):
            out[names[j]] = float(coefs[k])

    pred = full_design @ np.array([out[name] for name in names])
    sd = float(np.sqrt(np.mean((y - pred) ** 2)))
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
    the model spec. The user callable looks its positional arguments up against
    the full `(*latent, *observed)` factor order; a factor that the simulated
    panel does not provide is read past the end of the (narrower) design row and
    clamped by `jax` -- a throwaway seed value the CHS MLE re-fits.
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


def _is_elasticity_param(name: str) -> bool:
    """Return True for CES exponent / outside-coefficient parameters.

    Covers the single-exponent forms (`phi`, `rho`, `sigma`) and
    `log_ces_general`'s per-factor exponents `sigma_<factor>` and outside
    coefficient `tfp`. These must be seeded nonzero so the CES log expression is
    finite at the start point.
    """
    return name in {"phi", "rho", "sigma", "tfp"} or name.startswith("sigma_")


def _seed_generic_nls_theta0(
    param_names: tuple[str, ...],
    init_overrides: dict[str, float],
    *,
    n_unknowns: int,
) -> np.ndarray:
    """Seed the NLS start vector for `_fit_generic_nls`.

    Applies `init_overrides`, then seeds elasticity/outside-style params at 0.5
    so CES / general-CES log expressions are finite at the start, and gives the
    remaining (simplex-style "gamma") params a strictly positive uniform initial
    share when the function looks CES-shaped. Elasticity-style names cover the
    single-exponent forms (`phi`, `rho`, `sigma`) AND `log_ces_general`'s
    per-factor exponents `sigma_<factor>` and outside coefficient `tfp` -- the
    latter were previously unrecognized, leaving every parameter at 0 so that
    `tfp * log(sum gamma_i ...) = 0 * log(0) = NaN` (Pro F4). The trailing
    cf-coefficient slot (if `n_unknowns > len(param_names)`) stays at zero.
    """
    theta0 = np.zeros(n_unknowns)
    for name, val in init_overrides.items():
        if name in param_names:
            theta0[param_names.index(name)] = val
    for j, name in enumerate(param_names):
        if _is_elasticity_param(name) and name not in init_overrides:
            theta0[j] = 0.5
    has_elasticity = any(_is_elasticity_param(n) for n in param_names)
    if has_elasticity:
        share_candidates = [
            j
            for j, n in enumerate(param_names)
            if not _is_elasticity_param(n)
            and n != "constant"
            and n not in init_overrides
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
    fixed: Mapping[str, float] | None = None,
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
        fixed: optional ``{name: value}`` pinning a subset of
            ``param_names``. Pinned entries are held at their values and only
            the remaining (free) parameters are optimised; the pinned values
            are reconstructed into the full param vector at every residual
            evaluation so the free fit is conditional on the pins.

    """
    init_overrides = init_overrides or {}
    fixed = fixed or {}
    has_cf = cf is not None
    free_names = [n for n in param_names if n not in fixed]
    n_free = len(free_names)
    kappa_idx = n_free
    # Baseline full param vector carrying the pinned values; free positions
    # are overwritten from the optimiser's vector at each evaluation.
    base_theta = np.array([float(fixed.get(n, 0.0)) for n in param_names])
    free_positions = [i for i, n in enumerate(param_names) if n not in fixed]

    @jax.jit
    def predict_batch(theta: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(transition_func, in_axes=(0, None))(states, theta)

    states_jnp = jnp.asarray(states_panel)

    def residuals(theta_np: np.ndarray) -> np.ndarray:
        full = base_theta.copy()
        full[free_positions] = theta_np[:n_free]
        preds = np.asarray(predict_batch(jnp.asarray(full), states_jnp))
        if has_cf:
            preds = preds + theta_np[kappa_idx] * cf
        return preds - y

    # Seed using the full param layout (preserves elasticity / CES-share
    # seeding) then select the free positions plus the trailing cf slot.
    full_seed = _seed_generic_nls_theta0(
        param_names,
        init_overrides,
        n_unknowns=len(param_names) + (1 if has_cf else 0),
    )
    theta0 = np.array(
        [full_seed[i] for i in free_positions] + ([0.0] if has_cf else [])
    )

    result = least_squares(residuals, theta0, method="lm", max_nfev=5000)
    theta = result.x
    resid = residuals(theta)
    sd = float(np.sqrt(np.mean(resid**2)))
    out = {n: float(fixed[n]) for n in param_names if n in fixed}
    for i, n in enumerate(free_names):
        out[n] = float(theta[i])
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
    fixed: Mapping[str, float] | None = None,
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

    When ``fixed`` pins a subset of the transition coefficients (by their
    parameter name), those are held at their values inside the regression and
    only the remaining coefficients are fit. The `log_ces`-family fitters do
    not yet support pinning and raise if asked to.
    """
    fixed = dict(fixed) if fixed else {}
    if transition_name == "linear":
        if cf is not None:
            x_design = np.column_stack([x_design, cf])
            regressor_names = [*regressor_names, "cf"]
        return _fit_linear(y, x_design, regressor_names, fixed=fixed)
    # log_ces_af shares log_ces's math (CES over production factors only), so it
    # routes to the same specialised fitter (Pro F6); without this it would fall
    # through to _resolve_transition_callable as a bare string and raise.
    if transition_name in ("log_ces", "log_ces_af", "log_ces_with_constant"):
        if fixed:
            msg = (
                f"fixed_params for the '{transition_name}' transition of factor "
                f"'{factor}' is not supported; pinning is implemented for linear "
                "and the generic NLS transitions (translog etc.)."
            )
            raise NotImplementedError(msg)
        with_constant = transition_name == "log_ces_with_constant"
        return _fit_log_ces(
            y, x_design, regressor_names, with_constant=with_constant, cf=cf
        )

    func, param_names = _resolve_transition_callable(
        transition_name, factor, processed_model, model_spec
    )
    return _fit_generic_nls(func, param_names, y, x_design, cf=cf, fixed=fixed)


def _transition_fixes_for_period(
    fixed_params: pd.DataFrame | None,
    period: int,
) -> dict[str, dict[str, float]]:
    """Collect ``transition`` pins for ``period`` as ``{factor: {regname: value}}``.

    Reads the rows of ``fixed_params`` whose category is ``"transition"`` and
    whose period equals ``period`` (the AMN params index labels this level
    ``aug_period``, but AMN keys it by calendar period). Returns an empty dict
    when there is nothing to pin.
    """
    if fixed_params is None or fixed_params.empty:
        return {}
    idx = fixed_params.index
    mask = (idx.get_level_values(0) == "transition") & (
        idx.get_level_values(1) == period
    )
    out: dict[str, dict[str, float]] = {}
    for label, value in fixed_params.loc[mask, "value"].items():
        _cat, _p, factor, regname = label  # ty: ignore[not-iterable]
        out.setdefault(factor, {})[regname] = float(value)
    return out


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
    state_predictors: list[str],
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
        state_predictors: CorrectionSpec first-stage state predictors present at
            `period`, in design order.
        present_observed: CorrectionSpec instruments present at `period` (the
            excluded observed factors).

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
            "AMN control function with more than one present investment factor at "
            f"period {period} is unsupported: the control-function choice for the "
            "state-factor production regressions is ambiguous."
        )
        raise NotImplementedError(msg)

    cf_by_factor: dict[str, np.ndarray] = {}
    investment_rows: list[tuple[str, int, str, str, float]] = []
    for inv_factor in present_investment:
        if not present_observed:
            msg = (
                "The AMN control function requires at least one present instrument "
                f"at period {period} to identify the control-function coefficient "
                f"for '{inv_factor}': without an excluded instrument the residual "
                "eta_{I,t} is collinear with the production inputs (theta_t, I_t)."
            )
            raise ValueError(msg)
        determinant_names = [*state_predictors, *present_observed]
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
    x_design_production: np.ndarray
    """Production design under the control-function path: present factors with the
    excluded *instruments* removed. Latent factors and any non-instrument observed
    factors (genuine production controls) are kept."""
    production_factor_names: list[str]
    """Names matching `x_design_production`'s columns."""
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
    context: _ProductionContext,
    run_cf: bool,
    targets: list[str],
    fixed_by_factor: Mapping[str, Mapping[str, float]] | None = None,
) -> list[tuple[str, int, str, str, float]]:
    """Run the production regressions for every latent outcome at `period`.

    Under `run_cf` the production inputs are the present factors with the excluded
    *instruments* removed -- latent factors plus any non-instrument observed
    factors (genuine production controls) -- and the control-function residual is
    injected as a `kappa*cf` covariate into *state* outcomes only, not the
    investment factor's own transition. Only the `CorrectionSpec` instruments are
    excluded; other observed factors remain production inputs.
    Without `run_cf` the regressors are all present factors (legacy behaviour).

    Return:
        Transition / shock_sd parameter rows for this period.

    """
    if run_cf:
        fit_x_design = context.x_design_production
        fit_names = context.production_factor_names
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

        inject_cf = run_cf and factor in targets
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
            fixed=fixed_by_factor.get(factor) if fixed_by_factor else None,
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
    linearize_control_function: bool = False,
    fixed_params: pd.DataFrame | None = None,
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
        linearize_control_function: When True, fit only the single linear `cf`
            term and skip the higher-order `kappa_terms`
            `NotImplementedError` gate (used when AMN seeds `estimate_chs`).
        fixed_params: optional params frame whose ``transition`` rows pin
            production-function coefficients. Each pinned coefficient is held at
            its value inside the per-period production regression while the
            remaining coefficients are fit conditional on the pins. Rows of
            other categories are ignored here (they belong to other stages).

    The AMN eq.-7-8 investment control-function correction (AF Sec. 3.5) runs iff
    the model declares a `CorrectionSpec` (presence is the single trigger). A
    contemporaneous first-stage investment equation is then fitted per investment
    factor over the spec's `state_predictors` + excluded `instruments`, and its
    in-sample residual `eta_{I,t}` is added as an additive `kappa * cf` covariate
    to each `targets` factor's production regression. AMN implements only the
    linear `cf` term; higher-order `kappa_terms` raise `NotImplementedError`.

    Return:
        ProductionFitResult with production-function and investment-equation
        parameter DataFrames.

    """
    endog_info = processed_model.endogenous_factors_info
    control_function = endog_info.control_function
    # CorrectionSpec presence is the single trigger; there is no separate flag.
    run_cf = control_function is not None

    if control_function is not None and not linearize_control_function:
        # AMN implements only a single linear cf term per target; the higher-order
        # (translog) kappa_terms basis needs estimate_chs. When used to *seed*
        # estimate_chs (`linearize_control_function=True`), AMN instead fits only
        # the linear cf term and leaves the higher-order kappa terms for the
        # start-value defaults.
        for target, terms in control_function.kappa_terms.items():
            if tuple(terms) != ("cf",):
                msg = (
                    "AMN implements only a linear control function (kappa * cf). "
                    f"Target {target!r} requests higher-order terms {tuple(terms)}; "
                    "use estimate_chs for the full polynomial basis."
                )
                raise NotImplementedError(msg)

    investment_factors = (
        [control_function.investment_factor] if control_function is not None else []
    )
    cf_targets = list(control_function.targets) if control_function is not None else []
    cf_predictors = (
        list(control_function.state_predictors) if control_function is not None else []
    )
    cf_instruments = (
        list(control_function.instruments) if control_function is not None else []
    )

    panel = _draw_factor_panel(structural, mixture_weights, n_draws=n_draws, seed=seed)

    periods = processed_model.labels.periods
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

        # Production design used under the control-function path: drop only the
        # excluded instruments, keeping latent factors and any non-instrument
        # observed factors (genuine production controls). When every observed
        # factor is an instrument this reduces to the latent-only design.
        instrument_set = set(cf_instruments)
        production_pairs = [(f, c) for f, c in present_pairs if f not in instrument_set]
        production_factor_names = [f for f, _ in production_pairs]
        x_design_production = (
            panel[[c for _, c in production_pairs]].to_numpy()
            if production_pairs
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
                state_predictors=[
                    f for f in cf_predictors if f in present_factor_names
                ],
                present_observed=[
                    f for f in cf_instruments if f in present_factor_names
                ],
            )
            investment_rows.extend(period_investment_rows)

        context = _ProductionContext(
            x_design=x_design,
            present_factor_names=present_factor_names,
            x_design_production=x_design_production,
            production_factor_names=production_factor_names,
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
                context=context,
                run_cf=run_cf,
                targets=cf_targets,
                fixed_by_factor=_transition_fixes_for_period(fixed_params, t),
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
