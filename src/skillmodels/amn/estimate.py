"""Top-level orchestration for the three-stage AMN estimator.

Chains the three stages:

1. `mixture_em.fit_mixture_em` -> reduced-form Pi, Psi
2. `minimum_distance.solve_minimum_distance` -> structural Lambda, A, Sigma, mu, Omega
3. `simulate_and_regress.simulate_and_regress` -> production-function params

and merges the resulting parameter pieces into a single skillmodels
params DataFrame.
"""

import numpy as np
import optimagic as om
import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import ESTIMATION_CONF
from skillmodels.amn.minimum_distance import (
    _STAGE2_FIXED_CATEGORIES,
    solve_minimum_distance,
)
from skillmodels.amn.mixture_em import (
    build_augmented_measure_layout,
    build_augmented_measure_matrix,
    fit_mixture_em,
)
from skillmodels.amn.simulate_and_regress import simulate_and_regress
from skillmodels.amn.types import (
    AMNEstimationOptions,
    AMNEstimationResult,
    AMNStageResults,
    MinimumDistanceResult,
    MixtureFitResult,
)
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.types import ProcessedModel

# Parameter categories whose pins estimate_amn can honour in Stage 3 (the
# simulate-and-regress production step). The production regression holds these
# coefficients at their pinned values while fitting the remaining ones.
_STAGE3_FIXED_CATEGORIES = frozenset({"transition"})

# All fixed_params categories estimate_amn can route to the stage that fits
# them: Stage 2 (structural measurement system) plus Stage 3 (production).
_SUPPORTED_FIXED_CATEGORIES = _STAGE2_FIXED_CATEGORIES | _STAGE3_FIXED_CATEGORIES

# Restricted-CES transitions AMN cannot consistently estimate standalone: the
# Stage-3 CES regression omits Freyberger's primitive-scale recovery. The
# generalized form `log_ces_general` is fine (it can express the transformed CES),
# and custom `@register_params` transitions are out of scope for this guard.
_RESTRICTED_CES_TRANSITIONS = frozenset(
    {"log_ces", "log_ces_af", "log_ces_with_constant"}
)


def _fail_if_standalone_unsupported(
    processed_model: ProcessedModel,
    *,
    for_start_values: bool,
) -> None:
    """Refuse standalone AMN on a model it cannot consistently estimate.

    AMN's restricted-CES (`log_ces` / `log_ces_with_constant`) Stage-3 regression
    does not perform Freyberger's primitive-scale recovery, so the returned CES
    parameters are not consistent estimates. That is acceptable only when AMN is
    producing start values for another estimator that re-fits every parameter
    (`estimate_chs` and `estimate_af` both seed from AMN); `for_start_values=True`
    flags that context, so the guard is skipped there. Standalone use (the
    default) raises.
    """
    if for_start_values:
        return
    info = processed_model.transition_info
    if info is None:
        return
    bad = sorted(
        factor
        for factor, name in info.function_names.items()
        if name in _RESTRICTED_CES_TRANSITIONS
    )
    if bad:
        msg = (
            f"estimate_amn cannot consistently estimate the restricted-CES "
            f"transition on factor(s) {bad}: AMN's Stage-3 CES regression omits the "
            "primitive-scale recovery (Freyberger 2025), so the standalone result "
            "would be inconsistent. Use 'log_ces_general', or use AMN only to seed "
            "estimate_chs (which re-fits every parameter)."
        )
        raise NotImplementedError(msg)


def _measurement_params_dataframe(
    structural: MinimumDistanceResult,
) -> pd.DataFrame:
    """Translate Stage 2 outputs into rows of the standard params DataFrame."""
    rows: list[tuple[str, int, str, str, float]] = []
    for idx, row in structural.loadings.iterrows():
        period, meas, factor = idx  # ty: ignore[not-iterable]
        rows.append(
            ("loadings", int(period), str(meas), str(factor), float(row["loading"]))
        )
    for idx, row in structural.measurement_intercepts.iterrows():
        period, meas = idx  # ty: ignore[not-iterable]
        rows.append(
            ("controls", int(period), str(meas), "constant", float(row["intercept"]))
        )
    for idx, row in structural.measurement_sds.iterrows():
        period, meas = idx  # ty: ignore[not-iterable]
        rows.append(("meas_sds", int(period), str(meas), "-", float(row["sd"])))
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


def _fit_stage1_mixture(
    processed_model: ProcessedModel,
    data: pd.DataFrame,
    amn_options: AMNEstimationOptions,
) -> MixtureFitResult:
    """Fit the Stage-1 mixture on the augmented measure vector.

    The EM method is taken verbatim from `amn_options.mixture_em_method`:
    `"complete_case"` (the default) raises `InsufficientCompleteCasesError` on an
    unbalanced panel with too few complete rows, while `"missing_data"`
    marginalises over each row's missing entries. An optional
    `mixture_em_max_rows` cap subsamples rows first so the per-restart cost stays
    bounded (used by CHS seeding); standalone estimation keeps the full sample.
    """
    n_components = processed_model.dimensions.n_mixtures
    layout = build_augmented_measure_layout(processed_model)
    augmented = build_augmented_measure_matrix(data, processed_model, layout)

    max_rows = amn_options.mixture_em_max_rows
    if max_rows is not None and augmented.shape[0] > max_rows:
        rng = np.random.default_rng(amn_options.seed)
        keep = rng.choice(augmented.shape[0], max_rows, replace=False)
        augmented = augmented[keep]

    return fit_mixture_em(
        augmented,
        n_components=n_components,
        max_iter=amn_options.em_max_iter,
        tol=amn_options.em_tol,
        n_init=amn_options.em_n_init,
        reg_covar=amn_options.em_reg_covar,
        seed=amn_options.seed,
        layout=layout,
        method=amn_options.mixture_em_method,
        allow_never_observed=amn_options.allow_never_observed_measurements,
    )


@beartype(conf=ESTIMATION_CONF)
def estimate_amn(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    options: AMNEstimationOptions | None = None,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
    *,
    linearize_control_function: bool = False,
    for_start_values: bool = False,
) -> AMNEstimationResult:
    """Estimate a latent factor model using the Attanasio-Meghir-Nix method.

    Args:
        model_spec: Same model spec used by CHS and AF.
        data: Panel dataset in long format with MultiIndex (id, period).
        options: AMN-specific options. If None, uses defaults.
        start_params: Not honoured -- raises `NotImplementedError` when
            non-null. The three-stage estimator has no single free
            optimisation to warm-start.
        fixed_params: Pins for parameters AMN can hold in the stage that fits
            them. Currently the `transition` category (production-function
            coefficients) is honoured: each pinned coefficient is held at its
            value inside the Stage-3 production regression while the remaining
            coefficients are fit conditional on the pins, so the reported
            params stay consistent with the criterion. Pins for other
            categories raise `NotImplementedError` -- pin them on the returned
            params yourself, or pass them to `estimate_chs` when seeding it.
        constraints: Not honoured -- raises `NotImplementedError` when a
            non-empty list is passed. The AMN stages have no optimiser in
            which to impose equality/other constraints.
        linearize_control_function: When True, fit only the linear `cf` term
            of any `CorrectionSpec` and skip the higher-order
            `NotImplementedError` gate. Used when AMN seeds `estimate_chs`:
            the higher-order `kappa_terms` then fall back to small start
            defaults rather than being estimated here.
        for_start_values: When True, the result is consumed only as start values
            for an estimator that re-fits every parameter (`estimate_chs` and
            `estimate_af` both seed from AMN), so the standalone guard against
            models AMN cannot consistently estimate (restricted CES) is skipped.
            Standalone callers leave this False.

    Return:
        AMNEstimationResult containing per-stage outputs and the combined
        params DataFrame.

    """
    if start_params is not None or constraints:
        raise NotImplementedError(
            "estimate_amn does not honour start_params or constraints. The "
            "three-stage estimator has no single free optimisation in which to "
            "warm-start or impose cross-parameter constraints. Use fixed_params "
            "to pin individual parameters, or pass start_params / constraints to "
            "estimate_chs when seeding it from AMN."
        )
    if fixed_params is not None and not fixed_params.empty:
        # Pins are honoured inside the stage that fits each parameter, so they
        # stay consistent with the criterion (unlike overlaying them on the
        # result). Stage 2 (minimum distance) owns the structural measurement
        # categories; Stage 3 (production regression) owns `transition`.
        categories = set(fixed_params.index.get_level_values(0))
        unsupported = categories - _SUPPORTED_FIXED_CATEGORIES
        if unsupported:
            raise NotImplementedError(
                "estimate_amn honours fixed_params for the "
                f"{sorted(_SUPPORTED_FIXED_CATEGORIES)} categories (held in the "
                "stage that fits each: Stage 2 for the measurement system, "
                "Stage 3 for production). Pins for "
                f"{sorted(unsupported)} are not supported (they are derived "
                "outputs, not free parameters); pin them on the returned params "
                "yourself, or pass them to estimate_chs when seeding it from AMN."
            )
    if options is None:
        options = AMNEstimationOptions()
    amn_options = options

    processed_model = process_model(model_spec)
    _fail_if_standalone_unsupported(processed_model, for_start_values=for_start_values)
    mixture = _fit_stage1_mixture(processed_model, data, amn_options)

    structural = solve_minimum_distance(
        mixture,
        processed_model,
        weighting=amn_options.minimum_distance_weighting,
        algorithm=amn_options.optimizer_algorithm,
        allow_overnormalization=amn_options.allow_ces_overnormalization,
        algo_options=dict(amn_options.optimizer_options) or None,
        fixed_params=fixed_params,
    )

    production = simulate_and_regress(
        structural,
        processed_model,
        model_spec,
        mixture_weights=mixture.weights,
        n_draws=amn_options.n_simulation_draws,
        seed=amn_options.seed,
        linearize_control_function=linearize_control_function,
        fixed_params=fixed_params,
    )

    measurement = _measurement_params_dataframe(structural)
    all_params = pd.concat(
        [measurement, production.production_params, production.investment_params]
    ).sort_index()

    success = structural.success and mixture.converged

    return AMNEstimationResult(
        model_spec=model_spec,
        stages=AMNStageResults(
            mixture=mixture,
            structural=structural,
            production=production,
        ),
        params=all_params,
        success=success,
        md_criterion=float(structural.objective_value),
        synthetic_panel=None,
    )
