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
from skillmodels.amn.minimum_distance import solve_minimum_distance
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
) -> AMNEstimationResult:
    """Estimate a latent factor model using the Attanasio-Meghir-Nix method.

    Args:
        model_spec: Same model spec used by CHS and AF.
        data: Panel dataset in long format with MultiIndex (id, period).
        options: AMN-specific options. If None, uses defaults.
        start_params: Not honoured -- raises `NotImplementedError` when
            non-null. The three-stage estimator has no single free
            optimisation to warm-start.
        fixed_params: Not honoured -- raises `NotImplementedError` when
            non-null. Pin parameters on the returned params yourself, or
            constrain `estimate_chs` when seeding it from AMN.
        constraints: Not honoured -- raises `NotImplementedError` when a
            non-empty list is passed. The AMN stages have no optimiser in
            which to impose equality/other constraints.
        linearize_control_function: When True, fit only the linear `cf` term
            of any `CorrectionSpec` and skip the higher-order
            `NotImplementedError` gate. Used when AMN seeds `estimate_chs`:
            the higher-order `kappa_terms` then fall back to small start
            defaults rather than being estimated here.

    Return:
        AMNEstimationResult containing per-stage outputs and the combined
        params DataFrame.

    """
    if start_params is not None or fixed_params is not None or constraints:
        raise NotImplementedError(
            "estimate_amn does not honour start_params, fixed_params or "
            "constraints. The three-stage minimum-distance estimator has no "
            "single free optimisation in which to warm-start, pin or constrain "
            "parameters, and overlaying them on the result would make the "
            "reported params inconsistent with the fitted stages and the "
            "criterion. Apply such overrides to the returned params yourself, or "
            "pass them to estimate_chs when seeding it from AMN."
        )
    if options is None:
        options = AMNEstimationOptions()
    amn_options = options

    processed_model = process_model(model_spec)
    mixture = _fit_stage1_mixture(processed_model, data, amn_options)

    structural = solve_minimum_distance(
        mixture,
        processed_model,
        weighting=amn_options.minimum_distance_weighting,
        algorithm=amn_options.optimizer_algorithm,
        allow_overnormalization=amn_options.allow_ces_overnormalization,
        algo_options=dict(amn_options.optimizer_options) or None,
    )

    production = simulate_and_regress(
        structural,
        processed_model,
        model_spec,
        mixture_weights=mixture.weights,
        n_draws=amn_options.n_simulation_draws,
        seed=amn_options.seed,
        linearize_control_function=linearize_control_function,
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
