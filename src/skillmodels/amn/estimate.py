"""Top-level orchestration for the three-stage AMN estimator.

Chains the three stages:

1. `mixture_em.fit_mixture_em` -> reduced-form Pi, Psi
2. `minimum_distance.solve_minimum_distance` -> structural Lambda, A, Sigma, mu, Omega
3. `simulate_and_regress.simulate_and_regress` -> production-function params

and merges the resulting parameter pieces into a single skillmodels
params DataFrame.
"""

import warnings

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
    reduce_to_seedable_measurements,
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
from skillmodels.common.selector import align_index_names
from skillmodels.common.types import ProcessedModel

# Row cap for the missing-data Stage-1 EM: a seed needs only a representative
# subsample, and the EM cost scales with the number of distinct missing patterns.
_MAX_MISSING_DATA_SEED_ROWS = 5000


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


def _apply_overrides(
    params: pd.DataFrame,
    *,
    fixed_params: pd.DataFrame | None,
    start_params: pd.DataFrame | None,
) -> pd.DataFrame:
    """Overlay user-supplied fixed_params and start_params on `params`.

    `fixed_params` wins over `start_params`, which wins over the
    estimated values. Rows in the overrides not present in `params` are
    added; rows in `params` not present in the overrides are kept.
    """
    out = params.copy()
    if start_params is not None and not start_params.empty:
        aligned = align_index_names(start_params, target_names=out.index.names)
        merged = out.reindex(out.index.union(aligned.index))
        merged.loc[aligned.index, "value"] = aligned["value"]
        out = merged
    if fixed_params is not None and not fixed_params.empty:
        aligned = align_index_names(fixed_params, target_names=out.index.names)
        merged = out.reindex(out.index.union(aligned.index))
        merged.loc[aligned.index, "value"] = aligned["value"]
        out = merged
    return out.sort_index()


def _seed_stage1_mixture(
    processed_model: ProcessedModel,
    data: pd.DataFrame,
    amn_options: AMNEstimationOptions,
) -> MixtureFitResult:
    """Fit the Stage-1 mixture, choosing the EM method for the data's missingness.

    `"complete_case"` fits on listwise-complete rows after the subsample drop;
    `"missing_data"` always marginalises over missing entries; `"auto"` (default)
    uses complete-case when a feasible complete-case subset exists and otherwise
    falls back to the missing-data EM over the full measurement set -- the regime
    of an unbalanced panel where no individual spans every period.
    """
    n_components = processed_model.dimensions.n_mixtures
    full_layout = build_augmented_measure_layout(processed_model)
    full_augmented = build_augmented_measure_matrix(data, processed_model, full_layout)
    method = amn_options.mixture_em_method

    layout, augmented, fit_method = full_layout, full_augmented, "missing_data"
    if method in ("complete_case", "auto"):
        layout, augmented, _dropped = reduce_to_seedable_measurements(
            full_layout,
            full_augmented,
            processed_model,
            n_components=n_components,
            min_complete_cases=amn_options.seed_min_complete_cases,
        )
        n_complete = int((~np.isnan(augmented).any(axis=1)).sum())
        if method == "complete_case" or n_complete >= n_components:
            fit_method = "complete_case"
        else:
            warnings.warn(
                "AMN Stage 1: no complete-case subset is feasible for the "
                f"{n_components}-component mixture (unbalanced panel: too few "
                "individuals span every period). Falling back to the missing-data "
                "EM over the full measurement set.",
                RuntimeWarning,
                stacklevel=2,
            )
            layout, augmented = full_layout, full_augmented

    if (
        fit_method == "missing_data"
        and augmented.shape[0] > _MAX_MISSING_DATA_SEED_ROWS
    ):
        # The missing-data EM cost scales with the number of distinct missing
        # patterns (worst case: one per row). A seed does not need the full
        # sample, so cap the rows to keep Stage-1 seeding tractable on large
        # unbalanced panels.
        rng = np.random.default_rng(amn_options.seed)
        keep = rng.choice(
            augmented.shape[0], _MAX_MISSING_DATA_SEED_ROWS, replace=False
        )
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
        method=fit_method,
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
        start_params: Optional starting parameter values; overlaid on the
            estimated combined params DataFrame as well as on Stage 1 EM
            starts (the latter not yet wired).
        fixed_params: Parameters to pin during estimation. Currently
            applied as a post-hoc override on the combined params
            DataFrame; future revisions may enforce them inside each
            stage's optimizer.
        constraints: Reserved for forward-compatibility (equality
            constraints from optimagic). Not yet honoured inside the AMN
            stages; pass-through only.
        linearize_control_function: When True, fit only the linear `cf` term
            of any `CorrectionSpec` and skip the higher-order
            `NotImplementedError` gate. Used when AMN seeds `estimate_chs`:
            the higher-order `kappa_terms` then fall back to small start
            defaults rather than being estimated here.

    Return:
        AMNEstimationResult containing per-stage outputs and the combined
        params DataFrame.

    """
    del constraints  # forward-compat hook; AMN stages do not yet honour these
    if options is None:
        options = AMNEstimationOptions()
    amn_options = options

    processed_model = process_model(model_spec)
    mixture = _seed_stage1_mixture(processed_model, data, amn_options)

    structural = solve_minimum_distance(
        mixture,
        processed_model,
        weighting=amn_options.minimum_distance_weighting,
        algorithm=amn_options.optimizer_algorithm,
        allow_overnormalization=amn_options.allow_ces_overnormalization,
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
    all_params = _apply_overrides(
        all_params, fixed_params=fixed_params, start_params=start_params
    )

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
