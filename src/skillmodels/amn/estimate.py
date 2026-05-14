"""Top-level orchestration for the three-stage AMN estimator.

Chains the three stages:

1. `mixture_em.fit_mixture_em` -> reduced-form Pi, Psi
2. `minimum_distance.solve_minimum_distance` -> structural Lambda, A, Sigma, mu, Omega
3. `simulate_and_regress.simulate_and_regress` -> production-function params

and merges the resulting parameter pieces into a single skillmodels
params DataFrame.
"""

import optimagic as om
import pandas as pd

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
)
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.selector import align_index_names


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


def estimate_amn(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    amn_options: AMNEstimationOptions | None = None,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> AMNEstimationResult:
    """Estimate a latent factor model using the Attanasio-Meghir-Nix method.

    Args:
        model_spec: Same model spec used by CHS and AF.
        data: Panel dataset in long format with MultiIndex (id, period).
        amn_options: AMN-specific options. If None, uses defaults.
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

    Return:
        AMNEstimationResult containing per-stage outputs and the combined
        params DataFrame.

    """
    del constraints  # forward-compat hook; AMN stages do not yet honour these
    if amn_options is None:
        amn_options = AMNEstimationOptions()

    processed_model = process_model(model_spec)
    layout = build_augmented_measure_layout(processed_model)
    augmented = build_augmented_measure_matrix(data, processed_model, layout)

    mixture = fit_mixture_em(
        augmented,
        n_components=amn_options.n_mixture_components,
        max_iter=amn_options.em_max_iter,
        tol=amn_options.em_tol,
        n_init=amn_options.em_n_init,
        reg_covar=amn_options.em_reg_covar,
        seed=amn_options.seed,
        layout=layout,
    )

    structural = solve_minimum_distance(
        mixture,
        processed_model,
        weighting=amn_options.minimum_distance_weighting,
        algorithm=amn_options.optimizer_algorithm,
    )

    production = simulate_and_regress(
        structural,
        processed_model,
        model_spec,
        mixture_weights=mixture.weights,
        n_draws=amn_options.n_simulation_draws,
        seed=amn_options.seed,
        investment_endogeneity=amn_options.investment_endogeneity,
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
        all_params=all_params,
        success=success,
        synthetic_panel=None,
    )
