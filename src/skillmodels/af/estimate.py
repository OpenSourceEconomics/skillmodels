"""Main driver for the AF estimation procedure."""

import dataclasses
import gc

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
from jax import Array

from skillmodels.af.initial_period import estimate_initial_period
from skillmodels.af.params import get_measurements_per_factor
from skillmodels.af.transition_period import estimate_transition_period
from skillmodels.af.types import (
    AFEstimationOptions,
    AFEstimationResult,
    AFPeriodResult,
    ChainLink,
    ConditionalDistribution,
    MixtureComponent,
)
from skillmodels.af.validate import validate_af_model
from skillmodels.amn.estimate import estimate_amn
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model


def estimate_af(  # noqa: PLR0915
    model_spec: ModelSpec,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> AFEstimationResult:
    """Estimate a latent factor model using the Antweiler-Freyberger method.

    Sequential period-by-period MLE with Halton quadrature for numerical
    integration, following Antweiler and Freyberger (2025).

    The procedure estimates one period at a time:
    - Step 0: Fit initial distribution and measurement params for period 0
    - Step t (t >= 1): Estimate transition and measurement params using the
      estimated distribution from previous periods

    Args:
        model_spec: Model specification (same as for CHS estimation).
        data: Dataset in long format with MultiIndex (id, period).
        af_options: AF-specific estimation options. If None, uses defaults.
        start_params: Optional starting parameter values. If provided, any
            matching index entries override the heuristic defaults. Uses the
            same 4-level MultiIndex as CHS params (category, period, name1,
            name2). Unmatched entries keep their heuristic values.
        fixed_params: Optional DataFrame with a "value" column pinning
            specified parameters to fixed values. Bounds are clamped equal
            to the value so the optimizer excludes them. Used, e.g., to pin
            time-invariant latent factors to identity transitions with zero
            shocks (same convention as CHS augmented periods).
        constraints: Optional list of optimagic Constraint objects. Only
            `om.EqualityConstraint` entries that select via
            `skillmodels.common.constraints.select_by_loc` are honoured.
            Two regimes:
            * **Cross-period**: members straddle multiple AF steps. Once
              any member is estimated in an earlier step, every other
              member (in not-yet-estimated periods) is pinned to that
              value via `fixed_params`.
            * **Within-step**: all members lie in the same step's
              params index (e.g. `investment_sds` at period t-1 and
              `meas_sds` at period t both live in step t). The constraint
              is forwarded verbatim to that step's `om.minimize` so the
              optimizer enforces equality during fitting.
            Other constraint types are ignored (AF's per-period MLE
            handles model-implied within-period constraints internally).

    Return:
        AFEstimationResult with per-period results and combined parameters.

    """
    jax.config.update("jax_enable_x64", val=True)

    if af_options is None:
        af_options = AFEstimationOptions()

    validate_af_model(model_spec)
    processed_model = process_model(model_spec)

    # If AMN-based starts are requested, run the full AMN three-stage
    # estimator upfront and overlay its parameter estimates onto the
    # caller-supplied `start_params` (user values win on overlap).
    # After this the per-period MLE proceeds with `initialization_strategy
    # = "constant"` internally so the within-period Spearman pre-pass is
    # skipped (AMN's values are already in the optimizer's starting
    # neighbourhood).
    if af_options.initialization_strategy == "amn":
        amn_result = estimate_amn(model_spec=model_spec, data=data)
        amn_start = amn_result.all_params[["value"]]
        if start_params is not None:
            user_idx = start_params.index
            amn_start = amn_start.drop(
                index=amn_start.index.intersection(user_idx),
                errors="ignore",
            )
            start_params = pd.concat([amn_start, start_params]).sort_index()
        else:
            start_params = amn_start
        af_options = AFEstimationOptions(
            n_halton_points=af_options.n_halton_points,
            n_halton_points_shock=af_options.n_halton_points_shock,
            n_mixture_components=af_options.n_mixture_components,
            optimizer_algorithm=af_options.optimizer_algorithm,
            optimizer_options=dict(af_options.optimizer_options),
            two_stage=af_options.two_stage,
            coarse_fraction=af_options.coarse_fraction,
            stability_floor=af_options.stability_floor,
            n_obs_per_batch=af_options.n_obs_per_batch,
            initialization_strategy="constant",
        )

    # Extract data arrays per period
    n_periods = processed_model.dimensions.n_periods
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    observed_factors = processed_model.labels.observed_factors

    # Identify endogenous (investment) factors
    endog_info = processed_model.endogenous_factors_info
    endogenous_factors = tuple(
        f
        for f in factors
        if f in endog_info.factor_info and endog_info.factor_info[f].is_endogenous
    )
    state_factors = tuple(f for f in factors if f not in endogenous_factors)

    period_data = _extract_period_data(
        data,
        n_periods,
        factors,
        controls_names,
        model_spec,
        observed_factors=observed_factors,
    )

    equality_groups = _extract_equality_groups(constraints)

    # Step 0: Initial period
    period_0_result, cond_dist = estimate_initial_period(
        model_spec=model_spec,
        processed_model=processed_model,
        measurements=period_data[0]["measurements"],
        controls=period_data[0]["controls"],
        af_options=af_options,
        state_factors=state_factors,
        start_params=start_params,
        fixed_params=fixed_params,
        observed_factors=observed_factors,
        observed_factor_values=period_data[0].get("observed_factors"),
        user_constraints=constraints,
    )

    period_results: list[AFPeriodResult] = [period_0_result]
    conditional_dists: list[ConditionalDistribution] = [cond_dist]
    fixed_params = _propagate_equality_groups(
        period_results=period_results,
        fixed_params=fixed_params,
        equality_groups=equality_groups,
    )

    # Steps 1..T-1: Transition periods
    for t in range(1, n_periods):
        measurements_pt = get_measurements_per_factor(model_spec.factors, period=t)
        if not measurements_pt:
            break

        prev_period_params = period_results[-1].params

        period_t_result, cond_dist = estimate_transition_period(
            period=t,
            model_spec=model_spec,
            processed_model=processed_model,
            measurements=period_data[t]["measurements"],
            controls=period_data[t]["controls"],
            prev_measurements=period_data[t - 1]["measurements"],
            prev_controls=period_data[t - 1]["controls"],
            prev_period_params=prev_period_params,
            prev_distribution=cond_dist,
            af_options=af_options,
            endogenous_factors=endogenous_factors,
            observed_factors=observed_factors,
            observed_factor_data=period_data.get(t - 1, {}).get(
                "observed_factors", None
            ),
            start_params=start_params,
            fixed_params=fixed_params,
            user_constraints=constraints,
        )
        period_results.append(period_t_result)
        conditional_dists.append(cond_dist)
        fixed_params = _propagate_equality_groups(
            period_results=period_results,
            fixed_params=fixed_params,
            equality_groups=equality_groups,
        )

    # Combine parameters from all periods
    all_params = pd.concat([r.params for r in period_results])

    # Free the XLA compilation cache + any unreferenced device buffers
    # before materialising the result. The per-period likelihoods and
    # gradients leave hundreds of MB of compiled executables and stale
    # intermediates on the device; without this the GPU→host copy in
    # `_to_numpy(...)` has been observed to OOM on a host-side staging
    # allocation, even though the arrays themselves are small.
    jax.clear_caches()
    gc.collect()

    # Drop `samples_per_component` (the multi-GB per-period
    # `(n_halton, n_obs, n_state)` importance buffer) from every
    # conditional distribution BEFORE materialising anything else.
    # Otherwise the next `_to_numpy(c.mean)` call has to fit a staging
    # buffer alongside live `samples_per_component` device arrays and
    # OOMs. Mutating the list in place and forcing a GC pass releases
    # the underlying device buffers immediately; only the small
    # summary stats and chain history remain on the GPU when conversion
    # starts.
    for idx, cd in enumerate(conditional_dists):
        conditional_dists[idx] = dataclasses.replace(cd, samples_per_component=())
    del cd
    gc.collect()
    jax.clear_caches()

    # Materialise every remaining JAX array in the result as a numpy
    # array. Downstream consumers (pickling, plotting, posterior_states)
    # don't need GPU residency, and leaving the arrays as `jax.Array`
    # would force materialisation at pickle time -- which on a busy
    # device routinely OOMs inside `__reduce__`.
    #
    # Skip this transfer entirely when the caller has opted out via
    # `keep_conditional_distributions=False` (e.g. small-GPU runs on
    # P100 12 GB where the per-component arrays alone exceed the
    # remaining device free memory).
    if af_options.keep_conditional_distributions:
        conditional_dists_compact = tuple(
            _to_numpy_conditional_distribution(cd) for cd in conditional_dists
        )
    else:
        conditional_dists_compact = ()
        del conditional_dists
        gc.collect()
        jax.clear_caches()

    return AFEstimationResult(
        period_results=tuple(period_results),
        all_params=all_params,
        model_spec=model_spec,
        conditional_distributions=conditional_dists_compact,
    )


def _to_numpy(value: Array | np.ndarray | None) -> np.ndarray | None:
    """Materialise a JAX array as numpy; pass `None` through."""
    if value is None:
        return None
    return np.asarray(jax.device_get(value))


def _to_numpy_chain_link(link: ChainLink) -> ChainLink:
    """Convert every JAX field of a `ChainLink` to numpy."""
    return dataclasses.replace(
        link,
        transition_params=_to_numpy(link.transition_params),
        shock_sds=_to_numpy(link.shock_sds),
        shock_factor_indices=_to_numpy(link.shock_factor_indices),
        inv_eq_params=_to_numpy(link.inv_eq_params),
        inv_sds=_to_numpy(link.inv_sds),
        obs_factor_values=_to_numpy(link.obs_factor_values),
    )


def _to_numpy_conditional_distribution(
    cond_dist: ConditionalDistribution,
) -> ConditionalDistribution:
    """Convert all arrays to numpy and drop `samples_per_component`."""
    new_components = tuple(
        MixtureComponent(
            mean=_to_numpy(c.mean),  # ty: ignore[invalid-argument-type]
            chol_cov=_to_numpy(c.chol_cov),  # ty: ignore[invalid-argument-type]
        )
        for c in cond_dist.components
    )
    new_chain_links = tuple(_to_numpy_chain_link(cl) for cl in cond_dist.chain_links)
    return dataclasses.replace(
        cond_dist,
        mixture_weights=_to_numpy(cond_dist.mixture_weights),
        components=new_components,
        samples_per_component=(),
        conditional_weights=_to_numpy(cond_dist.conditional_weights),
        cond_means=_to_numpy(cond_dist.cond_means),
        cond_chols=_to_numpy(cond_dist.cond_chols),
        chain_links=new_chain_links,
    )


def _extract_period_data(
    data: pd.DataFrame,
    n_periods: int,
    _factors: tuple[str, ...],
    controls_names: tuple[str, ...],
    model_spec: ModelSpec,
    observed_factors: tuple[str, ...] = (),
) -> dict[int, dict[str, Array]]:
    """Extract measurement, control, and observed factor arrays per period.

    Return:
        Dict mapping period -> {"measurements": Array, "controls": Array,
        "observed_factors": Array (if any)}.

    """
    period_data: dict[int, dict[str, Array]] = {}

    idx_names = data.index.names
    period_col = str(idx_names[1])

    for t in range(n_periods):
        measurements_pt = get_measurements_per_factor(model_spec.factors, period=t)
        if not measurements_pt:
            continue

        all_measures: list[str] = []
        seen: set[str] = set()
        for measures in measurements_pt.values():
            for m in measures:
                if m not in seen:
                    seen.add(m)
                    all_measures.append(m)

        period_mask = data.index.get_level_values(period_col) == t
        period_df = data.loc[period_mask]

        meas_cols = [c for c in all_measures if c in period_df.columns]
        meas_array = jnp.array(
            period_df[meas_cols].to_numpy(dtype=np.float64, na_value=np.nan),
        )

        ctrl_arrays = []
        for ctrl in controls_names:
            if ctrl == "constant":
                ctrl_arrays.append(np.ones(len(period_df)))
            elif ctrl in period_df.columns:
                ctrl_arrays.append(period_df[ctrl].to_numpy(dtype=np.float64))
            else:
                ctrl_arrays.append(np.zeros(len(period_df)))
        ctrl_array = jnp.array(np.column_stack(ctrl_arrays))

        entry: dict[str, Array] = {
            "measurements": meas_array,
            "controls": ctrl_array,
        }

        if observed_factors:
            entry["observed_factors"] = _extract_observed_factors(
                period_df, observed_factors
            )

        period_data[t] = entry

    return period_data


def _extract_observed_factors(
    period_df: pd.DataFrame,
    observed_factors: tuple[str, ...],
) -> Array:
    """Extract observed factor values from a period's DataFrame."""
    obs_arrays = [
        period_df[of].to_numpy(dtype=np.float64)
        if of in period_df.columns
        else np.zeros(len(period_df))
        for of in observed_factors
    ]
    return jnp.array(np.column_stack(obs_arrays))


def _extract_equality_groups(
    constraints: list[om.constraints.Constraint] | None,
) -> list[pd.MultiIndex]:
    """Pull cross-period equality groups out of an optimagic constraints list.

    Honours `om.EqualityConstraint` instances whose selector is built via
    `functools.partial(skillmodels.common.constraints.select_by_loc, loc=...)`.
    The `loc` keyword carries the `pd.MultiIndex` of params that must be
    equal — those are the equality groups returned here.
    """
    if not constraints:
        return []
    groups: list[pd.MultiIndex] = []
    for c in constraints:
        if not isinstance(c, om.EqualityConstraint):
            continue
        selector = c.selector
        keywords = getattr(selector, "keywords", None)
        if not keywords or "loc" not in keywords:
            continue
        loc = keywords["loc"]
        if isinstance(loc, pd.MultiIndex) and len(loc) > 1:
            groups.append(loc)
    return groups


def _propagate_equality_groups(
    *,
    period_results: list[AFPeriodResult],
    fixed_params: pd.DataFrame | None,
    equality_groups: list[pd.MultiIndex],
) -> pd.DataFrame | None:
    """Propagate just-estimated values to all members of cross-period equality groups.

    For each equality group: if any member is in the union of
    `period_results[*].params`, pin every other member of the group
    (that is not already pinned by `fixed_params`) to that member's
    estimated value via additions to `fixed_params`. Subsequent
    periods' MLEs see those entries as fixed, enforcing equality
    across the chain.
    """
    if not equality_groups:
        return fixed_params

    estimated = pd.concat([r.params for r in period_results])
    if "value" in estimated.columns:
        estimated_series = estimated["value"]
    else:
        estimated_series = estimated.iloc[:, 0]

    if fixed_params is None or len(fixed_params) == 0:
        index_names = ["category", "period", "name1", "name2"]
        running = pd.DataFrame(
            {"value": []},
            index=pd.MultiIndex.from_tuples([], names=index_names),
        )
    else:
        running = fixed_params.copy()

    new_locs: list[tuple] = []
    new_values: list[float] = []
    for group in equality_groups:
        in_estimated = [loc for loc in group if loc in estimated_series.index]
        if not in_estimated:
            continue
        anchor_value = float(estimated_series.loc[in_estimated[0]])
        for loc in group:
            if loc in running.index:
                continue
            new_locs.append(loc)
            new_values.append(anchor_value)
    if not new_locs:
        return running

    addition = pd.DataFrame(
        {"value": new_values},
        index=pd.MultiIndex.from_tuples(new_locs, names=running.index.names),
    )
    return pd.concat([running, addition])
