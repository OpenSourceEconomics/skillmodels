"""Functions to create inputs for optimization of the log-likelihood."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from numpy.typing import NDArray

import skillmodels.chs.likelihood as lf
import skillmodels.chs.likelihood_debug as lfd
from skillmodels.amn.estimate import estimate_amn
from skillmodels.amn.start_values import get_spearman_start_params
from skillmodels.chs.kalman_filters import (
    calculate_sigma_scaling_factor_and_weights,
    is_all_linear,
    kalman_predict,
    linear_kalman_predict,
)
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.process_debug_data import process_debug_data
from skillmodels.common.constraints import (
    FixedConstraintWithValue,
    add_bounds,
    align_index_names,
    enforce_fixed_constraints,
    get_constraints,
    project_to_probability_constraints,
)
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.params_index import get_params_index
from skillmodels.common.parse_params import create_parsing_info
from skillmodels.common.process_data import process_data
from skillmodels.common.process_model import process_model
from skillmodels.common.types import ParsingInfo, ProcessedModel

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def get_maximization_inputs(  # noqa: C901, PLR0915
    model_spec: ModelSpec,
    data: pd.DataFrame,
    split_dataset: int = 1,
    *,
    chs_options: CHSEstimationOptions | None = None,
    fixed_params: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Create inputs for optimagic's maximize function.

    Args:
        model_spec: The model specification. See: :ref:`model_specs`
        data: Dataset in long format.
        split_dataset: Controls into how many slices to split the dataset
            during the gradient computation.
        chs_options: CHS-specific tuning parameters. Defaults to
            ``CHSEstimationOptions()`` when not provided.
        fixed_params: Optional DataFrame with a ``"value"`` column pinning
            specified parameters to fixed values. Uses the same 4-level
            MultiIndex as the returned ``params_template``. Each matching
            entry becomes a `FixedConstraintWithValue` in the returned
            constraints list, so optimagic holds the parameter at the given
            value during optimization. When a fix overlaps a
            `ProbabilityConstraint` selector (e.g., a gamma of a ``log_ces``
            transition), optimagic's fold machinery keeps the remaining free
            entries on the implied simplex (see
            ``optimagic.ProbabilityConstraint``).

    Returns a dictionary with keys:
        loglike: A jax jitted function that takes an optimagic-style
            params dataframe as only input and returns a dict with entries:
            - "value": The scalar log likelihood
            - "contributions": An array with the log likelihood per observation
        debug_loglike: Similar to loglike, with the following differences:
            - It is not jitted and thus faster on the first call and debuggable
            - It will add intermediate results as additional entries in the returned
              dictionary. Those can be used for debugging and plotting.
        gradient: The gradient of the scalar log likelihood
            function with respect to the parameters.
        loglike_and_gradient: Combination of loglike and
            loglike_gradient that is faster than calling the two functions separately.
        constraints: List of optimagic constraints that are implied by the
            model specification, extended by any user-supplied ``fixed_params``.
        params_template: Parameter DataFrame with correct index and
            bounds. The value column is empty except for the fixed constraints, which
            are set including the bounds.
        data_aug: DataFrame with augmented data. If model contains
            endogenous factors, we double up the number of periods in order to add

    """
    chs_options = chs_options or CHSEstimationOptions()
    processed_model = process_model(model_spec)
    p_index = get_params_index(
        update_info=processed_model.update_info,
        labels=processed_model.labels,
        dimensions=processed_model.dimensions,
        transition_info=processed_model.transition_info,
        endogenous_factors_info=processed_model.endogenous_factors_info,
    )

    parsing_info = create_parsing_info(
        params_index=p_index,
        update_info=processed_model.update_info,
        labels=processed_model.labels,
        anchoring=processed_model.anchoring,
        has_endogenous_factors=processed_model.endogenous_factors_info.has_endogenous_factors,
    )
    processed_data = process_data(
        df=data,
        has_endogenous_factors=processed_model.endogenous_factors_info.has_endogenous_factors,
        labels=processed_model.labels,
        update_info=processed_model.update_info,
        anchoring_info=processed_model.anchoring,
        purpose="estimation",
    )

    sigma_scaling_factor, sigma_weights = calculate_sigma_scaling_factor_and_weights(
        n_states=processed_model.dimensions.n_latent_factors,
        kappa=chs_options.sigma_points_scale,
    )

    partialed_get_jnp_params_vec = functools.partial(
        _get_jnp_params_vec,
        target_index=p_index,
    )

    partialed_loglikes = {}
    for n, fun in {
        "ll": lf.log_likelihood,
        "llo": lf.log_likelihood_obs,
        "debug_ll": lfd.log_likelihood,
    }.items():
        partialed_loglikes[n] = _partial_some_log_likelihood(
            fun=fun,
            parsing_info=parsing_info,
            measurements=processed_data["measurements"],
            controls=processed_data["controls"],
            observed_factors=processed_data["observed_factors"],
            model=processed_model,
            sigma_weights=sigma_weights,
            sigma_scaling_factor=sigma_scaling_factor,
            chs_options=chs_options,
        )

    _jitted_loglike = jax.jit(partialed_loglikes["ll"])
    _jitted_loglikeobs = jax.jit(partialed_loglikes["llo"])
    _gradient = jax.jit(jax.grad(partialed_loglikes["ll"]))

    def loglike(params: pd.DataFrame) -> float:
        params_vec = partialed_get_jnp_params_vec(params)
        return float(_jitted_loglike(params_vec))

    def loglikeobs(params: pd.DataFrame) -> NDArray[np.floating]:
        params_vec = partialed_get_jnp_params_vec(params)
        return _to_numpy(_jitted_loglikeobs(params_vec))

    def loglike_and_gradient(
        params: pd.DataFrame,
    ) -> tuple[float, NDArray[np.floating]]:
        params_vec = partialed_get_jnp_params_vec(params)
        crit = float(_jitted_loglike(params_vec))
        n_obs = processed_data["measurements"].shape[1]
        _grad = jnp.zeros_like(params_vec)
        start = 0
        stop = int(n_obs / split_dataset)
        step = int(n_obs / split_dataset)
        for i in range(split_dataset):
            stop = n_obs if i == split_dataset - 1 else stop
            measurements_slice = processed_data["measurements"][:, start:stop]
            controls_slice = processed_data["controls"][:, start:stop, :]
            observed_factors_slice = processed_data["observed_factors"][
                :, start:stop, :
            ]
            _grad += _gradient(
                params_vec,
                measurements=measurements_slice,
                controls=controls_slice,
                observed_factors=observed_factors_slice,
            )
            start += step
            stop += step
        grad = _to_numpy(_grad)
        return crit, grad

    def debug_loglike(params: pd.DataFrame) -> dict[str, Any]:
        params_vec = partialed_get_jnp_params_vec(params)
        jax_output = partialed_loglikes["debug_ll"](params_vec)
        tmp = _to_numpy(jax_output)
        tmp["value"] = float(tmp["value"])
        return process_debug_data(debug_data=tmp, model=processed_model)

    constraints = get_constraints(
        dimensions=processed_model.dimensions,
        labels=processed_model.labels,
        anchoring_info=processed_model.anchoring,
        update_info=processed_model.update_info,
        normalizations=processed_model.normalizations,
        endogenous_factors_info=processed_model.endogenous_factors_info,
        bounds_distance=chs_options.bounds_distance,
    )

    if fixed_params is not None:
        fixed_constraints = _build_fixed_constraints_from_params(
            fixed_params, params_index=p_index
        )
        constraints = list(constraints) + fixed_constraints

    params_template = pd.DataFrame(columns=["value"], index=p_index)
    params_template = add_bounds(
        params=params_template,
        bounds_distance=chs_options.bounds_distance,
    )
    params_template = enforce_fixed_constraints(
        params_template=params_template,
        constraints=constraints,
    )
    if not params_template.index.equals(p_index):
        raise ValueError("params_template index is not equal to p_index")

    strategy = chs_options.start_params_strategy
    if strategy == "spearman":
        params_template = get_spearman_start_params(
            model_spec=model_spec,
            data=data,
            params_template=params_template,
        )
    elif strategy == "amn":
        amn_result = estimate_amn(model_spec=model_spec, data=data)
        # First fill template via Spearman for entries AMN doesn't touch
        # (mixture weights, initial Cholesky diagonals not directly
        # produced by AMN's three stages); then overlay AMN values onto
        # the common index. Skip indices pre-pinned by
        # `enforce_fixed_constraints`.
        pre_pinned = params_template["value"].notna()
        params_template = get_spearman_start_params(
            model_spec=model_spec,
            data=data,
            params_template=params_template,
        )
        common = amn_result.all_params.index.intersection(params_template.index)
        free_common = common[~pre_pinned.reindex(common, fill_value=False)]
        params_template.loc[free_common, "value"] = amn_result.all_params.loc[
            free_common, "value"
        ]

    params_template = project_to_probability_constraints(
        params_template=params_template, constraints=constraints
    )

    return {
        "loglike": loglike,
        "loglikeobs": loglikeobs,
        "debug_loglike": debug_loglike,
        "loglike_and_gradient": loglike_and_gradient,
        "constraints": constraints,
        "params_template": params_template,
    }


def _build_fixed_constraints_from_params(
    fixed_params: pd.DataFrame,
    params_index: pd.MultiIndex,
) -> list[FixedConstraintWithValue]:
    """Convert a user-provided ``fixed_params`` DataFrame into constraints.

    Each matching row becomes a ``FixedConstraintWithValue`` so optimagic
    can treat user fixes uniformly with model-implied fixes (normalisations,
    anchoring, augmented periods, ...). Entries whose index is not in
    ``params_index`` are ignored.

    Users typically key `fixed_params` by the public-facing `period`
    level name, while `params_index` uses `aug_period` internally.
    `MultiIndex.intersection` silently returns an empty index when
    the operands' level names differ, so the level names are
    normalised first via `align_index_names`.
    """
    aligned = align_index_names(fixed_params, target_names=params_index.names)
    common = params_index.intersection(aligned.index)
    return [
        FixedConstraintWithValue(
            loc=idx,
            value=float(aligned.loc[idx, "value"]),
        )
        for idx in common
    ]


def _partial_some_log_likelihood(
    fun: Callable,
    parsing_info: ParsingInfo,
    measurements: Array,
    controls: Array,
    observed_factors: Array,
    model: ProcessedModel,
    sigma_weights: Array,
    sigma_scaling_factor: Array,
    chs_options: CHSEstimationOptions,
) -> Callable:
    update_info = model.update_info
    is_measurement_iteration = (update_info["purpose"] == "measurement").to_numpy()
    _aug_periods = pd.Series(
        update_info.index.get_level_values("aug_period").to_numpy()
    )
    is_predict_iteration = ((_aug_periods - _aug_periods.shift(-1)) == -1).to_numpy()
    # iteration_to_period is used as an indexer to loop over arrays of different lengths
    # in a jax.lax.scan. It needs to work for arrays of length n_aug_periods and not
    # raise IndexErrors on tracer arrays of length n_aug_periods - 1 (i.e.
    # n_transitions). To achieve that, we replace the last aug_period by -1. If there
    # are endogenous factors, the last aug_period is found at index -2 (there should not
    # be measurements for endogenous factors in the "second half" of the last period).
    last_aug_period = (
        model.labels.aug_periods[-2]
        if parsing_info.has_endogenous_factors
        else model.labels.aug_periods[-1]
    )
    iteration_to_period = _aug_periods.replace(last_aug_period, -1).to_numpy()
    if max(iteration_to_period) != last_aug_period - 1:
        raise ValueError("Unexpected iteration_to_period configuration")

    if is_all_linear(model.transition_info.function_names):
        constant_factor_indices = frozenset(
            i
            for i, f in enumerate(model.labels.latent_factors)
            if model.transition_info.function_names[f] == "constant"
        )
        predict_func = functools.partial(
            linear_kalman_predict,
            model.transition_info.func,
            latent_factors=model.labels.latent_factors,
            constant_factor_indices=constant_factor_indices,
            n_all_factors=model.dimensions.n_all_factors,
        )
    else:
        predict_func = functools.partial(kalman_predict, model.transition_info.func)

    return functools.partial(
        fun,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        predict_func=predict_func,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        dimensions=model.dimensions,
        labels=model.labels,
        chs_estimation_options=chs_options,
        is_measurement_iteration=is_measurement_iteration,
        is_predict_iteration=is_predict_iteration,
        iteration_to_period=iteration_to_period,
        observed_factors=observed_factors,
    )


def _to_numpy(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, dict):
        res = {}
        for key, value in obj.items():
            if np.isscalar(value):
                res[key] = value
            else:
                res[key] = np.array(value)

    elif np.isscalar(obj):
        res = obj
    else:
        res = np.array(obj)

    return res


def _get_jnp_params_vec(params: pd.DataFrame, target_index: pd.MultiIndex) -> Array:
    if set(params.index) != set(target_index):
        additional_entries = params.index.difference(target_index).tolist()
        missing_entries = target_index.difference(params.index).tolist()
        msg = "Invalid params DataFrame. "
        if additional_entries:
            msg += f"Your params have additional entries: {additional_entries}. "
        if missing_entries:
            msg += f"Your params have missing entries: {missing_entries}. "
        raise ValueError(msg)

    return jnp.array(params.reindex(target_index)["value"].to_numpy())
