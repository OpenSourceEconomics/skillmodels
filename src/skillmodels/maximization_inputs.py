"""Functions to create inputs for optimization of the log-likelihood."""

import functools
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

import skillmodels.likelihood_function as lf
import skillmodels.likelihood_function_debug as lfd
from skillmodels.constraints import (
    add_bounds,
    constraints_dicts_to_om,
    enforce_fixed_constraints,
    get_constraints_dicts,
)
from skillmodels.kalman_filters import calculate_sigma_scaling_factor_and_weights
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.process_data import process_data
from skillmodels.process_debug_data import process_debug_data
from skillmodels.process_model import process_model

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from skillmodels.types import ProcessedModel

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def get_maximization_inputs(
    model_dict: dict,
    data: pd.DataFrame,
    split_dataset: int = 1,
) -> dict[str, Any]:
    """Create inputs for optimagic's maximize function.

    Args:
        model_dict: The model specification. See: :ref:`model_specs`
        data: dataset in long format.
        split_dataset(Int): Controls into how many sclices to split the dataset
            during the gradient computation.

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
            model specification.
        params_template: Parameter DataFrame with correct index and
            bounds. The value column is empty except for the fixed constraints, which
            are set including the bounds.
        data_aug: DataFrame with augmented data. If model contains
            endogenous factors, we double up the number of periods in order to add

    """
    model = process_model(model_dict)
    p_index = get_params_index(
        update_info=model.update_info,
        labels=model.labels,
        dimensions=model.dimensions,
        transition_info=model.transition_info,
        endogenous_factors_info=model.endogenous_factors_info,
    )

    parsing_info = create_parsing_info(
        params_index=p_index,
        update_info=model.update_info,
        labels=model.labels,
        anchoring=model.anchoring,
        has_endogenous_factors=model.endogenous_factors_info.has_endogenous_factors,
    )
    processed_data = process_data(
        df=data,
        has_endogenous_factors=model.endogenous_factors_info.has_endogenous_factors,
        labels=model.labels,
        update_info=model.update_info,
        anchoring_info=model.anchoring,
        purpose="estimation",
    )

    sigma_scaling_factor, sigma_weights = calculate_sigma_scaling_factor_and_weights(
        model.dimensions.n_latent_factors,
        model.estimation_options.sigma_points_scale,
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
            model=model,
            sigma_weights=sigma_weights,
            sigma_scaling_factor=sigma_scaling_factor,
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
        return process_debug_data(debug_data=tmp, model=model)

    _constraints_dicts = get_constraints_dicts(
        dimensions=model.dimensions,
        labels=model.labels,
        anchoring_info=model.anchoring,
        update_info=model.update_info,
        normalizations=model.normalizations,
        endogenous_factors_info=model.endogenous_factors_info,
    )

    constraints = constraints_dicts_to_om(_constraints_dicts)

    params_template = pd.DataFrame(columns=["value"], index=p_index)
    params_template = add_bounds(
        params=params_template,
        bounds_distance=model.estimation_options.bounds_distance,
    )
    params_template = enforce_fixed_constraints(
        params_template=params_template,
        constraints_dicts=_constraints_dicts,
    )
    if not params_template.index.equals(p_index):
        raise ValueError("params_template index is not equal to p_index")
    return {
        "loglike": loglike,
        "loglikeobs": loglikeobs,
        "debug_loglike": debug_loglike,
        "loglike_and_gradient": loglike_and_gradient,
        "constraints": constraints,
        "params_template": params_template,
    }


def _partial_some_log_likelihood(
    fun: Callable,
    parsing_info: dict[str, Any],
    measurements: Array,
    controls: Array,
    observed_factors: Array,
    model: ProcessedModel,
    sigma_weights: Array,
    sigma_scaling_factor: Array,
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
        if parsing_info["has_endogenous_factors"]
        else model.labels.aug_periods[-1]
    )
    iteration_to_period = _aug_periods.replace(last_aug_period, -1).to_numpy()
    if max(iteration_to_period) != last_aug_period - 1:
        raise ValueError("Unexpected iteration_to_period configuration")

    return functools.partial(
        fun,
        parsing_info=parsing_info,
        measurements=measurements,
        controls=controls,
        transition_func=model.transition_info.func,
        sigma_scaling_factor=sigma_scaling_factor,
        sigma_weights=sigma_weights,
        dimensions=model.dimensions,
        labels=model.labels,
        estimation_options=model.estimation_options,
        is_measurement_iteration=is_measurement_iteration,
        is_predict_iteration=is_predict_iteration,
        iteration_to_period=iteration_to_period,
        observed_factors=observed_factors,
    )


def _to_numpy(obj: Any) -> Any:
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
