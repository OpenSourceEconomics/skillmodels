"""Functions to parse parameter vectors into structured dictionaries."""

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from skillmodels.types import LoadingsParsingInfo, ParsedParams, ParsingInfo

if TYPE_CHECKING:
    from skillmodels.types import Anchoring, Dimensions, Labels


def create_parsing_info(
    params_index: pd.MultiIndex,
    update_info: pd.DataFrame,
    labels: Labels,
    anchoring: Anchoring,
    *,
    has_endogenous_factors: bool,
) -> ParsingInfo:
    """Create a dataclass with information how the parameter vector has to be parsed.

    Args:
        params_index: It has the levels ["category", "aug_period",
            "name1", "name2"]
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        labels: Labels dataclass with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        anchoring: Anchoring dataclass with anchoring settings.
        has_endogenous_factors: Whether the model includes endogenous factors.

    Returns:
        ParsingInfo dataclass that maps model quantities to positions or slices of the
            parameter vector.

    """
    range_sr = pd.Series(data=np.arange(len(params_index)), index=params_index)

    # Simple quantities
    initial_states = _get_positional_selector_from_loc(range_sr, "initial_states")
    initial_cholcovs = _get_positional_selector_from_loc(range_sr, "initial_cholcovs")
    mixture_weights = _get_positional_selector_from_loc(range_sr, "mixture_weights")
    controls = _get_positional_selector_from_loc(range_sr, "controls")
    meas_sds = _get_positional_selector_from_loc(range_sr, "meas_sds")
    shock_sds = _get_positional_selector_from_loc(range_sr, "shock_sds")

    # loadings:
    mask = update_info[list(labels.latent_factors)].to_numpy()
    helper = np.arange(mask.size).reshape(mask.shape)
    flat_indices = helper[mask]

    loadings = LoadingsParsingInfo(
        slice=_get_positional_selector_from_loc(range_sr, "loadings"),
        flat_indices=jnp.array(flat_indices),
        shape=mask.shape,
        size=mask.size,
    )

    # transition coefficients
    transition: dict[str, Array | slice] = {}
    for factor in list(labels.latent_factors):
        helper_df = pd.DataFrame(index=params_index)
        loc = helper_df.query(f"category == 'transition' & name1 == '{factor}'").index
        transition[factor] = _get_positional_selector_from_loc(range_sr, loc)

    # anchoring_scaling_factors
    is_free_loading = update_info[list(labels.latent_factors)].to_numpy()
    is_anchoring = (update_info["purpose"] == "anchoring").to_numpy().reshape(-1, 1)
    is_anchoring_loading = jnp.array(is_free_loading & is_anchoring)
    is_anchored_factor = jnp.array(
        update_info.query("purpose == 'anchoring'")[list(labels.latent_factors)].any(
            axis=0,
        ),
    )
    is_anchoring_update = jnp.array(is_anchoring.flatten())

    return ParsingInfo(
        initial_states=initial_states,
        initial_cholcovs=initial_cholcovs,
        mixture_weights=mixture_weights,
        controls=controls,
        meas_sds=meas_sds,
        shock_sds=shock_sds,
        loadings=loadings,
        transition=transition,
        is_anchoring_loading=is_anchoring_loading,
        is_anchored_factor=is_anchored_factor,
        is_anchoring_update=is_anchoring_update,
        ignore_constant_when_anchoring=anchoring.ignore_constant_when_anchoring,
        has_endogenous_factors=has_endogenous_factors,
    )


def _get_positional_selector_from_loc(
    range_sr: pd.Series,
    loc: str | pd.MultiIndex | pd.Index,
) -> Array | slice:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing past lexsort depth may impact performance.",
        )
        try:
            ilocs = jnp.array(range_sr.loc[loc])
        except KeyError:
            ilocs = slice(0, 0)
        except Exception:
            raise
    return ilocs


def parse_params(
    params: Array,
    parsing_info: ParsingInfo,
    dimensions: Dimensions,
    labels: Labels,
    n_obs: int,
) -> tuple[Array, Array, Array, ParsedParams]:
    """Parse params into the quantities that depend on it.

    Args:
        params: 1d array with model parameters.
        parsing_info: ParsingInfo dataclass with information on how the parameters
            have to be parsed.
        dimensions: Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        labels: Labels dataclass with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        n_obs: Number of observations.

    Returns:
        Tuple of:
            - Array of shape (n_obs, n_mixtures, n_states) with initial state estimates.
            - Array of shape (n_obs, n_mixtures, n_states, n_states) with the transpose
              of the lower triangular cholesky factors of the initial covariance
              matrices.
            - Array of shape (n_obs, n_mixtures) with the log of the initial weight for
              each element in the finite mixture of normals.
            - ParsedParams dataclass with other model parameters.

    """
    states = _get_initial_states(params, parsing_info, dimensions, n_obs)
    upper_chols = _get_initial_upper_chols(params, parsing_info, dimensions, n_obs)
    log_weights = _get_initial_log_mixture_weights(params, parsing_info, n_obs)

    controls = _get_control_params(params, parsing_info, dimensions)
    loadings = _get_loadings(params, parsing_info)
    meas_sds = _get_meas_sds(params, parsing_info)
    shock_sds = _get_shock_sds(params, parsing_info, dimensions)
    transition = _get_transition_params(params, parsing_info, labels)

    anchoring_scaling_factors = _get_anchoring_scaling_factors(
        loadings,
        parsing_info,
        dimensions,
    )

    anchoring_constants = _get_anchoring_constants(
        controls,
        parsing_info,
        dimensions,
    )

    parsed = ParsedParams(
        controls=controls,
        loadings=loadings,
        meas_sds=meas_sds,
        shock_sds=shock_sds,
        transition=transition,
        anchoring_scaling_factors=anchoring_scaling_factors,
        anchoring_constants=anchoring_constants,
    )

    return states, upper_chols, log_weights, parsed


def _get_initial_states(
    params: Array,
    info: ParsingInfo,
    dimensions: Dimensions,
    n_obs: int,
) -> Array:
    """Create the array of initial states."""
    state = params[info.initial_states].reshape(
        1,
        dimensions.n_mixtures,
        dimensions.n_latent_factors,
    )
    return jnp.repeat(state, n_obs, axis=0)


def _get_initial_upper_chols(
    params: Array,
    info: ParsingInfo,
    dimensions: Dimensions,
    n_obs: int,
) -> Array:
    """Create the array with cholesky factors of the initial states covariance matrix.

    Note: The matrices contain the transpose of the lower triangular cholesky factors.

    """
    n_states, n_mixtures = dimensions.n_latent_factors, dimensions.n_mixtures
    chol_params = params[info.initial_cholcovs].reshape(n_mixtures, -1)
    upper_chols = jnp.zeros((n_obs, n_mixtures, n_states, n_states))
    for i in range(n_mixtures):
        filler = jnp.zeros((n_states, n_states))
        filler = filler.at[jnp.tril_indices(n_states)].set(chol_params[i])  # noqa: PD008
        upper_chols = upper_chols.at[:, i].set(filler.T)  # noqa: PD008
    return upper_chols


def _get_initial_log_mixture_weights(
    params: Array,
    info: ParsingInfo,
    n_obs: int,
) -> Array:
    """Create the array with the log of initial mixture weights."""
    log_weights = jnp.log(params[info.mixture_weights]).reshape(1, -1)
    return jnp.repeat(log_weights, n_obs, axis=0)


def _get_control_params(
    params: Array,
    info: ParsingInfo,
    dimensions: Dimensions,
) -> Array:
    """Create the parameters for control variables in measurement equations."""
    return params[info.controls].reshape(-1, dimensions.n_controls)


def _get_loadings(
    params: Array,
    info: ParsingInfo,
) -> Array:
    """Create the array of factor loadings."""
    loadings_info = info.loadings
    free = params[loadings_info.slice]
    extended = jnp.zeros(loadings_info.size).at[loadings_info.flat_indices].set(free)  # noqa: PD008
    return extended.reshape(loadings_info.shape)


def _get_meas_sds(
    params: Array,
    info: ParsingInfo,
) -> Array:
    """Create the array of standard deviations of the measurement errors."""
    return params[info.meas_sds]


def _get_shock_sds(
    params: Array,
    info: ParsingInfo,
    dimensions: Dimensions,
) -> Array:
    """Create the array of standard deviations of the shocks in transition functions."""
    return params[info.shock_sds].reshape(-1, dimensions.n_latent_factors)


def _get_transition_params(
    params: Array,
    info: ParsingInfo,
    labels: Labels,
) -> dict[str, Array]:
    """Create a list of arrays with transition equation parameters."""
    trans_params = {}
    n_aug_periods = len(labels.aug_periods)

    len_reduction = 2 if info.has_endogenous_factors else 1

    for factor in list(labels.latent_factors):
        ilocs = info.transition[factor]
        trans_params[factor] = params[ilocs].reshape(n_aug_periods - len_reduction, -1)
    return trans_params


def _get_anchoring_scaling_factors(
    loadings: Array,
    info: ParsingInfo,
    dimensions: Dimensions,
) -> Array:
    """Create an array of anchoring scaling factors.

    Note: Parameters are not taken from the parameter vector but from the loadings.

    """
    scaling_factors = jnp.ones(
        (dimensions.n_aug_periods, dimensions.n_latent_factors),
    )
    free_anchoring_loadings = loadings[info.is_anchoring_loading].reshape(
        dimensions.n_aug_periods,
        -1,
    )
    scaling_factors = scaling_factors.at[:, info.is_anchored_factor].set(  # noqa: PD008
        free_anchoring_loadings,
    )

    scaling_for_observed = jnp.ones(
        (dimensions.n_aug_periods, dimensions.n_observed_factors),
    )

    return jnp.hstack([scaling_factors, scaling_for_observed])


def _get_anchoring_constants(
    controls: Array,
    info: ParsingInfo,
    dimensions: Dimensions,
) -> Array:
    """Create an array of anchoring constants.

    Note: Parameters are not taken from the parameter vector but from the controls.

    """
    constants = jnp.zeros((dimensions.n_aug_periods, dimensions.n_latent_factors))
    if not info.ignore_constant_when_anchoring:
        values = controls[:, 0][info.is_anchoring_update].reshape(
            dimensions.n_aug_periods,
            -1,
        )
        constants = constants.at[:, info.is_anchored_factor].set(values)  # noqa: PD008

    constants_for_observed = jnp.zeros(
        (dimensions.n_aug_periods, dimensions.n_observed_factors),
    )

    return jnp.hstack([constants, constants_for_observed])
