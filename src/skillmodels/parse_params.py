import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd


def create_parsing_info(params_index, update_info, labels, anchoring):
    """Create a dictionary with information how the parameter vector has to be parsed.

    Args:
        params_index (pandas.MultiIndex): It has the levels ["category", "period",
            "name1", "name2"]
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`

    Returns:
        dict: dictionary that maps model quantities to positions or slices of the
            parameter vector.

    """
    range_sr = pd.Series(data=np.arange(len(params_index)), index=params_index)

    parsing_info = {}

    simple_ones = [
        "initial_states",
        "initial_cholcovs",
        "mixture_weights",
        "controls",
        "meas_sds",
        "shock_sds",
    ]

    for quantity in simple_ones:
        parsing_info[quantity] = _get_positional_selector_from_loc(range_sr, quantity)

    # loadings:
    mask = update_info[labels["latent_factors"]].to_numpy()
    helper = np.arange(mask.size).reshape(mask.shape)
    flat_indices = helper[mask]

    parsing_info["loadings"] = {
        "slice": _get_positional_selector_from_loc(range_sr, "loadings"),
        "flat_indices": jnp.array(flat_indices),
        "shape": mask.shape,
        "size": mask.size,
    }

    # "trans_coeffs"
    pos_dict = {}
    for factor in labels["latent_factors"]:
        helper = pd.DataFrame(index=params_index)
        loc = helper.query(f"category == 'transition' & name1 == '{factor}'").index
        pos_dict[factor] = _get_positional_selector_from_loc(range_sr, loc)

    parsing_info["transition"] = pos_dict

    # anchoring_scaling_factors
    is_free_loading = update_info[labels["latent_factors"]].to_numpy()
    is_anchoring = (update_info["purpose"] == "anchoring").to_numpy().reshape(-1, 1)
    is_anchoring_loading = jnp.array(is_free_loading & is_anchoring)
    parsing_info["is_anchoring_loading"] = is_anchoring_loading
    parsing_info["is_anchored_factor"] = jnp.array(
        update_info.query("purpose == 'anchoring'")[labels["latent_factors"]].any(
            axis=0,
        ),
    )
    parsing_info["is_anchoring_update"] = is_anchoring.flatten()
    parsing_info["ignore_constant_when_anchoring"] = anchoring[
        "ignore_constant_when_anchoring"
    ]

    return parsing_info


def _get_positional_selector_from_loc(range_sr, loc):
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


def parse_params(params, parsing_info, dimensions, labels, n_obs):
    """Parse params into the quantities that depend on it.

    Args:
        params (jax.numpy.array): 1d array with model parameters.
        parsing_info (dict): Dictionary with information on how the parameters
            have to be parsed.
        dimensions (dict): Dimensional information like n_states, n_periods, n_controls,
            n_mixtures. See :ref:`dimensions`.
        n_obs (int): Number of observations.

    Returns:
        jax.numpy.array: Array of shape (n_obs, n_mixtures, n_states) with initial
            state estimates.
        jax.numpy.array: Array of shape (n_obs, n_mixtures, n_states, n_states) with the
            transpose of the lower triangular cholesky factors of the initial covariance
            matrices.
        jax.numpy.array: Array of shape (n_obs, n_mixtures) with the log of the initial
            weight for each element in the finite mixture of normals.
        dict: Dictionary with other parameters. It has the following key-value pairs:
            - "control_params":
            - "loadings":
            - "meas_sds":
            - "shock_sds":
            - "trans_params":
            - "anchoring_scaling_factors":
            - "anchoring_constants":

    """
    states = _get_initial_states(params, parsing_info, dimensions, n_obs)
    upper_chols = _get_initial_upper_chols(params, parsing_info, dimensions, n_obs)
    log_weights = _get_initial_log_mixture_weights(params, parsing_info, n_obs)
    pardict = {
        "controls": _get_control_params(params, parsing_info, dimensions),
        "loadings": _get_loadings(params, parsing_info),
        "meas_sds": _get_meas_sds(params, parsing_info),
        "shock_sds": _get_shock_sds(params, parsing_info, dimensions),
        "transition": _get_transition_params(params, parsing_info, labels),
    }

    pardict["anchoring_scaling_factors"] = _get_anchoring_scaling_factors(
        pardict["loadings"],
        parsing_info,
        dimensions,
    )

    pardict["anchoring_constants"] = _get_anchoring_constants(
        pardict["controls"],
        parsing_info,
        dimensions,
    )

    return states, upper_chols, log_weights, pardict


def _get_initial_states(params, info, dimensions, n_obs):
    """Create the array of initial states."""
    state = params[info["initial_states"]].reshape(
        1,
        dimensions["n_mixtures"],
        dimensions["n_latent_factors"],
    )
    return jnp.repeat(state, n_obs, axis=0)


def _get_initial_upper_chols(params, info, dimensions, n_obs):
    """Create the array with cholesky factors of the initial states covariance matrix.

    Note: The matrices contain the transpose of the lower triangular cholesky factors.

    """
    n_states, n_mixtures = dimensions["n_latent_factors"], dimensions["n_mixtures"]
    chol_params = params[info["initial_cholcovs"]].reshape(n_mixtures, -1)
    upper_chols = jnp.zeros((n_obs, n_mixtures, n_states, n_states))
    for i in range(n_mixtures):
        filler = jnp.zeros((n_states, n_states))
        filler = filler.at[jnp.tril_indices(n_states)].set(chol_params[i])
        upper_chols = upper_chols.at[:, i].set(filler.T)
    return upper_chols


def _get_initial_log_mixture_weights(params, info, n_obs):
    """Create the array with the log of initial mixture weights."""
    log_weights = jnp.log(params[info["mixture_weights"]]).reshape(1, -1)
    return jnp.repeat(log_weights, n_obs, axis=0)


def _get_control_params(params, info, dimensions):
    """Create the parameters for control variables in measurement equations."""
    return params[info["controls"]].reshape(-1, dimensions["n_controls"])


def _get_loadings(params, info):
    """Create the array of factor loadings."""
    info = info["loadings"]
    free = params[info["slice"]]
    extended = jnp.zeros(info["size"]).at[info["flat_indices"]].set(free)
    out = extended.reshape(info["shape"])
    return out


def _get_meas_sds(params, info):
    """Create the array of standard deviations of the measurement errors."""
    return params[info["meas_sds"]]


def _get_shock_sds(params, info, dimensions):
    """Create the array of standard deviations of the shocks in transition functions."""
    return params[info["shock_sds"]].reshape(-1, dimensions["n_latent_factors"])


def _get_transition_params(params, info, labels):
    """Create a list of arrays with transition equation parameters."""
    trans_params = {}
    t_info = info["transition"]
    n_periods = len(labels["periods"])
    for factor in labels["latent_factors"]:
        ilocs = t_info[factor]
        trans_params[factor] = params[ilocs].reshape(n_periods - 1, -1)
    return trans_params


def _get_anchoring_scaling_factors(loadings, info, dimensions):
    """Create an array of anchoring scaling factors.

    Note: Parameters are not taken from the parameter vector but from the loadings.

    """
    scaling_factors = jnp.ones(
        (dimensions["n_periods"], dimensions["n_latent_factors"]),
    )
    free_anchoring_loadings = loadings[info["is_anchoring_loading"]].reshape(
        dimensions["n_periods"],
        -1,
    )
    scaling_factors = scaling_factors.at[:, info["is_anchored_factor"]].set(
        free_anchoring_loadings,
    )

    scaling_for_observed = jnp.ones(
        (dimensions["n_periods"], dimensions["n_observed_factors"]),
    )

    scaling_factors = jnp.hstack([scaling_factors, scaling_for_observed])

    return scaling_factors


def _get_anchoring_constants(controls, info, dimensions):
    """Create an array of anchoring constants.

    Note: Parameters are not taken from the parameter vector but from the controls.

    """
    constants = jnp.zeros((dimensions["n_periods"], dimensions["n_latent_factors"]))
    if not info["ignore_constant_when_anchoring"]:
        values = controls[:, 0][info["is_anchoring_update"]].reshape(
            dimensions["n_periods"],
            -1,
        )
        constants = constants.at[:, info["is_anchored_factor"]].set(values)

    constants_for_observed = jnp.zeros(
        (dimensions["n_periods"], dimensions["n_observed_factors"]),
    )

    constants = jnp.hstack([constants, constants_for_observed])

    return constants
