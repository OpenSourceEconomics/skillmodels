import jax
import jax.numpy as jnp
from jax.ops import index
from jax.ops import index_add
from jax.ops import index_update


array_qr_jax = jax.vmap(jax.vmap(jnp.linalg.qr))


# ======================================================================================
# Update Step
# ======================================================================================


def kalman_update(
    states,
    upper_chols,
    loadings,
    control_params,
    meas_sd,
    measurements,
    controls,
    log_mixture_weights,
    debug,
):
    """Perform a Kalman update with likelihood evaluation.

    Args:
        states (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states) with
            pre-update states estimates.
        upper_chols (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states,
            n_states) with the transpose of the lower triangular cholesky factor
            of the pre-update covariance matrix of the state estimates.
        loadings (jax.numpy.array): 1d array of length n_states with factor loadings.
        control_params (jax.numpy.array): 1d array of length n_controls.
        meas_sd (float): Standard deviation of the measurement error.
        measurements (jax.numpy.array): 1d array of length n_obs with measurements.
            May contain NaNs if no measurement was observed.
        controls (jax.numpy.array): Array of shape (n_obs, n_controls) with data on the
            control variables.
        log_mixture_weights (jax.numpy.array): Array of shape (n_obs, n_mixtures) with
            the natural logarithm of the weights of each element of the mixture of
            normals distribution.
        debug (bool): If true, the debug_info contains the residuals of the update and
            their standard deviations. Otherwise, it is an empty dict.

    Returns:
        states (jax.numpy.array): Same format as states.
        new_states (jax.numpy.array): Same format as states.
        new_upper_chols (jax.numpy.array): Same format as upper_chols
        new_log_mixture_weights: (jax.numpy.array): Same format as log_mixture_weights
        new_loglikes: (jax.numpy.array): 1d array of length n_obs
        debug_info (dict): Empty or containing residuals and residual_sds

    """
    # find out which measurements are missing
    not_missing = jnp.isfinite(measurements)

    # reduce everything to non-missing entries. Variables that refer to the reduced
    # arrays or dimensions have a leading underscore in their name
    _states = states[not_missing]
    _upper_chols = upper_chols[not_missing]
    _measurements = measurements[not_missing]
    _log_mixture_weights = log_mixture_weights[not_missing]
    _controls = controls[not_missing]
    _n_obs, _n_mixtures, _n_states = _states.shape

    # actual square-root Kalman updates
    _expected_measurements = jnp.dot(_states, loadings) + jnp.dot(
        _controls, control_params
    ).reshape(_n_obs, 1)
    _residuals = _measurements.reshape(_n_obs, 1) - _expected_measurements
    _f_stars = jnp.dot(_upper_chols, loadings.reshape(_n_states, 1))

    _m = jnp.zeros((_n_obs, _n_mixtures, _n_states + 1, _n_states + 1))
    _m = index_update(_m, index[..., 0, 0], meas_sd)
    _m = index_update(_m, index[..., 1:, :1], _f_stars)
    _m = index_update(_m, index[..., 1:, 1:], _upper_chols)

    _r = array_qr_jax(_m)[1]

    _new_upper_chols = _r[..., 1:, 1:]
    _root_sigmas = _r[..., 0, 0]
    _abs_root_sigmas = jnp.abs(_root_sigmas)
    # it is important not to divide by the absolute value of _root_sigmas in order
    # to recover the sign of the Kalman gain.
    _kalman_gains = _r[..., 0, 1:] / _root_sigmas.reshape(_n_obs, _n_mixtures, 1)
    _new_states = _states + _kalman_gains * _residuals.reshape(_n_obs, _n_mixtures, 1)

    # calculate log likelihood per individual and update mixture weights
    _loglikes_per_dist = jax.scipy.stats.norm.logpdf(_residuals, 0, _abs_root_sigmas)
    if _n_mixtures >= 2:
        _weighted_loglikes_per_dist = _loglikes_per_dist + _log_mixture_weights
        _loglikes = jax.scipy.special.logsumexp(_weighted_loglikes_per_dist, axis=1)
        _new_log_mixture_weights = _weighted_loglikes_per_dist - _loglikes.reshape(
            -1, 1
        )

    else:
        _loglikes = _loglikes_per_dist.flatten()
        _new_log_mixture_weights = _log_mixture_weights

    # combine pre-update quantities for missing observations with updated quantities
    new_states = index_update(states, index[not_missing], _new_states)
    new_upper_chols = index_update(upper_chols, index[not_missing], _new_upper_chols)
    new_loglikes = index_update(
        jnp.zeros(len(measurements)), index[not_missing], _loglikes
    )
    new_log_mixture_weights = index_update(
        log_mixture_weights, index[not_missing], _new_log_mixture_weights
    )

    debug_info = {}
    if debug:
        n_obs, n_mixtures = new_log_mixture_weights.shape

        residuals = jnp.full((n_obs, n_mixtures), jnp.nan)
        residuals = index_update(residuals, index[not_missing], _residuals)
        debug_info["residuals"] = residuals

        residual_sds = jnp.full((n_obs, n_mixtures), jnp.nan)
        residual_sds = index_update(residual_sds, index[not_missing], _abs_root_sigmas)
        debug_info["residual_sds"] = residual_sds

    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )


# ======================================================================================
# Update Step
# ======================================================================================


def calculate_sigma_scaling_factor_and_weights(n_states, kappa=2):
    """Calculate the scaling factor and weights for sigma points according to Julier.

    There are other sigma point algorithms, but many of them possibly have negative
    weights which makes the unscented predict step more complicated.

    Args:
        n_states (int): Number of states.
        kappa (float): Spreading factor of the sigma points.

    Returns:
        float: Scaling factor
        jax.numpy.array: Sigma weights of length 2 * n_states + 1

    """
    scaling_factor = jnp.sqrt(kappa + n_states)
    n_sigma = 2 * n_states + 1
    weights = 0.5 * jnp.ones(n_sigma) / (n_states + kappa)
    weights = index_update(weights, index[0], kappa / (n_states + kappa))
    return scaling_factor, weights


def kalman_predict(
    states,
    upper_chols,
    sigma_scaling_factor,
    sigma_weights,
    transition_functions,
    trans_coeffs,
    shock_sds,
    anchoring_scaling_factors,
    anchoring_variables,
):
    """Make a unscented Kalman predict.

    Args:
        states (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states) with
            pre-update states estimates.
        upper_chols (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states,
            n_states) with the transpose of the lower triangular cholesky factor
            of the pre-update covariance matrix of the state estimates.
        sigma_scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights (jax.numpy.array): 1d array of length n_sigma with non-negative
            sigma weights.
        transition_functions (tuple): tuple of tuples where the first element is the
            name of the transition function and the second the actual transition
            function. Order is important and corresponds to the latent
            factors in alphabetical order.
        trans_coeffs (tuple): Tuple of 1d jax.numpy.arrays with transition parameters.
        anchoring_scaling_factors (jax.numpy.array): Array of shape (2, n_fac) with
            the scaling factors for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        anchoring_variables (jax.numpy.array): Array of shape (2, n_obs, n_fac) with
            anchoring outcomes. Can be 0 for unanchored factors or if no centering is
            desired. The first element corresponds to the input period, the second to
            the output period (i.e. input period + 1)

    Returns:
        jax.numpy.array: Predicted states, same shape as states.
        jax.numpy.array: Predicted upper_chols, same shape as upper_chols.

    """
    sigma_points = _calculate_sigma_points(states, upper_chols, sigma_scaling_factor)
    transformed = _transform_sigma_points(
        sigma_points,
        transition_functions,
        trans_coeffs,
        anchoring_scaling_factors,
        anchoring_variables,
    )

    n_obs, n_mixtures, n_sigma, n_fac = sigma_points.shape

    predicted_states = jnp.dot(sigma_weights, transformed)

    devs = transformed - predicted_states.reshape(n_obs, n_mixtures, 1, n_fac)

    qr_weights = jnp.sqrt(sigma_weights).reshape(n_sigma, 1)
    qr_points = jnp.zeros((n_obs, n_mixtures, n_sigma + n_fac, n_fac))
    qr_points = index_update(qr_points, index[:, :, 0:n_sigma], devs * qr_weights)
    qr_points = index_update(qr_points, index[:, :, n_sigma:], jnp.diag(shock_sds))
    predicted_covs = array_qr_jax(qr_points)[1][:, :, :n_fac]

    return predicted_states, predicted_covs


def _calculate_sigma_points(states, upper_chols, scaling_factor):
    """Calculate the array of sigma_points for the unscented transform.

    Args:
        states (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states) with
            pre-update states estimates.
        upper_chols (jax.numpy.array): Array of shape (n_obs, n_mixtures, n_states,
            n_states) with the transpose of the lower triangular cholesky factor
            of the pre-update covariance matrix of the state estimates.
        scaling_factor (float): A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.

    Returns:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with sigma points.

    """
    n_obs, n_mixtures, n_fac = states.shape
    n_sigma = 2 * n_fac + 1

    scaled_upper_chols = upper_chols * scaling_factor
    sigma_points = jnp.repeat(states, n_sigma, axis=1).reshape(
        n_obs, n_mixtures, n_sigma, n_fac
    )
    sigma_points = index_add(
        sigma_points, index[:, :, 1 : n_fac + 1], scaled_upper_chols
    )
    sigma_points = index_add(
        sigma_points, index[:, :, n_fac + 1 :], -scaled_upper_chols
    )
    return sigma_points


def _transform_sigma_points(
    sigma_points,
    transition_functions,
    trans_coeffs,
    anchoring_scaling_factors,
    anchoring_variables,
):
    """Anchor sigma points, transform them and unanchor the transformed sigma points.

    Args:
        sigma_points (jax.numpy.array) of shape n_obs, n_mixtures, n_sigma, n_fac.
        transition_functions (tuple): tuple of tuples where the first element is the
            name of the transition function and the second the actual transition
            function. Order is important and corresponds to the latent
            factors in alphabetical order.
        trans_coeffs (tuple): Tuple of 1d jax.numpy.arrays with transition parameters.
        anchoring_scaling_factors (jax.numpy.array): Array of shape (2, n_fac) with
            the scaling factors for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        anchoring_variables (jax.numpy.array): Array of shape (2, n_obs, n_fac) with
            anchoring outcomes. Can be 0 for unanchored factors or if no centering is
            desired. The first element corresponds to the input period, the second to
            the output period (i.e. input period + 1)

    Returns:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with transformed sigma points.

    """
    n_obs, n_mixtures, n_sigma, n_states = sigma_points.shape
    anchored = sigma_points * anchoring_scaling_factors[0] - anchoring_variables[
        0
    ].reshape(n_obs, 1, 1, n_states)

    transformed_anchored = anchored
    for i, ((name, func), coeffs) in enumerate(zip(transition_functions, trans_coeffs)):
        if name != "constant":
            output = func(anchored, coeffs)
            transformed_anchored = index_update(
                transformed_anchored, index[..., i], output
            )

    transformed_unanchored = (
        transformed_anchored + anchoring_variables[1].reshape(n_obs, 1, 1, n_states)
    ) / anchoring_scaling_factors[1]

    return transformed_unanchored
