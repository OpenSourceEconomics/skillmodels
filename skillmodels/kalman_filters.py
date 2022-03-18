import jax
import jax.numpy as jnp


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
    n_obs, n_mixtures, n_states = states.shape

    not_missing = jnp.isfinite(measurements)

    # replace missing measurements and controls by reasonable fill values to avoid NaNs
    # in the gradient calculation. All values that are influenced by this, are
    # replaced by other values later. Choosing the average expected
    # expected measurements without controls as fill value ensures that all numbers
    # are well defined because the fill values have a reasonable order of magnitude.
    # See https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
    # and https://jax.readthedocs.io/en/latest/faq.html
    # for more details on the issue of NaNs in gradient calculations.
    _safe_controls = jnp.where(not_missing.reshape(n_obs, 1), controls, 0)

    _safe_expected_measurements = jnp.dot(states, loadings) + jnp.dot(
        _safe_controls, control_params
    ).reshape(n_obs, 1)

    _safe_measurements = jnp.where(
        not_missing, measurements, _safe_expected_measurements.mean(axis=1)
    )

    _residuals = _safe_measurements.reshape(n_obs, 1) - _safe_expected_measurements
    _f_stars = jnp.dot(upper_chols, loadings.reshape(n_states, 1))

    _m = jnp.zeros((n_obs, n_mixtures, n_states + 1, n_states + 1))
    _m = _m.at[..., 0, 0].set(meas_sd)
    _m = _m.at[..., 1:, :1].set(_f_stars)
    _m = _m.at[..., 1:, 1:].set(upper_chols)

    _r = array_qr_jax(_m)[1]

    _new_upper_chols = _r[..., 1:, 1:]
    _root_sigmas = _r[..., 0, 0]
    _abs_root_sigmas = jnp.abs(_root_sigmas)
    # it is important not to divide by the absolute value of _root_sigmas in order
    # to recover the sign of the Kalman gain.
    _kalman_gains = _r[..., 0, 1:] / _root_sigmas.reshape(n_obs, n_mixtures, 1)
    _new_states = states + _kalman_gains * _residuals.reshape(n_obs, n_mixtures, 1)

    # calculate log likelihood per individual and update mixture weights
    _loglikes_per_dist = jax.scipy.stats.norm.logpdf(_residuals, 0, _abs_root_sigmas)
    if n_mixtures >= 2:
        _weighted_loglikes_per_dist = _loglikes_per_dist + log_mixture_weights
        _loglikes = jax.scipy.special.logsumexp(_weighted_loglikes_per_dist, axis=1)
        _new_log_mixture_weights = _weighted_loglikes_per_dist - _loglikes.reshape(
            -1, 1
        )

    else:
        _loglikes = _loglikes_per_dist.flatten()
        _new_log_mixture_weights = log_mixture_weights

    # combine pre-update quantities for missing observations with updated quantities
    new_states = jnp.where(not_missing.reshape(n_obs, 1, 1), _new_states, states)
    new_upper_chols = jnp.where(
        not_missing.reshape(n_obs, 1, 1, 1), _new_upper_chols, upper_chols
    )
    new_loglikes = jnp.where(not_missing, _loglikes, 0)
    new_log_mixture_weights = jnp.where(
        not_missing.reshape(n_obs, 1), _new_log_mixture_weights, log_mixture_weights
    )

    debug_info = {}
    if debug:
        residuals = jnp.where(not_missing.reshape(n_obs, 1), _residuals, jnp.nan)
        debug_info["residuals"] = residuals
        residual_sds = jnp.where(
            not_missing.reshape(n_obs, 1), _abs_root_sigmas, jnp.nan
        )
        debug_info["residual_sds"] = residual_sds

    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )


# ======================================================================================
# Predict Step
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
    weights = weights.at[0].set(kappa / (n_states + kappa))
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
    anchoring_constants,
    observed_factors,
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
        anchoring_constants (jax.numpy.array): Array of shape (2, n_states) with the
            constants for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        observed_factors (jax.numpy.array): Array of shape (n_obs, n_observed_factors)
            with data on the observed factors in period t.

    Returns:
        jax.numpy.array: Predicted states, same shape as states.
        jax.numpy.array: Predicted upper_chols, same shape as upper_chols.

    """
    sigma_points = _calculate_sigma_points(
        states, upper_chols, sigma_scaling_factor, observed_factors
    )
    transformed = _transform_sigma_points(
        sigma_points,
        transition_functions,
        trans_coeffs,
        anchoring_scaling_factors,
        anchoring_constants,
    )

    # do not use sigma_points.shape because sigma_points contain observed factors
    n_obs, n_mixtures, n_sigma, n_fac = transformed.shape

    predicted_states = jnp.dot(sigma_weights, transformed)

    devs = transformed - predicted_states.reshape(n_obs, n_mixtures, 1, n_fac)

    qr_weights = jnp.sqrt(sigma_weights).reshape(n_sigma, 1)
    qr_points = jnp.zeros((n_obs, n_mixtures, n_sigma + n_fac, n_fac))
    qr_points = qr_points.at[:, :, 0:n_sigma].set(devs * qr_weights)
    qr_points = qr_points.at[:, :, n_sigma:].set(jnp.diag(shock_sds))
    predicted_covs = array_qr_jax(qr_points)[1][:, :, :n_fac]

    return predicted_states, predicted_covs


def _calculate_sigma_points(states, upper_chols, scaling_factor, observed_factors):
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
        observed_factors (jax.numpy.array): Array of shape (n_obs, n_observed_factors)
            with data on the observed factors in period t.

    Returns:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with sigma points.

    """
    n_obs, n_mixtures, n_fac = states.shape
    n_sigma = 2 * n_fac + 1
    n_observed = observed_factors.shape[1]

    scaled_upper_chols = upper_chols * scaling_factor
    sigma_points = jnp.repeat(states, n_sigma, axis=1).reshape(
        n_obs, n_mixtures, n_sigma, n_fac
    )
    sigma_points = sigma_points.at[:, :, 1 : n_fac + 1].add(scaled_upper_chols)
    sigma_points = sigma_points.at[:, :, n_fac + 1 :].add(-scaled_upper_chols)

    observed_part = observed_factors.repeat(n_sigma, axis=0).reshape(
        n_obs, n_mixtures, n_sigma, n_observed
    )

    sigma_points = jnp.concatenate([sigma_points, observed_part], axis=-1)
    return sigma_points


def _transform_sigma_points(
    sigma_points,
    transition_info,
    trans_coeffs,
    anchoring_scaling_factors,
    anchoring_constants,
):
    """Anchor sigma points, transform them and unanchor the transformed sigma points.

    Args:
        sigma_points (jax.numpy.array) of shape n_obs, n_mixtures, n_sigma, n_fac.
        transition_info (dict): Dict with the entries "func", "columns".
        trans_coeffs (tuple): Tuple of 1d jax.numpy.arrays with transition parameters.
        anchoring_scaling_factors (jax.numpy.array): Array of shape (2, n_states) with
            the scaling factors for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        anchoring_constants (jax.numpy.array): Array of shape (2, n_states) with the
            constants for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).

    Returns:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with transformed sigma points.

    """
    n_obs, n_mixtures, n_sigma, n_fac = sigma_points.shape

    flat_sigma_points = sigma_points.reshape(-1, n_fac)

    anchored = flat_sigma_points * anchoring_scaling_factors[0] + anchoring_constants[0]

    kwargs = {"sigma_points": anchored, "params": trans_coeffs}

    for factor_name, position in transition_info["columns"].items():
        kwargs[factor_name] = anchored[:, position]

    transition_function = transition_info["func"]

    transformed_anchored = transition_function(**kwargs)

    n_observed = transformed_anchored.shape[-1]

    transformed_unanchored = (
        transformed_anchored - anchoring_constants[1][:n_observed]
    ) / anchoring_scaling_factors[1][:n_observed]

    out_shape = (n_obs, n_mixtures, n_sigma, -1)
    out = transformed_unanchored.reshape(out_shape)

    return out
