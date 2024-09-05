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
):
    """Perform a Kalman update with likelihood evaluation, returning debug info on top.

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
        _safe_controls,
        control_params,
    ).reshape(n_obs, 1)

    _safe_measurements = jnp.where(
        not_missing,
        measurements,
        _safe_expected_measurements.mean(axis=1),
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
            -1,
            1,
        )

    else:
        _loglikes = _loglikes_per_dist.flatten()
        _new_log_mixture_weights = log_mixture_weights

    # combine pre-update quantities for missing observations with updated quantities
    new_states = jnp.where(not_missing.reshape(n_obs, 1, 1), _new_states, states)
    new_upper_chols = jnp.where(
        not_missing.reshape(n_obs, 1, 1, 1),
        _new_upper_chols,
        upper_chols,
    )
    new_loglikes = jnp.where(not_missing, _loglikes, 0)
    new_log_mixture_weights = jnp.where(
        not_missing.reshape(n_obs, 1),
        _new_log_mixture_weights,
        log_mixture_weights,
    )

    debug_info = {}
    residuals = jnp.where(not_missing.reshape(n_obs, 1), _residuals, jnp.nan)
    debug_info["residuals"] = residuals
    residual_sds = jnp.where(
        not_missing.reshape(n_obs, 1),
        _abs_root_sigmas,
        jnp.nan,
    )
    debug_info["residual_sds"] = residual_sds
    debug_info["log_mixture_weights"] = new_log_mixture_weights

    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
        debug_info,
    )
