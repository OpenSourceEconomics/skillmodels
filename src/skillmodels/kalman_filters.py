"""Kalman filter operations for state estimation using the square-root form."""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jax import Array

from skillmodels.qr import qr_gpu

LINEAR_FUNCTION_NAMES = frozenset({"linear", "constant"})


def is_all_linear(function_names: Mapping[str, str]) -> bool:
    """Return True if every factor uses a linear or constant transition function."""
    return all(name in LINEAR_FUNCTION_NAMES for name in function_names.values())


array_qr_jax = (
    jax.vmap(jax.vmap(qr_gpu))
    if jax.default_backend() == "gpu"
    else jax.vmap(jax.vmap(jnp.linalg.qr))
)


# ======================================================================================
# Update Step
# ======================================================================================
def kalman_update(
    states: Array,
    upper_chols: Array,
    loadings: Array,
    control_params: Array,
    meas_sd: Array,
    measurements: Array,
    controls: Array,
    log_mixture_weights: Array,
) -> tuple[Array, Array, Array, Array]:
    """Perform a Kalman update with likelihood evaluation.

    Args:
        states: Array of shape (n_obs, n_mixtures, n_states) with
            pre-update states estimates.
        upper_chols: Array of shape (n_obs, n_mixtures, n_states,
            n_states) with the transpose of the lower triangular cholesky factor
            of the pre-update covariance matrix of the state estimates.
        loadings: 1d array of length n_states with factor loadings.
        control_params: 1d array of length n_controls.
        meas_sd: Standard deviation of the measurement error.
        measurements: 1d array of length n_obs with measurements.
            May contain NaNs if no measurement was observed.
        controls: Array of shape (n_obs, n_controls) with data on the
            control variables.
        log_mixture_weights: Array of shape (n_obs, n_mixtures) with
            the natural logarithm of the weights of each element of the mixture of
            normals distribution.

    Returns:
        new_states: Same format as states.
        new_upper_chols: Same format as upper_chols
        new_log_mixture_weights: (jax.numpy.array): Same format as log_mixture_weights
        new_loglikes: (jax.numpy.array): 1d array of length n_obs

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

    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
    )


# ======================================================================================
# Predict Step
# ======================================================================================
def calculate_sigma_scaling_factor_and_weights(
    n_states: int,
    kappa: float = 2,
) -> tuple[Array, Array]:
    """Calculate the scaling factor and weights for sigma points according to Julier.

    There are other sigma point algorithms, but many of them possibly have negative
    weights which makes the unscented predict step more complicated.

    Args:
        n_states: Number of states.
        kappa: Spreading factor of the sigma points.

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
    transition_func: Callable,
    states: Array,
    upper_chols: Array,
    sigma_scaling_factor: float,
    sigma_weights: Array,
    trans_coeffs: dict[str, Array],
    shock_sds: Array,
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
    observed_factors: Array,
) -> tuple[Array, Array]:
    """Make a unscented Kalman predict.

    Args:
        transition_func: The transition function.
        states: Array of shape (n_obs, n_mixtures, n_states) with
            pre-update states estimates.
        upper_chols: Array of shape (n_obs, n_mixtures, n_states,
            n_states) with the transpose of the lower triangular cholesky factor
            of the pre-update covariance matrix of the state estimates.
        sigma_scaling_factor: A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        sigma_weights: 1d array of length n_sigma with non-negative
            sigma weights.
        trans_coeffs: Tuple of 1d jax.numpy.arrays with transition parameters.
        shock_sds: 1d array of length n_fac with shock standard
            deviations.
        anchoring_scaling_factors: Array of shape (2, n_fac) with
            the scaling factors for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        anchoring_constants: Array of shape (2, n_states) with the
            constants for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        observed_factors: Array of shape (n_obs, n_observed_factors)
            with data on the observed factors in period t.

    Returns:
        jax.numpy.array: Predicted states, same shape as states.
        jax.numpy.array: Predicted upper_chols, same shape as upper_chols.

    """
    sigma_points = _calculate_sigma_points(
        states=states,
        upper_chols=upper_chols,
        scaling_factor=sigma_scaling_factor,
        observed_factors=observed_factors,
    )
    transformed = transform_sigma_points(
        sigma_points=sigma_points,
        transition_func=transition_func,
        trans_coeffs=trans_coeffs,
        anchoring_scaling_factors=anchoring_scaling_factors,
        anchoring_constants=anchoring_constants,
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


def linear_kalman_predict(
    transition_func: Callable | None,  # noqa: ARG001
    states: Array,
    upper_chols: Array,
    sigma_scaling_factor: float,  # noqa: ARG001
    sigma_weights: Array,  # noqa: ARG001
    trans_coeffs: dict[str, Array],
    shock_sds: Array,
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
    observed_factors: Array,
    *,
    latent_factors: tuple[str, ...],
    constant_factor_indices: frozenset[int],
    n_all_factors: int,
) -> tuple[Array, Array]:
    """Make a linear Kalman predict (square-root form).

    Much cheaper than the unscented predict because it avoids sigma point
    generation and transformation. Only valid when every factor uses a `linear`
    or `constant` transition function.

    The positional parameters `transition_func`, `sigma_scaling_factor` and
    `sigma_weights` are accepted for signature compatibility with
    `kalman_predict` but are ignored.

    Args:
        transition_func: Ignored (kept for signature compatibility).
        states: Array of shape (n_obs, n_mixtures, n_states).
        upper_chols: Array of shape (n_obs, n_mixtures, n_states, n_states).
        sigma_scaling_factor: Ignored.
        sigma_weights: Ignored.
        trans_coeffs: Dict mapping factor name to 1d coefficient array.
        shock_sds: 1d array of length n_states.
        anchoring_scaling_factors: Array of shape (2, n_states).
        anchoring_constants: Array of shape (2, n_states).
        observed_factors: Array of shape (n_obs, n_observed_factors).
        latent_factors: Tuple of latent factor names.
        constant_factor_indices: Indices of factors with `constant` transition.
        n_all_factors: Total number of factors (latent + observed).

    Returns:
        Predicted states, same shape as states.
        Predicted upper_chols, same shape as upper_chols.

    """
    n_latent = len(latent_factors)

    # Build F (n_latent x n_all) and c (n_latent,) from trans_coeffs.
    # linear factor i: F[i] = trans_coeffs[factor_i][:-1], c[i] = last element
    # constant factor i: F[i] = e_i (unit vector), c[i] = 0
    f_rows = []
    c_vals = []
    for i, factor in enumerate(latent_factors):
        if i in constant_factor_indices:
            row = jnp.zeros(n_all_factors).at[i].set(1.0)
            f_rows.append(row)
            c_vals.append(0.0)
        else:
            coeffs = trans_coeffs[factor]
            f_rows.append(coeffs[:-1])
            c_vals.append(coeffs[-1])

    f_mat = jnp.stack(f_rows)  # (n_latent, n_all)
    c_vec = jnp.array(c_vals)  # (n_latent,)

    s_in = anchoring_scaling_factors[0][:n_latent]  # (n_latent,) for input period
    s_out = anchoring_scaling_factors[1][:n_latent]  # (n_latent,) for output period
    c_in = anchoring_constants[0][:n_latent]  # (n_latent,)
    c_out = anchoring_constants[1][:n_latent]  # (n_latent,)

    # Mean prediction
    anchored_states = states * s_in + c_in  # (n_obs, n_mix, n_latent)
    # Concatenate with observed factors to get full state vector
    n_obs, n_mix, _ = states.shape
    obs_expanded = jnp.broadcast_to(
        observed_factors[:, jnp.newaxis, :], (n_obs, n_mix, observed_factors.shape[1])
    )
    full_states = jnp.concatenate([anchored_states, obs_expanded], axis=-1)

    predicted_anchored = full_states @ f_mat.T + c_vec  # (n_obs, n_mix, n_latent)
    predicted_states = (predicted_anchored - c_out) / s_out

    # Covariance prediction (square-root form)
    # G = diag(1/s_out) @ F_latent @ diag(s_in) where F_latent is the first
    # n_latent columns of F
    f_latent = f_mat[:, :n_latent]  # (n_latent, n_latent)
    g_mat = (f_latent * s_in) / s_out[:, jnp.newaxis]  # (n_latent, n_latent)

    # Stack: [upper_chol @ G.T ; diag(shock_sds / s_out)]
    chol_g = upper_chols @ g_mat.T  # (n_obs, n_mix, n_latent, n_latent)
    shock_diag = jnp.diag(shock_sds / s_out)  # (n_latent, n_latent)

    stack = jnp.concatenate(
        [chol_g, jnp.broadcast_to(shock_diag, chol_g.shape)], axis=-2
    )  # (n_obs, n_mix, 2*n_latent, n_latent)

    predicted_covs = array_qr_jax(stack)[1][:, :, :n_latent]

    return predicted_states, predicted_covs


def _calculate_sigma_points(
    states: Array,
    upper_chols: Array,
    scaling_factor: float,
    observed_factors: Array,
) -> Array:
    """Calculate the array of sigma_points for the unscented transform.

    Args:
        states: Array of shape (n_obs, n_mixtures, n_states) with
            pre-update states estimates.
        upper_chols: Array of shape (n_obs, n_mixtures, n_states,
            n_states) with the transpose of the lower triangular cholesky factor
            of the pre-update covariance matrix of the state estimates.
        scaling_factor: A scaling factor that controls the spread of the
            sigma points. Bigger means that sigma points are further apart. Depends on
            the sigma_point algorithm chosen.
        observed_factors: Array of shape (n_obs, n_observed_factors)
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
        n_obs,
        n_mixtures,
        n_sigma,
        n_fac,
    )
    sigma_points = sigma_points.at[:, :, 1 : n_fac + 1].add(scaled_upper_chols)
    sigma_points = sigma_points.at[:, :, n_fac + 1 :].add(-scaled_upper_chols)

    observed_part = jnp.broadcast_to(
        observed_factors[:, jnp.newaxis, jnp.newaxis, :],
        (n_obs, n_mixtures, n_sigma, n_observed),
    )

    return jnp.concatenate([sigma_points, observed_part], axis=-1)


def transform_sigma_points(
    sigma_points: Array,
    transition_func: Callable,
    trans_coeffs: dict[str, Array],
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
) -> Array:
    """Anchor sigma points, transform them and unanchor the transformed sigma points.

    Args:
        sigma_points: Array of shape n_obs, n_mixtures, n_sigma, n_fac.
        transition_func: The transition function.
        trans_coeffs: Tuple of 1d jax.numpy.arrays with transition parameters.
        anchoring_scaling_factors: Array of shape (2, n_states) with
            the scaling factors for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).
        anchoring_constants: Array of shape (2, n_states) with the
            constants for anchoring. The first row corresponds to the input
            period, the second to the output period (i.e. input period + 1).

    Returns:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with transformed sigma points.

    """
    n_obs, n_mixtures, n_sigma, n_fac = sigma_points.shape

    flat_sigma_points = sigma_points.reshape(-1, n_fac)

    anchored = flat_sigma_points * anchoring_scaling_factors[0] + anchoring_constants[0]

    transformed_anchored = transition_func(trans_coeffs, anchored)

    n_observed = transformed_anchored.shape[-1]

    transformed_unanchored = (
        transformed_anchored - anchoring_constants[1][:n_observed]
    ) / anchoring_scaling_factors[1][:n_observed]

    out_shape = (n_obs, n_mixtures, n_sigma, -1)
    return transformed_unanchored.reshape(out_shape)
