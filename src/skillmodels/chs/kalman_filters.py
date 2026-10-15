"""Kalman filter operations for state estimation using the square-root form."""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jax import Array

from skillmodels.chs.qr import qr_gpu
from skillmodels.common.transitions import apply_anchored_transition

LINEAR_FUNCTION_NAMES = frozenset({"linear", "constant"})


def is_all_linear(function_names: Mapping[str, str]) -> bool:
    """Return True if every factor uses a linear or constant transition function."""
    return all(name in LINEAR_FUNCTION_NAMES for name in function_names.values())


array_qr_jax = (
    jax.vmap(jax.vmap(qr_gpu))
    if jax.default_backend() == "gpu"
    else jax.vmap(jax.vmap(jnp.linalg.qr))
)


def kalman_update(
    states: Array,
    upper_chols: Array,
    loadings: Array,
    control_params: Array,
    meas_sd: float | Array,
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

    if n_mixtures >= 2:
        new_states, new_upper_chols = reset_negligible_mixture_components(
            states=new_states,
            upper_chols=new_upper_chols,
            log_mixture_weights=new_log_mixture_weights,
        )

    return (
        new_states,
        new_upper_chols,
        new_log_mixture_weights,
        new_loglikes,
    )


def reset_negligible_mixture_components(
    states: Array,
    upper_chols: Array,
    log_mixture_weights: Array,
) -> tuple[Array, Array]:
    """Give components whose weight underflows the dominant component's state.

    A component whose weight is below the smallest normal float contributes exactly
    nothing to the likelihood, but its state is still propagated. Under a nonlinear
    (e.g. translog) transition it can then explode to inf over later periods, and
    the gradient through its zero weight becomes 0 * inf = NaN for every parameter.
    Overwriting its state and covariance with the dominant component's keeps all
    values finite; the likelihood is unchanged and the gradient is exact up to the
    underflowed weight. The weights themselves are left untouched.

    Args:
        states: Array of shape (n_obs, n_mixtures, n_states).
        upper_chols: Array of shape (n_obs, n_mixtures, n_states, n_states).
        log_mixture_weights: Array of shape (n_obs, n_mixtures).

    Returns:
        states and upper_chols with negligible components replaced.

    """
    log_weights = jax.lax.stop_gradient(log_mixture_weights)
    negligible = log_weights < jnp.log(jnp.finfo(log_weights.dtype).tiny)
    dominant = jnp.argmax(log_weights, axis=1)
    dominant_states = jnp.take_along_axis(states, dominant[:, None, None], axis=1)
    dominant_chols = jnp.take_along_axis(
        upper_chols, dominant[:, None, None, None], axis=1
    )
    return (
        jnp.where(negligible[..., None], dominant_states, states),
        jnp.where(negligible[..., None, None], dominant_chols, upper_chols),
    )


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
    sigma_scaling_factor: float | Array,
    sigma_weights: Array,
    trans_coeffs: dict[str, Array],
    shock_sds: Array,
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
    observed_factors: Array,
    observed_factor_vars: Array | None = None,
    *,
    missing_observed_indices: tuple[int, ...] = (),
) -> tuple[Array, Array]:
    """Make a unscented Kalman predict.

    Observed factors at `missing_observed_indices` may be missing. They augment
    the sigma-point input as extra Gaussian dimensions, independent of the latent
    states, with mean `observed_factors` (the fill mean where missing) and
    variance `observed_factor_vars` (0 where observed, so observed values pass
    through unchanged). The number of sigma points, and hence `sigma_weights`,
    is based on `n_latent + len(missing_observed_indices)` for every observation.

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
        observed_factor_vars: Array of shape
            (n_obs, len(missing_observed_indices)) with the variance of each
            possibly-missing observed factor (0 where it is observed).
        missing_observed_indices: Positions (among the observed factors) of the
            observed factors that may be missing.

    Returns:
        jax.numpy.array: Predicted states, same shape as states.
        jax.numpy.array: Predicted upper_chols, same shape as upper_chols.

    """
    if missing_observed_indices:
        states, upper_chols = _augment_by_missing_observed_factors(
            states=states,
            upper_chols=upper_chols,
            observed_factors=observed_factors,
            observed_factor_vars=observed_factor_vars,
            missing_observed_indices=missing_observed_indices,
        )
    sigma_points = _calculate_sigma_points(
        states=states,
        upper_chols=upper_chols,
        scaling_factor=sigma_scaling_factor,
        observed_factors=observed_factors,
        missing_observed_indices=missing_observed_indices,
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
    sigma_scaling_factor: float | Array,  # noqa: ARG001
    sigma_weights: Array,  # noqa: ARG001
    trans_coeffs: dict[str, Array],
    shock_sds: Array,
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
    observed_factors: Array,
    observed_factor_vars: Array | None = None,
    *,
    latent_factors: tuple[str, ...],
    constant_factor_indices: frozenset[int],
    n_all_factors: int,
    missing_observed_indices: tuple[int, ...] = (),
) -> tuple[Array, Array]:
    """Make a linear Kalman predict (square-root form).

    Much cheaper than the unscented predict because it avoids sigma point
    generation and transformation. Only valid when every factor uses a `linear`
    or `constant` transition function.

    The positional parameters `transition_func`, `sigma_scaling_factor` and
    `sigma_weights` are accepted for signature compatibility with
    `kalman_predict` but are ignored.

    Observed factors at `missing_observed_indices` may be missing, in which case
    they are `N(m, v)` independent of the latent states, with `m` given in
    `observed_factors` and `v` in `observed_factor_vars`. The prediction is exact:
    the mean uses `m`, and the covariance adds `B diag(v) B'` with `B` the
    unanchored coefficients of those factors, so the cross-covariance induced by
    the same factor entering several equations is kept.

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
        observed_factor_vars: Array of shape
            (n_obs, len(missing_observed_indices)) with the variance of each
            possibly-missing observed factor (0 where it is observed).
        latent_factors: Tuple of latent factor names.
        constant_factor_indices: Indices of factors with `constant` transition.
        n_all_factors: Total number of factors (latent + observed).
        missing_observed_indices: Positions (among the observed factors) of the
            observed factors that may be missing.

    Returns:
        Predicted states, same shape as states.
        Predicted upper_chols, same shape as upper_chols.

    """
    n_latent = len(latent_factors)

    f_mat, c_vec = _build_f_and_c(
        latent_factors, constant_factor_indices, n_all_factors, trans_coeffs
    )

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

    blocks = [chol_g, jnp.broadcast_to(shock_diag, chol_g.shape)]
    if missing_observed_indices:
        if observed_factor_vars is None:
            msg = "observed_factor_vars is required with missing_observed_indices."
            raise ValueError(msg)
        # Row j is sqrt(v_j) * B[:, j], so the rows add B diag(v) B' to the cov.
        b_missing = f_mat[:, n_latent + jnp.array(missing_observed_indices)]
        b_missing = b_missing / s_out[:, jnp.newaxis]  # (n_latent, n_missing)
        # Promote before the root: loader moments are float32, while the
        # covariance factor can be float64. A later cast cannot undo root rounding.
        missing_sds = jnp.sqrt(observed_factor_vars.astype(chol_g.dtype))
        rows = missing_sds[..., jnp.newaxis] * b_missing.T
        blocks.append(
            jnp.broadcast_to(rows[:, jnp.newaxis], (n_obs, n_mix, *b_missing.T.shape))
        )
    # Stack shape is (n_obs, n_mix, 2 * n_latent + n_missing, n_latent).
    stack = jnp.concatenate(blocks, axis=-2)

    predicted_covs = array_qr_jax(stack)[1][:, :, :n_latent]

    return predicted_states, predicted_covs


def _build_f_and_c(
    latent_factors: tuple[str, ...],
    constant_factor_indices: frozenset[int],
    n_all_factors: int,
    trans_coeffs: dict[str, Array],
) -> tuple[Array, Array]:
    """Build F matrix and c vector from transition coefficients.

    Stack all coefficient arrays, build identity rows for constant factors,
    and select via a boolean mask.

    Args:
        latent_factors: Tuple of latent factor names.
        constant_factor_indices: Indices of factors with `constant` transition.
        n_all_factors: Total number of factors (latent + observed).
        trans_coeffs: Dict mapping factor name to 1d coefficient array.

    Returns:
        f_mat: Array of shape (n_latent, n_all_factors).
        c_vec: Array of shape (n_latent,).

    """
    n_latent = len(latent_factors)
    identity = jnp.eye(n_latent, n_all_factors)

    # Will be of shape (n_latent, n_all+1)
    all_coeffs = jnp.stack(
        [
            trans_coeffs[f]
            if i not in constant_factor_indices
            else jnp.zeros(n_all_factors + 1)
            for i, f in enumerate(latent_factors)
        ]
    )

    f_from_coeffs = all_coeffs[:, :-1]
    c_from_coeffs = all_coeffs[:, -1]

    is_constant = jnp.array([i in constant_factor_indices for i in range(n_latent)])
    mask = is_constant[:, None]  # (n_latent, 1)

    f_mat = jnp.where(mask, identity, f_from_coeffs)
    c_vec = jnp.where(is_constant, 0.0, c_from_coeffs)

    return f_mat, c_vec


def _calculate_sigma_points(
    states: Array,
    upper_chols: Array,
    scaling_factor: float | Array,
    observed_factors: Array,
    missing_observed_indices: tuple[int, ...] = (),
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
        missing_observed_indices: Positions (among the observed factors) of the
            observed factors held in the trailing `len(missing_observed_indices)`
            entries of `states`. Their sigma-point values replace the data.

    Returns:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with sigma points.

    """
    n_obs, n_mixtures, n_fac = states.shape
    n_sigma = 2 * n_fac + 1
    n_observed = observed_factors.shape[1]
    n_missing = len(missing_observed_indices)

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
    if n_missing:
        # Promote first: observed factors are stored as float32, and writing the
        # float64 sigma-point values into them would round those values.
        observed_part = observed_part.astype(sigma_points.dtype)
        observed_part = observed_part.at[..., jnp.array(missing_observed_indices)].set(
            sigma_points[..., n_fac - n_missing :]
        )
        sigma_points = sigma_points[..., : n_fac - n_missing]

    return jnp.concatenate([sigma_points, observed_part], axis=-1)


def _augment_by_missing_observed_factors(
    states: Array,
    upper_chols: Array,
    observed_factors: Array,
    observed_factor_vars: Array | None,
    missing_observed_indices: tuple[int, ...],
) -> tuple[Array, Array]:
    """Append possibly-missing observed factors as independent Gaussian states.

    Args:
        states: Array of shape (n_obs, n_mixtures, n_latent).
        upper_chols: Array of shape (n_obs, n_mixtures, n_latent, n_latent).
        observed_factors: Array of shape (n_obs, n_observed_factors) with the fill
            mean where a factor is missing.
        observed_factor_vars: Array of shape (n_obs, n_missing) with the fill
            variance where a factor is missing and 0 where it is observed.
        missing_observed_indices: Positions of the possibly-missing factors among
            the observed factors.

    Returns:
        Augmented states of shape (n_obs, n_mixtures, n_latent + n_missing) and
        block-diagonal upper Cholesky factors of matching shape.

    """
    if observed_factor_vars is None:
        msg = "observed_factor_vars is required with missing_observed_indices."
        raise ValueError(msg)
    n_obs, n_mixtures, n_latent = states.shape
    n_missing = len(missing_observed_indices)
    n_aug = n_latent + n_missing
    means = observed_factors[:, jnp.array(missing_observed_indices)]
    aug_states = jnp.concatenate(
        [states, jnp.broadcast_to(means[:, None], (n_obs, n_mixtures, n_missing))],
        axis=-1,
    )
    # Preserve the covariance-factor precision during the root, not just scatter.
    sds = jnp.sqrt(observed_factor_vars.astype(upper_chols.dtype))
    aug_chols = jnp.zeros((n_obs, n_mixtures, n_aug, n_aug), dtype=upper_chols.dtype)
    aug_chols = aug_chols.at[..., :n_latent, :n_latent].set(upper_chols)
    diag = jnp.arange(n_latent, n_aug)
    aug_chols = aug_chols.at[..., diag, diag].set(sds[:, None, :])
    return aug_states, aug_chols


def transform_sigma_points(
    sigma_points: Array,
    transition_func: Callable,
    trans_coeffs: dict[str, Array],
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
) -> Array:
    """Anchor sigma points, transform them and unanchor the transformed sigma points.

    Thin sigma-points-shape wrapper around
    `skillmodels.common.transitions.apply_anchored_transition`: flattens
    `(n_obs, n_mixtures, n_sigma, n_fac)` to `(N, n_fac)` for the
    common-core anchor → transition → unanchor pipeline, then restores
    the sigma-points layout for the UKF caller. Code paths that don't
    care about sigma-points layout (e.g. `simulate_dataset`) should
    call `apply_anchored_transition` directly.

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

    Return:
        jax.numpy.array: Array of shape n_obs, n_mixtures, n_sigma, n_fac (where n_sigma
        equals 2 * n_fac + 1) with transformed sigma points.

    """
    n_obs, n_mixtures, n_sigma, n_fac = sigma_points.shape

    transformed_unanchored = apply_anchored_transition(
        states=sigma_points.reshape(-1, n_fac),
        transition_func=transition_func,
        trans_coeffs=trans_coeffs,
        anchoring_scaling_factors=anchoring_scaling_factors,
        anchoring_constants=anchoring_constants,
    )

    return transformed_unanchored.reshape((n_obs, n_mixtures, n_sigma, -1))
