"""Contains Kalman Update and Predict functions in several flavors."""

from numba import float64 as f64
from numba import int64 as i64
from numba import guvectorize
import numpy as np
from skillmodels.fast_routines.transform_sigma_points import transform_sigma_points
from skillmodels.fast_routines.qr_decomposition import array_qr


@guvectorize(
    [
        (
            f64[:, :],
            f64[:, :, :],
            f64[:],
            f64[:],
            f64[:],
            f64[:],
            f64[:],
            f64[:],
            i64[:],
            f64[:],
        )
    ],
    (
        "(nemf, nfac), (nemf, nfac_, nfac_), (), (), (ncon), "
        "(ncon), (nfac), (), (ninc), (nemf)"
    ),
    target="cpu",
    nopython=True,
)
def sqrt_linear_update(
    state, cov, like_vec, y, c, delta, h, sqrt_r, positions, weights
):
    """Make a linear Kalman update in square root form and evaluate likelihood.

    The square-root form of the Kalman update is much more robust than the
    usual form and almost as fast.

    All quantities (states, covariances likelihood and weights) are updated in
    place. The function follows the usual numpy broadcast rules.

    Args:
        state (np.ndarray): numpy array of (..., nemf, nfac).

        cov (np.ndarray): numpy array of (..., nemf, nfac, nfac).

        like_vec (np.ndarray): a scalar in form of a length one numpy array.

        y (np.ndarray): a scalar in form of a length one numpy array.

        c (np.ndarray): numpy array of (..., ncontrols) with control variables.

        delta (np.ndarray): estimated parameters of the control variables.

        h (np.ndarray): numpy array of length nfac with factor loadings.

        sqrt_r (np.ndarray): a scalar in form of a length one numpy array.

        positions (np.ndarray): the positions of the factors measured by y.

        weights (np.ndarray): numpy array of (nemf, nind)

    References:
        Robert Grover Brown. Introduction to Random Signals and Applied
            Kalman Filtering. Wiley and sons, 2012.

    """
    nemf, nfac = state.shape
    m = nfac + 1
    ncontrol = delta.shape[0]
    # invariant = 0.398942280401432702863218082711682654917240142822265625
    invariant = 1 / (2 * np.pi) ** 0.5
    invar_diff = y[0]
    if np.isfinite(invar_diff):
        # same for all factor distributions
        for cont in range(ncontrol):
            invar_diff -= c[cont] * delta[cont]

        # per distribution stuff
        for emf in range(nemf):
            diff = invar_diff
            for pos in positions:
                diff -= state[emf, pos] * h[pos]

            cov[emf, 0, 0] = sqrt_r[0]

            for f in range(1, m):
                cov[emf, 0, f] = 0.0

            for f in range(1, m):
                for pos in positions:
                    cov[emf, f, 0] += cov[emf, f, pos + 1] * h[pos]

            for f in range(m):
                for g in range(m - 1, f, -1):
                    b = cov[emf, g, f]
                    if b != 0.0:
                        a = cov[emf, g - 1, f]
                        if abs(b) > abs(a):
                            r_ = a / b
                            s_ = 1 / (1 + r_ ** 2) ** 0.5
                            c_ = s_ * r_
                        else:
                            r_ = b / a
                            c_ = 1 / (1 + r_ ** 2) ** 0.5
                            s_ = c_ * r_
                        for k_ in range(m):
                            helper1 = cov[emf, g - 1, k_]
                            helper2 = cov[emf, g, k_]
                            cov[emf, g - 1, k_] = c_ * helper1 + s_ * helper2
                            cov[emf, g, k_] = -s_ * helper1 + c_ * helper2

            sigma = cov[emf, 0, 0]
            prob = invariant / np.abs(sigma) * np.exp(-diff ** 2 / (2 * sigma ** 2))

            diff /= sigma
            for f in range(nfac):
                state[emf, f] += cov[emf, 0, f + 1] * diff

            if nemf == 1:
                like_vec[0] *= prob
            else:
                weights[emf] *= max(prob, 1e-250)

        if nemf >= 2:
            sum_wprob = 0.0
            for emf in range(nemf):
                sum_wprob += weights[emf]

            like_vec[0] *= sum_wprob

            for emf in range(nemf):
                weights[emf] /= sum_wprob


@guvectorize(
    [
        (
            f64[:, :],
            f64[:, :, :],
            f64[:],
            f64[:],
            f64[:],
            f64[:],
            f64[:],
            f64[:],
            i64[:],
            f64[:],
            f64[:],
        )
    ],
    (
        "(nemf, nfac), (nemf, nfac, nfac), (), (), (ncon), "
        "(ncon), (nfac), (), (ninc), (nemf), (nfac)"
    ),
    target="cpu",
    nopython=True,
)
def normal_linear_update(
    state, cov, like_vec, y, c, delta, h, r, positions, weights, kf
):
    """Make a linear Kalman update and evaluate likelihood.

    All quantities (states, covariances likelihood and weights) are updated in
    place. The function follows the usual numpy broadcast rules.

    Args:
        state (np.ndarray): numpy array of (..., nemf, nfac).

        cov (np.ndarray): numpy array of (..., nemf, nfac, nfac).

        like_vec (np.ndarray): a scalar in form of a length one numpy array.

        y (np.ndarray): a scalar in form of a length one numpy array.

        c (np.ndarray): numpy array of (..., ncontrols) with control variables.

        delta (np.ndarray): estimated parameters of the control variables.

        h (np.ndarray): numpy array of length nfac with factor loadings.

        r (np.ndarray): a scalar in form of a length one numpy array.

        positions (np.ndarray): the positions of the factors measured by y.

        weights (np.ndarray): numpy array of (nemf, nind)

        kf (np.ndarray): an intermediate array of lenght nfac.

    References:
        Robert Grover Brown. Introduction to Random Signals and Applied
            Kalman Filtering. Wiley and sons, 2012.

    """
    nemf, nfac = state.shape
    ncontrol = delta.shape[0]
    # invariant = 0.398942280401432702863218082711682654917240142822265625
    invariant = 1 / (2 * np.pi) ** 0.5
    invar_diff = y[0]
    if np.isfinite(invar_diff):
        # same for all factor distributions
        for cont in range(ncontrol):
            invar_diff -= c[cont] * delta[cont]

        # per distribution stuff
        for emf in range(nemf):
            diff = invar_diff
            for pos in positions:
                diff -= state[emf, pos] * h[pos]

            for f in range(nfac):
                kf[f] = 0.0
                for pos in positions:
                    kf[f] += cov[emf, f, pos] * h[pos]

            sigma_squared = r[0]
            for pos in positions:
                sigma_squared += kf[pos] * h[pos]

            prob = (
                invariant
                / np.sqrt(sigma_squared)
                * np.exp(-diff ** 2 / (2 * sigma_squared))
            )

            diff /= sigma_squared
            for f in range(nfac):
                state[emf, f] += kf[f] * diff

            for row in range(nfac):
                for col in range(nfac):
                    cov[emf, row, col] -= kf[row] * kf[col] / sigma_squared

            if nemf == 1:
                pass
                like_vec[0] *= prob
            else:
                weights[emf] *= max(prob, 1e-250)

        if nemf >= 2:
            sum_wprob = 0.0
            for emf in range(nemf):
                sum_wprob += weights[emf]

            like_vec[0] *= sum_wprob

            for emf in range(nemf):
                weights[emf] /= sum_wprob


def normal_linear_predict(state, cov, shocks_sds, transition_matrix):
    """Make a linear kalman predict step in linear form.
    Args:
        state (np.ndarray): numpy array of (nemf * nobs, nfac).
        cov (np.ndarray): numpy array of (nemf * nobs, nfac, nfac).
        shocks_sds (np.ndarray): numpy array of (nfac).
        transition_matrix (np.ndarray): state transition matrix of (nfac, nfac), 
            the same for all obs.
                
    References:
    https://en.wikipedia.org/wiki/Kalman_filter
    
    
    """
    nstates, nfac = state.shape
    predicted_states = np.dot(transition_matrix, state.T).T
    Q = np.diag(shocks_sds ** 2)
    # reshape transition matrix to get the same dimension as cov
    predicted_covs = (
        np.matmul(np.matmul(transition_matrix, cov), transition_matrix.T) + Q
    )

    return predicted_states, predicted_covs


def normal_unscented_predict(
    stage,
    sigma_points,
    flat_sigma_points,
    s_weights_m,
    s_weights_c,
    Q,
    transform_sigma_points_args,
    out_flat_states,
    out_flat_covs,
):
    """Make a unscented Kalman filter predict step in square-root form.

    Args:
        stage (int): the development stage in which the predict step is done.
        sigma_points (np.ndarray): numpy array of (nemf * nind, nsigma, nfac)
        flat_sigma_points (np.ndarray): array of (nemf * nind * nsigma, nfac).
            It is a view on sigma_points.
        s_weights_m (np.ndarray): numpy array of length nsigma with sigma
            weights for the means.
        s_weights_c (np.ndarray): numpy array of length nsigma with sigma
            weights for the covariances.
        Q (np.ndarray): numpy array of (nstages, nfac, nfac) with vaiances of
            the transition equation shocks.
        transform_sigma_points_args (dict): (see transform_sigma_points).
        out_flat_states (np.ndarray): output array of (nind * nemf, nfac).
        out_flat_covs (np.ndarray): output array of (nind * nemf, nfac, nfac).

    References:
        Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman
        Filter for State and Parameter-Estimation. 2001.

    """
    nemf_times_nind, nsigma, nfac = sigma_points.shape
    q = Q[stage]
    transform_sigma_points(stage, flat_sigma_points, **transform_sigma_points_args)
    # get them back into states
    predicted_states = np.dot(s_weights_m, sigma_points, out=out_flat_states)
    devs = sigma_points - predicted_states.reshape(nemf_times_nind, 1, nfac)
    # dev_outerprod has dimensions (nemf_times_nind, nsigma, nfac, nfac)
    dev_outerprod = devs.reshape(nemf_times_nind, nsigma, 1, nfac) * devs.reshape(
        nemf_times_nind, nsigma, nfac, 1
    )
    out_flat_covs[:] = (
        np.sum((s_weights_c.reshape(nsigma, 1, 1) * dev_outerprod), axis=1) + q
    )


def sqrt_linear_predict(state, root_cov, shocks_sds, transition_matrix):
    """Make a linear kalman predict step in linear form.
    Args:
        state (np.ndarray): numpy array of (nemf * nobs, nfac).
        root_cov (np.ndarray): upper triangular cholesky factor of the covariance 
            matrix of (nemf * nobs, nfac, nfac).
        shocks_sds (np.ndarray): numpy array of (nfac).
        transition_matrix (np.ndarray): state transition matrix of (nfac, nfac), 
            the same for all obs.
                
    References:
    Robert Grover Brown. Introduction to Random Signals and Applied
        Kalman Filtering. Wiley and sons, 2012.
    
    
    """
    nstates, nfac = state.shape
    predicted_states = np.dot(transition_matrix, state.T).T
    root_Q = np.diag(shocks_sds)

    m = np.empty([nstates, 2 * nfac, nfac])
    m[:, :nfac] = np.matmul(root_cov, transition_matrix.T)
    m[:, nfac:] = root_Q
    # array_qr modifies matrix m in place
    predicted_root_covs = array_qr(m)[:, :nfac, :]

    return predicted_states, predicted_root_covs


def sqrt_unscented_predict(
    stage,
    sigma_points,
    flat_sigma_points,
    s_weights_m,
    s_weights_c,
    Q,
    transform_sigma_points_args,
    out_flat_states,
    out_flat_covs,
):

    """Make a unscented Kalman filter predict step in square-root form.

    The square-root form of the Kalman predict is much more robust than the
    usual form and also much faster.

    Args:
        stage (int): the development stage in which the predict step is done.
        sigma_points (np.ndarray): numpy array of (nemf * nind, nsigma, nfac)
        flat_sigma_points (np.ndarray): array of (nemf * nind * nsigma, nfac).
            It is a view on sigma_points.
        s_weights_m (np.ndarray): numpy array of length nsigma with sigma
            weights for the means.
        s_weights_c (np.ndarray): numpy array of length nsigma with sigma
            weights for the covariances.
        Q (np.ndarray): numpy array of (nstages, nfac, nfac) with vaiances of
            the transition equation shocks.
        transform_sigma_points_args (dict): (see transform_sigma_points).
        out_flat_states (np.ndarray): output array of (nind * nemf, nfac).
        out_flat_covs (np.ndarray): output array of (nind * nemf, nfac, nfac).

    References:
        Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman
        Filter for State and Parameter-Estimation. 2001.

    """
    nemf_times_nind, nsigma, nfac = sigma_points.shape
    q = Q[stage]
    transform_sigma_points(stage, flat_sigma_points, **transform_sigma_points_args)

    # get them back into states
    predicted_states = np.dot(s_weights_m, sigma_points, out=out_flat_states)
    devs = sigma_points - predicted_states.reshape(nemf_times_nind, 1, nfac)

    qr_weights = np.sqrt(s_weights_c).reshape(nsigma, 1)
    qr_points = np.zeros((nemf_times_nind, 3 * nfac + 1, nfac))
    qr_points[:, 0:nsigma, :] = devs * qr_weights
    qr_points[:, nsigma:, :] = np.sqrt(q)
    out_flat_covs[:, 1:, 1:] = array_qr(qr_points)[:, :nfac, :]


def sqrt_probit_update(
    k, t, j, states, covs, mix_weights, like_vec, y_data, c_data, deltas, H, R
):
    raise NotImplementedError("probit updates are not yet implemented")


def normal_probit_update(
    k, t, j, states, covs, mix_weights, like_vec, y_data, c_data, deltas, H, R
):
    raise NotImplementedError("probit updates are not yet implemented")
