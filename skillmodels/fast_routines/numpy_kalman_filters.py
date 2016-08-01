from skillmodels.fast_routines.transform_sigma_points import transform_sigma_points
from skillmodels.fast_routines.qr_decomposition import numba_array_qr
from skillmodels.fast_routines.choldate import numba_array_choldate
import numpy as np


def normal_unscented_predict(stage, sigma_points, flat_sigma_points,
                             s_weights_m, s_weights_c, Q,
                             transform_sigma_points_args,
                             out_flat_states, out_flat_covs):
    """Predicted states and covs according to the unscented Kalman filter.

    Args:
        sigma_points (np.ndarray):

    """
    nemf_times_nind, nsigma, nfac = sigma_points.shape
    q = Q[stage]
    transform_sigma_points(stage, flat_sigma_points,
                           **transform_sigma_points_args)
    # get them back into states
    predicted_states = np.dot(s_weights_m, sigma_points, out=out_flat_states)
    devs = sigma_points - predicted_states.reshape(nemf_times_nind, 1, nfac)
    # dev_outerprod has dimensions (nemf_times_nind, nsigma, nfac, nfac)
    dev_outerprod = devs.reshape(nemf_times_nind, nsigma, 1, nfac) \
        * devs.reshape(nemf_times_nind, nsigma, nfac, 1)
    out_flat_covs[:] = \
        np.sum((s_weights_c.reshape(nsigma, 1, 1) * dev_outerprod), axis=1) + q


def sqrt_unscented_predict(stage, sigma_points, flat_sigma_points, s_weights_m,
                           s_weights_c, Q, transform_sigma_points_args,
                           out_flat_states, out_flat_covs):
    """Return predicted states and square-roots of covs.

    The algorithm used comes from:
    Van Der Merwe, R. and Wan, E.A. The Square-Root Unscented Kalman
    Filter for State and Parameter-Estimation. 2001.

    """
    nemf_times_nind, nsigma, nfac = sigma_points.shape
    q = Q[stage]
    transform_sigma_points(stage, flat_sigma_points,
                           **transform_sigma_points_args)

    # get them back into states
    predicted_states = np.dot(s_weights_m, sigma_points, out=out_flat_states)
    devs = sigma_points - predicted_states.reshape(nemf_times_nind, 1, nfac)

    if (s_weights_c >= 0).all():
        qr_weights = np.sqrt(s_weights_c).reshape(nsigma, 1)
        qr_points = np.zeros((nemf_times_nind, 3 * nfac + 1, nfac))
        qr_points[:, 0: nsigma, :] = devs * qr_weights
        qr_points[:, nsigma:, :] = np.sqrt(q)
        out_flat_covs[:, 1:, 1:] = numba_array_qr(qr_points)[:, :nfac, :]
    else:
        raise NotImplementedError('This has to be adjusted to the new form of the cov matrices')
        qr_weight = np.sqrt(s_weights_c[1])
        # create the array of compound matrices on which to perform qr
        qr_points = np.zeros((nemf_times_nind, 3 * nfac, nfac))
        # the first block of each matrix are devs, starting from dev 1,
        # (i.e. excluding the zero_th row), weighted with qr_weight.
        qr_points[:, 0: 2 * nfac, :] = devs[:, 1:, :] * qr_weight
        # the others are the transpose of the cholesky factor of q.
        # Note that in in our case q is diagonal, such that a element-wise
        # square root of q is the same as the (transpose of the) cholesky
        # factor but also defined if q is only positive semi-definite. This
        # is important because constant transition equations imply zero
        # diagonal elements of q.which makes it not positive definite.
        qr_points[:, 2 * nfac:, :] = np.sqrt(q)

        update_points = devs[:, 0, :]
        update_weight = s_weights_c[0]

        out_flat_covs[:, 1:, 1:] = numba_array_qr(qr_points)[:, :, :nfac, :]
        out_flat_covs = numba_array_choldate(
            out_flat_covs, update_points, update_weight)


def sqrt_probit_update(k, t, j, states, covs, mix_weights, like_vec, y_data,
                       c_data, deltas, H, R):
    raise NotImplementedError('probit updates are not yet implemented')


def normal_probit_update(k, t, j, states, covs, mix_weights, like_vec, y_data,
                         c_data, deltas, H, R):
    raise NotImplementedError('probit updates are not yet implemented')

