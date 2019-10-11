def calculate_sigma_points(states, flat_covs, scaling_factor, out):
    """Calculate the array of sigma_points for the unscented transform.

    Args:
        states (np.ndarray): numpy array of (nind, nmixtures, nfac)
        flat_covs (np.ndarray): numpy array of (nind * nmixtures, nfac, nfac)
        scaling_factor (float): a constant scaling factor for sigma points that
            depends on the sigma_point algorithm chosen.
        out (np.ndarray): numpy array of (nmixtures * nind, nsigma, nfac) with
            sigma_points.

    """
    cholcovs_t = flat_covs[:, 1:, 1:]

    nmixtures_times_nind, nsigma, nfac = out.shape
    out[:] = states.reshape(nmixtures_times_nind, 1, nfac)
    cholcovs_t *= scaling_factor
    out[:, 1 : nfac + 1, :] += cholcovs_t
    out[:, nfac + 1 :, :] -= cholcovs_t
