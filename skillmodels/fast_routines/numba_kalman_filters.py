from numba import float64 as f64
from numba import int64 as i64
from numba import guvectorize
import numpy as np


@guvectorize([(f64[:, :], f64[:, :, :], f64[:], f64[:], f64[:],
               f64[:], f64[:], f64[:], i64[:], f64[:])],
             ('(nemf, nfac), (nemf, nfac_, nfac_), (), (), (ncon), '
              '(ncon), (nfac), (), (ninc), (nemf)'),
             target='cpu', nopython=True)
def sqrt_linear_update(state, cov, like_vec, y, c, delta, h, sqrt_r,
                       positions, weights):
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
            prob = invariant / np.abs(sigma) * np.exp(
                - diff ** 2 / (2 * sigma ** 2))

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


@guvectorize([(f64[:, :], f64[:, :, :], f64[:], f64[:], f64[:],
               f64[:], f64[:], f64[:], i64[:], f64[:], f64[:])],
             ('(nemf, nfac), (nemf, nfac, nfac), (), (), (ncon), '
              '(ncon), (nfac), (), (ninc), (nemf), (nfac)'),
             target='cpu', nopython=True)
def normal_linear_update(state, cov, like_vec, y, c, delta, h, r, positions,
                         weights, kf):
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

            prob = invariant / np.sqrt(sigma_squared) * np.exp(
                - diff ** 2 / (2 * sigma_squared))

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
