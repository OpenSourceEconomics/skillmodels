from numba import jit
import numpy as np


@jit
def array_choldate(to_update, update_with, weight):
    """Make a cholesky up- or downdate on all matrices in a numpy array.

    args:
        to_update: [nemf * nind, nfac, nfac]
        update_with: [nemf * nind, nfac]
        weight: a scalar

    The square matrices in to_update have to be UPPER TRIANGULAR
    cholesky factors.

    The function is based on the Matlab code from the article on cholesky
    decomposition on Wikipedia but all slicing is replaced by explicit
    for loops in order to use nopython mode in numba.

    Original Matlab code::

        function [L] = cholupdate(L,x)
            p = length(x);
            for k=1:p
                r = sqrt(L(k,k)^2 + x(k)^2);
                c = r / L(k, k);
                s = x(k) / L(k, k);
                L(k, k) = r;
                L(k+1:p,k) = (L(k+1:p,k) + s*x(k+1:p)) / c;
                x(k+1:p) = c*x(k+1:p) - s*L(k+1:p,k);
            end
        end

    """
    long_side, nfac = update_with.shape

    sign = np.sign(weight)
    weight = abs(weight) ** 0.5
    update_with *= weight

    for u in range(long_side):
        for k in range(nfac):
            d = to_update[u, k, k]
            r_squared = d ** 2 + sign * update_with[u, k] ** 2
            if r_squared < 0.0:
                r = 0.0
            else:
                r = r_squared ** 0.5
            c = r / d
            s = update_with[u, k] / d
            to_update[u, k, k] = r
            for i in range(k + 1, nfac):
                to_update[u, k, i] = (
                    to_update[u, k, i] + sign * s * update_with[u, i]
                ) / c
                update_with[u, i] = c * update_with[u, i] - s * to_update[u, k, i]
    return to_update
