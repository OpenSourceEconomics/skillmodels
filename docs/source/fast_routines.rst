.. _fast_routines:


***************************
The fast_routines directory
***************************

The directory *skillmodels.fast_routines* contains very low level functions that are optimized for speed. Examples are a QR-decomposition, Cholesky up- and downdates and Kalman filters. Users usually won't call these functions directly.


The choldate module
*******************

.. automodule:: skillmodels.fast_routines.choldate
    :members:


The kalman_filters module
*************************

.. automodule:: skillmodels.fast_routines.kalman_filters
    :members:

.. autofunction:: sqrt_linear_update(state, cov, like_vec, y, c, delta, h, sqrt_r, positions, weights)

.. autofunction:: normal_linear_update(state, cov, like_vec, y, c, delta, h, sqrt_r, positions, weights, kf)


The qr_decomposition module
***************************

.. automodule:: skillmodels.fast_routines.qr_decomposition
    :members:

The sigma_points module
***********************

.. automodule:: skillmodels.fast_routines.sigma_points
    :members:

The transform_sigma_points module
*********************************

.. automodule:: skillmodels.fast_routines.transform_sigma_points
    :members:
