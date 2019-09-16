from skillmodels.estimation.parse_params import parse_params
from skillmodels.estimation.parse_params import restore_unestimated_quantities
from skillmodels.fast_routines.kalman_filters import normal_linear_update
from skillmodels.fast_routines.kalman_filters import normal_unscented_predict
from skillmodels.fast_routines.kalman_filters import sqrt_linear_update
from skillmodels.fast_routines.kalman_filters import sqrt_unscented_predict
from skillmodels.fast_routines.sigma_points import calculate_sigma_points


def log_likelihood_contributions(
    params,
    like_contributions,
    parse_params_args,
    periods,
    nmeas_list,
    anchoring,
    square_root_filters,
    update_args,
    predict_args,
    calculate_sigma_points_args,
    restore_args,
):
    """Return the log likelihood contributions per update and individual in the sample.

    Users do not have to call this function directly and do not have to bother
    about its arguments but the function nicely shows how the likelihood
    interpretation of the Kalman filter allow to break the large likelihood
    problem of the model into many smaller problems.

    First the params vector is parsed into the many quantities that depend on
    it. See :ref:`params_and_quants` for details.

    Then, for each period of the model first all Kalman updates for the
    measurement equations are done. Each Kalman update updates the following
    quantities:

        * the state array X
        * the covariance matrices P
        * the likelihood vector like_vec
        * the weights for the mixture distribution of the factors W

    Then the predict step of the Unscented Kalman filter is applied. The
    predict step propagates the following quantities to the next period:

        * the state array X
        * the covariance matrices P

    In the last period an additional update is done to incorporate the
    anchoring equation into the likelihood.

    """
    restore_unestimated_quantities(**restore_args)
    like_contributions[:] = 0.0

    parse_params(params, **parse_params_args)

    k = 0
    for t in periods:
        for _j in range(nmeas_list[t]):
            # measurement updates
            update(square_root_filters, update_args[k])
            k += 1
        if t < periods[-1]:
            calculate_sigma_points(**calculate_sigma_points_args)
            predict(t, square_root_filters, predict_args)
    if anchoring is True:
        update(square_root_filters, update_args[k])

    return like_contributions


def update(square_root_filters, update_args):
    """Select and call the correct update function.

    The actual update functions are implemented in several modules in
    :ref:`fast_routines`

    """
    if square_root_filters is True:
        sqrt_linear_update(*update_args)
    else:
        normal_linear_update(*update_args)


def predict(period, square_root_filters, predict_args):
    """Select and call the correct predict function.

    The actual predict functions are implemented in several modules in
    :ref:`fast_routines`

    """
    if square_root_filters is True:
        sqrt_unscented_predict(period, **predict_args)
    else:
        normal_unscented_predict(period, **predict_args)
