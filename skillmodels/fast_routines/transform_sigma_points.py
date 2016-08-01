import skillmodels.model_functions.anchoring_functions as anch
import skillmodels.model_functions.transition_functions as trans
import skillmodels.model_functions.endogeneity_functions as endog
import numpy as np


def transform_sigma_points(
        stage, flat_sigma_points, transition_argument_dicts,
        transition_function_names,
        anchoring_type=None, anchoring_positions=None,
        anch_params=None, intercept=None,
        psi=None, tau=None, endog_position=None, correction_func=None):
    """Transform an array of sigma_points for the unscented predict.

    Note: This function will probably be replaced by a more optimized one.
    I write this mostly to have a baseline for optimization.

    """
    nfac = flat_sigma_points.shape[1]
    intermediate_array = np.empty_like(flat_sigma_points)

    # anchor the flat_sigma_points
    if anchoring_type is not None:
        anch_func = 'anchor_flat_sigma_points_{}'.format(anchoring_type)
        getattr(anch, anch_func)(
            flat_sigma_points, anchoring_positions, anch_params, intercept)

    # apply transition function of endog_factor
    if endog_position is not None:
        p = endog_position
        intermediate_array[:, p] = getattr(trans, transition_function_names[p])(
            flat_sigma_points, **transition_argument_dicts[stage][p])
        # store the unchanged endog_factor
        heterogeneity = flat_sigma_points[:, p].copy()

        # call the endogeneity correction function()
        flat_sigma_points[:, p] = getattr(endog, correction_func)(
            flat_sigma_points, psi, endog_position)

    # apply the other transition equations
    for f in range(nfac):
        if f != endog_position:
            intermediate_array[:, f] = \
                getattr(trans, transition_function_names[f])(
                    flat_sigma_points, **transition_argument_dicts[stage][f])

    # copy them into the sigma_point array, adding the additively separable
    # part of the endogeneity correction where necessary
    for f in range(nfac):
        if tau is None or tau[f] == 0.0:
            flat_sigma_points[:, f] = intermediate_array[:, f]
        else:
            flat_sigma_points[:, f] = \
                intermediate_array[:, f] + tau[f] * heterogeneity

    if anchoring_type is not None:
        unanch_func = 'unanchor_flat_sigma_points_{}'.format(anchoring_type)
        getattr(anch, unanch_func)(
            flat_sigma_points, anchoring_positions, anch_params, intercept)

