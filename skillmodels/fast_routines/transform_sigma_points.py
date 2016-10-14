import skillmodels.model_functions.anchoring_functions as anch
import skillmodels.model_functions.transition_functions as trans
import skillmodels.model_functions.endogeneity_functions as endog
import numpy as np


def transform_sigma_points(
        stage, flat_sigma_points, transition_argument_dicts,
        transition_function_names,
        anchoring_type=None, anchoring_positions=None,
        anch_params=None, intercept=None,
        psi=None, endog_position=None, correction_func=None):
    """Transform an array of sigma_points for the unscented predict.

    This function automatically anchors the sigma points and unanchors the
    results if the necessary arguments are provided.

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

        # call the endogeneity correction function()
        flat_sigma_points[:, p] = getattr(endog, correction_func)(
            flat_sigma_points, psi, endog_position)

    # apply the other transition equations
    for f in range(nfac):
        if f != endog_position:
            intermediate_array[:, f] = \
                getattr(trans, transition_function_names[f])(
                    flat_sigma_points, **transition_argument_dicts[stage][f])

    # copy them into the sigma_point array
    flat_sigma_points[:] = intermediate_array[:]

    if anchoring_type is not None:
        unanch_func = 'unanchor_flat_sigma_points_{}'.format(anchoring_type)
        getattr(anch, unanch_func)(
            flat_sigma_points, anchoring_positions, anch_params, intercept)

