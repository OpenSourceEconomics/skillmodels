import skillmodels.model_functions.anchoring_functions as anch
import skillmodels.model_functions.transition_functions as trans
import numpy as np


def transform_sigma_points(
    period,
    flat_sigma_points,
    transition_argument_dicts,
    transition_function_names,
    anchoring_positions=None,
    anch_params=None,
    intercept=None,
):
    """Transform an array of sigma_points for the unscented predict.

    This function automatically anchors the sigma points and unanchors the
    results if the necessary arguments are provided.

    """
    nfac = flat_sigma_points.shape[1]
    intermediate_array = np.empty_like(flat_sigma_points)

    # anchor the flat_sigma_points
    if anch_params is not None:
        anch_func = "anchor_flat_sigma_points_linear"
        getattr(anch, anch_func)(
            flat_sigma_points, anchoring_positions, anch_params, intercept
        )

    for f in range(nfac):
        intermediate_array[:, f] = getattr(trans, transition_function_names[f])(
            flat_sigma_points, **transition_argument_dicts[period][f]
        )

    # copy them into the sigma_point array
    flat_sigma_points[:] = intermediate_array[:]

    if anch_params is not None:
        unanch_func = "unanchor_flat_sigma_points_linear"
        getattr(anch, unanch_func)(
            flat_sigma_points, anchoring_positions, anch_params, intercept
        )
