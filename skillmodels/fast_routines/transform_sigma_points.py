import numpy as np

import skillmodels.model_functions.transition_functions as trans


def transform_sigma_points(
    period,
    flat_sigma_points,
    transition_argument_dicts,
    transition_function_names,
    anchoring_loadings=None,
    anchoring_positions=None,
    anchoring_variables=None,
):
    """Transform an array of sigma_points for the unscented predict.

    This function automatically anchors the sigma points and unanchors the
    results if the necessary arguments are provided.

    """
    nsigma_times_nind, nfac = flat_sigma_points.shape
    nsigma = int(2 * nfac + 1)
    nind = int(nsigma_times_nind / nsigma)

    intermediate_array = np.empty_like(flat_sigma_points)
    sigma_points = flat_sigma_points.reshape(nind, nsigma, nfac)

    if anchoring_loadings is not None:
        anchor_sigma_points(
            sigma_points,
            anchoring_loadings[period],
            anchoring_positions,
            anchoring_variables[period],
        )

    for f in range(nfac):
        intermediate_array[:, f] = getattr(trans, transition_function_names[f])(
            flat_sigma_points, **transition_argument_dicts[period][f]
        )

    # copy them into the sigma_point array
    flat_sigma_points[:] = intermediate_array[:]

    if anchoring_loadings is not None:
        unanchor_sigma_points(
            sigma_points,
            anchoring_loadings[period + 1],
            anchoring_positions,
            anchoring_variables[period + 1],
        )


def anchor_sigma_points(sigma_points, loadings, positions, variables):
    for p, pos in enumerate(positions):
        sigma_points[:, :, pos] *= loadings[p, pos]
        if variables is not None:
            sigma_points[:, :, pos] -= variables[p]


def unanchor_sigma_points(sigma_points, loadings, positions, variables):
    for p, pos in enumerate(positions):
        sigma_points[:, :, pos] /= loadings[p, pos]
        if variables is not None:
            sigma_points[:, :, pos] += variables[p]
