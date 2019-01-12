"""Define several anchor and unanchor functions for the unscented predict.

Each function takes the following arguments:

    * flat_sigma_points: A 2d array with the sigma points that have to be
      anchored or unanchored
    * positions: a list with the positions of the factors (in the
      alphabetically ordered factor list) that have to be anchored
    * anch_params: the last row of the factor loading matrix H
    * intercept: the intercept of the anchoring equation

.. Note:: It is possible that this module changes quite heavily when I optimize
    the predict step for speed.

"""

#anchoring part one
def anchor_flat_sigma_points_linear(flat_sigma_points, positions,
                                    anch_params=None, intercept=None):
    if anch_params is not None and intercept is not None:
        flat_sigma_points[:, positions] = \
            flat_sigma_points[:, positions] * \
            anch_params[positions] + intercept
    elif anch_params is not None:
        flat_sigma_points[:, positions] *= anch_params[positions]
    else:
        flat_sigma_points[:, positions] += intercept


def unanchor_flat_sigma_points_linear(flat_sigma_points, positions,
                                      anch_params=None, intercept=None):
    if anch_params is not None and intercept is not None:
        flat_sigma_points[:, positions] = \
            (flat_sigma_points[:, positions] - intercept) \
            / anch_params[positions]
    elif anch_params is not None:
        flat_sigma_points[:, positions] /= anch_params[positions]
    else:
        flat_sigma_points[:, positions] -= intercept


def anchor_flat_sigma_points_probability(flat_sigma_points, positions, cov,
                                         anch_params=None, intercept=None):
    raise NotImplementedError('currently only linear anchoring is implemented')


def unanchor_flat_sigma_points_probability(flat_sigma_points, positions, cov,
                                           anch_params=None, intercept=None):
    raise NotImplementedError('currently only linear anchoring is implemented')


def anchor_flat_sigma_points_odds_ratio(flat_sigma_points, positions, cov,
                                        anch_params=None, intercept=None):
    raise NotImplementedError('currently only linear anchoring is implemented')


def unanchor_flat_sigma_points_odds_ratio(flat_sigma_points, positions, cov,
                                          anch_params=None, intercept=None):
    raise NotImplementedError('currently only linear anchoring is implemented')


