import pandas as pd

import skillmodels.model_functions.transition_functions as tf


def params_index(
    update_info, controls, factors, nemf, transition_names, included_factors
):
    """Generate index for the params_df for estimagic.

    The index has four levels. The first is the parameter category. The second is the
    period in which the parameters are used. The third and fourth are additional
    descriptors that depend on the category. If the fourth level is not really needed,
    it contains an empty string.

    Args:
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.
        controls (list): List of lists. There is one sublist per period which contains
            the names of the control variables in that period. Constant not included.
        factors (list): The latent factors of the model
        nemf (int): Number of elements in the mixture distribution of the factors.
        transition_names (list): name of the transition equation of each factor
        included_factors (list): the factors that appear on the right hand side of
            the transition equations of the latent factors.

    Returns:
        params_index (pd.MultiIndex)

    """

    periods = list(range(len(controls)))

    ind_tups = _delta_index_tuples(controls, update_info)
    ind_tups += _h_index_tuples(factors, update_info)
    ind_tups += _r_index_tuples(update_info)
    ind_tups += _q_index_tuples(periods, factors)
    ind_tups += _x_index_tuples(nemf, factors)
    ind_tups += _w_index_tuples(nemf)
    ind_tups += _p_index_tuples(nemf, factors)
    ind_tups += _trans_coeffs_index_tuples(
        factors, periods, transition_names, included_factors
    )

    index = pd.MultiIndex.from_tuples(
        ind_tups, names=["category", "period", "name1", "name2"]
    )
    return index


def _delta_index_tuples(controls, update_info):
    """Index tuples for delta.

    Args:
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.
        controls (list): List of lists. There is one sublist per period which contains
            the names of the control variables in that period. Constant not included.

    """
    ind_tups = []
    for period, meas in update_info.index:
        for cont in ["constant"] + list(controls[period]):
            ind_tups.append(("delta", period, meas, cont))
    return ind_tups


def _h_index_tuples(factors, update_info):
    """Index tuples for h.

    Args:
        factors (list): The latent factors of the model
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period, meas in update_info.index:
        for factor in factors:
            ind_tups.append(("h", period, meas, factor))
    return ind_tups


def _r_index_tuples(update_info):
    """Index tuples for r.

    Args:
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period, meas in update_info.index:
        ind_tups.append(("r", period, meas, "-"))
    return ind_tups


def _q_index_tuples(periods, factors):
    """Index tuples for q.

    Args:
        periods (list): The periods of the model.
        factors (list): The latent factors of the model.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period in periods[:-1]:
        for factor in factors:
            ind_tups.append(("q", period, factor, "-"))
    return ind_tups


def _x_index_tuples(nemf, factors):
    """Index tuples for x.

    Args:
        nemf (int): Number of elements in the mixture distribution of the factors.
        factors (list): The latent factors of the model

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(nemf):
        for factor in factors:
            ind_tups.append(("x", 0, f"mixture_{emf}", factor))
    return ind_tups


def _w_index_tuples(nemf):
    """Index tuples for w.

    Args:
        nemf (int): Number of elements in the mixture distribution of the factors.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(nemf):
        ind_tups.append(("w", 0, f"mixture_{emf}", "-"))
    return ind_tups


def _p_index_tuples(nemf, factors):
    """Index tuples for p.

    Args:
        nemf (int): Number of elements in the mixture distribution of the factors.
        factors (list): The latent factors of the model

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(nemf):
        for row, factor1 in enumerate(factors):
            for col, factor2 in enumerate(factors):
                if col <= row:
                    ind_tups.append(("p", 0, f"mixture_{emf}", f"{factor1}-{factor2}"))
    return ind_tups


def _trans_coeffs_index_tuples(factors, periods, transition_names, included_factors):
    """Index tuples for transition equation coefficients.

    Args:
        factors (list): The latent factors of the model
        periods (list): The periods of the model
        transition_names (list): name of the transition equation of each factor
        included_factors (list): the factors that appear on the right hand side of
            the transition equations of the latent factors.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period in periods[:-1]:
        for f, factor in enumerate(factors):
            func = getattr(tf, "index_tuples_{}".format(transition_names[f]))
            ind_tups += func(factor, included_factors[f], period)
    return ind_tups
