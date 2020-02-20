import pandas as pd

import skillmodels.model_functions.transition_functions as tf


def params_index(
    update_info, controls, factors, nmixtures, transition_names, included_factors
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
        nmixtures (int): Number of elements in the mixture distribution of the factors.
        transition_names (list): name of the transition equation of each factor
        included_factors (list): the factors that appear on the right hand side of
            the transition equations of the latent factors.

    Returns:
        params_index (pd.MultiIndex)

    """

    periods = list(range(len(controls)))

    ind_tups = _control_coeffs_index_tuples(controls, update_info)
    ind_tups += _loading_index_tuples(factors, update_info)
    ind_tups += _meas_sd_index_tuples(update_info)
    ind_tups += _shock_sd_index_tuples(periods, factors)
    ind_tups += _initial_mean_index_tuples(nmixtures, factors)
    ind_tups += _mixture_weight_index_tuples(nmixtures)
    ind_tups += _initial_cov_index_tuples(nmixtures, factors)
    ind_tups += _trans_coeffs_index_tuples(
        factors, periods, transition_names, included_factors
    )

    index = pd.MultiIndex.from_tuples(
        ind_tups, names=["category", "period", "name1", "name2"]
    )
    return index


def _control_coeffs_index_tuples(controls, update_info):
    """Index tuples for control coeffs.

    Args:
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.
        controls (list): List of lists. There is one sublist per period which contains
            the names of the control variables in that period. Constant not included.

    """
    ind_tups = []
    for period, meas in update_info.index:
        for cont in ["constant"] + list(controls[period]):
            ind_tups.append(("control_coeffs", period, meas, cont))
    return ind_tups


def _loading_index_tuples(factors, update_info):
    """Index tuples for loading.

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
            ind_tups.append(("loading", period, meas, factor))
    return ind_tups


def _meas_sd_index_tuples(update_info):
    """Index tuples for meas_sd.

    Args:
        update_info (DataFrame): DataFrame with one row per update. It has aMultiIndex
            that indicates the period and name of the measurement for that update.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period, meas in update_info.index:
        ind_tups.append(("meas_sd", period, meas, "-"))
    return ind_tups


def _shock_sd_index_tuples(periods, factors):
    """Index tuples for shock_sd.

    Args:
        periods (list): The periods of the model.
        factors (list): The latent factors of the model.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period in periods[:-1]:
        for factor in factors:
            ind_tups.append(("shock_sd", period, factor, "-"))
    return ind_tups


def _initial_mean_index_tuples(nmixtures, factors):
    """Index tuples for initial_mean.

    Args:
        nmixtures (int): Number of elements in the mixture distribution of the factors.
        factors (list): The latent factors of the model

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(nmixtures):
        for factor in factors:
            ind_tups.append(("initial_mean", 0, f"mixture_{emf}", factor))
    return ind_tups


def _mixture_weight_index_tuples(nmixtures):
    """Index tuples for mixture_weight.

    Args:
        nmixtures (int): Number of elements in the mixture distribution of the factors.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(nmixtures):
        ind_tups.append(("mixture_weight", 0, f"mixture_{emf}", "-"))
    return ind_tups


def _initial_cov_index_tuples(nmixtures, factors):
    """Index tuples for initial_cov.

    Args:
        nmixtures (int): Number of elements in the mixture distribution of the factors.
        factors (list): The latent factors of the model

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(nmixtures):
        for row, factor1 in enumerate(factors):
            for col, factor2 in enumerate(factors):
                if col <= row:
                    ind_tups.append(
                        ("initial_cov", 0, f"mixture_{emf}", f"{factor1}-{factor2}")
                    )
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
