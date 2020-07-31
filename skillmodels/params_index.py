import pandas as pd

import skillmodels.transition_functions as tf


def params_index(update_info, labels, dims):
    """Generate index for the params_df for estimagic.

    The index has four levels. The first is the parameter category. The second is the
    period in which the parameters are used. The third and fourth are additional
    descriptors that depend on the category. If the fourth level is not really needed,
    it contains an empty string.

    Args:
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
        in the likelihood function. See :ref:`update_info`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        options (dict): Tuning parameters for the estimation. See :ref:`options`.

    Returns:
        params_index (pd.MultiIndex)

    """
    ind_tups = control_coeffs_index_tuples(labels["controls"], update_info)
    ind_tups += loading_index_tuples(labels["factors"], update_info)
    ind_tups += meas_sd_index_tuples(update_info)
    ind_tups += shock_sd_index_tuples(labels["periods"], labels["factors"])
    ind_tups += initial_mean_index_tuples(dims["n_mixtures"], labels["factors"])
    ind_tups += mixture_weight_index_tuples(dims["n_mixtures"])
    ind_tups += initial_cov_index_tuples(dims["n_mixtures"], labels["factors"])
    ind_tups += trans_coeffs_index_tuples(
        labels["factors"], labels["periods"], labels["transition_names"]
    )

    index = pd.MultiIndex.from_tuples(
        ind_tups, names=["category", "period", "name1", "name2"]
    )
    return index


def control_coeffs_index_tuples(controls, update_info):
    """Index tuples for control coeffs.

    Args:
        controls (list): List of lists. There is one sublist per period which contains
            the names of the control variables in that period. Constant not included.
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.

    """
    ind_tups = []
    for period, meas in update_info.index:
        for cont in controls:
            ind_tups.append(("controls", period, meas, cont))
    return ind_tups


def loading_index_tuples(factors, update_info):
    """Index tuples for loading.

    Args:
        factors (list): The latent factors of the model
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period, meas in update_info.index:
        for factor in factors:
            ind_tups.append(("loadings", period, meas, factor))
    return ind_tups


def meas_sd_index_tuples(update_info):
    """Index tuples for meas_sd.

    Args:
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for period, meas in update_info.index:
        ind_tups.append(("meas_sds", period, meas, "-"))
    return ind_tups


def shock_sd_index_tuples(periods, factors):
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
            ind_tups.append(("shock_sds", period, factor, "-"))
    return ind_tups


def initial_mean_index_tuples(n_mixtures, factors):
    """Index tuples for initial_mean.

    Args:
        n_mixtures (int): Number of elements in the mixture distribution of the factors.
        factors (list): The latent factors of the model

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(n_mixtures):
        for factor in factors:
            ind_tups.append(("initial_states", 0, f"mixture_{emf}", factor))
    return ind_tups


def mixture_weight_index_tuples(n_mixtures):
    """Index tuples for mixture_weight.

    Args:
        n_mixtures (int): Number of elements in the mixture distribution of the factors.

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(n_mixtures):
        ind_tups.append(("mixture_weights", 0, f"mixture_{emf}", "-"))
    return ind_tups


def initial_cov_index_tuples(n_mixtures, factors):
    """Index tuples for initial_cov.

    Args:
        n_mixtures (int): Number of elements in the mixture distribution of the factors.
        factors (list): The latent factors of the model

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for emf in range(n_mixtures):
        for row, factor1 in enumerate(factors):
            for col, factor2 in enumerate(factors):
                if col <= row:
                    ind_tups.append(
                        (
                            "initial_cholcovs",
                            0,
                            f"mixture_{emf}",
                            f"{factor1}-{factor2}",
                        )
                    )
    return ind_tups


def trans_coeffs_index_tuples(factors, periods, transition_names):
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
            ind_tups += func(factor, factors, period)
    return ind_tups
