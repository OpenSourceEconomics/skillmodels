import pandas as pd


def get_params_index(update_info, labels, dimensions, transition_info):
    """Generate index for the params_df for optimagic.

    The index has four levels. The first is the parameter category. The second is the
    period in which the parameters are used. The third and fourth are additional
    descriptors that depend on the category. If the fourth level is not really needed,
    it contains an empty string.

    Args:
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        labels (dict): Dict of lists with labels for the model quantities like
            factors, periods, controls, stagemap and stages. See :ref:`labels`
        options (dict): Tuning parameters for the estimation.
            See :ref:`estimation_options`.

    Returns:
        params_index (pd.MultiIndex)

    """
    ind_tups = get_control_params_index_tuples(labels["controls"], update_info)
    ind_tups += get_loadings_index_tuples(labels["latent_factors"], update_info)
    ind_tups += get_meas_sds_index_tuples(update_info)
    ind_tups += get_shock_sds_index_tuples(labels["periods"], labels["latent_factors"])
    ind_tups += initial_mean_index_tuples(
        dimensions["n_mixtures"],
        labels["latent_factors"],
    )
    ind_tups += get_mixture_weights_index_tuples(dimensions["n_mixtures"])
    ind_tups += get_initial_cholcovs_index_tuples(
        dimensions["n_mixtures"],
        labels["latent_factors"],
    )
    ind_tups += get_transition_index_tuples(transition_info, labels["periods"])

    index = pd.MultiIndex.from_tuples(
        ind_tups,
        names=["category", "period", "name1", "name2"],
    )
    return index


def get_control_params_index_tuples(controls, update_info):
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


def get_loadings_index_tuples(factors, update_info):
    """Index tuples for loading.

    Args:
        factors (list): The latent factors of the model
        update_info (pandas.DataFrame): DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.

    Returns:
        ind_tups (list)

    """
    mask = update_info[factors].to_numpy()
    ind_tups = []
    for i, (period, meas) in enumerate(update_info.index):
        for f, factor in enumerate(factors):
            if mask[i, f]:
                ind_tups.append(("loadings", period, meas, factor))
    return ind_tups


def get_meas_sds_index_tuples(update_info):
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


def get_shock_sds_index_tuples(periods, factors):
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


def get_mixture_weights_index_tuples(n_mixtures):
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


def get_initial_cholcovs_index_tuples(n_mixtures, factors):
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
                        ),
                    )
    return ind_tups


def get_transition_index_tuples(transition_info, periods):
    """Index tuples for transition equation coefficients.

    Args:
        latent_factors (list): The latent factors of the model
        all_factors (list): The latent and observed factors of the model.
        periods (list): The periods of the model
        transition_names (list): name of the transition equation of each factor

    Returns:
        ind_tups (list)

    """
    ind_tups = []
    for factor, names in transition_info["param_names"].items():
        for period in periods[:-1]:
            for name in names:
                ind_tups.append(("transition", period, factor, name))
    return ind_tups
