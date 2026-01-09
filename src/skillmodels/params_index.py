from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from skillmodels.types import (
        Dimensions,
        EndogenousFactorsInfo,
        Labels,
        TransitionInfo,
    )


def get_params_index(
    update_info: pd.DataFrame,
    labels: Labels,
    dimensions: Dimensions,
    transition_info: TransitionInfo,
    endogenous_factors_info: EndogenousFactorsInfo,
) -> pd.MultiIndex:
    """Generate index for the params_df for optimagic.

    The index has four levels. The first is the parameter category. The second is the
    period in which the parameters are used. The third and fourth are additional
    descriptors that depend on the category. If the fourth level is not really needed,
    it contains an empty string.

    Args:
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.
        labels: Labels for model quantities.
        dimensions: Dimensional information.
        transition_info: Information about the transition equations.
        endogenous_factors_info: Information about endogenous factors, if any.

    Returns:
        params_index (pd.MultiIndex)

    """
    ind_tups = get_control_params_index_tuples(
        controls=labels.controls, update_info=update_info
    )
    ind_tups += get_loadings_index_tuples(
        factors=labels.latent_factors, update_info=update_info
    )
    ind_tups += get_meas_sds_index_tuples(update_info=update_info)
    ind_tups += get_shock_sds_index_tuples(
        aug_periods=labels.aug_periods,
        factors=labels.latent_factors,
        has_endogenous_factors=endogenous_factors_info.has_endogenous_factors,
    )
    ind_tups += initial_mean_index_tuples(
        n_mixtures=dimensions.n_mixtures,
        factors=labels.latent_factors,
    )
    ind_tups += get_mixture_weights_index_tuples(n_mixtures=dimensions.n_mixtures)
    ind_tups += get_initial_cholcovs_index_tuples(
        n_mixtures=dimensions.n_mixtures,
        factors=labels.latent_factors,
    )
    ind_tups += get_transition_index_tuples(
        transition_info=transition_info,
        aug_periods=labels.aug_periods,
        has_endogenous_factors=endogenous_factors_info.has_endogenous_factors,
    )

    index = pd.MultiIndex.from_tuples(
        ind_tups,
        names=["category", "aug_period", "name1", "name2"],
    )
    return index


def get_control_params_index_tuples(
    controls: tuple[str, ...],
    update_info: pd.DataFrame,
) -> list[tuple[str, int, str, str]]:
    """Index tuples for control coeffs.

    Args:
        controls: Names of the control variables. Constant not included.
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.

    """
    ind_tups = []
    for aug_period, meas in update_info.index:
        for cont in controls:
            ind_tups.append(("controls", aug_period, meas, cont))
    return ind_tups


def get_loadings_index_tuples(
    factors: tuple[str, ...],
    update_info: pd.DataFrame,
) -> list[tuple[str, int, str, str]]:
    """Index tuples for loading.

    Args:
        factors: The latent factors of the model.
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.

    """
    mask = update_info[list(factors)].to_numpy()
    ind_tups = []
    for i, (aug_period, meas) in enumerate(update_info.index):
        for f, factor in enumerate(factors):
            if mask[i, f]:
                ind_tups.append(("loadings", aug_period, meas, factor))
    return ind_tups


def get_meas_sds_index_tuples(
    update_info: pd.DataFrame,
) -> list[tuple[str, int, str, str]]:
    """Index tuples for meas_sd.

    Args:
        update_info: DataFrame with one row per Kalman update needed
            in the likelihood function. See :ref:`update_info`.

    """
    ind_tups = []
    for aug_period, meas in update_info.index:
        ind_tups.append(("meas_sds", aug_period, meas, "-"))
    return ind_tups


def get_shock_sds_index_tuples(
    aug_periods: tuple[int, ...],
    factors: tuple[str, ...],
    has_endogenous_factors: bool,
) -> list[tuple[str, int, str, str]]:
    """Index tuples for shock_sd.

    Args:
        aug_periods: The augmented periods of the model.
        factors: The latent factors of the model.
        has_endogenous_factors: Whether the model has endogenous factors.

    """
    end = -2 if has_endogenous_factors else -1
    ind_tups = []
    for aug_period in aug_periods[:end]:
        for factor in factors:
            ind_tups.append(("shock_sds", aug_period, factor, "-"))
    return ind_tups


def initial_mean_index_tuples(
    n_mixtures: int,
    factors: tuple[str, ...],
) -> list[tuple[str, int, str, str]]:
    """Index tuples for initial_mean.

    Args:
        n_mixtures: Number of elements in the mixture distribution of the factors.
        factors: The latent factors of the model.

    """
    ind_tups = []
    for emf in range(n_mixtures):
        for factor in factors:
            ind_tups.append(("initial_states", 0, f"mixture_{emf}", factor))
    return ind_tups


def get_mixture_weights_index_tuples(
    n_mixtures: int,
) -> list[tuple[str, int, str, str]]:
    """Index tuples for mixture_weight.

    Args:
        n_mixtures: Number of elements in the mixture distribution of the factors.

    """
    ind_tups = []
    for emf in range(n_mixtures):
        ind_tups.append(("mixture_weights", 0, f"mixture_{emf}", "-"))
    return ind_tups


def get_initial_cholcovs_index_tuples(
    n_mixtures: int,
    factors: tuple[str, ...],
) -> list[tuple[str, int, str, str]]:
    """Index tuples for initial_cov.

    Args:
        n_mixtures: Number of elements in the mixture distribution of the factors.
        factors: The latent factors of the model.

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


def get_transition_index_tuples(
    transition_info: TransitionInfo,
    aug_periods: tuple[int, ...],
    has_endogenous_factors: bool,
) -> list[tuple[str, int, str, str]]:
    """Index tuples for transition equation coefficients.

    Args:
        transition_info: Information about transition equations.
        aug_periods: The augmented periods of the model.
        has_endogenous_factors: Whether the model has endogenous factors.

    """
    end = -2 if has_endogenous_factors else -1
    ind_tups = []
    for factor, names in transition_info.param_names.items():
        for aug_period in aug_periods[:end]:
            for name in names:
                ind_tups.append(("transition", aug_period, factor, name))
    return ind_tups
