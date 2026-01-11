"""Functions to compute and process filtered latent states."""

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info, parse_params
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model

if TYPE_CHECKING:
    import pandas as pd


def get_filtered_states(
    model_dict: dict,
    data: pd.DataFrame,
    params: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute filtered latent states given data and estimated parameters."""
    max_inputs = get_maximization_inputs(model_dict=model_dict, data=data)
    params = params.loc[max_inputs["params_template"].index]
    debug_loglike = max_inputs["debug_loglike"]
    debug_data = debug_loglike(params)
    unanchored_states_df = debug_data["filtered_states"]
    unanchored_ranges = debug_data["state_ranges"]
    model = process_model(model_dict)

    anchored_states_df = anchor_states_df(
        states_df=unanchored_states_df,
        model_dict=model_dict,
        params=params,
        use_aug_period=True,
    )

    anchored_ranges = create_state_ranges(
        filtered_states=anchored_states_df,
        factors=model.labels.latent_factors,
    )

    return {
        "anchored_states": {
            "states": anchored_states_df,
            "state_ranges": anchored_ranges,
        },
        "unanchored_states": {
            "states": unanchored_states_df,
            "state_ranges": unanchored_ranges,
        },
    }


def anchor_states_df(
    states_df: pd.DataFrame,
    model_dict: dict,
    params: pd.DataFrame,
    *,
    use_aug_period: bool,
) -> pd.DataFrame:
    """Anchor states in a DataFrame.

    The DataFrame is expected to have a column called "period" as well as one column
    for each latent factor.

    All other columns are not affected.

    This is a bit difficult because we need to re-use `parse_params` (which was meant
    as an internal function that only works with jax objects).

    """
    model = process_model(model_dict)

    p_index = get_params_index(
        update_info=model.update_info,
        labels=model.labels,
        dimensions=model.dimensions,
        transition_info=model.transition_info,
        endogenous_factors_info=model.endogenous_factors_info,
    )

    params = params.loc[p_index]

    parsing_info = create_parsing_info(
        params_index=p_index,
        update_info=model.update_info,
        labels=model.labels,
        anchoring=model.anchoring,
        has_endogenous_factors=model.endogenous_factors_info.has_endogenous_factors,
    )

    *_, parsed_params = parse_params(
        params=jnp.array(params["value"].to_numpy()),
        parsing_info=parsing_info,
        dimensions=model.dimensions,
        labels=model.labels,
        n_obs=1,
    )

    n_latent = model.dimensions.n_latent_factors

    _scaling_factors = np.array(parsed_params.anchoring_scaling_factors[:, :n_latent])
    _constants = np.array(parsed_params.anchoring_constants[:, :n_latent])
    if use_aug_period:
        period_arr = states_df["aug_period"].to_numpy()
        ap_to_p = model.labels.aug_periods_to_periods
        scaling_factors = np.empty(shape=(len(ap_to_p), n_latent))
        constants = np.empty(shape=(len(ap_to_p), n_latent))
        for ap, p in ap_to_p.items():
            scaling_factors[ap] = _scaling_factors[p]
            constants[ap] = _constants[p]
    else:
        period_arr = states_df["period"].to_numpy()
        scaling_factors = _scaling_factors
        constants = _constants

    scaling_arr = scaling_factors[period_arr]
    constants_arr = constants[period_arr]

    out = states_df.copy(deep=True)
    for pos, factor in enumerate(model.labels.latent_factors):
        out[factor] = constants_arr[:, pos] + states_df[factor] * scaling_arr[:, pos]

    return out[states_df.columns]
