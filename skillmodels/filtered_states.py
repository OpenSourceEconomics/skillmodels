import jax.numpy as jnp
import numpy as np

from skillmodels.likelihood_function import get_maximization_inputs
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info
from skillmodels.parse_params import parse_params
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model


def get_filtered_states(model_dict, data, params):
    max_inputs = get_maximization_inputs(model_dict=model_dict, data=data)
    debug_loglike = max_inputs["debug_loglike"]
    debug_data = debug_loglike(params)
    unanchored_states_df = debug_data["filtered_states"]
    unanchored_ranges = debug_data["state_ranges"]
    model = process_model(model_dict)

    anchored_states_df = anchor_states_df(
        states_df=unanchored_states_df, model_dict=model_dict, params=params
    )

    anchored_ranges = create_state_ranges(
        filtered_states=anchored_states_df, factors=model["labels"]["latent_factors"]
    )

    out = {
        "anchored": {"states": anchor_states_df, "state_ranges": anchored_ranges},
        "unanchored": {
            "states": unanchored_states_df,
            "state_ranges": unanchored_ranges,
        },
    }

    return out


def anchor_states_df(states_df, model_dict, params):
    """Anchor states in a DataFrame.

    The DataFrame is expected to have a column called "period" as well as one column
    for each latent factor.

    All other columns are not affected.

    This is a bit difficult because we need to re-use `parse_params` (which was meant
    as an internal function that only works with jax objects).

    """
    model = process_model(model_dict)

    p_index = get_params_index(
        model["update_info"],
        model["labels"],
        model["dimensions"],
        model["transition_info"],
    )

    params = params.loc[p_index]

    parsing_info = create_parsing_info(
        p_index, model["update_info"], model["labels"], model["anchoring"]
    )

    *_, pardict = parse_params(
        params=jnp.array(params["value"].to_numpy()),
        parsing_info=parsing_info,
        dimensions=model["dimensions"],
        labels=model["labels"],
        n_obs=1,
    )

    n_latent = model["dimensions"]["n_latent_factors"]

    scaling_factors = np.array(pardict["anchoring_scaling_factors"][:, :n_latent])
    constants = np.array(pardict["anchoring_constants"][:, :n_latent])

    period_arr = states_df["period"].to_numpy()
    scaling_arr = scaling_factors[period_arr]
    constants_arr = constants[period_arr]

    out = states_df.copy(deep=True)
    for pos, factor in enumerate(model["labels"]["latent_factors"]):
        out[factor] = constants_arr[:, pos] + states_df[factor] * scaling_arr[:, pos]

    out = out[states_df.columns]

    return out
