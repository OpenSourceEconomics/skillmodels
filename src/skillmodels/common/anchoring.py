"""Generic anchoring utilities, estimator-agnostic.

Anchoring maps unitless latent factors back to the unit of a designated
anchor measurement via a per-period (scale, offset) pair recovered from
`ModelSpec.anchoring`. The application is a pure scalar transformation
of a DataFrame's latent-factor columns; the implementation only depends
on `process_model` / `parse_params` (both common) and operates on a
`(obs x period x factor)` DataFrame irrespective of which estimator
produced it.

Historically this lived under `skillmodels.chs.filtered_states` but
the cross-subpackage import from `common.simulate_data` was a code
smell — the function is genuinely common.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.params_index import get_params_index
from skillmodels.common.parse_params import create_parsing_info, parse_params
from skillmodels.common.process_model import process_model


def anchor_states_df(
    states_df: pd.DataFrame,
    model_spec: ModelSpec,
    params: pd.DataFrame,
    *,
    use_aug_period: bool,
) -> pd.DataFrame:
    """Anchor states in a DataFrame.

    The DataFrame is expected to have a column called "period" (or
    "aug_period" when `use_aug_period=True`) as well as one column for
    each latent factor. All other columns are not affected.

    This is a bit difficult because we need to re-use `parse_params`
    (which was meant as an internal function that only works with jax
    objects).
    """
    processed_model = process_model(model_spec)

    p_index = get_params_index(
        update_info=processed_model.update_info,
        labels=processed_model.labels,
        dimensions=processed_model.dimensions,
        transition_info=processed_model.transition_info,
        endogenous_factors_info=processed_model.endogenous_factors_info,
    )

    params = params.loc[p_index]

    parsing_info = create_parsing_info(
        params_index=p_index,
        update_info=processed_model.update_info,
        labels=processed_model.labels,
        anchoring=processed_model.anchoring,
        has_endogenous_factors=processed_model.endogenous_factors_info.has_endogenous_factors,
    )

    *_, parsed_params = parse_params(
        params=jnp.array(params["value"].to_numpy()),
        parsing_info=parsing_info,
        dimensions=processed_model.dimensions,
        labels=processed_model.labels,
        n_obs=1,
    )

    n_latent = processed_model.dimensions.n_latent_factors

    _scaling_factors = np.array(parsed_params.anchoring_scaling_factors[:, :n_latent])
    _constants = np.array(parsed_params.anchoring_constants[:, :n_latent])
    if use_aug_period:
        # _scaling_factors is already indexed by aug_period, use directly
        period_arr = states_df["aug_period"].to_numpy()
        scaling_factors = _scaling_factors
        constants = _constants
    else:
        period_arr = states_df["period"].to_numpy()
        ap_to_p = processed_model.labels.aug_periods_to_periods
        n_periods = processed_model.dimensions.n_periods
        scaling_factors = np.empty(shape=(n_periods, n_latent))
        constants = np.empty(shape=(n_periods, n_latent))
        for ap, p in ap_to_p.items():
            # For endogenous models, multiple aug_periods map to the same
            # period; constraints ensure they have identical anchoring params,
            # so the last write per period is correct.
            scaling_factors[p] = _scaling_factors[ap]
            constants[p] = _constants[ap]

    scaling_arr = scaling_factors[period_arr]
    constants_arr = constants[period_arr]

    out = states_df.copy(deep=True)
    for pos, factor in enumerate(processed_model.labels.latent_factors):
        out[factor] = constants_arr[:, pos] + states_df[factor] * scaling_arr[:, pos]

    return out[states_df.columns]
