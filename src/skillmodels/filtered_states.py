"""Functions to compute and process filtered latent states."""

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from skillmodels.maximization_inputs import get_maximization_inputs
from skillmodels.model_spec import ModelSpec
from skillmodels.params_index import get_params_index
from skillmodels.parse_params import create_parsing_info, parse_params
from skillmodels.process_debug_data import create_state_ranges
from skillmodels.process_model import process_model

if TYPE_CHECKING:
    from skillmodels.af.types import AFEstimationResult


def get_filtered_states(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params: pd.DataFrame,
    af_result: AFEstimationResult | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute latent state estimates given data and estimated parameters.

    For CHS (Kalman filter) estimation, computes filtered states via the
    debug likelihood. For AF estimation, computes posterior means via
    Halton quadrature.

    Args:
        model_spec: Model specification.
        data: Dataset in long format with MultiIndex (id, period).
        params: Estimated parameter DataFrame.
        af_result: If provided, use AF posterior computation instead of
            CHS Kalman filtering. Should be an `AFEstimationResult`.

    Return:
        Dict with "unanchored_states" (always present) and
        "anchored_states" (CHS only), each containing "states"
        DataFrame and "state_ranges".

    """
    if af_result is not None:
        from skillmodels.af.posterior_states import (  # noqa: PLC0415
            get_af_posterior_states,
        )

        return get_af_posterior_states(
            af_result=af_result,
            model_spec=model_spec,
            data=data,
        )

    max_inputs = get_maximization_inputs(model_spec=model_spec, data=data)
    params = params.loc[max_inputs["params_template"].index]
    debug_loglike = max_inputs["debug_loglike"]
    debug_data = debug_loglike(params)
    unanchored_states_df = debug_data["filtered_states"]
    unanchored_ranges = debug_data["state_ranges"]
    processed_model = process_model(model_spec)

    anchored_states_df = anchor_states_df(
        states_df=unanchored_states_df,
        model_spec=model_spec,
        params=params,
        use_aug_period=True,
    )

    # Map aug_period → period for the public API
    ap_to_p = processed_model.labels.aug_periods_to_periods
    for df in (anchored_states_df, unanchored_states_df):
        df["period"] = df["aug_period"].map(ap_to_p)
    anchored_states_df = anchored_states_df.drop(columns="aug_period")
    unanchored_states_df = unanchored_states_df.drop(columns="aug_period")

    anchored_ranges = create_state_ranges(
        filtered_states=anchored_states_df,
        factors=processed_model.labels.latent_factors,
    )
    unanchored_ranges = create_state_ranges(
        filtered_states=unanchored_states_df,
        factors=processed_model.labels.latent_factors,
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
    model_spec: ModelSpec,
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
