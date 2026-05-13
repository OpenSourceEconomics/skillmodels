"""Functions to compute and process filtered latent states."""

from typing import TYPE_CHECKING, Any

import pandas as pd

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.anchoring import anchor_states_df
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.state_ranges import create_state_ranges

if TYPE_CHECKING:
    from skillmodels.af.types import AFEstimationResult
    from skillmodels.amn.types import AMNEstimationResult


def get_filtered_states(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params: pd.DataFrame,
    af_result: AFEstimationResult | None = None,
    amn_result: AMNEstimationResult | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute latent state estimates given data and estimated parameters.

    For CHS (Kalman filter) estimation, computes filtered states via the
    debug likelihood. For AF estimation, computes posterior means via
    Halton quadrature. For AMN estimation, computes mixture-Schur
    conditional posteriors of the latent factors given the augmented
    measure vector.

    Args:
        model_spec: Model specification.
        data: Dataset in long format with MultiIndex (id, period).
        params: Estimated parameter DataFrame.
        af_result: If provided, use AF posterior computation instead of
            CHS Kalman filtering.
        amn_result: If provided, use AMN mixture-Schur posteriors
            instead. Only one of `af_result` and `amn_result` may be
            set.

    Return:
        Dict with "unanchored_states" (always present) and
        "anchored_states" (CHS only), each containing "states"
        DataFrame and "state_ranges".

    """
    if af_result is not None and amn_result is not None:
        msg = "Pass only one of af_result / amn_result."
        raise ValueError(msg)

    if af_result is not None:
        from skillmodels.af.posterior_states import (  # noqa: PLC0415
            get_af_posterior_states,
        )

        return get_af_posterior_states(
            af_result=af_result,
            model_spec=model_spec,
            data=data,
        )

    if amn_result is not None:
        from skillmodels.amn.posterior_states import (  # noqa: PLC0415
            get_amn_posterior_states,
        )

        return get_amn_posterior_states(
            amn_result=amn_result,
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
