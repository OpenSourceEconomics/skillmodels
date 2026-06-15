"""Cross-estimator extraction of per-individual latent states.

Replaces the old `get_filtered_states` (the "filtered" name was a Kalman
term that did not fit AF/AMN). Two entry points:

* `get_individual_states(data, result)` dispatches on the estimation
  result type and is the harmonised, recommended call.
* `get_individual_states_from_params(model_spec, data, params)` is the
  CHS-only Kalman-from-raw-parameters escape hatch (AF/AMN need their
  result machinery, so there is no raw-params path for them).
"""

from typing import Any

import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import ESTIMATION_CONF

# Runtime imports (not `TYPE_CHECKING`-guarded) so the beartype perimeter
# can resolve the union annotation without a forward-ref string. The
# `types` modules are negligible-cost; AMN's pulls sklearn lazily.
from skillmodels.af.types import AFEstimationResult
from skillmodels.amn.types import AMNEstimationResult
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.anchoring import anchor_states_df
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.state_ranges import create_state_ranges


@beartype(conf=ESTIMATION_CONF)
def get_individual_states(
    data: pd.DataFrame,
    result: CHSEstimationResult | AFEstimationResult | AMNEstimationResult,
) -> dict[str, dict[str, Any]]:
    """Compute per-individual latent state estimates from a fitted result.

    Dispatches on the concrete result type; the `ModelSpec` and parameters
    are read off `result`. For CHS this filters via the Kalman debug
    likelihood; for AF it computes posterior means via Halton quadrature;
    for AMN it computes mixture-Schur conditional posteriors of the latent
    factors given the augmented measure vector.

    Args:
        data: Dataset in long format with MultiIndex (id, period).
        result: A `CHSEstimationResult`, `AFEstimationResult`, or
            `AMNEstimationResult`.

    Return:
        Nested dict. The return shape is **asymmetric** across estimators:

        * CHS returns both `"anchored_states"` and `"unanchored_states"`.
        * AF and AMN return `"unanchored_states"` only — neither anchors.

        Each present key maps to `{"states": DataFrame, "state_ranges": ...}`.

    """
    if isinstance(result, AFEstimationResult):
        from skillmodels.af.posterior_states import (  # noqa: PLC0415
            get_af_posterior_states,
        )

        return get_af_posterior_states(
            af_result=result,
            model_spec=result.model_spec,
            data=data,
        )

    if isinstance(result, AMNEstimationResult):
        from skillmodels.amn.posterior_states import (  # noqa: PLC0415
            get_amn_posterior_states,
        )

        return get_amn_posterior_states(
            amn_result=result,
            data=data,
        )

    return get_individual_states_from_params(
        model_spec=result.model_spec,
        data=data,
        params=result.params,
    )


@beartype(conf=ESTIMATION_CONF)
def get_individual_states_from_params(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    params: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute CHS filtered states directly from raw parameters.

    The escape hatch behind `get_individual_states`'s CHS branch: it runs
    the Kalman debug likelihood at `params` without needing a result
    object. Inherently CHS-only — AF and AMN reconstruct states from their
    result machinery, not from a flat parameter vector.

    Args:
        model_spec: Model specification.
        data: Dataset in long format with MultiIndex (id, period).
        params: Estimated parameter DataFrame (standard 4-level MultiIndex).

    Return:
        Dict with both `"anchored_states"` and `"unanchored_states"`, each
        `{"states": DataFrame, "state_ranges": ...}`.

    """
    max_inputs = get_maximization_inputs(model_spec=model_spec, data=data)
    params = params.loc[max_inputs["params_template"].index]
    debug_loglike = max_inputs["debug_loglike"]
    debug_data = debug_loglike(params)
    unanchored_states_df = debug_data["filtered_states"]
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
