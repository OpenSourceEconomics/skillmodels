"""One-call driver for the CHS Kalman-MLE estimator.

`estimate_chs` is a thin wrapper over `get_maximization_inputs` +
`optimagic.maximize`, giving CHS the same `estimate_*(model_spec, data,
options, ...) -> ...EstimationResult` surface as `estimate_af` and
`estimate_amn`. `get_maximization_inputs` stays public as the power-user
escape hatch for callers who want to drive the optimiser themselves.
"""

import optimagic as om
import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import ESTIMATION_CONF
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.types import to_plain_dict


@beartype(conf=ESTIMATION_CONF)
def estimate_chs(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    options: CHSEstimationOptions | None = None,
    start_params: pd.DataFrame | None = None,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> CHSEstimationResult:
    """Estimate a latent factor model by Cunha-Heckman-Schennach Kalman MLE.

    Args:
        model_spec: Model specification (same object the other estimators
            consume).
        data: Dataset in long format with MultiIndex (id, period).
        options: CHS-specific tuning parameters, including the
            `optimizer_algorithm` / `optimizer_options` driving the
            `optimagic.maximize` call and the `start_params_strategy`
            seeding the `params_template`. Defaults to
            `CHSEstimationOptions()`.
        start_params: Optional starting values. Entries whose index matches
            the `params_template` overwrite the seeded `value` column;
            unmatched template entries keep their seeded values. Uses the
            standard 4-level MultiIndex (category, period, name1, name2).
        fixed_params: Optional DataFrame with a `"value"` column pinning
            parameters; forwarded to `get_maximization_inputs`, which turns
            each into a `FixedConstraintWithValue` so `optimagic` holds it
            fixed.
        constraints: Optional extra `optimagic` constraints, appended to the
            model-implied constraints from `get_maximization_inputs`.

    Return:
        `CHSEstimationResult` with the estimated `params`, the `success`
        flag, the maximised `loglikelihood`, and the raw optimagic
        `optimize_result`.

    """
    options = options or CHSEstimationOptions()

    max_inputs = get_maximization_inputs(
        model_spec=model_spec,
        data=data,
        chs_options=options,
        fixed_params=fixed_params,
    )

    start = max_inputs["params_template"].copy()
    if start_params is not None:
        overlay = start_params["value"].reindex(start.index)
        start.loc[overlay.notna(), "value"] = overlay[overlay.notna()]

    all_constraints = [*max_inputs["constraints"], *(constraints or [])]

    res = om.maximize(
        fun=max_inputs["loglike"],
        params=start[["value"]],
        algorithm=options.optimizer_algorithm,
        bounds=om.Bounds(lower=start["lower_bound"], upper=start["upper_bound"]),
        constraints=all_constraints,
        fun_and_jac=max_inputs["loglike_and_gradient"],
        **to_plain_dict(options.optimizer_options),
    )

    loglikelihood = float(max_inputs["loglike"](res.params))

    return CHSEstimationResult(
        model_spec=model_spec,
        params=res.params,
        success=bool(res.success),
        loglikelihood=loglikelihood,
        optimize_result=res,
    )
