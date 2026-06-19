"""One-call driver for the CHS Kalman-MLE estimator.

`estimate_chs` is a thin wrapper over `get_maximization_inputs` +
`estimagic.estimate_ml`, giving CHS the same `estimate_*(model_spec, data,
options, ...) -> ...EstimationResult` surface as `estimate_af` and
`estimate_amn`. Running through `estimate_ml` (rather than a bare
`optimagic.maximize`) means the returned result carries full ML inference —
standard errors, covariances, summaries — so callers that need inference can
adopt `estimate_chs` instead of hand-rolling `estimate_ml` on top of the
inputs. `get_maximization_inputs` stays public as the power-user escape
hatch for callers who want to drive the optimiser themselves.
"""

import optimagic as om
import pandas as pd
from beartype import beartype
from estimagic import estimate_ml

from skillmodels._beartype_conf import ESTIMATION_CONF
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.constraints import (
    enforce_fixed_constraints,
    reconcile_start_to_equality,
)
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
        flag, the maximised `loglikelihood`, the raw optimagic
        `optimize_result`, and the estimagic `likelihood_result` carrying ML
        inference (`.se()` / `.cov()` / `.summary()`).

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

    # Write every `FixedConstraintWithValue`'s target into the start vector.
    # optimagic's plain "fixed" constraint pins a parameter at its *start* value,
    # so a user constraint's `.value` only takes effect once enforced here. User
    # `constraints=` are merged only now; `get_maximization_inputs` enforced just
    # the internal (model-implied / `fixed_params`) constraints, so without this
    # a user-supplied fixed parameter would be silently held at its seed instead
    # of the requested value.
    start = enforce_fixed_constraints(start, all_constraints)

    # `estimate_ml` (via `om.minimize`) raises `InvalidParamsError` if the start
    # point violates any equality constraint. Seeding strategies (AMN/Spearman)
    # and user start_params fill each member independently, so pool each equality
    # group's seeded value onto the constraint surface -- honouring any fixed
    # member's enforced value so the line above is not averaged away.
    start = reconcile_start_to_equality(start, all_constraints)

    optimize_options = {
        "algorithm": options.optimizer_algorithm,
        "algo_options": to_plain_dict(options.optimizer_options) or None,
        "fun_and_jac": max_inputs["loglike_and_gradient"],
    }

    # Default to OPG/jacobian-based inference (`hessian=False`): the numerical
    # Hessian costs O(n_params**2) Kalman passes and is prohibitive on real
    # models. Overridable via `options.estimate_ml_options`.
    estimate_ml_kwargs = {
        "hessian": False,
        **to_plain_dict(options.estimate_ml_options),
    }

    res = estimate_ml(
        loglike=max_inputs["loglikeobs"],
        params=start[["value"]],
        optimize_options=optimize_options,
        bounds=om.Bounds(lower=start["lower_bound"], upper=start["upper_bound"]),
        constraints=all_constraints,
        **estimate_ml_kwargs,
    )

    loglikelihood = float(max_inputs["loglike"](res.params))

    return CHSEstimationResult(
        model_spec=model_spec,
        params=res.params,
        success=bool(res.optimize_result.success),
        loglikelihood=loglikelihood,
        optimize_result=res.optimize_result,
        likelihood_result=res,
    )
