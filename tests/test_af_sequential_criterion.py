"""The AF result exposes its objective under an honest name.

The per-period AF objective is a per-observation *mean* log-likelihood criterion, and
the aggregate is their sum -- a sequential/composite criterion, not a joint sample
log-likelihood. `AFEstimationResult` exposes that value as `sequential_criterion` with a
per-period breakdown in `period_mean_criteria`, while `loglikelihood` is retained as an
equal-valued alias for protocol conformance and back-compat.
"""

import pandas as pd

from skillmodels.af.types import AFEstimationResult, AFPeriodResult
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations


def _model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("s1",), ("s1",)),
                normalizations=Normalizations(
                    loadings=({"s1": 1}, {"s1": 1}), intercepts=({"s1": 0}, {"s1": 0})
                ),
                transition_function="linear",
            ),
        },
    )


def test_sequential_criterion_aggregates_period_means() -> None:
    period_results = (
        AFPeriodResult(
            period=0,
            params=pd.DataFrame(),
            loglikelihood=-1.5,
            success=True,
            optimize_result=None,
        ),
        AFPeriodResult(
            period=1,
            params=pd.DataFrame(),
            loglikelihood=-2.0,
            success=True,
            optimize_result=None,
        ),
    )
    means = tuple(pr.loglikelihood for pr in period_results)
    total = sum(means)
    result = AFEstimationResult(
        period_results=period_results,
        params=pd.DataFrame(),
        model_spec=_model(),
        conditional_distributions=(),
        success=True,
        loglikelihood=total,
        sequential_criterion=total,
        period_mean_criteria=means,
    )

    assert result.sequential_criterion == total
    assert result.loglikelihood == result.sequential_criterion
    assert result.period_mean_criteria == means
    assert sum(result.period_mean_criteria) == result.sequential_criterion
