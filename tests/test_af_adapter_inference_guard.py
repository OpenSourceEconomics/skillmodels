"""AF inference and posterior states reject calendar-adapter models for now.

`compute_af_standard_errors` and `get_af_posterior_states` still reconstruct the
single-period measurement layout, which is wrong for the source/destination calendar
adapter (source-investment params live in the next step's result and the per-step parser
counts differ). Until they consume the compiled `AFStepLayout`, they must raise on an
adapter model rather than return standard errors that differentiate the wrong objective
or posterior rows attached to the wrong latent.
"""

import pandas as pd
import pytest

from skillmodels.af.inference import compute_af_standard_errors
from skillmodels.af.posterior_states import get_af_posterior_states
from skillmodels.af.step_layout import (
    fail_if_calendar_adapter_unsupported,
    model_uses_calendar_adapter,
)
from skillmodels.af.types import AFEstimationResult
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations


def _plain_model() -> ModelSpec:
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


def _adapter_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("s1",), ("s1",)),
                normalizations=Normalizations(
                    loadings=({"s1": 1}, {"s1": 1}), intercepts=({"s1": 0}, {"s1": 0})
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=(("i1",), ("i1",)),
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}), intercepts=({"i1": 0}, {"i1": 0})
                ),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=False,
            ),
        },
    )


def test_model_uses_calendar_adapter_detects_reconstructed_endogenous() -> None:
    assert model_uses_calendar_adapter(_adapter_model()) is True
    assert model_uses_calendar_adapter(_plain_model()) is False


def test_raiser_is_noop_for_plain_model() -> None:
    fail_if_calendar_adapter_unsupported(_plain_model(), "feature")


def test_raiser_rejects_adapter_model() -> None:
    with pytest.raises(NotImplementedError, match="calendar adapter"):
        fail_if_calendar_adapter_unsupported(_adapter_model(), "feature")


def _adapter_result() -> AFEstimationResult:
    return AFEstimationResult(
        period_results=(),
        params=pd.DataFrame(),
        model_spec=_adapter_model(),
        conditional_distributions=(),
        success=True,
        loglikelihood=0.0,
    )


def test_standard_errors_reject_adapter_model() -> None:
    with pytest.raises(NotImplementedError, match="calendar adapter"):
        compute_af_standard_errors(_adapter_result(), pd.DataFrame())


def test_posterior_states_reject_adapter_model() -> None:
    with pytest.raises(NotImplementedError, match="calendar adapter"):
        get_af_posterior_states(_adapter_result(), _adapter_model(), pd.DataFrame())
