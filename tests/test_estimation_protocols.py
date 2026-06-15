"""Conformance tests for the cross-estimator structural Protocols.

Guards that the three concrete result / options dataclasses keep
satisfying `CommonEstimationResult` / `CommonEstimationOptions` (so
harmonised post-estimation code can rely on the shared surface) and that
the exactly-one-objective invariant holds for the likelihood estimators.
"""

import dataclasses

import pandas as pd
import pytest

from skillmodels.af.types import AFEstimationOptions, AFEstimationResult
from skillmodels.amn.types import AMNEstimationOptions, AMNEstimationResult
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.chs.types import CHSEstimationResult
from skillmodels.common.estimation import (
    CommonEstimationOptions,
    CommonEstimationResult,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations

RESULT_CLASSES = [AFEstimationResult, AMNEstimationResult, CHSEstimationResult]
OPTION_CLASSES = [AFEstimationOptions, AMNEstimationOptions, CHSEstimationOptions]

REQUIRED_RESULT_FIELDS = frozenset(
    {"model_spec", "params", "success", "loglikelihood", "md_criterion"}
)
REQUIRED_OPTION_FIELDS = frozenset({"optimizer_algorithm", "optimizer_options"})


def _tiny_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),),
                normalizations=Normalizations(
                    loadings=({"y1": 1},),
                    intercepts=({},),
                ),
                transition_function="linear",
            ),
        },
    )


@pytest.mark.parametrize("cls", RESULT_CLASSES)
def test_result_class_declares_common_result_fields(cls):
    names = {f.name for f in dataclasses.fields(cls)}
    assert REQUIRED_RESULT_FIELDS <= names


@pytest.mark.parametrize("cls", OPTION_CLASSES)
def test_options_class_declares_common_option_fields(cls):
    names = {f.name for f in dataclasses.fields(cls)}
    assert REQUIRED_OPTION_FIELDS <= names


@pytest.mark.parametrize("options", [AFEstimationOptions(), AMNEstimationOptions(), CHSEstimationOptions()])
def test_options_instances_satisfy_common_options_protocol(options):
    assert isinstance(options, CommonEstimationOptions)


def _make_chs_result() -> CHSEstimationResult:
    params = pd.DataFrame({"value": [0.5]})
    return CHSEstimationResult(
        model_spec=_tiny_model(),
        params=params,
        success=True,
        loglikelihood=-123.0,
        optimize_result=None,
    )


def _make_af_result() -> AFEstimationResult:
    params = pd.DataFrame({"value": [0.5]})
    return AFEstimationResult(
        period_results=(),
        params=params,
        model_spec=_tiny_model(),
        conditional_distributions=(),
        success=True,
        loglikelihood=-456.0,
    )


def test_chs_and_af_results_satisfy_common_result_protocol():
    for result in (_make_chs_result(), _make_af_result()):
        assert isinstance(result, CommonEstimationResult)


def test_exactly_one_objective_is_non_none_for_likelihood_results():
    for result in (_make_chs_result(), _make_af_result()):
        assert result.loglikelihood is not None
        assert result.md_criterion is None
