"""Tests for the public measurement-family interface on `ModelSpec`.

A measurement variable's observation model (Gaussian / probit / Tobit) is a
property of the variable, not of the factor it loads on, so it is configured via
`ModelSpec.measurement_models` (a `name -> MeasurementModel` mapping). A variable
omitted from the mapping is Gaussian, so existing specs are unchanged. The marker
classes resolve to the internal `MeasurementFamily` code + censoring bounds that the
shared likelihood kernel consumes.
"""

import math

import numpy as np
import pytest

from skillmodels.common.measurement_models import (
    GaussianMeasurement,
    MeasurementFamily,
    ProbitMeasurement,
    TobitMeasurement,
    measurement_family_arrays,
    resolve_measurement_family,
)
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _model(measurement_models=None) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("test_score", "passed_grade"),) * 2,
                normalizations=Normalizations(
                    loadings=({"test_score": 1},) * 2,
                    intercepts=({"test_score": 0},) * 2,
                ),
                transition_function="translog",
            ),
        },
        measurement_models=measurement_models,
    )


def test_omitted_measurement_models_defaults_to_gaussian() -> None:
    model = _model()
    assert model.measurement_model("test_score") == GaussianMeasurement()
    assert model.measurement_model("passed_grade") == GaussianMeasurement()


def test_listed_measurement_models_are_returned() -> None:
    model = _model({"passed_grade": ProbitMeasurement()})
    assert model.measurement_model("passed_grade") == ProbitMeasurement()
    # An unlisted measure is still Gaussian.
    assert model.measurement_model("test_score") == GaussianMeasurement()


def test_measurement_models_mapping_is_immutable() -> None:
    model = _model({"passed_grade": ProbitMeasurement()})
    with pytest.raises(TypeError):
        # Intentional illegal mutation: the mapping must be read-only.
        model.measurement_models["test_score"] = ProbitMeasurement()  # ty: ignore[invalid-assignment]


def test_unknown_measurement_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a measurement"):
        _model({"not_a_real_measure": ProbitMeasurement()})


def test_resolve_gaussian_family() -> None:
    family, lower, upper = resolve_measurement_family(GaussianMeasurement())
    assert family is MeasurementFamily.GAUSSIAN
    assert lower == -math.inf
    assert upper == math.inf


def test_resolve_probit_family() -> None:
    family, lower, upper = resolve_measurement_family(ProbitMeasurement())
    assert family is MeasurementFamily.PROBIT
    assert lower == -math.inf
    assert upper == math.inf


def test_resolve_one_sided_tobit_family() -> None:
    family, lower, upper = resolve_measurement_family(TobitMeasurement(lower=0.0))
    assert family is MeasurementFamily.TOBIT
    assert lower == 0.0
    assert upper == math.inf


def test_resolve_double_censored_tobit_family() -> None:
    family, lower, upper = resolve_measurement_family(
        TobitMeasurement(lower=0.0, upper=10.0)
    )
    assert family is MeasurementFamily.TOBIT
    assert (lower, upper) == (0.0, 10.0)


def test_tobit_rejects_no_bounds() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TobitMeasurement(lower=None, upper=None)


def test_tobit_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match=r"lower.*upper"):
        TobitMeasurement(lower=5.0, upper=1.0)


def test_from_dict_parses_measurement_models() -> None:
    model = ModelSpec.from_dict(
        {
            "factors": {
                "skills": {
                    "measurements": [
                        ["test_score", "passed_grade"],
                        ["test_score", "passed_grade"],
                    ],
                    "normalizations": {
                        "loadings": [{"test_score": 1}, {"test_score": 1}],
                        "intercepts": [{"test_score": 0}, {"test_score": 0}],
                    },
                    "transition_function": "translog",
                },
            },
            "measurement_models": {
                "passed_grade": {"family": "probit"},
                "test_score": {"family": "tobit", "lower": 0.0},
            },
        }
    )
    assert model.measurement_model("passed_grade") == ProbitMeasurement()
    assert model.measurement_model("test_score") == TobitMeasurement(lower=0.0)


def test_from_dict_without_measurement_models_is_all_gaussian() -> None:
    model = ModelSpec.from_dict(
        {
            "factors": {
                "skills": {
                    "measurements": [["test_score"], ["test_score"]],
                    "normalizations": {
                        "loadings": [{"test_score": 1}, {"test_score": 1}],
                        "intercepts": [{"test_score": 0}, {"test_score": 0}],
                    },
                    "transition_function": "translog",
                },
            },
        }
    )
    assert model.measurement_model("test_score") == GaussianMeasurement()


def test_measurement_family_arrays_aligns_and_defaults_gaussian() -> None:
    models = {
        "passed_grade": ProbitMeasurement(),
        "amount": TobitMeasurement(lower=0.0, upper=5.0),
    }
    names = ["test_score", "passed_grade", "amount"]
    codes, lowers, uppers = measurement_family_arrays(models, names)
    np.testing.assert_array_equal(
        codes,
        [
            int(MeasurementFamily.GAUSSIAN),
            int(MeasurementFamily.PROBIT),
            int(MeasurementFamily.TOBIT),
        ],
    )
    assert lowers.tolist() == [-math.inf, -math.inf, 0.0]
    assert uppers.tolist() == [math.inf, math.inf, 5.0]
