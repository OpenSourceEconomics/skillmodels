"""Observation events of limited (probit / Tobit) measurements.

The event of an observation -- missing, a density value, or a lower / upper
half-line -- is decided from the submitted float64 value and the declared float64
bounds, before any narrowing of the data. A float32 cast followed by a bound
comparison would turn an observation at a bound into an interior density value,
and an interior value next to a bound into a censoring mass.
"""

import math

import numpy as np
import pandas as pd
import pytest

from skillmodels.common.measurement_models import (
    MeasurementFamily,
    ObservationEvent,
    ProbitMeasurement,
    TobitMeasurement,
    classify_observations,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.process_data import process_data
from skillmodels.common.process_model import process_model

_INF = math.inf
E = ObservationEvent


def _classify(values, family, lower=-_INF, upper=_INF):
    return classify_observations(
        np.asarray(values, dtype=np.float64), family, lower, upper
    ).tolist()


def test_probit_zero_and_one_are_lower_and_upper_half_lines() -> None:
    got = _classify([0.0, 1.0, np.nan], MeasurementFamily.PROBIT)
    assert got == [E.LOWER_TAIL, E.UPPER_TAIL, E.MISSING]


@pytest.mark.parametrize("value", [0.5, -1.0, 2.0, _INF, -_INF])
def test_probit_rejects_values_outside_zero_one(value) -> None:
    with pytest.raises(ValueError, match="probit"):
        _classify([0.0, value], MeasurementFamily.PROBIT)


def test_gaussian_values_are_densities_and_nan_is_missing() -> None:
    got = _classify([0.3, np.nan, -2.0], MeasurementFamily.GAUSSIAN)
    assert got == [E.DENSITY, E.MISSING, E.DENSITY]


@pytest.mark.parametrize(
    ("value", "lower", "upper", "expected"),
    [
        (0.1, 0.1, _INF, E.LOWER_TAIL),
        (0.7, -_INF, 0.7, E.UPPER_TAIL),
        (math.nextafter(0.25, _INF), 0.25, 1.0, E.DENSITY),
        (math.nextafter(0.75, -_INF), 0.0, 0.75, E.DENSITY),
        (math.nextafter(0.1, -_INF), -_INF, 0.1, E.DENSITY),
        (0.5, 0.0, 1.0, E.DENSITY),
        (np.nan, 0.0, 1.0, E.MISSING),
    ],
)
def test_tobit_events_use_exact_float64_bounds(value, lower, upper, expected) -> None:
    assert _classify([value], MeasurementFamily.TOBIT, lower, upper) == [expected]


def test_tobit_follows_the_shared_kernel_outside_the_bounds() -> None:
    """`measurement_loglik` treats `y <= lower` / `y >= upper` as censored."""
    got = _classify([-1.0, 2.0], MeasurementFamily.TOBIT, 0.0, 1.0)
    assert got == [E.LOWER_TAIL, E.UPPER_TAIL]


def test_tobit_rejects_infinite_observations() -> None:
    with pytest.raises(ValueError, match="finite"):
        _classify([_INF], MeasurementFamily.TOBIT, 0.0, _INF)


def test_tobit_events_are_invariant_to_scaling_by_powers_of_two() -> None:
    values = np.array([0.1, math.nextafter(0.1, _INF), 0.7, math.nextafter(0.7, -_INF)])
    base = _classify(values, MeasurementFamily.TOBIT, 0.1, 0.7)
    for k in (-3, 5, 40):
        scale = 2.0**k
        scaled = _classify(
            values * scale, MeasurementFamily.TOBIT, 0.1 * scale, 0.7 * scale
        )
        assert scaled == base


def _model(measurement_models) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 2, intercepts=({"y1": 0},) * 2
                ),
                transition_function="linear",
            ),
        },
        measurement_models=measurement_models,
    )


def _data(y2, y3) -> pd.DataFrame:
    n = len(y2)
    rng = np.random.default_rng(0)
    rows = [
        {
            "caseid": i,
            "period": p,
            "y1": rng.normal(),
            "y2": y2[i],
            "y3": y3[i],
        }
        for i in range(n)
        for p in (0, 1)
    ]
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def _process(model, data):
    pm = process_model(model)
    return process_data(
        df=data,
        has_endogenous_factors=False,
        labels=pm.labels,
        update_info=pm.update_info,
        anchoring_info=pm.anchoring,
        measurement_models=model.measurement_models,
    )


def test_process_data_carries_events_decided_before_narrowing() -> None:
    near_lower = math.nextafter(0.25, _INF)
    y3 = [0.1, 0.7, near_lower, 0.5]
    model = _model({"y2": ProbitMeasurement(), "y3": TobitMeasurement(0.1, 0.7)})
    out = _process(model, _data([0.0, 1.0, np.nan, 1.0], y3))
    events = np.asarray(out["events"])
    # update rows: (0,y1), (0,y2), (0,y3), (1,y1), ...
    assert events.shape == (6, 4)
    assert events[1].tolist() == [E.LOWER_TAIL, E.UPPER_TAIL, E.MISSING, E.UPPER_TAIL]
    assert events[2].tolist() == [E.LOWER_TAIL, E.UPPER_TAIL, E.DENSITY, E.DENSITY]
    assert set(events[0].tolist()) == {E.DENSITY}
    # The limited route keeps the observed values in float64.
    assert np.asarray(out["measurements"]).dtype == np.float64
    assert float(out["measurements"][2, 2]) == near_lower


def test_process_data_without_limited_measurements_is_unchanged() -> None:
    model = _model(None)
    out = _process(model, _data([0.3, 1.0, 0.2, 1.0], [0.1, 0.7, 0.3, 0.5]))
    assert "events" not in out
    assert np.asarray(out["measurements"]).dtype == np.float32


def test_constant_limited_samples_are_valid_data() -> None:
    """All-one probit and fully censored Tobit samples are not 'no variance'."""
    model = _model({"y2": ProbitMeasurement(), "y3": TobitMeasurement(0.0, None)})
    out = _process(model, _data([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))
    assert set(np.asarray(out["events"])[1].tolist()) == {E.UPPER_TAIL}
    assert set(np.asarray(out["events"])[2].tolist()) == {E.LOWER_TAIL}


def test_process_data_rejects_invalid_probit_values() -> None:
    model = _model({"y2": ProbitMeasurement()})
    with pytest.raises(ValueError, match="probit"):
        _process(model, _data([0.0, 1.0, 0.5], [0.1, 0.2, 0.3]))
