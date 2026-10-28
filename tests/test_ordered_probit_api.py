"""Public API of ordered-probit measurements.

An ordered-probit measurement `y` with `K` categories (integer codes `0..K-1`) has a
latent index `y* = c + x'beta + lambda'theta + e`, `e ~ N(0, 1)`, and `y = k` iff
`c_k < y* <= c_{k+1}` with `c_0 = -inf`, `c_K = +inf` and `c_1 = 0`. Its parameters are
the usual intercept/controls and loadings, a `meas_sds` row fixed to 1 (as for probit),
and one `("cutpoints", aug_period, y, "cutpoint_k")` row per `k = 1..K-1` and period,
with `cutpoint_1` fixed to 0 and the cutpoints of one period increasing.
"""

import math

import numpy as np
import optimagic as om
import pandas as pd
import pytest

import skillmodels
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.constraints import FixedConstraintWithValue, select_by_loc
from skillmodels.common.measurement_models import (
    MeasurementFamily,
    ObservationEvent,
    OrderedProbitMeasurement,
    classify_observations,
)
from skillmodels.common.model_spec import (
    AnchoringSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.simulate_data import simulate_dataset

E = ObservationEvent
_K = 4


def test_ordered_probit_measurement_stores_its_number_of_categories() -> None:
    assert OrderedProbitMeasurement(n_categories=3).n_categories == 3


@pytest.mark.parametrize("n_categories", [1, 0, -2])
def test_ordered_probit_needs_at_least_two_categories(n_categories) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        OrderedProbitMeasurement(n_categories=n_categories)


@pytest.mark.parametrize("n_categories", [2.0, 2.5, True, "3"])
def test_ordered_probit_number_of_categories_must_be_an_integer(n_categories) -> None:
    with pytest.raises((TypeError, ValueError)):
        OrderedProbitMeasurement(n_categories=n_categories)


def test_ordered_probit_is_exported_with_the_other_families() -> None:
    assert skillmodels.OrderedProbitMeasurement is OrderedProbitMeasurement


def _factors_dict() -> dict:
    return {
        "skills": {
            "measurements": [["y1", "y2", "y3"]] * 2,
            "normalizations": {
                "loadings": [{"y1": 1}] * 2,
                "intercepts": [{"y1": 0}] * 2,
            },
            "transition_function": "linear",
        }
    }


def test_from_dict_builds_an_ordered_probit_measurement() -> None:
    model = ModelSpec.from_dict(
        {
            "factors": _factors_dict(),
            "measurement_models": {
                "y3": {"family": "ordered_probit", "n_categories": _K}
            },
        }
    )
    assert model.measurement_models["y3"] == OrderedProbitMeasurement(n_categories=_K)


def test_from_dict_requires_the_number_of_categories() -> None:
    with pytest.raises(ValueError, match="n_categories"):
        ModelSpec.from_dict(
            {
                "factors": _factors_dict(),
                "measurement_models": {"y3": {"family": "ordered_probit"}},
            }
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            [0.0, 1.0, 2.0, 3.0, np.nan],
            [E.LOWER_TAIL, E.INTERVAL, E.INTERVAL, E.UPPER_TAIL, E.MISSING],
        ),
    ],
)
def test_ordered_probit_events_follow_the_category(values, expected) -> None:
    got = classify_observations(
        np.array(values),
        MeasurementFamily.ORDERED_PROBIT,
        -math.inf,
        math.inf,
        n_categories=_K,
    )
    assert got.tolist() == expected


@pytest.mark.parametrize("value", [1.5, -1.0, 4.0, math.inf])
def test_ordered_probit_rejects_values_that_are_not_category_codes(value) -> None:
    with pytest.raises(ValueError, match="ordered probit"):
        classify_observations(
            np.array([0.0, value]),
            MeasurementFamily.ORDERED_PROBIT,
            -math.inf,
            math.inf,
            n_categories=_K,
        )


def test_two_category_ordered_probit_events_equal_probit_events() -> None:
    values = np.array([0.0, 1.0, np.nan, 1.0])
    ordered = classify_observations(
        values, MeasurementFamily.ORDERED_PROBIT, -math.inf, math.inf, n_categories=2
    )
    probit = classify_observations(
        values, MeasurementFamily.PROBIT, -math.inf, math.inf
    )
    assert ordered.tolist() == probit.tolist()


def _model(**kwargs) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 3, intercepts=({"y1": 0},) * 3
                ),
                transition_function="linear",
            ),
        },
        measurement_models={"y3": OrderedProbitMeasurement(n_categories=_K)},
        **kwargs,
    )


def _data(y3=None, n=200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        skill = rng.normal()
        for p in range(3):
            rows.append(
                {
                    "caseid": i,
                    "period": p,
                    "y1": skill + rng.normal(scale=0.5),
                    "y2": 0.5 * skill + rng.normal(scale=0.5),
                    "y3": float(np.digitize(skill + rng.normal(), [-0.8, 0.0, 0.9])),
                }
            )
    df = pd.DataFrame(rows).set_index(["caseid", "period"])
    if y3 is not None:
        df["y3"] = y3
    return df


def _inputs(strategy="none", **kwargs):
    return get_maximization_inputs(
        model_spec=_model(),
        data=_data(**kwargs),
        chs_options=CHSEstimationOptions(start_params_strategy=strategy),
    )


def _cutpoint_locs(aug_period):
    return [("cutpoints", aug_period, "y3", f"cutpoint_{k}") for k in range(1, _K)]


def test_params_index_has_one_cutpoint_row_per_threshold_and_period() -> None:
    template = _inputs()["params_template"]
    cutpoints = template.query("category == 'cutpoints'")
    expected = [loc for p in range(3) for loc in _cutpoint_locs(p)]
    assert list(cutpoints.index) == expected


def test_first_cutpoint_and_error_sd_are_fixed() -> None:
    inputs = _inputs()
    fixed = {
        c.loc: c.value
        for c in inputs["constraints"]
        if isinstance(c, FixedConstraintWithValue)
    }
    for p in range(3):
        assert fixed[("cutpoints", p, "y3", "cutpoint_1")] == 0.0
        assert fixed[("meas_sds", p, "y3", "-")] == 1.0
        assert ("cutpoints", p, "y3", "cutpoint_2") not in fixed


def test_cutpoints_of_a_period_are_constrained_to_increase() -> None:
    increasing = [
        c for c in _inputs()["constraints"] if isinstance(c, om.IncreasingConstraint)
    ]
    template = _inputs()["params_template"]
    selected = [list(c.selector(template).index) for c in increasing]
    assert selected == [_cutpoint_locs(p) for p in range(3)]


def test_seeded_start_cutpoints_start_at_zero_and_increase() -> None:
    template = _inputs(strategy="spearman")["params_template"]
    for p in range(3):
        values = template.loc[_cutpoint_locs(p), "value"].to_numpy()
        assert values[0] == 0.0
        assert np.all(np.diff(values) > 0)


def test_ordering_composes_with_cutpoints_tied_across_periods() -> None:
    """User ties of cutpoints across periods and the ordering hold together."""
    inputs = _inputs(strategy="spearman")
    start = inputs["params_template"].copy()
    # Tied parameters need equal start values.
    for p in (1, 2):
        start.loc[_cutpoint_locs(p), "value"] = start.loc[
            _cutpoint_locs(0), "value"
        ].to_numpy()
    ties = [
        om.PairwiseEqualityConstraint(
            selectors=[
                lambda params, p=p: select_by_loc(params, _cutpoint_locs(p))
                for p in range(3)
            ]
        )
    ]
    # Targets that violate the ordering and differ across periods.
    target = {p: np.array([0.4, 1.5, 0.2]) + 0.3 * p for p in range(3)}
    locs = [loc for p in range(3) for loc in _cutpoint_locs(p)]
    target_vec = np.concatenate([target[p] for p in range(3)])

    def objective(params):
        return float(np.sum((params.loc[locs, "value"].to_numpy() - target_vec) ** 2))

    result = om.minimize(
        fun=objective,
        params=start[["value"]],
        algorithm="scipy_lbfgsb",
        constraints=[*inputs["constraints"], *ties],
    )
    got = np.array(
        [result.params.loc[_cutpoint_locs(p), "value"].to_numpy() for p in range(3)]
    )
    assert np.allclose(got, got[0])
    assert got[0, 0] == 0.0
    assert np.all(np.diff(got[0]) >= 0)


def test_process_data_rejects_non_integer_ordered_codes() -> None:
    y3 = _data()["y3"].to_numpy().copy()
    y3[0] = 1.5
    with pytest.raises(ValueError, match="ordered probit"):
        _inputs(y3=y3)


def test_ordered_probit_anchoring_outcome_is_rejected() -> None:
    model = _model().with_anchoring(
        AnchoringSpec(
            outcomes={"skills": "y3"},
            free_constant=True,
            free_loadings=True,
        )
    )
    with pytest.raises(NotImplementedError, match="anchoring"):
        get_maximization_inputs(
            model_spec=model,
            data=_data(),
            chs_options=CHSEstimationOptions(start_params_strategy="none"),
        )


def test_chs_likelihood_of_ordered_probit_is_not_available_yet() -> None:
    inputs = _inputs(strategy="spearman")
    with pytest.raises(NotImplementedError, match="ordered-probit"):
        inputs["loglike"](inputs["params_template"])


def test_af_rejects_ordered_probit_until_supported() -> None:
    with pytest.raises(NotImplementedError, match="ordered-probit"):
        skillmodels.estimate_af(model_spec=_model(), data=_data())


def test_simulation_rejects_ordered_probit_until_supported() -> None:
    params = _inputs(strategy="spearman")["params_template"]
    with pytest.raises(NotImplementedError, match="ordered-probit"):
        simulate_dataset(model_spec=_model(), params=params, n_obs=10)
