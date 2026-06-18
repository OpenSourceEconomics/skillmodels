"""Tests for `skillmodels.amn.estimate.estimate_amn` (end-to-end orchestration)."""

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.amn import estimate_amn
from skillmodels.amn.mixture_em import InsufficientCompleteCasesError
from skillmodels.amn.types import AMNEstimationOptions
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _tiny_model() -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
        },
        n_mixtures=2,
    )


def _tiny_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        f0 = rng.normal()
        f1 = 0.6 * f0 + rng.normal(0, 0.5)
        for period, f in [(0, f0), (1, f1)]:
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": f + rng.normal(0, 0.3),
                    "y2": 0.9 * f + rng.normal(0, 0.4),
                    "y3": 1.1 * f + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def test_estimate_amn_produces_combined_params_dataframe():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    result = estimate_amn(model, data, options)

    assert result.params.index.names == [
        "category",
        "aug_period",
        "name1",
        "name2",
    ]
    cats = set(result.params.index.get_level_values("category"))
    assert {"loadings", "meas_sds", "transition", "shock_sds"} <= cats
    # 6 measurement loadings, 6 meas_sds, 1 transition (slope on skills) +
    # constant for period 0, 1 shock_sds for period 0.
    assert "controls" in cats  # measurement intercepts collapse to controls


def _subsample_model() -> ModelSpec:
    """Model whose `skills` factor has one rotating-subsample measurement.

    Measured by full-sample y1 (normalization) and y2, plus y_sub, which is
    missing for almost every individual.
    """
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y_sub"), ("y1", "y2", "y_sub")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="linear",
            ),
        },
        n_mixtures=2,
    )


def _subsample_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        f0 = rng.normal()
        f1 = 0.6 * f0 + rng.normal(0, 0.5)
        for period, f in [(0, f0), (1, f1)]:
            # y_sub observed only for the very first individual: the full
            # complete-case count (1) is below the 2 mixture components, exactly
            # the rotating-subsample regime that makes complete-case EM infeasible.
            y_sub = f + rng.normal(0, 0.3) if caseid == 0 else np.nan
            rows.append(
                {
                    "caseid": caseid,
                    "period": period,
                    "y1": f + rng.normal(0, 0.3),
                    "y2": 0.9 * f + rng.normal(0, 0.4),
                    "y_sub": y_sub,
                }
            )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def test_estimate_amn_complete_case_raises_on_subsample_measurement():
    """Default complete-case Stage 1 raises an informative error on subsamples.

    With a rotating-subsample measurement the full augmented vector has too few
    complete cases (1 < 2 components) to fit the mixture. The default
    complete-case method must raise `InsufficientCompleteCasesError` pointing at
    the missing-data method, rather than silently dropping the measurement or
    switching methods.
    """
    model = _subsample_model()
    data = _subsample_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    with pytest.raises(InsufficientCompleteCasesError, match="missing_data"):
        estimate_amn(model, data, options)


def test_estimate_amn_missing_data_includes_subsample_measurement():
    """The missing-data method seeds on the full set, keeping the subsample.

    Marginalising over missing entries needs no complete cases, so the subsample
    measurement `y_sub` is retained in the recovered loadings rather than
    dropped. Same interface as the default -- only `mixture_em_method` differs.
    """
    model = _subsample_model()
    data = _subsample_data(n=1500)
    options = AMNEstimationOptions(
        n_simulation_draws=5000, seed=0, mixture_em_method="missing_data"
    )

    result = estimate_amn(model, data, options)

    meas = result.params.xs("loadings", level="category").index.get_level_values(
        "name1"
    )
    assert "y_sub" in set(meas)  # subsample measurement retained, not dropped
    assert "y2" in set(meas)


def _split_panel_data(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Unbalanced panel: each individual is observed in exactly one period.

    No individual spans both periods, so the augmented vector has zero
    complete-case rows -- the regime that forces the missing-data EM.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for caseid in range(n):
        f = rng.normal()
        period = caseid % 2
        rows.append(
            {
                "caseid": caseid,
                "period": period,
                "y1": f + rng.normal(0, 0.3),
                "y2": 0.9 * f + rng.normal(0, 0.4),
                "y3": 1.1 * f + rng.normal(0, 0.5),
            }
        )
    return pd.DataFrame(rows).set_index(["caseid", "period"])


def test_estimate_amn_complete_case_raises_on_unbalanced_panel():
    """With zero complete-case rows the default method raises, no silent switch."""
    model = _tiny_model()
    data = _split_panel_data(n=1000)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    with pytest.raises(InsufficientCompleteCasesError, match="missing_data"):
        estimate_amn(model, data, options)


def test_estimate_amn_missing_data_fits_unbalanced_panel():
    """The missing-data method fits the mixture even with no complete rows.

    Each individual is observed in exactly one period, so columns from the two
    periods are never co-observed: the EM still fits (and recovers the means)
    but warns that the cross-period covariances are unidentified.
    """
    model = _tiny_model()
    data = _split_panel_data(n=1000)
    options = AMNEstimationOptions(
        n_simulation_draws=2000, seed=0, mixture_em_method="missing_data"
    )

    with pytest.warns(RuntimeWarning, match="co-observation"):
        result = estimate_amn(model, data, options)

    assert isinstance(result.success, bool)
    assert result.stages.mixture.means.shape[0] == 2


@pytest.mark.parametrize("override", ["start_params", "fixed_params"])
def test_estimate_amn_rejects_param_overrides(override):
    """estimate_amn must refuse start/fixed params it cannot honour in-stage.

    The three-stage estimator has no single free optimisation to pin, so
    overlaying values after the fact would make the reported params inconsistent
    with the fitted stages and criterion. It raises instead of silently
    overwriting estimates.
    """
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    pin = pd.DataFrame(
        {"value": [0.42]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 1, "y2", "skills")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    callers = {
        "start_params": lambda: estimate_amn(model, data, options, start_params=pin),
        "fixed_params": lambda: estimate_amn(model, data, options, fixed_params=pin),
    }

    with pytest.raises(NotImplementedError):
        callers[override]()


def test_estimate_amn_rejects_constraints():
    """A non-empty constraints list is refused; the AMN stages cannot honour it."""
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    constraint = om.EqualityConstraint(
        selector=lambda params: params.loc[[("loadings", 1, "y2", "skills")]]
    )

    with pytest.raises(NotImplementedError):
        estimate_amn(model, data, options, constraints=[constraint])


def test_estimate_amn_returns_success_flag():
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=1)

    result = estimate_amn(model, data, options)

    assert isinstance(result.success, bool)
    assert result.stages.mixture.weights.shape == (2,)
    assert result.stages.structural.factor_period_slots == (
        (0, "skills"),
        (1, "skills"),
    )
