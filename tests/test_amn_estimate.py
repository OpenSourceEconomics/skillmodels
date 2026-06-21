"""Tests for `skillmodels.amn.estimate.estimate_amn` (end-to-end orchestration)."""

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.amn import estimate_amn
from skillmodels.amn.estimate import _fail_if_standalone_unsupported
from skillmodels.amn.mixture_em import InsufficientCompleteCasesError
from skillmodels.amn.types import AMNEstimationOptions
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.process_model import process_model


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


def _tiny_ces_model() -> ModelSpec:
    """A model whose skill transition is restricted CES (`log_ces`).

    AMN cannot consistently estimate this standalone (no primitive-scale
    recovery), so estimate_amn must refuse it unless it is seeding estimate_chs.
    """
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="log_ces",
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
    """estimate_amn refuses start_params wholesale and fixed_params it cannot honour.

    `start_params` has no single free optimisation to warm-start, so any pin is
    refused. `fixed_params` is honoured for the categories owned by the stage
    that fits them (transition, loadings, controls, meas_sds); a pin on a
    category AMN cannot hold in-stage (here a derived `shock_sds` residual SD)
    still raises rather than silently overwriting the estimate.
    """
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    pin = pd.DataFrame(
        {"value": [0.42]},
        index=pd.MultiIndex.from_tuples(
            [("shock_sds", 0, "skills", "-")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    callers = {
        "start_params": lambda: estimate_amn(model, data, options, start_params=pin),
        "fixed_params": lambda: estimate_amn(model, data, options, fixed_params=pin),
    }

    with pytest.raises(NotImplementedError):
        callers[override]()


def test_estimate_amn_honours_fixed_loading():
    """A pinned measurement loading is held in the Stage-2 minimum distance.

    The free fit would estimate the y2 loading from the moments; pinning it must
    hold it exactly while the other structural parameters adjust around it.
    """
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    pin = pd.DataFrame(
        {"value": [0.8]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "y2", "skills")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=pin)

    got = result.params.loc[("loadings", 0, "y2", "skills"), "value"]
    assert got == pytest.approx(0.8)


def test_estimate_amn_honours_fixed_meas_sd():
    """A pinned measurement SD is held in the Stage-2 minimum distance."""
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    pin = pd.DataFrame(
        {"value": [0.5]},
        index=pd.MultiIndex.from_tuples(
            [("meas_sds", 0, "y2", "-")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=pin)

    got = result.params.loc[("meas_sds", 0, "y2", "-"), "value"]
    assert got == pytest.approx(0.5)


def test_estimate_amn_rejects_pinning_normalized_loading():
    """Pinning a loading the model already normalizes is a clear error.

    `y1` is the loading-normalized measure (lambda = 1), so it is not a free
    Stage-2 parameter; trying to pin it must raise rather than silently no-op.
    """
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    pin = pd.DataFrame(
        {"value": [2.0]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "y1", "skills")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    with pytest.raises(ValueError, match="normaliz"):
        estimate_amn(model, data, options, fixed_params=pin)


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


def test_estimate_amn_standalone_rejects_restricted_ces():
    """Standalone AMN refuses a restricted-CES model it cannot consistently estimate.

    The restricted-CES (`log_ces`) Stage-3 regression omits Freyberger's
    primitive-scale recovery, so a standalone fit would return inconsistent CES
    parameters. estimate_amn must raise instead, pointing to log_ces_general or
    seeding estimate_chs.
    """
    model = _tiny_ces_model()
    data = _tiny_data(n=500)
    options = AMNEstimationOptions(n_simulation_draws=2000, seed=0)

    with pytest.raises(NotImplementedError, match="restricted-CES"):
        estimate_amn(model, data, options)


def test_amn_seeding_bypasses_restricted_ces_guard():
    """The standalone guard does not fire when AMN is seeding estimate_chs.

    `linearize_control_function=True` marks the CHS-seeding context (CHS re-fits
    every parameter), so a rough restricted-CES seed is acceptable; the guard
    must let it through. A non-CES model never trips the guard either.
    """
    ces = process_model(_tiny_ces_model())
    # Seeding context: no raise even though the transition is restricted CES.
    _fail_if_standalone_unsupported(ces, for_start_values=True)
    # Standalone non-CES model: no raise.
    _fail_if_standalone_unsupported(
        process_model(_tiny_model()), for_start_values=False
    )
    # Standalone CES: raises (mirrors the integration test above).
    with pytest.raises(NotImplementedError, match="restricted-CES"):
        _fail_if_standalone_unsupported(ces, for_start_values=False)


def test_estimate_amn_standalone_rejects_log_ces_af():
    """log_ces_af is restricted CES too (Pro F6) and must trip the guard."""
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"), ("y1", "y2", "y3")),
                normalizations=Normalizations(
                    loadings=({"y1": 1}, {"y1": 1}),
                    intercepts=({"y1": 0}, {}),
                ),
                transition_function="log_ces_af",
            ),
        },
        n_mixtures=2,
    )
    processed = process_model(model)
    with pytest.raises(NotImplementedError, match="restricted-CES"):
        _fail_if_standalone_unsupported(processed, for_start_values=False)


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


def test_estimate_amn_honours_fixed_transition_constant():
    """A pinned transition constant is held exactly in the Stage-3 regression.

    The free fit would put a nonzero intercept on the skills production
    regression; pinning it to 0 must partial the intercept out and report
    exactly 0, leaving the other coefficients free.
    """
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    pin = pd.DataFrame(
        {"value": [0.0]},
        index=pd.MultiIndex.from_tuples(
            [("transition", 0, "skills", "constant")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=pin)

    got = result.params.loc[("transition", 0, "skills", "constant"), "value"]
    assert got == 0.0


def test_estimate_amn_honours_fixed_transition_slope():
    """A pinned transition slope is held at its value, not re-estimated."""
    model = _tiny_model()
    data = _tiny_data(n=1500)
    options = AMNEstimationOptions(n_simulation_draws=5000, seed=0)

    pin = pd.DataFrame(
        {"value": [0.3]},
        index=pd.MultiIndex.from_tuples(
            [("transition", 0, "skills", "skills")],
            names=["category", "aug_period", "name1", "name2"],
        ),
    )

    result = estimate_amn(model, data, options, fixed_params=pin)

    got = result.params.loc[("transition", 0, "skills", "skills"), "value"]
    assert got == pytest.approx(0.3)
