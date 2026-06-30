"""Tests for `skillmodels.amn.inference.compute_amn_standard_errors`."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import skillmodels.amn.inference as inf
from skillmodels.amn import compute_amn_standard_errors, estimate_amn
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


def _tiny_data(n: int = 800, seed: int = 0) -> pd.DataFrame:
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


def test_bootstrap_returns_expected_shapes():
    model = _tiny_model()
    data = _tiny_data(n=500, seed=0)
    options = AMNEstimationOptions(n_simulation_draws=1000, seed=0)
    fit = estimate_amn(model, data, options)

    inference = compute_amn_standard_errors(fit, data, options, n_boot=5, seed=11)

    assert inference.n_boot == 5
    assert inference.n_clusters == 500
    assert inference.standard_errors.shape[0] == fit.params.shape[0]
    assert inference.replicate_params.shape == (5, fit.params.shape[0])
    assert inference.vcov.shape == (fit.params.shape[0], fit.params.shape[0])


def test_bootstrap_standard_errors_non_negative_and_finite_where_replicates_finite():
    model = _tiny_model()
    data = _tiny_data(n=500, seed=1)
    options = AMNEstimationOptions(n_simulation_draws=1000, seed=0)
    fit = estimate_amn(model, data, options)

    inference = compute_amn_standard_errors(fit, data, options, n_boot=8, seed=42)

    # Wherever we have at least two finite replicates for a parameter,
    # the std should be finite and non-negative.
    for col in inference.standard_errors.index:
        finite = inference.replicate_params[col].dropna()
        if len(finite) >= 2:
            se = inference.standard_errors[col]
            assert np.isfinite(se)
            assert se >= 0.0


def test_bootstrap_uses_distinct_reproducible_replicate_seeds(monkeypatch):
    model = _tiny_model()
    data = _tiny_data(n=500, seed=0)
    options = AMNEstimationOptions(n_simulation_draws=1000, seed=0)
    fit = estimate_amn(model, data, options)

    real = inf.estimate_amn

    def make_spy(record: list[int]):
        def spy(model_spec, boot_data, amn_options):
            record.append(amn_options.seed)
            return real(model_spec, boot_data, amn_options)

        return spy

    seen_seeds: list[int] = []
    monkeypatch.setattr(inf, "estimate_amn", make_spy(seen_seeds))
    compute_amn_standard_errors(fit, data, options, n_boot=4, seed=7)

    # All four replicate seeds are distinct (independent random streams).
    assert len(seen_seeds) == 4
    assert len(set(seen_seeds)) == 4

    # Same top-level seed reproduces the exact same replicate seeds.
    seen_again: list[int] = []
    monkeypatch.setattr(inf, "estimate_amn", make_spy(seen_again))
    compute_amn_standard_errors(fit, data, options, n_boot=4, seed=7)
    assert seen_again == seen_seeds


def test_bootstrap_excludes_nonconverged_replicate(monkeypatch):
    model = _tiny_model()
    data = _tiny_data(n=500, seed=0)
    options = AMNEstimationOptions(n_simulation_draws=1000, seed=0)
    fit = estimate_amn(model, data, options)

    real = inf.estimate_amn
    sentinel = -999.0
    call_counter = {"n": 0}

    def spy(model_spec, boot_data, amn_options):
        idx = call_counter["n"]
        call_counter["n"] += 1
        real_fit = real(model_spec, boot_data, amn_options)
        if idx == 1:
            # 2nd replicate: nonconverged with a recognizable sentinel value.
            bad_params = real_fit.params.copy()
            bad_params["value"] = sentinel
            return dataclasses.replace(real_fit, params=bad_params, success=False)
        return real_fit

    monkeypatch.setattr(inf, "estimate_amn", spy)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        inference = compute_amn_standard_errors(fit, data, options, n_boot=4, seed=3)

    # The nonconverged replicate's row is all NaN, not stored.
    assert inference.replicate_params.iloc[1].isna().all()
    # The sentinel params never leak into the bootstrap distribution.
    assert not (inference.replicate_params == sentinel).to_numpy().any()
