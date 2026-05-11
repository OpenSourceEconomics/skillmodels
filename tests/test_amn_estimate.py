"""Tests for the AMN (Attanasio-Meghir-Nix 2020) estimator."""

import numpy as np
import pandas as pd

from skillmodels.amn import AMNEstimationOptions, AMNEstimationResult, estimate_amn
from skillmodels.model_spec import (
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.params_index import get_params_index
from skillmodels.process_model import process_model


def _build_linear_t3_model() -> ModelSpec:
    """Two-factor T=3 linear-transition model used in several tests."""
    return ModelSpec(
        factors={
            "state": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 3,
                    intercepts=({"y1": 0},) * 3,
                ),
                transition_function="linear",
            ),
            "inv": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 3,
                normalizations=Normalizations(
                    loadings=({"z1": 1},) * 3,
                    intercepts=({"z1": 0},) * 3,
                ),
                transition_function="linear",
            ),
        },
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )


def _truth_params_linear_t3(model: ModelSpec) -> pd.DataFrame:
    processed = process_model(model)
    p_index = get_params_index(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        transition_info=processed.transition_info,
        endogenous_factors_info=processed.endogenous_factors_info,
    )
    df = pd.DataFrame({"value": np.zeros(len(p_index))}, index=p_index)
    cat = df.index.get_level_values("category")
    df.loc[cat == "loadings", "value"] = 1.0
    df.loc[cat == "meas_sds", "value"] = 0.3
    df.loc[cat == "shock_sds", "value"] = 0.4
    df.loc[cat == "mixture_weights", "value"] = 1.0
    for aug in range(2):
        for f, other in (("state", "inv"), ("inv", "state")):
            df.loc[("transition", aug, f, f), "value"] = 0.7
            df.loc[("transition", aug, f, other), "value"] = 0.2
            df.loc[("transition", aug, f, "constant"), "value"] = 0.1
    diag_mask = pd.Series(
        [
            idx[0] == "initial_cholcovs"
            and "-" in idx[3]
            and idx[3].split("-")[0] == idx[3].split("-")[1]
            for idx in df.index
        ],
        index=df.index,
    )
    df.loc[diag_mask, "value"] = 1.0
    return df


def _simulate_linear_t3(params: pd.DataFrame, n_obs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_periods = 3
    state = rng.normal(0.0, 1.0, size=(n_obs, 2))
    state_history = [state.copy()]

    def _val(loc: tuple) -> float:
        return float(params.loc[loc, "value"])

    for t in range(1, n_periods):
        prev = state_history[-1]
        new_state = np.zeros_like(prev)
        for f, idx in (("state", 0), ("inv", 1)):
            other_idx = 1 - idx
            other = "inv" if f == "state" else "state"
            a = _val(("transition", t - 1, f, f))
            b = _val(("transition", t - 1, f, other))
            c = _val(("transition", t - 1, f, "constant"))
            sigma = _val(("shock_sds", t - 1, f, "-"))
            new_state[:, idx] = (
                a * prev[:, idx]
                + b * prev[:, other_idx]
                + c
                + sigma * rng.normal(size=n_obs)
            )
        state_history.append(new_state)

    rows: list[dict] = []
    for obs_id in range(n_obs):
        for t in range(n_periods):
            row: dict[str, float | int] = {"caseid": obs_id, "period": t}
            st = state_history[t][obs_id]
            for f, idx in (("state", 0), ("inv", 1)):
                meas_prefix = "y" if f == "state" else "z"
                for k in (1, 2, 3):
                    meas = f"{meas_prefix}{k}"
                    lam = _val(("loadings", t, meas, f))
                    eps = _val(("meas_sds", t, meas, "-"))
                    row[meas] = lam * st[idx] + eps * rng.normal()
            rows.append(row)
    return pd.DataFrame.from_records(rows).set_index(["caseid", "period"])


def test_estimate_amn_returns_result_with_full_params() -> None:
    """`estimate_amn` returns an `AMNEstimationResult` with no NaN entries."""
    model = _build_linear_t3_model()
    truth = _truth_params_linear_t3(model)
    data = _simulate_linear_t3(truth, n_obs=300, seed=20260511)

    result = estimate_amn(model_spec=model, data=data)

    assert isinstance(result, AMNEstimationResult)
    assert not result.params["value"].isna().any()
    assert result.n_obs == 300


def test_estimate_amn_recovers_linear_transition_within_15_percent() -> None:
    """Recover linear-transition coefficients within 15% of truth.

    On a linear-transition DGP with EIV correction, transition
    coefficients should land within 15% of truth on a moderate sample.
    """
    model = _build_linear_t3_model()
    truth = _truth_params_linear_t3(model)
    data = _simulate_linear_t3(truth, n_obs=2000, seed=20260511)

    result = estimate_amn(model_spec=model, data=data)
    params = result.params
    truth_loc_pairs = [
        (("transition", 0, "state", "state"), 0.7),
        (("transition", 0, "state", "inv"), 0.2),
        (("transition", 0, "inv", "state"), 0.2),
        (("transition", 0, "inv", "inv"), 0.7),
    ]
    for loc, true_value in truth_loc_pairs:
        est = float(params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
        rel = abs(est - true_value) / abs(true_value)
        assert rel < 0.15, (
            f"AMN estimate {est:.3f} at {loc} is {rel:.1%} off truth {true_value:.3f}"
        )


def test_amn_bias_correction_pulls_coefficient_closer_to_truth() -> None:
    """The EIV-corrected coefficient is closer to truth than the raw OLS.

    OLS on noisy proxies is attenuated toward zero; the EIV
    correction undoes (most of) that attenuation. We verify this on
    a single coefficient with a measurement-noise-heavy DGP.
    """
    model = _build_linear_t3_model()
    truth = _truth_params_linear_t3(model)
    # Inflate measurement noise to make the attenuation bias bite.
    cat = truth.index.get_level_values("category")
    truth.loc[cat == "meas_sds", "value"] = 0.8
    data = _simulate_linear_t3(truth, n_obs=2000, seed=20260511)

    raw = estimate_amn(
        model_spec=model,
        data=data,
        amn_options=AMNEstimationOptions(use_bias_correction=False),
    )
    corrected = estimate_amn(
        model_spec=model,
        data=data,
        amn_options=AMNEstimationOptions(use_bias_correction=True),
    )
    loc = ("transition", 0, "state", "state")
    raw_est = float(raw.params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
    corr_est = float(corrected.params.loc[loc, "value"])  # ty: ignore[invalid-argument-type]
    truth_value = 0.7

    raw_err = abs(raw_est - truth_value)
    corr_err = abs(corr_est - truth_value)
    assert raw_est < truth_value, (
        f"Uncorrected AMN should attenuate toward 0; "
        f"got {raw_est:.3f} >= {truth_value:.3f}"
    )
    assert corr_err < raw_err, (
        f"EIV correction did not reduce bias: raw_err={raw_err:.3f}, "
        f"corr_err={corr_err:.3f}"
    )


def test_estimate_amn_respects_fixed_params() -> None:
    """User-supplied `fixed_params` overwrite the AMN point estimate."""
    model = _build_linear_t3_model()
    truth = _truth_params_linear_t3(model)
    data = _simulate_linear_t3(truth, n_obs=500, seed=20260511)

    pinned_loc = ("transition", 0, "state", "state")
    fixed = pd.DataFrame(
        {"value": [99.0]},
        index=pd.MultiIndex.from_tuples(
            [pinned_loc], names=["category", "period", "name1", "name2"]
        ),
    )
    result = estimate_amn(model_spec=model, data=data, fixed_params=fixed)
    assert float(result.params.loc[pinned_loc, "value"]) == 99.0  # ty: ignore[invalid-argument-type]


def test_amn_proxies_and_variance_present() -> None:
    """The result carries proxies and EIV variances per (period, factor)."""
    model = _build_linear_t3_model()
    truth = _truth_params_linear_t3(model)
    data = _simulate_linear_t3(truth, n_obs=400, seed=20260511)

    result = estimate_amn(model_spec=model, data=data)
    assert len(result.factor_proxies) > 0
    for key, proxy in result.factor_proxies.items():
        assert proxy.shape == (400,)
        assert key in result.proxy_meas_err_var
        assert result.proxy_meas_err_var[key] > 0


def test_amn_default_options_use_bias_correction() -> None:
    assert AMNEstimationOptions().use_bias_correction is True
