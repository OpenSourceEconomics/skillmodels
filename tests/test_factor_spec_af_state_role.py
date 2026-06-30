"""Tests for the explicit `af_state_role` metadata on `FactorSpec`.

A time-invariant factor (e.g. MC/MN) whose period-0 measurement density must be
re-applied as an AF importance factor at every step is declared explicitly via
`af_state_role="static_persistent"`, rather than inferred from an identity transition
(which may be supplied through `fixed_params` and is therefore not reliably detectable).
Factors default to `"dynamic"`, so existing specs are unchanged.
"""

import pytest

from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations


def test_factor_spec_defaults_to_dynamic_af_state_role() -> None:
    spec = FactorSpec(measurements=(("m",), ("m",)), transition_function="linear")
    assert spec.af_state_role == "dynamic"


def test_factor_spec_accepts_static_persistent_role() -> None:
    spec = FactorSpec(
        measurements=(("mc_1",), ()),
        transition_function="linear",
        af_state_role="static_persistent",
    )
    assert spec.af_state_role == "static_persistent"


def test_factor_spec_rejects_unknown_af_state_role() -> None:
    with pytest.raises((ValueError, TypeError)):
        FactorSpec(
            measurements=(("m",),),
            transition_function="linear",
            # Intentionally invalid: exercises beartype's runtime rejection.
            af_state_role="bogus",  # ty: ignore[invalid-argument-type]
        )


def test_from_dict_parses_af_state_role() -> None:
    model = ModelSpec.from_dict(
        {
            "factors": {
                "MC": {
                    "measurements": [["mc_1"], []],
                    "normalizations": {
                        "loadings": [{"mc_1": 1}, {}],
                        "intercepts": [{"mc_1": 0}, {}],
                    },
                    "transition_function": "linear",
                    "af_state_role": "static_persistent",
                },
            },
        }
    )
    assert model.factors["MC"].af_state_role == "static_persistent"


def test_from_dict_defaults_af_state_role_to_dynamic() -> None:
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("test_score",), ("test_score",)),
                normalizations=Normalizations(
                    loadings=({"test_score": 1},) * 2,
                    intercepts=({"test_score": 0},) * 2,
                ),
                transition_function="linear",
            ),
        },
    )
    assert model.factors["skills"].af_state_role == "dynamic"
