"""Tests for model_spec module."""

import pytest

from skillmodels.common.model_spec import (
    AnchoringSpec,
    CorrectionSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _minimal_dict():
    """Return a minimal model dict for from_dict tests."""
    return {
        "factors": {
            "f1": {
                "measurements": [["y1", "y2"], ["y1", "y2"]],
                "transition_function": "linear",
            },
        },
    }


def test_from_dict_minimal() -> None:
    d = _minimal_dict()
    spec = ModelSpec.from_dict(d)
    assert "f1" in spec.factors
    assert spec.factors["f1"].measurements == (("y1", "y2"), ("y1", "y2"))
    assert spec.factors["f1"].transition_function == "linear"
    assert spec.anchoring is None
    assert spec.controls == ()


def test_from_dict_with_normalizations() -> None:
    d = _minimal_dict()
    d["factors"]["f1"]["normalizations"] = {
        "loadings": [{"y1": 1}, {}],
        "intercepts": [{}, {}],
    }
    spec = ModelSpec.from_dict(d)
    norms = spec.factors["f1"].normalizations
    assert norms is not None
    assert len(norms.loadings) == 2


def test_from_dict_normalizations_default_intercepts() -> None:
    d = _minimal_dict()
    d["factors"]["f1"]["normalizations"] = {
        "loadings": [{"y1": 1}, {}],
    }
    spec = ModelSpec.from_dict(d)
    norms = spec.factors["f1"].normalizations
    assert norms is not None
    assert len(norms.intercepts) == 2
    # Default intercepts are empty dicts
    assert dict(norms.intercepts[0]) == {}


def test_from_dict_with_anchoring() -> None:
    d = _minimal_dict()
    d["anchoring"] = {"outcomes": {"f1": "Q1"}, "free_controls": True}
    spec = ModelSpec.from_dict(d)
    assert spec.anchoring is not None
    assert dict(spec.anchoring.outcomes) == {"f1": "Q1"}
    assert spec.anchoring.free_controls is True


def test_from_dict_with_n_mixtures() -> None:
    d = _minimal_dict()
    d["n_mixtures"] = 2
    spec = ModelSpec.from_dict(d)
    assert spec.n_mixtures == 2


def test_from_dict_with_stagemap() -> None:
    d = _minimal_dict()
    d["stagemap"] = [0]
    spec = ModelSpec.from_dict(d)
    assert spec.stagemap == (0,)


def test_from_dict_correction_block_not_yet_supported() -> None:
    d = _minimal_dict()
    d["factors"]["f1"]["is_endogenous"] = True
    d["factors"]["f1"]["correction"] = {"instruments": ["z1"]}
    with pytest.raises(NotImplementedError, match="correction"):
        ModelSpec.from_dict(d)


def test_correction_spec_defaults_are_empty() -> None:
    cf = CorrectionSpec()
    assert cf.state_predictors == ()
    assert cf.instruments == ()
    assert cf.targets == ()
    assert dict(cf.kappa_terms) == {}


def test_correction_spec_stores_fields_and_makes_kappa_terms_immutable() -> None:
    cf = CorrectionSpec(
        state_predictors=("health_mom", "health_kid"),
        instruments=("sum_inv_paid_log", "sum_inv_private_log"),
        targets=("health_mom", "health_kid"),
        kappa_terms={"health_mom": ("cf",), "health_kid": ("cf", "cf ** 2")},
    )
    assert cf.state_predictors == ("health_mom", "health_kid")
    assert cf.instruments == ("sum_inv_paid_log", "sum_inv_private_log")
    assert cf.targets == ("health_mom", "health_kid")
    assert cf.kappa_terms["health_kid"] == ("cf", "cf ** 2")
    # kappa_terms must be converted to an immutable mapping.
    with pytest.raises(TypeError):
        cf.kappa_terms["health_mom"] = ("cf", "cf ** 2")  # ty: ignore[invalid-assignment]


def test_correction_spec_is_frozen() -> None:
    cf = CorrectionSpec()
    with pytest.raises(AttributeError):
        cf.targets = ("health_mom",)  # ty: ignore[invalid-assignment]


def test_factor_spec_correction_defaults_to_none() -> None:
    spec = FactorSpec(measurements=(("y1",),))
    assert spec.correction is None


def test_factor_spec_accepts_correction() -> None:
    cf = CorrectionSpec(
        instruments=("z1",),
        targets=("health_mom",),
    )
    spec = FactorSpec(
        measurements=(("ln_inv",),),
        is_endogenous=True,
        correction=cf,
    )
    assert spec.correction is cf


def test_with_added_factor(model2) -> None:
    new_factor = FactorSpec(
        measurements=(("z1", "z2"),) * 8,
        transition_function="linear",
    )
    result = model2.with_added_factor("fac4", new_factor)
    assert len(result.factors) == 4
    assert "fac4" in result.factors


def test_with_added_observed_factors(model2) -> None:
    result = model2.with_added_observed_factors("obs1", "obs2")
    assert result.observed_factors == ("obs1", "obs2")


def test_with_anchoring(model2) -> None:
    anch = AnchoringSpec(outcomes={"fac2": "Q2"})
    result = model2.with_anchoring(anch)
    assert result.anchoring is not None
    assert dict(result.anchoring.outcomes) == {"fac2": "Q2"}


def test_with_controls(model2) -> None:
    result = model2.with_controls(("x1", "x2"))
    assert result.controls == ("x1", "x2")


def test_with_stagemap(model2) -> None:
    result = model2.with_stagemap((0, 1, 2, 3, 4, 5, 6))
    assert result.stagemap == (0, 1, 2, 3, 4, 5, 6)


def test_without_correction_strips_correction_from_all_factors() -> None:
    corrected = FactorSpec(
        measurements=(("ln_inv",),),
        is_endogenous=True,
        correction=CorrectionSpec(instruments=("z1",), targets=("skills",)),
    )
    plain = FactorSpec(measurements=(("y1",),))
    model = ModelSpec(factors={"skills": plain, "investment": corrected})

    stripped = model.without_correction()

    assert all(f.correction is None for f in stripped.factors.values())
    # The rest of the spec is preserved.
    assert stripped.factors["investment"].is_endogenous is True
    assert tuple(stripped.factors) == ("skills", "investment")


def test_with_transition_functions_valid(model2) -> None:
    funcs = {"fac1": "linear", "fac2": "translog", "fac3": "constant"}
    result = model2.with_transition_functions(funcs)
    assert result.factors["fac1"].transition_function == "linear"
    assert result.factors["fac2"].transition_function == "translog"


def test_factor_spec_with_normalizations() -> None:
    fspec = FactorSpec(measurements=(("y1", "y2"),))
    norms = Normalizations(loadings=({"y1": 1},), intercepts=({},))
    result = fspec.with_normalizations(norms)
    assert result.normalizations is not None
    assert dict(result.normalizations.loadings[0]) == {"y1": 1}


def test_with_transition_functions_mismatched_keys_raises(model2) -> None:
    funcs = {"fac1": "linear", "wrong": "translog"}
    with pytest.raises(ValueError, match="do not match"):
        model2.with_transition_functions(funcs)
