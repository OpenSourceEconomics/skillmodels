"""Tests for model_spec module."""

import pytest

from skillmodels.common.model_spec import (
    AnchoringSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)
from skillmodels.common.types import EstimationOptions


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
    assert spec.estimation_options is None
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


def test_from_dict_with_estimation_options() -> None:
    d = _minimal_dict()
    d["estimation_options"] = {"n_mixtures": 2, "robust_bounds": False}
    spec = ModelSpec.from_dict(d)
    assert spec.estimation_options is not None
    assert spec.estimation_options.n_mixtures == 2
    assert spec.estimation_options.robust_bounds is False


def test_from_dict_with_stagemap() -> None:
    d = _minimal_dict()
    d["stagemap"] = [0]
    spec = ModelSpec.from_dict(d)
    assert spec.stagemap == (0,)


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


def test_with_estimation_options(model2) -> None:
    opts = EstimationOptions(n_mixtures=3)
    result = model2.with_estimation_options(opts)
    assert result.estimation_options is not None
    assert result.estimation_options.n_mixtures == 3


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
