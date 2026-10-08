"""Regression guard: documentation must not teach stale or false API/behavior.

Each assertion encodes a fixed Pro-review finding (F1-F9): a phrase the docs must
no longer contain, or a corrected phrase/field they must now contain. The audience is a
user copying the docs; a stale claim makes them build the wrong model or call a missing
API. Keep these checks textual and cheap so they run in any environment.
"""

from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _read(rel: str) -> str:
    return (DOCS / rel).read_text(encoding="utf-8")


def test_model_specs_uses_correction_not_is_correction() -> None:
    text = _read("how_to_guides/model_specs.md")
    assert "is_correction" not in text
    assert "correction" in text


def test_names_and_concepts_uses_current_option_names() -> None:
    text = _read("explanations/names_and_concepts.md")
    assert "n_mixture_components" not in text
    assert "initialization_strategy" not in text
    assert "investment_endogeneity" not in text
    assert "start_params_strategy" in text


def test_amn_guide_does_not_claim_chs_is_one_component() -> None:
    text = _read("how_to_guides/how_to_estimate_amn.md")
    assert "one mixture component" not in text
    assert "n_mixtures" in text


def test_af_guide_does_not_overclaim_constraint_support() -> None:
    text = _read("how_to_guides/how_to_estimate_af.md")
    assert "All optimagic constraint kinds are supported" not in text
    assert "select_by_loc" in text


def test_compare_guide_does_not_claim_identical_point_estimates() -> None:
    text = _read("how_to_guides/how_to_compare_estimators.md")
    assert "same point estimate" not in text


def test_transition_functions_does_not_say_amn_lacks_custom_transitions() -> None:
    text = _read("reference_guides/transition_functions.md")
    assert "not yet with AMN" not in text


def test_tutorial_activates_the_af_model_it_describes() -> None:
    text = _read("getting_started/tutorial.ipynb")
    assert "af_state_role" in text
    assert "has_initial_distribution=False" in text
    assert "is_endogenous=True" in text


def test_estimator_prerequisites_reference_page_exists() -> None:
    text = _read("reference_guides/estimator_prerequisites.md")
    assert "prerequisite" in text.lower()
    # A table comparing the three estimators.
    for estimator in ("CHS", "AF", "AMN"):
        assert estimator in text
    # AF non-Gaussian support is initial-period only — must be stated, not blanket.
    assert "initial" in text.lower()


def test_amn_guide_documents_missing_data_and_fixed_param_prerequisites() -> None:
    text = _read("how_to_guides/how_to_estimate_amn.md")
    assert "mixture_em_method" in text
    assert "allow_never_observed_measurements" in text
