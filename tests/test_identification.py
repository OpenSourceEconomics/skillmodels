"""Tests for the reusable transition-aware identification anchor check (audit F6).

`check_identification` is the estimator-agnostic core of the AF period-0 anchor
diagnostic: given a ModelSpec plus any fixed_params / constraints, it returns a
list of human-readable problems with the initial-period affine anchoring. Every
factor with an initial distribution needs both a loading (scale) and an intercept
(location) anchor at period 0; the CES simplex does NOT supply the initial
location anchor (Pro F1). It is exposed for CHS/common tooling but is deliberately
NOT wired into the default process_model path (no pipeline-wide gating).
"""

import functools
from collections.abc import Mapping

import optimagic as om
import pandas as pd
import pytest

from skillmodels.common.constraints import select_by_loc
from skillmodels.common.identification import (
    check_identification,
    fail_if_not_identified,
    find_excess_initial_restrictions,
)
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _model(
    *,
    skills_loadings: tuple[Mapping[str, float], ...],
    skills_intercepts: tuple[Mapping[str, float], ...],
    skills_transition: str = "translog",
) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=skills_loadings,
                    intercepts=skills_intercepts,
                ),
                transition_function=skills_transition,
            ),
            "investment": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"z1": 1},) * 2,
                    intercepts=({"z1": 0},) * 2,
                ),
                transition_function="linear",
                is_endogenous=True,
            ),
        },
    )


def test_check_identification_clean_translog_returns_no_problems() -> None:
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({"y1": 0},) * 2,
    )
    assert check_identification(model) == []


def test_check_identification_flags_missing_period0_loading() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    problems = check_identification(model)
    assert any("period 0" in p for p in problems)


def test_check_identification_log_ces_requires_initial_intercept_anchor() -> None:
    # The CES simplex replaces only the CROSS-PERIOD location alternative, not
    # the absolute initial anchor mu_theta,0,1=0 (Pro F1: f(x+c,i+c)=f(x,i)+c
    # leaves a common-shift orbit). So plain log_ces still needs a period-0
    # intercept anchor.
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({}, {}),
        skills_transition="log_ces",
    )
    problems = check_identification(model)
    assert any("period 0" in p and "location" in p for p in problems)


def test_check_identification_equality_of_free_loadings_does_not_anchor() -> None:
    # An equality among free loadings reduces dimension but leaves their common
    # scale free (Pro F3): it is NOT an anchor unless connected to a fixed or
    # normalized member. period-0 skills loading is empty and the group ties two
    # otherwise-free loadings, so the scale stays unanchored -> a problem.
    model = _model(
        skills_loadings=({}, {}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    group = pd.MultiIndex.from_tuples(
        [
            ("loadings", 0, "y1", "skills"),
            ("loadings", 1, "y1", "skills"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    constraints: list[om.constraints.Constraint] = [
        om.EqualityConstraint(selector=functools.partial(select_by_loc, loc=group))
    ]
    problems = check_identification(model, constraints=constraints)
    assert any("period 0" in p and "scale" in p for p in problems)


def test_check_identification_zero_loading_pin_is_not_a_scale_anchor() -> None:
    # A loading fixed to 0 is invariant to every rescaling of the factor and
    # cannot pin its scale (Pro F1). It must NOT count as a scale anchor.
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    fixed = pd.DataFrame(
        {"value": [0.0]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "y1", "skills")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    problems = check_identification(model, fixed_params=fixed)
    assert any("period 0" in p and "scale" in p for p in problems)


def test_check_identification_nonfinite_loading_pin_is_not_an_anchor() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    fixed = pd.DataFrame(
        {"value": [float("inf")]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "y1", "skills")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    problems = check_identification(model, fixed_params=fixed)
    assert any("period 0" in p and "scale" in p for p in problems)


def test_check_identification_equality_anchor_is_transitive() -> None:
    # A=normalized, A=B, B=C across two constraints: the anchor must propagate
    # transitively to C (Pro F2). Here the only period-0 loading anchor is
    # reachable only through a two-hop equality chain.
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    g1 = pd.MultiIndex.from_tuples(
        [("loadings", 1, "y1", "skills"), ("loadings", 1, "y2", "skills")],
        names=["category", "period", "name1", "name2"],
    )
    g2 = pd.MultiIndex.from_tuples(
        [("loadings", 1, "y2", "skills"), ("loadings", 0, "y1", "skills")],
        names=["category", "period", "name1", "name2"],
    )
    constraints: list[om.constraints.Constraint] = [
        om.EqualityConstraint(selector=functools.partial(select_by_loc, loc=g1)),
        om.EqualityConstraint(selector=functools.partial(select_by_loc, loc=g2)),
    ]
    assert check_identification(model, constraints=constraints) == []


def test_check_identification_honours_fixed_params_anchor() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    fixed = pd.DataFrame(
        {"value": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "y1", "skills")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    assert check_identification(model, fixed_params=fixed) == []


def test_fail_if_not_identified_raises_when_location_unanchored() -> None:
    """A factor with a scale anchor but no period-0 location anchor is rejected."""
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({}, {}),
    )
    with pytest.raises(ValueError, match="not identified"):
        fail_if_not_identified(model)


def test_fail_if_not_identified_passes_a_fully_anchored_model() -> None:
    """A model with both period-0 scale and location anchors passes silently."""
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({"y1": 0},) * 2,
    )
    fail_if_not_identified(model)


_INDEX_NAMES = ["category", "period", "name1", "name2"]


def _fixed(rows: list[tuple[tuple[object, ...], float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"value": [value for _, value in rows]},
        index=pd.MultiIndex.from_tuples([key for key, _ in rows], names=_INDEX_NAMES),
    )


def _equality(*keys: tuple[object, ...]) -> om.constraints.Constraint:
    loc = pd.MultiIndex.from_tuples(list(keys), names=_INDEX_NAMES)
    return om.EqualityConstraint(selector=functools.partial(select_by_loc, loc=loc))


def test_check_identification_accepts_fixed_initial_mean_as_location_anchor() -> None:
    """A fixed period-0 component mean removes the location orbit (CHS convention).

    `skills` normalizes its period-0 loading but leaves the measurement intercept
    free; pinning one initial-component latent mean supplies the location anchor, so
    the model is fully anchored.
    """
    model = _model(
        skills_loadings=({"y1": 1.0},) * 2,
        skills_intercepts=({}, {}),
    )
    fixed = _fixed([(("initial_states", 0, "mixture_0", "skills"), 0.0)])
    assert check_identification(model, fixed_params=fixed) == []


def test_check_identification_flags_factor_with_no_normalizations() -> None:
    """A factor whose `normalizations is None` with no other anchor is rejected.

    An absent `Normalizations` object is not "out of scope": the initial affine
    orbit is still unpinned, so the gate must report the missing scale/location
    anchors rather than silently skip the factor.
    """
    model = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=None,
                transition_function="translog",
            ),
            "investment": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"z1": 1.0},) * 2,
                    intercepts=({"z1": 0.0},) * 2,
                ),
                transition_function="linear",
                is_endogenous=True,
            ),
        },
    )
    problems = check_identification(model)
    assert any("skills" in problem for problem in problems)


def test_check_identification_rejects_zero_loading_tied_to_intercept() -> None:
    """A loading equated to a zero intercept does not pin scale.

    `skills` has a period-0 intercept normalization (location) but no period-0
    loading normalization (scale). Tying that loading equal to the zero intercept
    leaves it fixed at zero, which is invariant under rescaling, so the factor is
    still scale-under-identified.
    """
    model = _model(
        skills_loadings=({}, {"y1": 1.0}),
        skills_intercepts=({"y1": 0.0}, {"y1": 0.0}),
    )
    constraints = [
        _equality(
            ("controls", 0, "y1", "constant"),
            ("loadings", 0, "y1", "skills"),
        )
    ]
    problems = check_identification(model, constraints=constraints)
    assert any("scale anchor" in problem for problem in problems)


def test_find_excess_initial_restrictions_flags_second_scale_and_location_pin() -> None:
    """A second loading pin and an extra initial-mean pin are testable restrictions.

    The factor pins two period-0 loadings (`y1`, `y2`) and both a measurement
    intercept and an initial-component mean, so each one-dimensional orbit direction
    is pinned twice; the surplus pins are testable, not normalizations.
    """
    model = _model(
        skills_loadings=({"y1": 1.0, "y2": 1.0}, {"y1": 1.0}),
        skills_intercepts=({"y1": 0.0}, {"y1": 0.0}),
    )
    fixed = _fixed([(("initial_states", 0, "mixture_0", "skills"), 0.0)])
    excess = find_excess_initial_restrictions(model, fixed_params=fixed)
    assert any("scale" in item and "testable" in item for item in excess)
    assert any("location" in item and "testable" in item for item in excess)


def test_find_excess_initial_restrictions_clean_model_is_empty() -> None:
    """One scale pin and one location pin per factor are normalizations, not excess."""
    model = _model(
        skills_loadings=({"y1": 1.0}, {"y1": 1.0}),
        skills_intercepts=({"y1": 0.0}, {"y1": 0.0}),
    )
    assert find_excess_initial_restrictions(model) == []


def test_find_excess_initial_restrictions_ignores_equality_tied_pins() -> None:
    """Two loadings tied equal are one restriction, so they are not flagged."""
    model = _model(
        skills_loadings=({"y1": 1.0, "y2": 1.0}, {"y1": 1.0}),
        skills_intercepts=({"y1": 0.0}, {"y1": 0.0}),
    )
    constraints = [
        _equality(
            ("loadings", 0, "y1", "skills"),
            ("loadings", 0, "y2", "skills"),
        )
    ]
    excess = find_excess_initial_restrictions(model, constraints=constraints)
    assert not any("scale" in item for item in excess)


def _ces_two_factor_model(
    *,
    investment_loadings: tuple[Mapping[str, float], ...],
    investment_intercepts: tuple[Mapping[str, float], ...],
) -> ModelSpec:
    """A restricted-CES `skills` output plus a latent `investment` input.

    Both factors carry an initial distribution, so investment is a CES input
    that has its own period-0 measurement system.
    """
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"y1": 1.0},) * 2,
                    intercepts=({"y1": 0.0},) * 2,
                ),
                transition_function="log_ces",
            ),
            "investment": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 2,
                normalizations=Normalizations(
                    loadings=investment_loadings,
                    intercepts=investment_intercepts,
                ),
                transition_function="linear",
            ),
        },
    )


def test_find_excess_flags_second_scale_anchor_in_restricted_ces_system() -> None:
    """A second scale anchor across a restricted-CES system is a testable restriction.

    `skills` (restricted CES) anchors its own period-0 scale; `investment` is a
    CES input that also carries an initial distribution and redundantly anchors a
    second scale. The CES restrictions identify investment's scale from skills'
    single anchor, so the extra anchor constrains an identified feature.
    """
    model = _ces_two_factor_model(
        investment_loadings=({"z1": 1.0},) * 2,
        investment_intercepts=({"z1": 0.0},) * 2,
    )
    excess = find_excess_initial_restrictions(model)
    assert any(
        "CES" in item and "scale" in item and "testable" in item for item in excess
    )


def test_find_excess_allows_single_scale_anchor_in_restricted_ces_system() -> None:
    """One scale anchor for the whole restricted-CES system is not excess.

    Only `skills` anchors the shared CES scale; investment's scale is identified
    through the CES restrictions, so there is no excess scale restriction.
    """
    model = _ces_two_factor_model(
        investment_loadings=({}, {}),
        investment_intercepts=({"z1": 0.0}, {"z1": 0.0}),
    )
    excess = find_excess_initial_restrictions(model)
    assert not any("scale" in item for item in excess)
