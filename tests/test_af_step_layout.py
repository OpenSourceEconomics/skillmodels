"""Tests for the AF source/destination step-layout compiler.

The AF estimator is sequential: step `s -> d = s+1` estimates the transition, the
source-period investment equation, and a measurement block. Under the contemporaneous
public `ModelSpec`, an investment indicator declared at calendar period `c` measures
`I_c`. The compiler re-times these calendar declarations onto AF's sequential steps so
that, for step `s -> d`:

- the FREE target block scores destination dynamic-state (skill) indicators on
  `theta_d` and source endogenous (investment) indicators on `I_s`;
- the FIXED importance block scores source dynamic-state indicators on `theta_s` and
  every static-persistent factor's period-0 indicators on its time-invariant value.

This reproduces MATLAB `likelihood_01`/`likelihood_12` (target = {skill_{s+1}, inv_s},
importance = {skill_s, MC_0, MN_0}) while the public spec stays contemporaneous.
"""

from skillmodels.af.step_layout import (
    AFEval,
    AFFactorInfo,
    AFFactorRole,
    compile_af_step_layouts,
)


def _cnlsy_factor_infos() -> tuple[AFFactorInfo, ...]:
    """A minimal 3-period CNLSY-shaped factor set (skills, MC, MN, investment)."""
    return (
        AFFactorInfo(
            name="skills",
            role=AFFactorRole.DYNAMIC,
            measurements_by_period=(
                ("sk_a", "sk_b"),
                ("sk_a", "sk_b"),
                ("sk_a", "sk_b"),
            ),
        ),
        AFFactorInfo(
            name="MC",
            role=AFFactorRole.STATIC_PERSISTENT,
            measurements_by_period=(("mc_1", "mc_2"), (), ()),
        ),
        AFFactorInfo(
            name="MN",
            role=AFFactorRole.STATIC_PERSISTENT,
            measurements_by_period=(("mn_1",), (), ()),
        ),
        AFFactorInfo(
            name="investment",
            role=AFFactorRole.ENDOGENOUS,
            measurements_by_period=(("inv_a", "inv_b"), ("inv_a", "inv_b"), ()),
        ),
    )


def _term_keys(terms, role):
    """Set of (measurement, data_period, param_period, eval, free) for one role."""
    return {
        (t.measurement, t.data_period, t.param_period, t.eval_node, t.free)
        for t in terms
        if t.role == role
    }


def test_compiles_one_layout_per_transition() -> None:
    layouts = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)
    assert len(layouts) == 2
    assert (layouts[0].source_period, layouts[0].destination_period) == (0, 1)
    assert (layouts[1].source_period, layouts[1].destination_period) == (1, 2)


def test_step_0_to_1_target_block() -> None:
    layout = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)[0]
    # Destination skills on theta_d, source investment (age-7, calendar 0) on I_0.
    assert _term_keys(layout.terms, "target") == {
        ("sk_a", 1, 1, AFEval.THETA_DEST, True),
        ("sk_b", 1, 1, AFEval.THETA_DEST, True),
        ("inv_a", 0, 0, AFEval.INV_SRC, True),
        ("inv_b", 0, 0, AFEval.INV_SRC, True),
    }


def test_step_0_to_1_importance_block_includes_static_factors() -> None:
    layout = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)[0]
    # Source skills on theta_s plus the static MC_0/MN_0 densities (the F3 fix).
    assert _term_keys(layout.terms, "importance") == {
        ("sk_a", 0, 0, AFEval.THETA_SRC, False),
        ("sk_b", 0, 0, AFEval.THETA_SRC, False),
        ("mc_1", 0, 0, AFEval.STATIC, False),
        ("mc_2", 0, 0, AFEval.STATIC, False),
        ("mn_1", 0, 0, AFEval.STATIC, False),
    }


def test_step_1_to_2_target_uses_age9_investment_on_i1() -> None:
    layout = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)[1]
    # Destination skills on theta_2, source investment (age-9, calendar 1) on I_1.
    assert _term_keys(layout.terms, "target") == {
        ("sk_a", 2, 2, AFEval.THETA_DEST, True),
        ("sk_b", 2, 2, AFEval.THETA_DEST, True),
        ("inv_a", 1, 1, AFEval.INV_SRC, True),
        ("inv_b", 1, 1, AFEval.INV_SRC, True),
    }


def test_step_1_to_2_importance_reapplies_period0_static_factors() -> None:
    layout = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)[1]
    # Source skills are now calendar 1; MC_0/MN_0 are re-applied unchanged.
    assert _term_keys(layout.terms, "importance") == {
        ("sk_a", 1, 1, AFEval.THETA_SRC, False),
        ("sk_b", 1, 1, AFEval.THETA_SRC, False),
        ("mc_1", 0, 0, AFEval.STATIC, False),
        ("mc_2", 0, 0, AFEval.STATIC, False),
        ("mn_1", 0, 0, AFEval.STATIC, False),
    }


def test_static_persistent_does_not_leak_future_periods() -> None:
    """A static factor contributes only its period-0 rows to every importance block.

    A static-persistent factor whose declarations span periods 0 and 2 re-applies its
    period-0 measurement as a fixed importance term at every step, and never lets its
    period-2 declaration leak forward into an earlier step's importance block.
    """
    factor_infos = (
        AFFactorInfo(
            name="skills",
            role=AFFactorRole.DYNAMIC,
            measurements_by_period=(("sk0",), ("sk1",), ("sk2",)),
        ),
        AFFactorInfo(
            name="MC",
            role=AFFactorRole.STATIC_PERSISTENT,
            measurements_by_period=(("mc0",), (), ("mc2",)),
        ),
        AFFactorInfo(
            name="MN",
            role=AFFactorRole.STATIC_PERSISTENT,
            measurements_by_period=(("mn0",), (), ()),
        ),
    )
    layouts = compile_af_step_layouts(factor_infos, n_periods=3)
    step_01_static = {
        (t.measurement, t.data_period)
        for t in layouts[0].terms
        if t.role == "importance" and t.eval_node == AFEval.STATIC
    }
    assert ("mc0", 0) in step_01_static
    assert ("mc2", 2) not in step_01_static
    assert ("mn0", 0) in step_01_static

    for layout in layouts:
        step_static = {
            (t.measurement, t.data_period)
            for t in layout.terms
            if t.role == "importance" and t.eval_node == AFEval.STATIC
        }
        assert ("mn0", 0) in step_static


def test_each_investment_calendar_wave_is_a_target_exactly_once() -> None:
    layouts = compile_af_step_layouts(_cnlsy_factor_infos(), n_periods=3)
    inv_targets = [
        (t.measurement, t.data_period)
        for layout in layouts
        for t in layout.terms
        if t.role == "target" and t.eval_node == AFEval.INV_SRC
    ]
    # age-7 (calendar 0) and age-9 (calendar 1) each used once; no calendar-2 wave.
    assert sorted(inv_targets) == [
        ("inv_a", 0),
        ("inv_a", 1),
        ("inv_b", 0),
        ("inv_b", 1),
    ]
