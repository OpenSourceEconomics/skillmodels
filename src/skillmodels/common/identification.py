"""Transition-aware identification anchor diagnostics (audit F6/F7/F8).

The initial-period latent distribution is not produced by any transition, so
its affine orbit (scale + location) must be pinned directly. `check_identification`
verifies that, dispatching on each factor's transition type:

Every factor with an initial distribution needs BOTH a scale anchor and a location
anchor at the initial period, regardless of transition. The scale anchor is a
finite, nonzero loading; the location anchor is a finite measurement intercept OR,
equivalently, a finite fixed initial-component latent mean (the CHS convention pins
the latter and leaves the measurement constants free). The CES simplex
`sum_i gamma_i = 1` does NOT supply the initial location anchor (Pro F1: plain CES
obeys f(x+c,i+c)=f(x,i)+c, so a common shift of all latent inputs leaves observables
unchanged while preserving the simplex); it only replaces the *cross-period*
skills-location alternative, which this period-0 precheck does not police.

An anchor may come from a `Normalizations` map, a `fixed_params` pin, or a
`select_by_loc` equality constraint connected to a numerically pinned member. The
pinned VALUE (not just the key) is propagated through equality components and then
tested against the category rule, so a loading tied to a zero intercept does not
masquerade as a scale anchor. A factor whose `normalizations is None` is checked the
same way against its fixed/equality anchors, not skipped. Periods t>0 are
intentionally not checked here: an unrestricted
transition transforms along with a later latent's affine orbit, so verifying
those needs a full transition-aware diagnostic (a larger, separate piece). This
is therefore an INITIAL-ANCHOR PRECHECK, not a complete identification proof: an
empty result means `initial_anchor_ok`, with later identification unverified.

This is the estimator-agnostic core: identification of the initial latent
distribution is a property of the model, not of any estimator. `fail_if_not_identified`
wraps it as a hard gate that `estimate_af` (via `validate_af_model`), `estimate_chs`
and `estimate_amn` all run by default, so an unanchored initial distribution raises in
every estimator. The gate lives at the estimator entry (not in `process_model`) because
the alternative anchor sources (`fixed_params`, equality `constraints`) are only known
there, not from the bare `ModelSpec`. Each estimator exposes a `require_identification`
switch to turn the gate off for models that are intentionally location-under-identified
-- e.g. the original CHS replication convention, where the initial latent mean is a free
parameter seeded to 0 rather than pinned by an intercept normalization.

RESTRICTED-CES SCALE SHARING (Pro F5). For a restricted-CES skill factor, the
production restrictions identify the relative skill/investment scales, so only
ONE primitive scale anchor is needed across the connected skill-investment system
(on lambda_theta,0,1 OR lambda_I,0,1); a second loading pin is then a testable
restriction, not a normalization. This is split deliberately between the two
sides of the diagnostic:

- The HARD GATE (`check_identification` / `fail_if_not_identified`) stays
  per-factor and CONSERVATIVE: it requires each factor with an initial
  distribution to carry its own scale anchor. A precise per-system requirement
  needs the production graph plus nonzero-share information, which is not
  available from the bare `ModelSpec`; a crude graph could make the gate PASS a
  genuinely scale-under-identified model, which is worse than over-requiring. In
  the standard setup investment is endogenous with
  `has_initial_distribution=False`, so it is skipped and exactly one scale anchor
  (on skills) is required -- correct. The only cost is over-requiring a redundant
  anchor in the exotic case where a CES input factor ALSO carries its own initial
  distribution; pass `require_identification=False` (or add the redundant anchor)
  there.
- The WARN side (`find_excess_initial_restrictions` /
  `warn_if_overrestricted`) IS restricted-CES aware: when a restricted-CES
  production is present it accounts scale anchors once across the CES system and
  flags a second independent scale pin as a testable restriction. Being
  imprecise here only over- or under-emits a warning, never weakens the gate.
"""

import math
import warnings
from collections.abc import Iterable

import optimagic as om
import pandas as pd

from skillmodels.common.constraints import _equality_constraint_loc
from skillmodels.common.model_spec import FactorSpec, ModelSpec

# Restricted CES (psi=1) production is affine in its latent inputs, so the
# production shares identify the relative scales of the combined factors; one
# primitive scale anchor propagates to the whole system. The general
# `log_ces_general` form is excluded (it does not impose the restriction).
_RESTRICTED_CES_TRANSITIONS = frozenset(
    {"log_ces", "log_ces_af", "log_ces_with_constant"}
)


def _restricted_ces_scale_system(model_spec: ModelSpec) -> frozenset[str]:
    """Return latent factors that share one scale through restricted-CES production.

    When any factor uses a restricted-CES transition, the production combines the
    latent factors into a system whose relative scales the CES restrictions identify
    from a single primitive anchor (see `recover_primitive_ces_scales`). skillmodels
    CES production is over the full latent state, so the system is the set of latent
    factors that carry an initial distribution. Returns an empty set when no
    restricted-CES transition is present (the translog default: each factor anchors
    its own scale) or when fewer than two such factors exist (no sharing to report).
    """
    has_restricted_ces = any(
        isinstance(spec.transition_function, str)
        and spec.transition_function in _RESTRICTED_CES_TRANSITIONS
        for spec in model_spec.factors.values()
    )
    if not has_restricted_ces:
        return frozenset()
    members = frozenset(
        name
        for name, spec in model_spec.factors.items()
        if spec.has_initial_distribution
    )
    return members if len(members) >= 2 else frozenset()


def _is_scale_anchor_value(value: float) -> bool:
    """A loading (scale) anchor must be finite and nonzero (Pro F1).

    A zero loading is invariant to every rescaling of the latent factor and so
    cannot pin its scale; non-finite values do not define a valid model.
    """
    return math.isfinite(value) and value != 0.0


def _is_location_anchor_value(value: float) -> bool:
    """An intercept (location) anchor must be finite (Pro F1)."""
    return math.isfinite(value)


def _pinned_values(
    model_spec: ModelSpec,
    fixed_params: pd.DataFrame | None,
) -> dict[tuple[object, ...], float]:
    """Map every numerically pinned parameter key to its value.

    Combines `Normalizations` (loadings to `("loadings", period, meas, factor)`,
    intercepts to `("controls", period, meas, "constant")`) with `fixed_params`
    rows. Keeping the value -- not just the key -- lets the caller apply the
    category-specific anchor rule: a loading pins scale only when finite and
    nonzero, a location anchor needs only finiteness.
    """
    pinned: dict[tuple[object, ...], float] = {}
    for factor, spec in model_spec.factors.items():
        norms = spec.normalizations
        if norms is None:
            continue
        for period, loadings in enumerate(norms.loadings or ()):
            for meas, value in loadings.items():
                pinned[("loadings", period, meas, factor)] = float(value)
        for period, intercepts in enumerate(norms.intercepts or ()):
            for meas, value in intercepts.items():
                pinned[("controls", period, meas, "constant")] = float(value)
    if fixed_params is not None:
        values = fixed_params["value"].to_numpy()
        for idx, value in zip(fixed_params.index, values, strict=True):
            pinned[tuple(idx)] = float(value)
    return pinned


def _equality_components(
    constraints: list[om.constraints.Constraint] | None,
) -> list[frozenset[tuple[object, ...]]]:
    """Return connected components of keys tied equal by `select_by_loc` groups.

    Two keys share a component when one equality group lists them together or a
    chain of overlapping groups connects them; standalone keys are omitted.
    """
    components: list[set[tuple[object, ...]]] = []
    for constraint in constraints or []:
        loc = _equality_constraint_loc(constraint)
        if loc is None:
            continue
        merged = {tuple(idx) for idx in loc}
        disjoint: list[set[tuple[object, ...]]] = []
        for component in components:
            if component & merged:
                merged |= component
            else:
                disjoint.append(component)
        disjoint.append(merged)
        components = disjoint
    return [frozenset(component) for component in components]


def _propagate_equality_values(
    pinned: dict[tuple[object, ...], float],
    components: list[frozenset[tuple[object, ...]]],
) -> dict[tuple[object, ...], float]:
    """Spread each equality component's agreed value to all its members (Pro F3).

    A member inherits a value only when its component's pinned members agree on a
    single value; a component with conflicting pins fabricates no anchor. Carrying
    the VALUE -- not merely the membership -- stops a loading tied to a zero
    intercept from masquerading as a scale anchor.
    """
    propagated = dict(pinned)
    for component in components:
        component_values = {pinned[key] for key in component if key in pinned}
        if len(component_values) == 1:
            (value,) = component_values
            for key in component:
                propagated[key] = value
    return propagated


def fail_if_not_identified(
    model_spec: ModelSpec,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> None:
    """Raise if any factor's initial-period affine orbit is unanchored.

    The estimator-agnostic gate every entry point runs by default:
    `check_identification` collects the missing period-0 scale/location anchors
    and this wrapper turns a non-empty result into a `ValueError`. Identification
    of the initial latent distribution is a property of the model, not of the
    estimator, so AF, CHS and AMN all enforce it. Anchors may come from
    `Normalizations`, `fixed_params`, or a connected equality constraint (all
    passed through to `check_identification`).

    Raises:
        ValueError: If the initial-anchor precheck reports any problem.

    """
    problems = check_identification(model_spec, fixed_params, constraints)
    if problems:
        msg = "ModelSpec is not identified:\n" + "\n".join(
            f"  - {problem}" for problem in problems
        )
        raise ValueError(msg)


def check_identification(
    model_spec: ModelSpec,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> list[str]:
    """Return problems with the initial-period affine anchoring of each factor.

    Empty list means every factor with an initial distribution has both a period-0
    scale anchor (a finite, nonzero loading) and a period-0 location anchor (a
    finite measurement intercept or a finite fixed initial-component latent mean).
    Anchors come from `Normalizations`, `fixed_params`, or equality constraints
    connected to a numerically pinned member; the pinned VALUE -- not just the key
    -- decides whether the category-specific anchor rule is met. A factor whose
    `normalizations is None` is checked the same way, against whatever fixed/equality
    anchors it has, rather than skipped.
    """
    pinned = _pinned_values(model_spec, fixed_params)
    pinned = _propagate_equality_values(pinned, _equality_components(constraints))
    problems: list[str] = []
    for factor_name, spec in model_spec.factors.items():
        if not spec.has_initial_distribution:
            continue
        if len(spec.measurements) == 0 or len(spec.measurements[0]) == 0:
            continue
        measures = spec.measurements[0]

        # The initial distribution is not produced by a transition, so its scale
        # and location orbits must be pinned directly. A loading pins scale only
        # when finite and nonzero (the CES simplex does NOT supply the location
        # anchor: f(x+c,i+c)=f(x,i)+c leaves a common-shift orbit). The location
        # anchor is a finite measurement intercept OR a finite fixed
        # initial-component latent mean (the CHS convention leaves the measurement
        # constants free).
        has_loading = any(
            _is_scale_anchor_value(pinned[key])
            for meas in measures
            if (key := ("loadings", 0, meas, factor_name)) in pinned
        )
        has_intercept = any(
            _is_location_anchor_value(pinned[key])
            for meas in measures
            if (key := ("controls", 0, meas, "constant")) in pinned
        ) or any(
            _is_location_anchor_value(pinned[key])
            for component in range(model_spec.n_mixtures)
            if (key := ("initial_states", 0, f"mixture_{component}", factor_name))
            in pinned
        )

        if not has_loading:
            problems.append(
                f"Factor '{factor_name}' period 0: no loading normalization "
                f"(scale anchor). The initial distribution is not produced by a "
                f"transition, so its scale must be pinned directly. Add a "
                f"loading=1 normalization for one period-0 measurement, a "
                f"`fixed_params` loading pin, or an equality constraint."
            )
        if not has_intercept:
            problems.append(
                f"Factor '{factor_name}' period 0: no location anchor. The initial "
                f"distribution is not produced by a transition, so its location "
                f"must be pinned directly. Add an intercept=0 normalization for one "
                f"period-0 measurement, a `fixed_params` intercept pin, a "
                f"`fixed_params` pin on one initial-component mean "
                f"(`initial_states`, the CHS convention), or an equality constraint."
            )
    return problems


def _initial_anchor_keys(
    factor_name: str,
    spec: FactorSpec,
    pinned: dict[tuple[object, ...], float],
    n_mixtures: int,
) -> tuple[list[tuple[str, int, str, str]], list[tuple[str, int, str, str]]]:
    """Return the pinned period-0 scale and location anchor keys for a factor.

    Scale keys are finite-nonzero period-0 loadings; location keys are finite
    period-0 measurement intercepts plus finite fixed initial-component means.
    """
    measures = spec.measurements[0]
    scale_keys = [
        key
        for meas in measures
        if (key := ("loadings", 0, meas, factor_name)) in pinned
        and _is_scale_anchor_value(pinned[key])
    ]
    location_keys = [
        key
        for meas in measures
        if (key := ("controls", 0, meas, "constant")) in pinned
        and _is_location_anchor_value(pinned[key])
    ] + [
        key
        for component in range(n_mixtures)
        if (key := ("initial_states", 0, f"mixture_{component}", factor_name)) in pinned
        and _is_location_anchor_value(pinned[key])
    ]
    return scale_keys, location_keys


def _scale_excess_message(factor_name: str, n_scale: int) -> list[str]:
    """Return the per-factor scale over-restriction message, if any."""
    if n_scale <= 1:
        return []
    return [
        f"Factor '{factor_name}' period 0: {n_scale} independent scale pins "
        f"(loadings), but the initial scale orbit has one direction. "
        f"{n_scale - 1} of them are testable restrictions, not normalizations -- "
        f"they constrain identified features and can move the estimate under "
        f"misspecification."
    ]


def _location_excess_message(factor_name: str, n_location: int) -> list[str]:
    """Return the per-factor location over-restriction message, if any."""
    if n_location <= 1:
        return []
    return [
        f"Factor '{factor_name}' period 0: {n_location} independent location pins "
        f"(measurement intercept and/or initial-component mean), but the initial "
        f"location orbit has one direction. {n_location - 1} of them are testable "
        f"restrictions, not normalizations."
    ]


def find_excess_initial_restrictions(
    model_spec: ModelSpec,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
) -> list[str]:
    """Report period-0 anchoring restrictions that exceed the affine orbit (Pro F4).

    `check_identification` is one-sided: it flags *too few* anchors. This is the
    complementary side -- it flags *too many*. Each factor's initial affine orbit
    has exactly two free directions, one scale and one location, so exactly one
    independent scale pin and one independent location pin are normalizations. Any
    further independent pin is a *testable* restriction: it constrains an identified
    feature of the model and can move the pseudo-true estimate under misspecification,
    rather than merely choosing units or origin. A second period-0 loading pin and a
    simultaneous measurement-intercept + initial-mean pin are the common cases.

    Returns one human-readable string per over-restricted scale/location block; an
    empty list means each orbit direction is pinned at most once. Pins tied together
    by an equality constraint count as a single restriction, so consistent equality
    chains never trip the check.

    Restricted-CES scale sharing (Pro F5): when the model uses a restricted-CES
    production (`log_ces` / `log_ces_af` / `log_ces_with_constant`), the production
    shares identify the RELATIVE scales of the factors the CES combines, so a single
    primitive scale anchor across the whole CES system is the only scale
    normalization (see `recover_primitive_ces_scales`). A per-factor scale anchor on a
    second CES factor is therefore a testable restriction. The scale accounting for
    those factors is reported once at the SYSTEM level rather than per factor. This is
    the WARN side only -- the hard gate (`fail_if_not_identified`) stays per-factor and
    conservative, so it never passes a genuinely scale-under-identified model on the
    strength of an assumed CES link.
    """
    pinned = _pinned_values(model_spec, fixed_params)
    components = _equality_components(constraints)
    component_of: dict[tuple[object, ...], int] = {}
    for index, component in enumerate(components):
        for key in component:
            component_of[key] = index

    def _independent_count(keys: Iterable[tuple[object, ...]]) -> int:
        return len({component_of.get(key, key) for key in keys})

    ces_system = _restricted_ces_scale_system(model_spec)

    messages: list[str] = []
    system_scale_keys: list[tuple[object, ...]] = []
    for factor_name, spec in model_spec.factors.items():
        if not spec.has_initial_distribution:
            continue
        if len(spec.measurements) == 0 or len(spec.measurements[0]) == 0:
            continue
        scale_keys, location_keys = _initial_anchor_keys(
            factor_name, spec, pinned, model_spec.n_mixtures
        )

        if factor_name in ces_system:
            # Scale accounting is deferred to the system-level check below, since
            # the CES restrictions pin one scale across all member factors.
            system_scale_keys.extend(scale_keys)
        else:
            messages.extend(
                _scale_excess_message(factor_name, _independent_count(scale_keys))
            )
        messages.extend(
            _location_excess_message(factor_name, _independent_count(location_keys))
        )

    n_system_scale = _independent_count(system_scale_keys)
    if n_system_scale > 1:
        members = ", ".join(f"'{name}'" for name in sorted(ces_system))
        messages.append(
            f"Restricted-CES scale system ({members}) period 0: {n_system_scale} "
            f"independent scale pins (loadings) across the system, but the CES "
            f"restrictions identify the relative scales from a SINGLE primitive "
            f"anchor. {n_system_scale - 1} of them are testable restrictions, not "
            f"normalizations."
        )
    return messages


def warn_if_overrestricted(
    model_spec: ModelSpec,
    fixed_params: pd.DataFrame | None = None,
    constraints: list[om.constraints.Constraint] | None = None,
    *,
    stacklevel: int = 3,
) -> None:
    """Emit a `UserWarning` for each period-0 restriction beyond a normalization.

    The mirror image of `fail_if_not_identified`: that gate fires when an initial
    orbit direction is UNPINNED (under-identification); this one fires when a
    direction is pinned MORE THAN ONCE. A surplus pin constrains an identified
    feature, so it is a testable restriction rather than a free normalization and can
    move the estimate under misspecification. Warning (not raising) keeps such models
    estimable while flagging the implicit restriction to the user.
    """
    for restriction in find_excess_initial_restrictions(
        model_spec, fixed_params, constraints
    ):
        warnings.warn(restriction, UserWarning, stacklevel=stacklevel)
