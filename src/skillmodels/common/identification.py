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

KNOWN LIMITATION -- restricted-CES scale is per-COMPONENT, not per-factor (Pro
F2). For a restricted-CES skill factor, the production restrictions identify the
relative skill/investment scales, so only ONE primitive scale anchor is needed
across the connected skill-investment system (on lambda_theta,0,1 OR
lambda_I,0,1); a second loading pin is then a testable restriction, not a
normalization. This precheck does not build the production-graph connected
components: it checks each factor's OWN initial loading anchor. In the standard
setup -- investment is endogenous with `has_initial_distribution=False`, so it is
skipped here -- this happens to require exactly one scale anchor (on skills),
which is correct. But it does NOT flag an additional lambda_I,0,1 pin as an
over-normalizing testable restriction, and for a CES model where the input factor
DOES carry an initial distribution it would over-require (one anchor per factor).
The connected-component scale accounting belongs to the full identification
diagnostic.
"""

import math

import optimagic as om
import pandas as pd

from skillmodels.common.constraints import _equality_constraint_loc
from skillmodels.common.model_spec import ModelSpec


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
