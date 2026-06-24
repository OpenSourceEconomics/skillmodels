"""Transition-aware identification anchor diagnostics (audit F6/F7/F8).

The initial-period latent distribution is not produced by any transition, so
its affine orbit (scale + location) must be pinned directly. `check_identification`
verifies that, dispatching on each factor's transition type:

Every factor with an initial distribution needs BOTH a loading (scale) anchor and
an intercept (location) anchor at the initial period, regardless of transition.
The CES simplex `sum_i gamma_i = 1` does NOT supply the initial location anchor
(Pro F1: plain CES obeys f(x+c,i+c)=f(x,i)+c, so a common shift of all latent
inputs leaves observables unchanged while preserving the simplex); it only
replaces the *cross-period* skills-location alternative, which this period-0
precheck does not police.

An anchor may come from a `Normalizations` map, a `fixed_params` pin, or a
`select_by_loc` equality constraint that is connected to a numerically fixed
member. Periods t>0 are intentionally not checked here: an unrestricted
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


def _normalized_keys(model_spec: ModelSpec) -> set[tuple[object, ...]]:
    """Return the param keys pinned by `Normalizations` across all factor-periods.

    Loadings map to `("loadings", period, meas, factor)` and intercepts to
    `("controls", period, meas, "constant")` -- the conventions used by the
    params MultiIndex.
    """
    keys: set[tuple[object, ...]] = set()
    for factor, spec in model_spec.factors.items():
        norms = spec.normalizations
        if norms is None:
            continue
        for period, loadings in enumerate(norms.loadings or ()):
            for meas, value in loadings.items():
                if _is_scale_anchor_value(value):
                    keys.add(("loadings", period, meas, factor))
        for period, intercepts in enumerate(norms.intercepts or ()):
            for meas, value in intercepts.items():
                if _is_location_anchor_value(value):
                    keys.add(("controls", period, meas, "constant"))
    return keys


def _valid_fixed_keys(fixed_params: pd.DataFrame | None) -> set[tuple[object, ...]]:
    """Return the `fixed_params` keys that are VALID anchors by value (Pro F1).

    Loading pins count only when finite and nonzero; intercept (controls) pins
    count when finite; other categories pass through unchecked (they are not
    consulted as period-0 anchors).
    """
    keys: set[tuple[object, ...]] = set()
    if fixed_params is None:
        return keys
    values = fixed_params["value"].to_numpy()
    for idx, value in zip(fixed_params.index, values, strict=True):
        category = idx[0]
        if category == "loadings":
            if _is_scale_anchor_value(float(value)):
                keys.add(tuple(idx))
        elif category == "controls":
            if _is_location_anchor_value(float(value)):
                keys.add(tuple(idx))
        else:
            keys.add(tuple(idx))
    return keys


def _equality_closure(
    constraints: list[om.constraints.Constraint] | None,
    anchor_sources: set[tuple[object, ...]],
) -> set[tuple[object, ...]]:
    """Propagate anchors transitively through equality groups (Pro F2).

    A `select_by_loc` equality group ties its members equal. A group fixes its
    members to a number only when it is connected -- possibly through a chain of
    groups -- to a numerically fixed or normalized member. Iterate to a fixed
    point so that A fixed, A=B, B=C all anchor C regardless of how the single
    logical component is split across constraints.
    """
    groups = [
        {tuple(idx) for idx in loc}
        for constraint in constraints or []
        if (loc := _equality_constraint_loc(constraint)) is not None
    ]
    anchored = set(anchor_sources)
    changed = True
    while changed:
        changed = False
        for group in groups:
            if group & anchored and not group <= anchored:
                anchored |= group
                changed = True
    return anchored


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

    Empty list means every factor with an initial distribution has the anchors
    its transition requires (see the module docstring for the per-transition
    rules). Factors whose `normalizations is None` are skipped -- that is a
    separate, estimator-specific error.
    """
    # Anchor sources are the VALID (by-value) fixed pins and normalizations;
    # equality groups then transfer those anchors transitively.
    anchor_sources = _valid_fixed_keys(fixed_params) | _normalized_keys(model_spec)
    anchored = _equality_closure(constraints, anchor_sources)
    problems: list[str] = []
    for factor_name, spec in model_spec.factors.items():
        norms = spec.normalizations
        if norms is None:
            continue
        if not spec.has_initial_distribution:
            continue
        if len(spec.measurements) == 0 or len(spec.measurements[0]) == 0:
            continue
        measures = spec.measurements[0]

        # Every factor with an initial distribution needs an absolute initial
        # location anchor mu_theta,0,1=0 (the CES simplex does NOT supply it --
        # f(x+c,i+c)=f(x,i)+c leaves a common-shift orbit) and a scale anchor.
        # Both must be VALID by value (Pro F1) -- normalizations and fixed pins
        # are filtered above, so a plain set-membership test suffices here.
        has_loading = any(
            ("loadings", 0, meas, factor_name) in anchored for meas in measures
        )
        has_intercept = any(
            ("controls", 0, meas, "constant") in anchored for meas in measures
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
                f"Factor '{factor_name}' period 0: no intercept normalization "
                f"(location anchor). The initial distribution is not produced by "
                f"a transition, so its location must be pinned directly. Add an "
                f"intercept=0 normalization for one period-0 measurement, a "
                f"`fixed_params` intercept pin, or an equality constraint."
            )
    return problems
