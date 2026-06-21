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

This is the estimator-agnostic core used by `validate_af_model`. It is exposed
for CHS/common tooling but is deliberately NOT wired into the default
`process_model` path, so it does not gate the application PyTask pipelines.

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

import optimagic as om
import pandas as pd

from skillmodels.common.constraints import _equality_constraint_loc
from skillmodels.common.model_spec import ModelSpec


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
            for meas in loadings:
                keys.add(("loadings", period, meas, factor))
        for period, intercepts in enumerate(norms.intercepts or ()):
            for meas in intercepts:
                keys.add(("controls", period, meas, "constant"))
    return keys


def _anchored_param_keys(
    fixed_params: pd.DataFrame | None,
    constraints: list[om.constraints.Constraint] | None,
    anchor_sources: set[tuple[object, ...]],
) -> set[tuple[object, ...]]:
    """Collect the params anchored outside the period-0 normalization maps.

    A `fixed_params` row pins its parameter to a value. A `select_by_loc`
    equality group only *transfers* an anchor: it ties its members equal, which
    fixes their common value to a number ONLY when the group already contains a
    numerically fixed or normalized member (`anchor_sources`). An equality among
    otherwise-free loadings leaves their common scale free and is NOT an anchor
    (Pro F3). Keys are the params' index tuples `(category, period, name1,
    name2)`.
    """
    keys: set[tuple[object, ...]] = set()
    if fixed_params is not None:
        keys.update(tuple(idx) for idx in fixed_params.index)
    for constraint in constraints or []:
        loc = _equality_constraint_loc(constraint)
        if loc is None:
            continue
        members = {tuple(idx) for idx in loc}
        if members & anchor_sources:
            keys.update(members)
    return keys


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
    fixed_keys: set[tuple[object, ...]] = set()
    if fixed_params is not None:
        fixed_keys = {tuple(idx) for idx in fixed_params.index}
    anchor_sources = fixed_keys | _normalized_keys(model_spec)
    anchored = _anchored_param_keys(fixed_params, constraints, anchor_sources)
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

        has_loading = bool(norms.loadings and norms.loadings[0]) or any(
            ("loadings", 0, meas, factor_name) in anchored for meas in measures
        )
        # Every factor with an initial distribution needs an absolute initial
        # location anchor mu_theta,0,1=0. The CES simplex does NOT supply it
        # (Pro F1: f(x+c,i+c)=f(x,i)+c leaves a common-shift orbit); the simplex
        # only replaces the *cross-period* location alternative, which this
        # period-0 precheck does not police.
        has_intercept = bool(norms.intercepts and norms.intercepts[0]) or any(
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
