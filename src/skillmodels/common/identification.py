"""Transition-aware identification anchor diagnostics (audit F6/F7/F8).

The initial-period latent distribution is not produced by any transition, so
its affine orbit (scale + location) must be pinned directly. `check_identification`
verifies that, dispatching on each factor's transition type:

- trans-log (and other free-level transitions): need both a loading (scale) and
  an intercept (location) anchor at the initial period;
- plain `log_ces` / `log_ces_af`: the simplex `sum_i gamma_i = 1` IS the
  location restriction (Freyberger 2025), so only a loading anchor is needed;
- `log_ces_with_constant`: the free additive level means location must still be
  pinned by an intercept anchor.

An anchor may come from a `Normalizations` map, a `fixed_params` pin, or a
`select_by_loc` equality constraint. Periods t>0 are intentionally not checked
here: the transition can legitimately propagate the anchor, so verifying them
needs a full transition-aware local-rank diagnostic (a larger, separate piece).

This is the estimator-agnostic core used by `validate_af_model`. It is exposed
for CHS/common tooling but is deliberately NOT wired into the default
`process_model` path, so it does not gate the application PyTask pipelines.
"""

import optimagic as om
import pandas as pd

from skillmodels.common.constraints import _equality_constraint_loc
from skillmodels.common.model_spec import ModelSpec

# Transitions that bake in the CES simplex gamma_1+...+gamma_n=1 (no free
# additive level). The simplex IS the skills-location restriction (Freyberger
# Assumption a:ageinvariant_technology_skills_ces(b)), so these factors do NOT
# additionally need a measurement-intercept (location) anchor -- requiring one
# would impose both location alternatives and over-restrict (audit F8).
# `log_ces_with_constant` is deliberately excluded: its free additive constant
# absorbs the level, so location must still be pinned by an intercept anchor.
_SIMPLEX_LOCATION_TRANSITIONS = frozenset({"log_ces", "log_ces_af"})


def _anchored_param_keys(
    fixed_params: pd.DataFrame | None,
    constraints: list[om.constraints.Constraint] | None,
) -> set[tuple[object, ...]]:
    """Collect the params anchored outside the normalization maps.

    A `fixed_params` row pins its parameter to a value; a member of a
    `select_by_loc` equality group is tied to the rest of its group (treated
    here, conservatively, as anchored). Both supply an affine anchor a
    `Normalizations` map could otherwise be relied on for. Keys are the params'
    index tuples `(category, period, name1, name2)`.
    """
    keys: set[tuple[object, ...]] = set()
    if fixed_params is not None:
        keys.update(tuple(idx) for idx in fixed_params.index)
    for constraint in constraints or []:
        loc = _equality_constraint_loc(constraint)
        if loc is not None:
            keys.update(tuple(idx) for idx in loc)
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
    anchored = _anchored_param_keys(fixed_params, constraints)
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
        # The simplex of plain log_ces / log_ces_af already pins the location, so
        # such factors need no intercept anchor (audit F8); treat it as present.
        # @pro: the simplex gamma_1+...+gamma_n=1 is a per-PERIOD location
        # restriction, but here I only relax the period-0 intercept requirement.
        # Is relaxing period 0 (the only period this function checks) the correct
        # granularity, and is applying it to log_ces_af identical to log_ces right?
        simplex_locates = spec.transition_function in _SIMPLEX_LOCATION_TRANSITIONS
        has_intercept = (
            simplex_locates
            or bool(norms.intercepts and norms.intercepts[0])
            or any(("controls", 0, meas, "constant") in anchored for meas in measures)
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
