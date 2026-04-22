"""ModelSpec builders that mirror the MATLAB AF CES and translog runs.

The MATLAB setup has four factors in the initial joint distribution:
``skills`` (latent, non-trivial transition), ``MC`` and ``MN`` (latent,
time-invariant), and ``log_income`` (observed, enters the investment
equation). A fifth factor, ``investment``, is endogenous.

Both builders return a ``(ModelSpec, fixed_params)`` pair. The
``fixed_params`` DataFrame pins the parameters that need to be zeroed to
match the MATLAB production functions.
"""

from dataclasses import dataclass

import pandas as pd

from skillmodels.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.types import EstimationOptions

from .load_cnlsy import (
    INCOME_MEASURE,
    INV_MEASURES,
    MC_MEASURES,
    MN_MEASURES,
    SKILL_MEASURES,
)

_N_PERIODS = 3
_INV_PERIODS = (0, 1)


@dataclass(frozen=True)
class BuiltModel:
    """A ``ModelSpec`` plus the ``fixed_params`` DataFrame it expects."""

    model_spec: ModelSpec
    fixed_params: pd.DataFrame


def _measurements(
    per_period: tuple[str, ...], active_periods: tuple[int, ...] = (0, 1, 2)
) -> tuple[tuple[str, ...], ...]:
    """Build per-period measurement tuples, empty where the factor is inactive."""
    return tuple(per_period if t in active_periods else () for t in range(_N_PERIODS))


def _normalizations(
    per_period: tuple[str, ...], active_periods: tuple[int, ...] = (0, 1, 2)
) -> Normalizations:
    """Fix the first measurement's loading to 1 and its intercept to 0."""
    first = per_period[0]
    return Normalizations(
        loadings=tuple(
            {first: 1} if t in active_periods else {} for t in range(_N_PERIODS)
        ),
        intercepts=tuple(
            {first: 0} if t in active_periods else {} for t in range(_N_PERIODS)
        ),
    )


def _common_factor_specs() -> dict[str, FactorSpec]:
    """FactorSpecs shared by the CES and translog variants."""
    return {
        "MC": FactorSpec(
            measurements=_measurements(MC_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MC_MEASURES, active_periods=(0,)),
            transition_function="linear",
        ),
        "MN": FactorSpec(
            measurements=_measurements(MN_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MN_MEASURES, active_periods=(0,)),
            transition_function="linear",
        ),
        "investment": FactorSpec(
            measurements=_measurements(INV_MEASURES, active_periods=_INV_PERIODS),
            normalizations=_normalizations(INV_MEASURES, active_periods=_INV_PERIODS),
            is_endogenous=True,
            transition_function="linear",
        ),
    }


def _common_fixed_rows() -> list[tuple[tuple[str, int, str, str], float]]:
    """Fixed-parameter rows for time-invariant MC / MN and small shocks."""
    rows: list[tuple[tuple[str, int, str, str], float]] = []
    for t in range(_N_PERIODS - 1):
        # MC and MN are time-invariant: identity transition, near-zero shock.
        for factor in ("MC", "MN"):
            rows.append((("transition", t, factor, factor), 1.0))
            for other in ("skills", "MC", "MN", "investment"):
                if other != factor:
                    rows.append((("transition", t, factor, other), 0.0))
            rows.append((("transition", t, factor, "constant"), 0.0))
            rows.append((("shock_sds", t, factor, "-"), 1e-3))
    return rows


def build_ces_model() -> BuiltModel:
    """Build the MATLAB CES variant.

    ``skills`` uses ``log_ces`` over all latent factors (skills, MC, MN,
    investment); cross-factor gammas for ``MC`` and ``MN`` are pinned to
    ``0`` so the CES reduces to the MATLAB 2-input form on
    ``(skills, investment)``.
    """
    factors: dict[str, FactorSpec] = {
        "skills": FactorSpec(
            measurements=_measurements(SKILL_MEASURES),
            normalizations=_normalizations(SKILL_MEASURES),
            transition_function="log_ces",
        ),
        **_common_factor_specs(),
    }

    rows = _common_fixed_rows()
    for t in range(_N_PERIODS - 1):
        # Pin cross-factor gammas to 0: only skills and investment enter CES.
        rows.append((("transition", t, "skills", "MC"), 0.0))
        rows.append((("transition", t, "skills", "MN"), 0.0))

    fixed_idx = pd.MultiIndex.from_tuples(
        [r[0] for r in rows],
        names=["category", "period", "name1", "name2"],
    )
    fixed_params = pd.DataFrame(
        {"value": [r[1] for r in rows]},
        index=fixed_idx,
    )

    model = ModelSpec(
        factors=factors,
        observed_factors=(INCOME_MEASURE,),
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )
    return BuiltModel(model_spec=model, fixed_params=fixed_params)


def build_translog_model() -> BuiltModel:
    """Build the MATLAB translog variant.

    ``skills`` uses skillmodels' ``translog`` (polynomial in factors with
    squares and interactions). MATLAB's 2-input translog
    ``f = A + rho*log(theta) + delta*log(X) + phi*log(theta)*log(X)`` has
    no squared terms, so we pin:

    - all linear coefficients on MC / MN / investment off-factors not
      matching the MATLAB inputs (skills, investment) to 0;
    - all squared coefficients to 0 (the MATLAB form has no squares);
    - all interaction coefficients involving MC or MN to 0.

    The remaining free translog parameters are ``skills`` (= rho),
    ``investment`` (= delta), ``skills * investment`` (= phi), and
    ``constant`` (= A).
    """
    factors: dict[str, FactorSpec] = {
        "skills": FactorSpec(
            measurements=_measurements(SKILL_MEASURES),
            normalizations=_normalizations(SKILL_MEASURES),
            transition_function="translog",
        ),
        **_common_factor_specs(),
    }

    rows = _common_fixed_rows()
    all_factors = ("skills", "MC", "MN", "investment")
    keep_linear = {"skills", "investment"}
    for t in range(_N_PERIODS - 1):
        # Zero linear coefficients on non-input factors.
        for factor in all_factors:
            if factor not in keep_linear:
                rows.append((("transition", t, "skills", factor), 0.0))
        # Zero all squared coefficients (MATLAB translog has no squares).
        for factor in all_factors:
            rows.append((("transition", t, "skills", f"{factor} ** 2"), 0.0))
        # Zero every interaction that isn't skills * investment.
        combinations = [
            (a, b) for i, a in enumerate(all_factors) for b in all_factors[i + 1 :]
        ]
        for a, b in combinations:
            if {a, b} != {"skills", "investment"}:
                rows.append((("transition", t, "skills", f"{a} * {b}"), 0.0))

    fixed_idx = pd.MultiIndex.from_tuples(
        [r[0] for r in rows],
        names=["category", "period", "name1", "name2"],
    )
    fixed_params = pd.DataFrame(
        {"value": [r[1] for r in rows]},
        index=fixed_idx,
    )

    model = ModelSpec(
        factors=factors,
        observed_factors=(INCOME_MEASURE,),
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )
    return BuiltModel(model_spec=model, fixed_params=fixed_params)
