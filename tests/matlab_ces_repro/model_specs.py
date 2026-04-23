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
    per_period: tuple[str, ...],
    active_periods: tuple[int, ...] = (0, 1, 2),
    normalize_periods: tuple[int, ...] | None = None,
) -> Normalizations:
    """Fix the first measurement's loading to 1 and intercept to 0.

    Args:
        per_period: Tuple of measurement variable names.
        active_periods: Periods in which the factor is measured at all.
        normalize_periods: Periods in which to apply the normalisation. By
            default equals ``active_periods``. Set it to a subset (e.g.
            ``(0,)``) to match MATLAB's convention of normalising only at
            the initial period and letting the production function pin
            the scale of the factor thereafter.
    """
    if normalize_periods is None:
        normalize_periods = active_periods
    first = per_period[0]
    return Normalizations(
        loadings=tuple(
            {first: 1} if t in normalize_periods else {} for t in range(_N_PERIODS)
        ),
        intercepts=tuple(
            {first: 0} if t in normalize_periods else {} for t in range(_N_PERIODS)
        ),
    )


def _common_factor_specs() -> dict[str, FactorSpec]:
    """FactorSpecs shared by the CES and translog variants."""
    return {
        "MC": FactorSpec(
            measurements=_measurements(MC_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MC_MEASURES, active_periods=(0,)),
            transition_function="linear",
            has_production_shock=False,
        ),
        "MN": FactorSpec(
            measurements=_measurements(MN_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MN_MEASURES, active_periods=(0,)),
            transition_function="linear",
            has_production_shock=False,
        ),
        "investment": FactorSpec(
            # MATLAB places period-0 investment measurements in transition_01
            # and reconstructs investment deterministically from the state
            # factors + log_income at each period. Mirror that by keeping
            # investment out of the initial distribution (the has_initial_
            # distribution flag) and restricting its skillmodels measurements
            # to period 1 only.
            measurements=_measurements(INV_MEASURES, active_periods=(1,)),
            # MATLAB does not normalise the investment measurement model at
            # any period (all three loadings and intercepts are free); the
            # investment equation pins the scale of investment via the
            # coefficients on (skills, MC, MN, log_income). We follow
            # MATLAB's convention to make the param translation a direct
            # copy.
            normalizations=_normalizations(
                INV_MEASURES,
                active_periods=(1,),
                normalize_periods=(),
            ),
            transition_function="linear",
            is_endogenous=True,
            has_initial_distribution=False,
        ),
    }


def _common_fixed_rows() -> list[tuple[tuple[str, int, str, str], float]]:
    """Fixed-parameter rows for time-invariant MC / MN and the investment eq.

    - MC and MN are time-invariant with ``has_production_shock=False``: identity
      transition (self-coefficient 1, all others 0). No shock SD exists because
      the factor has no production shock in the AF params index.
    - Investment is endogenous (``is_endogenous=True``) with
      ``has_initial_distribution=False``; its equation lives in the
      ``investment_eq`` block. We pin its constant to 0 to match
      MATLAB's ``log(inv_t) = a_theta * theta + a_mc * MC + a_mn * MN +
      a_y * log_income + eta_I``.
    """
    rows: list[tuple[tuple[str, int, str, str], float]] = []
    for t in range(_N_PERIODS - 1):
        for factor in ("MC", "MN"):
            rows.append((("transition", t, factor, factor), 1.0))
            # MC / MN have linear transitions whose param names cover the
            # non-endogenous latents only after the is_endogenous flag on
            # investment takes it out of latent_factors for the transition
            # params index. Pin cross-coefficients to zero.
            for other in ("skills", "MC", "MN"):
                if other != factor:
                    rows.append((("transition", t, factor, other), 0.0))
            rows.append((("transition", t, factor, "constant"), 0.0))
        # Investment equation: no intercept (matches MATLAB).
        rows.append((("investment_eq", t, "investment", "constant"), 0.0))
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
            # MATLAB normalises skills only at period 0; the production
            # function ties the scale of skills at later periods.
            normalizations=_normalizations(SKILL_MEASURES, normalize_periods=(0,)),
            transition_function="log_ces",
        ),
        **_common_factor_specs(),
    }

    rows = _common_fixed_rows()
    for t in range(_N_PERIODS - 1):
        # MATLAB's CES is a 2-input form on (skills, investment). Pin all
        # other factor gammas in skills' production function to 0 so our
        # log_ces matches MATLAB's form exactly. In particular, MATLAB
        # *does not* use log_income as an input to the skills CES (it only
        # enters the investment equation). Leaving its gamma free would
        # make our model strictly richer and render the log-likelihood
        # comparison against MATLAB's optimum non-apples-to-apples.
        rows.append((("transition", t, "skills", "MC"), 0.0))
        rows.append((("transition", t, "skills", "MN"), 0.0))
        rows.append((("transition", t, "skills", INCOME_MEASURE), 0.0))

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
            # Same MATLAB convention as in CES: skills normalised only at
            # period 0; scale at later periods pinned by the production
            # function.
            normalizations=_normalizations(SKILL_MEASURES, normalize_periods=(0,)),
            transition_function="translog",
        ),
        **_common_factor_specs(),
    }

    rows = _common_fixed_rows()
    # MATLAB's translog is also a 2-input form on (skills, investment) with no
    # log_income term, so we pin log_income's translog coefficients in
    # exactly the same way as MC / MN. Leaving them free would make our
    # translog richer than MATLAB's and bias the comparison.
    all_factors_including_observed = (
        "skills",
        "MC",
        "MN",
        "investment",
        INCOME_MEASURE,
    )
    keep_linear = {"skills", "investment"}
    for t in range(_N_PERIODS - 1):
        # Zero linear coefficients on non-input factors.
        for factor in all_factors_including_observed:
            if factor not in keep_linear:
                rows.append((("transition", t, "skills", factor), 0.0))
        # Zero all squared coefficients (MATLAB translog has no squares).
        for factor in all_factors_including_observed:
            rows.append((("transition", t, "skills", f"{factor} ** 2"), 0.0))
        # Zero every interaction that isn't skills * investment.
        combinations = [
            (a, b)
            for i, a in enumerate(all_factors_including_observed)
            for b in all_factors_including_observed[i + 1 :]
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
