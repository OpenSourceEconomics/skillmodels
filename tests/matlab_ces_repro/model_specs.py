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
    *,
    pin_first_intercept: bool = True,
) -> Normalizations:
    """Fix the first measurement's loading to 1 and (optionally) intercept to 0.

    Args:
        per_period: Tuple of measurement variable names.
        active_periods: Periods in which the factor is measured at all.
        normalize_periods: Periods in which to apply the normalisation. By
            default equals ``active_periods``. Set it to a subset (e.g.
            ``(0,)``) to match MATLAB's convention of normalising only at
            the initial period and letting the production function pin
            the scale of the factor thereafter.
        pin_first_intercept: When True (default), pin the first
            measurement's intercept to 0 in the normalised periods. Set to
            False to match MATLAB's identification, which pins only the
            first loading and identifies the latent location via the
            latent factor mean instead.
    """
    if normalize_periods is None:
        normalize_periods = active_periods
    first = per_period[0]
    return Normalizations(
        loadings=tuple(
            {first: 1} if t in normalize_periods else {} for t in range(_N_PERIODS)
        ),
        intercepts=tuple(
            {first: 0} if (t in normalize_periods and pin_first_intercept) else {}
            for t in range(_N_PERIODS)
        ),
    )


def _common_factor_specs(
    *, match_matlab_normalisation: bool = False
) -> dict[str, FactorSpec]:
    """FactorSpecs shared by the CES and translog variants.

    Args:
        match_matlab_normalisation: When True, drop the first-intercept
            pin at period 0 for MC and MN, mirroring MATLAB's choice to
            identify those factors' location via the latent mean rather
            than the measurement intercept.
    """
    pin = not match_matlab_normalisation
    return {
        "MC": FactorSpec(
            measurements=_measurements(MC_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(
                MC_MEASURES, active_periods=(0,), pin_first_intercept=pin
            ),
            transition_function="linear",
            has_production_shock=False,
        ),
        "MN": FactorSpec(
            measurements=_measurements(MN_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(
                MN_MEASURES, active_periods=(0,), pin_first_intercept=pin
            ),
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


def _common_fixed_rows(
    *, pin_investment_eq_constant: bool = True
) -> list[tuple[tuple[str, int, str, str], float]]:
    """Fixed-parameter rows for time-invariant MC / MN and the investment eq.

    - MC and MN are time-invariant with ``has_production_shock=False``: identity
      transition (self-coefficient 1, all others 0). No shock SD exists because
      the factor has no production shock in the AF params index.
    - Investment is endogenous (``is_endogenous=True``) with
      ``has_initial_distribution=False``; its equation lives in the
      ``investment_eq`` block.

    Args:
        pin_investment_eq_constant: When True (default; matches MATLAB CES),
            pin the investment equation's constant to 0. When False (matches
            MATLAB translog), leave it free.
    """
    rows: list[tuple[tuple[str, int, str, str], float]] = []
    for t in range(_N_PERIODS - 1):
        for factor in ("MC", "MN"):
            rows.append((("transition", t, factor, factor), 1.0))
            for other in ("skills", "MC", "MN"):
                if other != factor:
                    rows.append((("transition", t, factor, other), 0.0))
            rows.append((("transition", t, factor, "constant"), 0.0))
        if pin_investment_eq_constant:
            rows.append((("investment_eq", t, "investment", "constant"), 0.0))
    return rows


def build_ces_model(*, match_matlab_normalisation: bool = False) -> BuiltModel:
    """Build the MATLAB CES variant.

    ``skills`` uses ``log_ces`` over all latent factors (skills, MC, MN,
    investment); cross-factor gammas for ``MC`` and ``MN`` are pinned to
    ``0`` so the CES reduces to the MATLAB 2-input form on
    ``(skills, investment)``.

    Args:
        match_matlab_normalisation: When True, drop the first-intercept
            pins at period 0 for skills, MC, MN, and instead pin the
            corresponding latent factor means and unit-variance Cholesky
            entries via fixed_params. This matches MATLAB's identification
            (latent location/scale fixed; measurement intercepts free) and
            makes period-0 parameter values directly comparable cell by
            cell. When False (default), use skillmodels' standard
            identification (first intercept = 0, latent mean free).
    """
    pin_intercept = not match_matlab_normalisation
    factors: dict[str, FactorSpec] = {
        "skills": FactorSpec(
            measurements=_measurements(SKILL_MEASURES),
            # MATLAB normalises skills only at period 0; the production
            # function ties the scale of skills at later periods.
            normalizations=_normalizations(
                SKILL_MEASURES,
                normalize_periods=(0,),
                pin_first_intercept=pin_intercept,
            ),
            transition_function="log_ces",
        ),
        **_common_factor_specs(match_matlab_normalisation=match_matlab_normalisation),
    }

    rows = _common_fixed_rows()
    if match_matlab_normalisation:
        rows.extend(_matlab_initial_normalisation_rows())
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


def _matlab_initial_normalisation_rows() -> list[
    tuple[tuple[str, int, str, str], float]
]:
    """Pin the period-0 latent means to MATLAB's identification choice.

    MATLAB identifies the location of `skills`, `MC`, and `MN` at period 0
    by pinning their latent means to 0; measurement intercepts are then
    free. The latent covariance is *not* pinned (MATLAB estimates 4 SDs
    and 6 correlations among `(skills, MC, MN, log_income)`). The
    `Sigma_Omega = I_4` constant in MATLAB's workspace is the
    standardised integration-grid covariance, not a pin on the actual
    latent covariance.
    """
    rows: list[tuple[tuple[str, int, str, str], float]] = []
    for factor in ("skills", "MC", "MN"):
        rows.append((("initial_states", 0, "mixture_0", factor), 0.0))
    return rows


def build_translog_model(*, match_matlab_normalisation: bool = False) -> BuiltModel:
    # Translog already matches MATLAB's identification by default (first
    # measurement intercepts pinned to 0; latent means free), so this flag
    # is a no-op here. We accept it for API symmetry with build_ces_model.
    del match_matlab_normalisation
    match_matlab_normalisation = False
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
    pin_intercept = not match_matlab_normalisation
    # MATLAB translog identification (verified from
    # AF_Application_One_Normal_Translog.m::likelihood_01/12):
    # at every period, the first skill measurement loading is pinned to 1
    # and the first skill intercept is pinned (to 0 at period 0; to
    # ``mu_skills_norm_0`` at periods 1+, which translog sets to 0). The
    # first investment loading is also pinned to 1 and the first
    # investment intercept to 0 at period 1 (the only period at which
    # investment has measurements). Apply the same per-period
    # normalisation to all active periods to match.
    factors: dict[str, FactorSpec] = {
        "skills": FactorSpec(
            measurements=_measurements(SKILL_MEASURES),
            normalizations=_normalizations(
                SKILL_MEASURES,
                pin_first_intercept=pin_intercept,
            ),
            transition_function="translog",
        ),
        **_common_factor_specs(match_matlab_normalisation=match_matlab_normalisation),
    }
    # Override investment to normalise at period 1 (its only active period)
    # to match MATLAB's translog convention.
    inv_factor = factors["investment"]
    factors["investment"] = type(inv_factor)(
        measurements=inv_factor.measurements,
        normalizations=_normalizations(
            INV_MEASURES,
            active_periods=(1,),
            normalize_periods=(1,),
            pin_first_intercept=pin_intercept,
        ),
        transition_function=inv_factor.transition_function,
        is_endogenous=inv_factor.is_endogenous,
        has_initial_distribution=inv_factor.has_initial_distribution,
        has_production_shock=getattr(inv_factor, "has_production_shock", True),
    )

    # Translog has a free investment-equation constant (CES pins it to 0).
    rows = _common_fixed_rows(pin_investment_eq_constant=False)
    if match_matlab_normalisation:
        rows.extend(_matlab_initial_normalisation_rows())
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
