"""Contains transition functions and corresponding helper functions.

Below the signature and purpose of a transition function and its helper
functions is explained with a transition function called example_func:
>

**example_func(** *states, params**)**:

    The actual transition function.

Args:
        * states: 1d numpy array of length n_all_factors
        * params: 1d numpy array with coefficients specific to this transition function

Return:
        * float
**names_example_func(** *factors* **)**:

    Generate a list of names for the params of the transition function.

    The names will be used to construct index tuples in the following way:

    ('transition', period, factor, NAME)

The transition functions have to be JAX jittable and differentiable. However, they
should not be jitted yet.

"""

import functools
from itertools import combinations

import jax
import jax.numpy as jnp
import optimagic as om
from jax import Array

from skillmodels.common.fixed_constraint import FixedConstraintWithValue
from skillmodels.common.selector import select_by_loc


def linear(states: Array, params: Array) -> Array:
    """Linear production function where the constant is the last parameter."""
    constant = params[-1]
    betas = params[:-1]
    return jnp.dot(states, betas) + constant


def params_linear(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for linear transition function."""
    return [*factors, "constant"]


def identity_constraints_linear(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[FixedConstraintWithValue]:
    """Identity constraints for linear transition function."""
    constraints: list[FixedConstraintWithValue] = []
    for regressor in params_linear(all_factors):
        val = 1.0 if factor == regressor else 0.0
        loc = ("transition", aug_period, factor, regressor)
        constraints.append(FixedConstraintWithValue(loc=loc, value=val))
    return constraints


def translog(states: Array, params: Array) -> Array:
    """Translog transition function.

    The name is a convention in the skill formation literature even though the function
    is better described as a linear in parameters transition function with squares and
    interaction terms of the states.

    This is the general-library specification: parameters are enumerated over ALL
    factors in `all_factors` (latent AND observed). Observed factors (e.g. income)
    therefore enter the production function with their own free linear, square and
    interaction coefficients. This is by design for the CHS estimator. For an AF
    production function that matches the paper's equation (6) (skill + investment
    only, NO squares), use `translog_af` and pass only the production factors.

    """
    nfac = len(states)
    constant = params[-1]
    lin_beta = params[:nfac]
    square_beta = params[nfac : 2 * nfac]
    inter_beta = params[2 * nfac : -1]

    res = jnp.dot(states, lin_beta)
    res += jnp.dot(states**2, square_beta)
    for p, (a, b) in zip(inter_beta, combinations(range(nfac), 2), strict=False):
        res += p * states[a] * states[b]
    res += constant
    return res


def params_translog(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for the translog production function."""
    return (
        list(factors)
        + [f"{factor} ** 2" for factor in factors]
        + [f"{a} * {b}" for a, b in combinations(factors, 2)]
        + ["constant"]
    )


def identity_constraints_translog(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[FixedConstraintWithValue]:
    """Identity constraints for translog transition function."""
    constraints: list[FixedConstraintWithValue] = []
    for regressor in params_translog(all_factors):
        val = 1.0 if factor == regressor else 0.0
        loc = ("transition", aug_period, factor, regressor)
        constraints.append(FixedConstraintWithValue(loc=loc, value=val))
    return constraints


def translog_af(states: Array, params: Array) -> Array:
    """AF (2020) production translog, equation (6): NO square terms.

    Implements `a_t + sum_i beta_i * states_i + sum_{i<j} delta_ij * states_i
    * states_j`, i.e. linear terms plus pairwise interactions only. Unlike the
    general-library `translog`, it omits the squared-factor terms, matching AF
    eq. (6) `a_t + g1 ln theta + g2 ln I + g3 ln theta ln I` for
    (skill, investment).

    Pass ONLY the production factors (skill + investment) as `states`; observed
    factors such as income must not enter the production function.
    """
    nfac = len(states)
    constant = params[-1]
    lin_beta = params[:nfac]
    inter_beta = params[nfac:-1]
    res = jnp.dot(states, lin_beta)
    for p, (a, b) in zip(inter_beta, combinations(range(nfac), 2), strict=False):
        res += p * states[a] * states[b]
    res += constant
    return res


def params_translog_af(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for `translog_af` (linear + interactions + constant)."""
    return (
        list(factors)
        + [f"{a} * {b}" for a, b in combinations(factors, 2)]
        + ["constant"]
    )


def identity_constraints_translog_af(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[FixedConstraintWithValue]:
    """Identity constraints for `translog_af` (carry-forward aug periods)."""
    constraints: list[FixedConstraintWithValue] = []
    for regressor in params_translog_af(all_factors):
        val = 1.0 if factor == regressor else 0.0
        loc = ("transition", aug_period, factor, regressor)
        constraints.append(FixedConstraintWithValue(loc=loc, value=val))
    return constraints


def log_ces(states: Array, params: Array) -> Array:
    """Log CES production function (KLS version).

    Computed as ``log(sum_i gamma_i * exp(states_i * phi)) / phi`` via a
    numerically stable weighted logsumexp. The weighted form keeps both the
    forward pass and the gradient finite when some ``gamma_i = 0``; the
    naive ``logsumexp(log(gamma) + states * phi)`` has a 1 / gamma term in
    the gradient that produces NaN at ``gamma_i = 0``.

    This is the general-library specification: the CES weights `gamma_i` are
    enumerated over ALL factors in `all_factors` (latent AND observed), so
    observed factors (e.g. income) receive a share of the probability simplex
    and enter the production aggregate. This is by design for the CHS
    estimator. For an AF production CES over the production factors only
    (skill + investment, matching the paper's equation (7)), use `log_ces_af`
    and pass only the production factors.

    Location restriction (Freyberger 2025): the simplex `sum_i gamma_i = 1`
    (with no free additive level) supplies the *cross-period* skills-location
    alternative (Assumption a:ageinvariant_technology_skills_ces(b)) -- i.e. it
    substitutes for age-invariance of the later skill measurement intercepts. It
    does NOT supply the absolute INITIAL location anchor mu_theta,0,1=0, which is
    still required separately: plain log_ces obeys f(x+c,i+c)=f(x,i)+c, so a
    common shift of all latent inputs would otherwise leave observables unchanged.
    So a `log_ces` model still pins a period-0 intercept, but need not also pin
    the later skill intercepts to be equal across periods.
    """
    phi = params[-1]
    gammas = params[:-1]
    scaling_factor = 1 / phi

    exponents = states * phi
    max_exp = jnp.max(exponents)
    shifted = jnp.exp(exponents - max_exp)
    unscaled = max_exp + jnp.log(jnp.sum(gammas * shifted))
    return unscaled * scaling_factor


def params_log_ces(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for the log_ces production function."""
    return [*factors, "phi"]


def constraints_log_ces(
    factor: str,
    factors: tuple[str, ...],
    aug_period: int,
) -> om.constraints.Constraint:
    """Constraints for log_ces production function."""
    names = params_log_ces(factors)
    loc = [("transition", aug_period, factor, name) for name in names[:-1]]
    return om.ProbabilityConstraint(selector=functools.partial(select_by_loc, loc=loc))


def identity_constraints_log_ces(
    factor: str,  # noqa: ARG001
    aug_period: int,  # noqa: ARG001
    all_factors: tuple[str, ...],  # noqa: ARG001
) -> list[om.constraints.Constraint]:
    """Identity constraints for log_ces in carry-forward aug periods.

    Returns an empty list. `log_ces` factors carry their own
    `ProbabilityConstraint` on the gammas (`constraints_log_ces`),
    which already pins the simplex. The carry-forward identity
    constraints used for `linear` / `translog` would conflict with
    that probability fold, so we no-op here -- the natural carry-
    forward in aug periods comes from the upstream model setup
    (e.g. `has_production_shock=False` for time-invariant factors).

    The signature matches `identity_constraints_linear` so callers
    can dispatch by name without case-splitting.
    """
    return []


def log_ces_af(states: Array, params: Array) -> Array:
    """AF (2020) production CES, equation (7): CES over production factors only.

    Identical math to `log_ces`; named separately so AF models can declare a
    production-only CES (skill + investment) without observed factors leaking
    in. Pass ONLY the production factors as `states`.
    """
    return log_ces(states, params)


def params_log_ces_af(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for the `log_ces_af` production function."""
    return params_log_ces(factors)


def constraints_log_ces_af(
    factor: str,
    factors: tuple[str, ...],
    aug_period: int,
) -> om.constraints.Constraint:
    """Constraints for `log_ces_af` production function (gammas on simplex)."""
    return constraints_log_ces(factor=factor, factors=factors, aug_period=aug_period)


def identity_constraints_log_ces_af(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[om.constraints.Constraint]:
    """Identity constraints for `log_ces_af` -- no-op.

    See :func:`identity_constraints_log_ces` for the rationale.
    """
    return identity_constraints_log_ces(
        factor=factor, aug_period=aug_period, all_factors=all_factors
    )


def log_ces_with_constant(states: Array, params: Array) -> Array:
    """Log CES production function with an additive level constant.

    Computed as ``A + (1/phi) * log(sum_i gamma_i * exp(states_i * phi))``,
    matching MATLAB's AF reference parametrisation
    ``log_skills_{t+1} = log(A_t) + (1/sigma) log(sum gamma_i theta_i^sigma)``.

    The plain ``log_ces`` lacks the constant ``A``, which forces models with
    a non-trivial ``A`` (e.g. AF Sec. 5.1's CES sims with ``A = e``) to
    absorb the level shift into the next-period skills measurement
    intercepts. When matching the MATLAB sim parametrisation exactly
    (all skill intercepts pinned to 0, ``A_t`` free per period), use
    this variant instead.

    Location (Freyberger 2025): unlike plain ``log_ces``, the free additive
    constant ``A`` adds a level degree of freedom, so the simplex on the weights
    is only a redundant parameterisation here and supplies no location
    restriction at all. Both the absolute initial location anchor and any
    cross-period location restriction must be imposed via measurement-intercept
    normalizations (the AF validator requires the initial intercept anchor).
    """
    constant_term = params[-1]
    phi = params[-2]
    gammas = params[:-2]
    scaling_factor = 1 / phi

    exponents = states * phi
    max_exp = jnp.max(exponents)
    shifted = jnp.exp(exponents - max_exp)
    unscaled = max_exp + jnp.log(jnp.sum(gammas * shifted))
    return constant_term + unscaled * scaling_factor


def params_log_ces_with_constant(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for ``log_ces_with_constant``."""
    return [*factors, "phi", "constant"]


def constraints_log_ces_with_constant(
    factor: str,
    factors: tuple[str, ...],
    aug_period: int,
) -> om.constraints.Constraint:
    """Constraints for ``log_ces_with_constant`` (gammas on the simplex)."""
    names = params_log_ces_with_constant(factors)
    # Gammas are everything except the last two entries (phi and constant).
    loc = [("transition", aug_period, factor, name) for name in names[:-2]]
    return om.ProbabilityConstraint(selector=functools.partial(select_by_loc, loc=loc))


def identity_constraints_log_ces_with_constant(
    factor: str,  # noqa: ARG001
    aug_period: int,  # noqa: ARG001
    all_factors: tuple[str, ...],  # noqa: ARG001
) -> list[om.constraints.Constraint]:
    """Identity constraints for ``log_ces_with_constant`` -- no-op.

    See :func:`identity_constraints_log_ces` for the rationale.
    """
    return []


def constant(state: Array, params: Array) -> Array:  # noqa: ARG001
    """Constant production function."""
    return state


def params_constant(factors: tuple[str, ...]) -> list[str]:  # noqa: ARG001
    """Index tuples for the constant production function."""
    return []


def robust_translog(states: Array, params: Array) -> Array:
    """Numerically robust version of the translog transition function.

    This function does a clipping of the state vector at +- 1e12 before calling
    the standard translog function. It has a no effect on the results if the
    states do not get close to the clipping values and prevents overflows otherwise.

    The name is a convention in the skill formation literature even though the function
    is better described as a linear in parameters transition function with squares and
    interaction terms of the states.

    """
    clipped_states = jnp.clip(states, -1e12, 1e12)
    return translog(states=clipped_states, params=params)


def params_robust_translog(factors: tuple[str, ...]) -> list[str]:
    """Return parameter names for robust translog transition function."""
    return params_translog(factors)


def identity_constraints_robust_translog(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[FixedConstraintWithValue]:
    """Identity constraints for robust_translog."""
    return identity_constraints_translog(
        factor=factor, aug_period=aug_period, all_factors=all_factors
    )


def linear_and_squares(states: Array, params: Array) -> Array:
    """linear_and_squares transition function."""
    nfac = len(states)
    constant = params[-1]
    lin_beta = params[:nfac]
    square_beta = params[nfac : 2 * nfac]

    res = jnp.dot(states, lin_beta)
    res += jnp.dot(states**2, square_beta)
    res += constant
    return res


def params_linear_and_squares(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for the linear_and_squares production function."""
    return list(factors) + [f"{factor} ** 2" for factor in factors] + ["constant"]


def identity_constraints_linear_and_squares(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[FixedConstraintWithValue]:
    """Identity constraints for linear_and_squares transition function."""
    constraints: list[FixedConstraintWithValue] = []
    for regressor in params_linear_and_squares(all_factors):
        val = 1.0 if factor == regressor else 0.0
        loc = ("transition", aug_period, factor, regressor)
        constraints.append(FixedConstraintWithValue(loc=loc, value=val))
    return constraints


def log_ces_general(states: Array, params: Array) -> Array:
    """Generalized log_ces production function without known location and scale."""
    n = states.shape[-1]
    tfp = params[-1]
    gammas = params[:n]
    sigmas = params[n : 2 * n]

    # note: once the b argument is supported in jax.scipy.special.logsumexp, we can set
    # b = gammas instead of adding the log of gammas to sigma_points * phi

    # the log step for gammas underflows for gamma = 0, but this is handled correctly
    # by logsumexp and does not raise a warning.
    unscaled = jax.scipy.special.logsumexp(jnp.log(gammas) + states * sigmas)
    return unscaled * tfp


def params_log_ces_general(factors: tuple[str, ...]) -> list[str]:
    """Index tuples for the generalized log_ces production function."""
    return list(factors) + [f"sigma_{fac}" for fac in factors] + ["tfp"]


def identity_constraints_log_ces_general(
    factor: str,  # noqa: ARG001
    aug_period: int,  # noqa: ARG001
    all_factors: tuple[str, ...],  # noqa: ARG001
) -> list[om.constraints.Constraint]:
    """Identity constraints for log_ces_general -- no-op.

    See :func:`identity_constraints_log_ces` for the rationale.
    """
    return []
