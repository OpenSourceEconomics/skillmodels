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
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optimagic as om
from jax import Array

if TYPE_CHECKING:
    from skillmodels.common.constraints import FixedConstraintWithValue


def select_by_loc(params: Any, loc: Any) -> Any:  # noqa: ANN401
    """Select parameters by location."""
    return params.loc[loc]


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
    from skillmodels.common.constraints import FixedConstraintWithValue  # noqa: PLC0415

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
    from skillmodels.common.constraints import FixedConstraintWithValue  # noqa: PLC0415

    constraints: list[FixedConstraintWithValue] = []
    for regressor in params_translog(all_factors):
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
    factors: tuple[str, ...],
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[om.constraints.Constraint]:
    """Identity constraints for log_ces."""
    raise NotImplementedError


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
    factors: tuple[str, ...],
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[om.constraints.Constraint]:
    """Identity constraints for ``log_ces_with_constant``."""
    raise NotImplementedError


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
    from skillmodels.common.constraints import FixedConstraintWithValue  # noqa: PLC0415

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
    factors: tuple[str, ...],
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[om.constraints.Constraint]:
    """Identity constraints for log_ces_general."""
    raise NotImplementedError
