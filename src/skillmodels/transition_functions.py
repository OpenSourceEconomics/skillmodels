"""Contains transition functions and corresponding helper functions.

Below the signature and purpose of a transition function and its helper
functions is explained with a transition function called example_func:
>

**example_func(** *states, params**)**:

    The actual transition function.

Args:
        * states: 1d numpy array of length n_all_factors
        * params: 1d numpy array with coefficients specific to this transition function

Returns:
        * float


**names_example_func(** *factors* **)**:

    Generate a list of names for the params of the transition function.

    The names will be used to construct index tuples in the following way:

    ('transition', period, factor, NAME)

The transition functions have to be JAX jittable and differentiable. However, they
should not be jitted yet.

"""

from itertools import combinations

import jax
import jax.numpy as jnp


def linear(states, params):
    """Linear production function where the constant is the last parameter."""
    constant = params[-1]
    betas = params[:-1]
    return jnp.dot(states, betas) + constant


def params_linear(factors):
    """Index tuples for linear transition function."""
    return [*factors, "constant"]


def translog(states, params):
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


def params_translog(factors):
    """Index tuples for the translog production function."""
    names = (
        factors
        + [f"{factor} ** 2" for factor in factors]
        + [f"{a} * {b}" for a, b in combinations(factors, 2)]
        + ["constant"]
    )
    return names


def log_ces(states, params):
    """Log CES production function (KLS version)."""
    phi = params[-1]
    gammas = params[:-1]
    scaling_factor = 1 / phi

    # note: once the b argument is supported in jax.scipy.special.logsumexp, we can set
    # b = gammas instead of adding the log of gammas to sigma_points * phi

    # the log step for gammas underflows for gamma = 0, but this is handled correctly
    # by logsumexp and does not raise a warning.
    unscaled = jax.scipy.special.logsumexp(jnp.log(gammas) + states * phi)
    result = unscaled * scaling_factor
    return result


def params_log_ces(factors):
    """Index tuples for the log_ces production function."""
    return [*factors, "phi"]


def constraints_log_ces(factor, factors, period):
    names = params_log_ces(factors)
    loc = [("transition", period, factor, name) for name in names[:-1]]
    return {"loc": loc, "type": "probability"}


def constant(state, params):  # noqa: ARG001
    """Constant production function."""
    return state


def params_constant(factors):  # noqa: ARG001
    """Index tuples for the constant production function."""
    return []


def robust_translog(states, params):
    """Numerically robust version of the translog transition function.

    This function does a clipping of the state vector at +- 1e12 before calling
    the standard translog function. It has a no effect on the results if the
    states do not get close to the clipping values and prevents overflows otherwise.

    The name is a convention in the skill formation literature even though the function
    is better described as a linear in parameters transition function with squares and
    interaction terms of the states.

    """
    clipped_states = jnp.clip(states, -1e12, 1e12)
    return translog(clipped_states, params)


def params_robust_translog(factors):
    return params_translog(factors)


def linear_and_squares(states, params):
    """linear_and_squares transition function."""
    nfac = len(states)
    constant = params[-1]
    lin_beta = params[:nfac]
    square_beta = params[nfac : 2 * nfac]

    res = jnp.dot(states, lin_beta)
    res += jnp.dot(states**2, square_beta)
    res += constant
    return res


def params_linear_and_squares(factors):
    """Index tuples for the linear_and_squares production function."""
    names = factors + [f"{factor} ** 2" for factor in factors] + ["constant"]
    return names


def log_ces_general(states, params):
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
    result = unscaled * tfp
    return result


def params_log_ces_general(factors):
    """Index tuples for the generalized log_ces production function."""
    return factors + [f"sigma_{fac}" for fac in factors] + ["tfp"]
