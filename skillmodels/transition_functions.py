"""Contains transition functions and corresponding helper functions.

Below the signature and purpose of a transition function and its helper
functions is explained with a transition function called example_func:


**example_func(** *sigma_points, params**)**:

    The actual transition function.

    Args:
        * sigma_points: 4d numpy array of sigma_points or states being transformed.
            The shape is n_obs, n_mixtures, n_sigma, n_fac.
        * params: 1d numpy array with coefficients specific to this transition function

    Returns
        * np.ndarray: Shape is n_obs, n_mixtures, n_sigma


**index_tuples_example_func(** *factor, factors, period* **)**:

    Generate a list of index tuples for the params of the transition function.

    Each index tuple contains four entries

    - 'transition' (fix)
    - period
    - factor
    - 'some-name'


The transition functions have to be JAX jittable and differentiable. However, they
should not be jitted yet.

"""
from itertools import combinations

import jax
import jax.numpy as jnp


def linear(sigma_points, params):
    """Linear production function where the constant is the last parameter."""
    constant = params[-1]
    betas = params[:-1]
    return jnp.dot(sigma_points, betas) + constant


def names_linear(factors):
    """Index tuples for linear transition function."""
    return factors + ["constant"]


def translog(sigma_points, params):
    """Translog transition function.

    The name is a convention in the skill formation literature even though the function
    is better described as a linear in parameters transition function with squares and
    interaction terms of the states.

    """
    nfac = sigma_points.shape[-1]
    constant = params[-1]
    lin_beta = params[:nfac]
    square_beta = params[nfac : 2 * nfac]
    inter_beta = params[2 * nfac : -1]

    res = jnp.dot(sigma_points, lin_beta)
    res += jnp.dot(sigma_points ** 2, square_beta)
    for p, (a, b) in zip(inter_beta, combinations(range(nfac), 2)):
        res += p * sigma_points[..., a] * sigma_points[..., b]
    res += constant
    return res


def names_translog(factors):
    """Index tuples for the translog production function."""
    names = (
        factors
        + [f"{factor} ** 2" for factor in factors]
        + [f"{a} * {b}" for a, b in combinations(factors, 2)]
        + ["constant"]
    )
    return names


def log_ces(sigma_points, params):
    """Log CES production function (KLS version)."""
    phi = params[-1]
    gammas = params[:-1]
    scaling_factor = 1 / phi

    # note: once the b argument is supported in jax.scipy.special.logsumexp, we can set
    # b = gammas instead of adding the log of gammas to sigma_points * phi

    # the log step for gammas underflows for gamma = 0, but this is handled correctly
    # by logsumexp and does not raise a warning.
    unscaled = jax.scipy.special.logsumexp(
        jnp.log(gammas) + sigma_points * phi, axis=-1
    )
    result = unscaled * scaling_factor
    return result


def names_log_ces(factors):
    """Index tuples for the log_ces production function."""
    return factors + ["phi"]


def constraints_log_ces(factor, factors, period):
    names = names_log_ces(factors)
    loc = [("transition", period, factor, name) for name in names[:-1]]
    return {"loc": loc, "type": "probability"}


def constant(sigma_points, params):
    """Constant production function should never be called."""
    raise NotImplementedError


def names_constant(factors):
    """Index tuples for the constant production function."""
    return []
