"""Halton quasi-random sequence generation for numerical quadrature."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.stats import qmc


def create_halton_nodes_and_weights(
    n_points: int,
    n_dim: int,
    *,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Create Halton quadrature nodes transformed to standard normal.

    Generate a low-discrepancy Halton sequence in [0, 1]^d, then transform
    to standard normal quantiles via the inverse CDF. Weights are uniform
    (1/n_points) since the Halton sequence provides quasi-uniform coverage.

    Args:
        n_points: Number of quadrature points.
        n_dim: Dimensionality of the sequence.
        seed: Seed for scrambled Halton sequence (for reproducibility).

    Return:
        Tuple of (nodes, weights) where:
        - nodes: shape (n_points, n_dim), standard normal quantiles
        - weights: shape (n_points,), uniform weights summing to 1

    """
    sampler = qmc.Halton(d=n_dim, scramble=True, seed=seed)
    # Generate uniform [0, 1] samples, skip first point (often degenerate)
    uniform_samples = sampler.random(n=n_points + 1)[1:]

    # Clip to avoid infinite values at 0 and 1
    uniform_samples = np.clip(uniform_samples, 1e-10, 1 - 1e-10)

    # Transform to standard normal via inverse CDF
    from scipy.stats import norm  # noqa: PLC0415

    normal_nodes = norm.ppf(uniform_samples)

    nodes = jnp.array(normal_nodes, dtype=jnp.float64)
    weights = jnp.ones(n_points, dtype=jnp.float64) / n_points

    return nodes, weights


def transform_nodes_to_conditional(
    standard_nodes: Array,
    mean: Array,
    chol_cov: Array,
) -> Array:
    """Transform standard normal nodes to a conditional distribution.

    Apply the affine transformation: x = mean + chol_cov @ z where z are
    standard normal nodes.

    Args:
        standard_nodes: Shape (n_points, n_dim), standard normal quantiles.
        mean: Shape (n_dim,), mean of the target distribution.
        chol_cov: Shape (n_dim, n_dim), lower Cholesky of the target covariance.

    Return:
        Transformed nodes, shape (n_points, n_dim).

    """
    return mean + standard_nodes @ chol_cov.T


def create_shock_nodes_and_weights(
    n_points: int,
    n_shocks: int,
    *,
    seed: int = 42,
) -> tuple[Array, Array]:
    """Create quadrature nodes for production shocks.

    Separate Halton sequence for the shock integration dimension, using
    a different seed to avoid correlation with the state nodes.

    Args:
        n_points: Number of quadrature points per shock dimension.
        n_shocks: Number of independent shock dimensions.
        seed: Seed for the Halton sequence.

    Return:
        Tuple of (nodes, weights) with nodes shape (n_points, n_shocks).

    """
    return create_halton_nodes_and_weights(n_points, n_shocks, seed=seed)
