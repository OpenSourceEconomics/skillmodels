"""Tests for the AF memory-aware batching helpers."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skillmodels.af.batching import (
    _DEFAULT_TARGET_BATCH_BYTES,
    _ENV_VAR_TARGET,
    auto_n_obs_per_batch,
    target_batch_bytes,
)
from skillmodels.af.likelihood import _map_over_obs

jax.config.update("jax_enable_x64", val=True)


def _square_sum(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)


def _two_arg(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x * y)


@pytest.mark.parametrize("batch_size", [None, 1, 3, 7, 100])
def test_map_over_obs_matches_vmap_for_every_batch_size(batch_size: int | None) -> None:
    """The chunked ``_map_over_obs`` must match ``jax.vmap`` exactly."""
    rng = np.random.default_rng(0)
    xs = jnp.asarray(rng.normal(size=(20, 5)))

    expected = jax.vmap(_square_sum)(xs)
    actual = _map_over_obs(_square_sum, xs, n_obs_per_batch=batch_size)

    # 1 ULP differences are allowed because `lax.map` may use a different
    # reduction order than `vmap`.
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-13
    )


@pytest.mark.parametrize("batch_size", [None, 1, 5])
def test_map_over_obs_two_args(batch_size: int | None) -> None:
    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.normal(size=(15, 3)))
    ys = jnp.asarray(rng.normal(size=(15, 3)))

    expected = jax.vmap(_two_arg)(xs, ys)
    actual = _map_over_obs(_two_arg, xs, ys, n_obs_per_batch=batch_size)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-14
    )


def test_map_over_obs_preserves_gradient() -> None:
    """Reverse-mode gradient must not depend on the chunk size."""
    rng = np.random.default_rng(2)
    xs = jnp.asarray(rng.normal(size=(12, 4)))

    def _loss(xs_flat: jnp.ndarray, batch: int | None) -> jnp.ndarray:
        xs_r = xs_flat.reshape((12, 4))
        return jnp.sum(_map_over_obs(_square_sum, xs_r, n_obs_per_batch=batch))

    g_full = jax.grad(lambda x: _loss(x, None))(xs.reshape(-1))
    g_chunked = jax.grad(lambda x: _loss(x, 3))(xs.reshape(-1))

    np.testing.assert_allclose(
        np.asarray(g_chunked), np.asarray(g_full), rtol=0, atol=1e-10
    )


def test_target_batch_bytes_default() -> None:
    os.environ.pop(_ENV_VAR_TARGET, None)
    assert target_batch_bytes() == _DEFAULT_TARGET_BATCH_BYTES


def test_target_batch_bytes_env_override() -> None:
    os.environ[_ENV_VAR_TARGET] = "1048576"
    try:
        assert target_batch_bytes() == 1_048_576
    finally:
        del os.environ[_ENV_VAR_TARGET]


def test_target_batch_bytes_rejects_junk() -> None:
    os.environ[_ENV_VAR_TARGET] = "not-a-number"
    try:
        assert target_batch_bytes() == _DEFAULT_TARGET_BATCH_BYTES
    finally:
        del os.environ[_ENV_VAR_TARGET]


def test_auto_n_obs_per_batch_small_problem_uses_all() -> None:
    """Tiny problems fit easily; the whole batch should run in one shot."""
    batch = auto_n_obs_per_batch(
        n_obs=100,
        n_halton_points=20,
        n_halton_points_shock=10,
        n_latent=2,
        n_endogenous=0,
    )
    assert batch == 100


def test_auto_n_obs_per_batch_large_problem_splits() -> None:
    """Large problems need to be chunked; the result is smaller than n_obs."""
    batch = auto_n_obs_per_batch(
        n_obs=1403,
        n_halton_points=20_000,
        n_halton_points_shock=20_000,
        n_latent=4,
        n_endogenous=1,
    )
    assert 1 <= batch < 1403


def test_auto_n_obs_per_batch_respects_target_bytes() -> None:
    """A bigger budget should allow a larger batch (monotone in the budget)."""
    small = auto_n_obs_per_batch(
        n_obs=10_000,
        n_halton_points=200,
        n_halton_points_shock=50,
        n_latent=2,
        n_endogenous=1,
        target_bytes=2**24,
    )
    large = auto_n_obs_per_batch(
        n_obs=10_000,
        n_halton_points=200,
        n_halton_points_shock=50,
        n_latent=2,
        n_endogenous=1,
        target_bytes=2**30,
    )
    assert small <= large
