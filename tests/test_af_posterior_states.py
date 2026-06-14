"""Tests for AF posterior state extraction.

Regression tests for the AF-F6 fix: posterior means are computed against the
per-observation, income-conditioned chained importance sample
(`samples_per_component`) weighted by `conditional_weights`, NOT from fresh
Halton draws against pooled per-component Gaussian summaries.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skillmodels.af.posterior_states import _compute_posterior_means
from skillmodels.af.types import (
    ConditionalDistribution,
    MixtureComponent,
)

jax.config.update("jax_enable_x64", True)


def _placeholder_cond_dist(
    *,
    samples_per_component: tuple,
    conditional_weights: jax.Array | None,
    mixture_weights: jax.Array,
    n_state: int,
) -> ConditionalDistribution:
    """Build a ConditionalDistribution with degenerate, unread placeholders.

    The fixed `_compute_posterior_means` must NOT read `components` /
    `cond_means` / `cond_chols`, so we fill them with placeholders.
    """
    n_components = len(mixture_weights)
    return ConditionalDistribution(
        mixture_weights=mixture_weights,
        components=tuple(
            MixtureComponent(mean=jnp.zeros(n_state), chol_cov=jnp.eye(n_state))
            for _ in range(n_components)
        ),
        samples_per_component=samples_per_component,
        conditional_weights=conditional_weights,
        cond_means=None,
        cond_chols=None,
        chain_links=(),
    )


def test_compute_posterior_means_respects_per_obs_income_conditioning() -> None:
    n_obs = 2
    n_state = 1
    n_summary = 8

    rng = np.random.default_rng(606)
    # Obs 0's samples centered near -2.0, obs 1's near +2.0: two different
    # income-conditioned priors.
    arr = np.zeros((n_summary, n_obs, n_state))
    arr[:, 0, 0] = -2.0 + rng.normal(0, 0.05, n_summary)
    arr[:, 1, 0] = 2.0 + rng.normal(0, 0.05, n_summary)
    samples_per_component = (jnp.asarray(arr),)

    cond_dist = _placeholder_cond_dist(
        samples_per_component=samples_per_component,
        conditional_weights=jnp.ones((n_obs, 1)),
        mixture_weights=jnp.array([1.0]),
        n_state=n_state,
    )

    # Identical measurement residual for both observations.
    measurements = jnp.array([[0.0], [0.0]])
    control_contrib = jnp.array([[0.0], [0.0]])
    full_loadings = jnp.array([[1.0]])
    meas_sds = jnp.array([0.5])

    out = _compute_posterior_means(
        cond_dist=cond_dist,
        measurements=measurements,
        control_contrib=control_contrib,
        full_loadings=full_loadings,
        meas_sds=meas_sds,
    )

    # The two observations must get DIFFERENT posterior means, each staying
    # on the side of its own income-conditioned prior cluster. Under the old
    # pooled code both would coincide.
    assert out[0, 0] < -0.5 < 0.5 < out[1, 0]


def test_compute_posterior_means_uses_measurements() -> None:
    n_summary = 16
    n_state = 1

    rng = np.random.default_rng(607)
    # Single obs, symmetric samples around 0.
    base = rng.normal(0, 1.0, n_summary)
    arr = base.reshape(n_summary, 1, 1)
    samples_per_component = (jnp.asarray(arr),)

    cond_dist = _placeholder_cond_dist(
        samples_per_component=samples_per_component,
        conditional_weights=jnp.ones((1, 1)),
        mixture_weights=jnp.array([1.0]),
        n_state=n_state,
    )

    full_loadings = jnp.array([[1.0]])
    meas_sds = jnp.array([0.5])
    control_contrib = jnp.array([[0.0]])

    out_low = _compute_posterior_means(
        cond_dist=cond_dist,
        measurements=jnp.array([[-1.5]]),
        control_contrib=control_contrib,
        full_loadings=full_loadings,
        meas_sds=meas_sds,
    )
    out_high = _compute_posterior_means(
        cond_dist=cond_dist,
        measurements=jnp.array([[1.5]]),
        control_contrib=control_contrib,
        full_loadings=full_loadings,
        meas_sds=meas_sds,
    )

    # The posterior mean tracks the measurement (loading = 1).
    assert out_low[0, 0] < 0.0 < out_high[0, 0]


def test_compute_posterior_means_raises_after_to_numpy() -> None:
    cond_dist = _placeholder_cond_dist(
        samples_per_component=(),  # as `to_numpy()` produces
        conditional_weights=None,
        mixture_weights=jnp.array([1.0]),
        n_state=1,
    )

    with pytest.raises(ValueError, match="to_numpy"):
        _compute_posterior_means(
            cond_dist=cond_dist,
            measurements=jnp.array([[0.0]]),
            control_contrib=jnp.array([[0.0]]),
            full_loadings=jnp.array([[1.0]]),
            meas_sds=jnp.array([0.5]),
        )
