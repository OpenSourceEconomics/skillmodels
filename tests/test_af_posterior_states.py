"""Tests for AF posterior state extraction.

Regression tests for the AF-F6 fix: posterior means are computed against the
per-observation, income-conditioned chained importance sample
(`samples_per_component`) weighted by `conditional_weights`, NOT from fresh
Halton draws against pooled per-component Gaussian summaries.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.af.posterior_states import (
    _compute_posterior_means,
    get_af_posterior_states,
)
from skillmodels.af.types import (
    ConditionalDistribution,
    MixtureComponent,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations

jax.config.update("jax_enable_x64", True)


def _estimate_reconstructed_endogenous_model() -> tuple:
    """Estimate a small reconstructed-endogenous (calendar-adapter) AF model.

    Investment is endogenous with `has_initial_distribution=False`, so the
    source/destination calendar adapter is active and `get_af_posterior_states`
    must consume the compiled layout rather than the single-period reconstruction.
    Returns `(af_result, model, data)`.
    """
    rng = np.random.default_rng(20240617)
    n_obs, n_periods = 250, 3
    theta = np.zeros((n_obs, n_periods))
    inv = np.zeros((n_obs, n_periods))
    income = rng.normal(1.0, 0.5, n_obs)
    theta[:, 0] = rng.normal(0, 1, n_obs)
    inv[:, 0] = 0.5 * theta[:, 0] + 0.2 * income + rng.normal(0, 0.25, n_obs)
    for t in range(n_periods - 1):
        theta[:, t + 1] = (
            0.05 + 0.6 * theta[:, t] + 0.3 * inv[:, t] + rng.normal(0, 0.3, n_obs)
        )
        inv[:, t + 1] = (
            0.5 * theta[:, t + 1] + 0.2 * income + rng.normal(0, 0.25, n_obs)
        )

    rows = []
    for i in range(n_obs):
        for t in range(n_periods):
            row = {
                "caseid": i,
                "period": t,
                "s1": theta[i, t] + rng.normal(0, 0.3),
                "s2": 0.3 + 0.8 * theta[i, t] + rng.normal(0, 0.35),
                "s3": -0.1 + 1.1 * theta[i, t] + rng.normal(0, 0.4),
                "income": income[i],
            }
            # Investment is measured only at the SOURCE periods (0, 1); it is
            # reconstructed at the terminal period, so no terminal indicators.
            if t < n_periods - 1:
                row["i1"] = inv[i, t] + rng.normal(0, 0.3)
                row["i2"] = 0.2 + 0.9 * inv[i, t] + rng.normal(0, 0.35)
                row["i3"] = -0.1 + 1.2 * inv[i, t] + rng.normal(0, 0.4)
            else:
                row["i1"] = row["i2"] = row["i3"] = np.nan
            rows.append(row)
    data = pd.DataFrame(rows).set_index(["caseid", "period"])

    inv_meas = (("i1", "i2", "i3"), ("i1", "i2", "i3"), ())
    model = ModelSpec(
        factors={
            "skill": FactorSpec(
                measurements=(("s1", "s2", "s3"),) * n_periods,
                normalizations=Normalizations(
                    loadings=({"s1": 1},) * n_periods,
                    intercepts=({"s1": 0},) * n_periods,
                ),
                transition_function="linear",
            ),
            "investment": FactorSpec(
                measurements=inv_meas,
                normalizations=Normalizations(
                    loadings=({"i1": 1}, {"i1": 1}, {}),
                    intercepts=({"i1": 0}, {"i1": 0}, {}),
                ),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=False,
            ),
        },
        observed_factors=("income",),
    )
    af_result = estimate_af(
        model_spec=model,
        data=data,
        options=AFEstimationOptions(
            n_halton_points=30,
            n_halton_points_shock=15,
            optimizer_algorithm="scipy_lbfgsb",
        ),
    )
    return af_result, model, data


def test_get_af_posterior_states_supports_calendar_adapter() -> None:
    # An adapter model (reconstructed-endogenous investment) must yield posterior
    # state means for the state factors, not raise the deferred-feature guard.
    af_result, model, data = _estimate_reconstructed_endogenous_model()

    out = get_af_posterior_states(af_result=af_result, model_spec=model, data=data)

    states = out["unanchored_states"]["states"]
    # State factors only (skill); the endogenous investment is not a state coordinate.
    assert "skill" in states.columns
    assert "investment" not in states.columns
    assert np.isfinite(states["skill"].to_numpy()).all()

    # The period-0 posterior skill mean must track the clean skill signal s1.
    p0 = states[states["period"] == 0].set_index("caseid")["skill"]
    s1_0 = data.xs(0, level="period")["s1"].reindex(p0.index)
    corr = np.corrcoef(p0.to_numpy(), s1_0.to_numpy())[0, 1]
    assert corr > 0.5, f"period-0 skill posterior should track s1 (corr={corr:.2f})"


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
