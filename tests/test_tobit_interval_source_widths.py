"""Observed Tobit intervals keep their source width under standardization.

An observed interval `lo < y* <= hi` enters the CHS update standardized by the
predicted location `mu` and scale `s`. Rounding `(lo - mu) / s` and `(hi - mu) / s`
separately can shrink the width of a narrow interval or remove it: at
`(lo, hi, mu, s) = (0, 1e-17, 1, 1)` both ends round to -1, although the interval has
the finite log mass -40.5629. The update takes the width as `(hi - lo) / s` instead.

The reference is `interval_reference`, a 250-digit decimal integral of the normal
density in source coordinates: the binary64 inputs are converted exactly before
centering and scaling, and the integral is a global antiderivative series, not the
production Hermite or endpoint-CDF formulas. It imports neither the project, JAX,
NumPy nor SciPy. Its domain is `|standardized end| <= 12` and standardized width
`>= 1e-40`. The oracle, the public reproducer and the mutation cases are adapted from
the external review of observed Tobit intervals (`filter_interval_oracle.py`,
`filter_public_witness.py`, `test_source_intervals.py`).
"""

import math
from decimal import Decimal, localcontext
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from skillmodels.chs.halfline import limited_update_from_loop_kwargs
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.measurement_models import ObservationEvent, TobitMeasurement
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations

jax.config.update("jax_enable_x64", True)


@cache
def _pi() -> Decimal:
    with localcontext() as context:
        context.prec = 250
        a = Decimal(1)
        b = Decimal(1) / Decimal(2).sqrt()
        t = Decimal(1) / 4
        p = Decimal(1)
        for _ in range(10):
            new_a = (a + b) / 2
            b = (a * b).sqrt()
            t -= p * (a - new_a) ** 2
            a = new_a
            p *= 2
        return (a + b) ** 2 / (4 * t)


def _antiderivative(x: Decimal) -> Decimal:
    """Integrate `exp(-z^2 / 2)` from 0 to `x`."""
    power_term = x
    total = x
    for n in range(1, 3000):
        power_term *= -x * x / (2 * n)
        term = power_term / (2 * n + 1)
        total += term
        if abs(term) < Decimal("1e-225"):
            return total
    raise ArithmeticError("Oracle series failed to converge")


@cache
def interval_reference(
    lo: float, hi: float, location: float = 0.0, scale: float = 1.0
) -> dict[str, float]:
    """Return log mass, standardized moments and location/scale scores of `(lo, hi]`.

    `y ~ N(location, scale^2)`; `z = (y - location) / scale`. `d_location` and
    `d_scale` are the derivatives of the log mass.
    """
    values = tuple(float(x) for x in (lo, hi, location, scale))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("The oracle requires finite inputs")
    if not values[0] < values[1] or not values[3] > 0:
        raise ValueError("Require lo < hi and scale > 0")
    with localcontext() as context:
        context.prec = 250
        lower, upper, mean, sd = map(Decimal.from_float, values)
        if (upper - lower) / sd < Decimal("1e-40"):
            raise ValueError("Outside the oracle domain: standardized width < 1e-40")
        a, b = (lower - mean) / sd, (upper - mean) / sd
        if max(abs(a), abs(b)) > 12:
            raise ValueError("Outside the oracle domain |z| <= 12")
        root = (2 * _pi()).sqrt()
        mass = (_antiderivative(b) - _antiderivative(a)) / root
        pa = (-a * a / 2).exp() / root
        pb = (-b * b / 2).exp() / root
        z_mean = (pa - pb) / mass
        z_second = 1 + (a * pa - b * pb) / mass
        return {
            "log_mass": float(mass.ln()),
            "z_mean": float(z_mean),
            "z_var": float(z_second - z_mean * z_mean),
            "d_location": float(z_mean / sd),
            "d_scale": float((z_second - 1) / sd),
        }


def source_cases() -> list[tuple[float, float, float, float]]:
    """Return `(lo, hi, location, scale)` source intervals.

    The cases vary the width, the offset, the reflection and the scale, and include
    adjacent floats.
    """
    result = [(0.0, 1e-17, 1.0, 1.0)]
    for width in (1e-5, 1e-9, 1e-13, 1e-17, 2.0**-52, 2.0**-54):
        for location in (-1.0, 0.0, 1.0, 3.0):
            result.append((0.0, width, location, 1.0))
            result.append((-width, 0.0, -location, 1.0))
    for exponent in (-8, -4, 0, 4, 20):
        factor = 2.0**exponent
        result.append((0.0, factor * 1e-17, factor, factor))
    for lower in (0.25, 0.5, 1.0):
        upper = math.nextafter(lower, math.inf)
        for location in (0.0, 1.0, 4.0):
            result.append((lower, upper, location, 1.0))
    return list(dict.fromkeys(result))


def test_standardized_ends_lose_the_width_of_source_intervals() -> None:
    """The class: rounded standardized ends collapse or distort narrow intervals."""
    collapsed = 0
    distorted = 0
    for lo, hi, location, scale in source_cases():
        a, b = (lo - location) / scale, (hi - location) / scale
        if a == b:
            collapsed += 1
        elif (
            abs(
                interval_reference(a, b)["log_mass"]
                - interval_reference(lo, hi, location, scale)["log_mass"]
            )
            > 1e-10
        ):
            distorted += 1
    assert (collapsed, distorted) == (22, 12)
    witness = interval_reference(0.0, 1e-17, 1.0, 1.0)
    assert abs(witness["log_mass"] + 40.56288511410345) < 1e-12
    assert abs(witness["d_location"] + 1.0) < 1e-15


def _build_inputs(lo=0.0, hi=1e-17, location=1.0, scale=1.0, loading=0.0, split=1):
    """Two persons, two periods, a normalized anchor and one interval-only Tobit."""
    spec = ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("anchor", "y"), ("anchor", "y")),
                normalizations=Normalizations(
                    loadings=({"anchor": 1}, {"anchor": 1}),
                    intercepts=({"anchor": 0}, {"anchor": 0}),
                ),
                transition_function="linear",
            )
        },
        measurement_models={
            "y": TobitMeasurement(
                lower=None, upper=None, interval_columns=("y_lo", "y_hi")
            )
        },
    )
    rows = [
        {
            "caseid": person,
            "period": period,
            "anchor": float(2 * person - 1),
            "y": np.nan,
            "y_lo": lo if period == 0 else np.nan,
            "y_hi": hi if period == 0 else np.nan,
        }
        for person in range(2)
        for period in range(2)
    ]
    inputs = get_maximization_inputs(
        model_spec=spec,
        data=pd.DataFrame(rows).set_index(["caseid", "period"]),
        chs_options=CHSEstimationOptions(
            start_params_strategy="none", split_dataset=split
        ),
    )
    params = inputs["params_template"].copy()
    params["value"] = 0.0
    for period in range(2):
        params.loc[("loadings", period, "anchor", "skills"), "value"] = 1.0
        params.loc[("meas_sds", period, "anchor", "-"), "value"] = 1.0
        params.loc[("meas_sds", period, "y", "-"), "value"] = scale
        params.loc[("loadings", period, "y", "skills"), "value"] = loading
        params.loc[("controls", period, "y", "constant"), "value"] = location
    params.loc[("shock_sds", 0, "skills", "-"), "value"] = 1.0
    params.loc[("initial_cholcovs", 0, "mixture_0", "skills-skills"), "value"] = 1.0
    params.loc[("mixture_weights", 0, "mixture_0", "-"), "value"] = 1.0
    return inputs, params


def _expected_contributions(lo, hi, location, scale, loading=0.0):
    """Per person: interval reference, posterior moments and total log likelihood.

    The first anchor gives `m = +/- 1/2` and `P = 1/2`; a zero transition makes the
    second period independent with the same anchor density, so the two anchor terms
    sum to `-log(4 pi) - 1/2`.
    """
    result = []
    for person in range(2):
        prior_mean = (2 * person - 1) / 2
        predictive_sd = math.sqrt(scale * scale + loading * loading / 2)
        reference = interval_reference(
            lo, hi, location + loading * prior_mean, predictive_sd
        )
        result.append(
            {
                **reference,
                "posterior_mean": prior_mean
                + loading / (2 * predictive_sd) * reference["z_mean"],
                "total_loglike": -math.log(4 * math.pi) - 0.5 + reference["log_mass"],
            }
        )
    return result


def test_public_loglikeobs_keeps_a_nonzero_source_interval() -> None:
    inputs, params = _build_inputs()
    reference = _expected_contributions(0.0, 1e-17, 1.0, 1.0)
    expected = np.array([r["total_loglike"] for r in reference])
    np.testing.assert_allclose(
        inputs["loglikeobs"](params), expected, rtol=0, atol=1e-10
    )


@pytest.mark.long_running
def test_public_debug_terms_and_score_keep_a_nonzero_source_interval() -> None:
    inputs, params = _build_inputs()
    reference = _expected_contributions(0.0, 1e-17, 1.0, 1.0)
    expected = np.array([r["total_loglike"] for r in reference])
    debug = inputs["debug_loglike"](params)
    contributions = debug["all_contributions"]
    raw = contributions.loc[
        (contributions["measurement"] == "y") & (contributions["aug_period"] == 0),
        "contribution",
    ]
    np.testing.assert_allclose(raw, reference[0]["log_mass"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(debug["contributions"], expected, rtol=0, atol=1e-10)
    value, gradient = inputs["loglike_and_gradient"](params)
    assert math.isfinite(value)
    assert np.isfinite(gradient).all()
    position = params.index.get_loc(("controls", 0, "y", "constant"))
    np.testing.assert_allclose(
        gradient[position],
        sum(r["d_location"] for r in reference),
        rtol=1e-5,
        atol=1e-8,
    )


def _scan_update(row: jax.Array) -> jax.Array:
    """Log likelihood, posterior mean and variance of one interval-only update."""
    lower, upper, location, sigma, loading = row
    kwargs = {
        "states": jnp.zeros((1, 1, 1)),
        "upper_chols": jnp.ones((1, 1, 1, 1)),
        "loadings": jnp.reshape(loading, (1,)),
        "control_params": jnp.reshape(location, (1,)),
        "meas_sd": sigma,
        "measurements": jnp.array([jnp.nan]),
        "events": jnp.array([ObservationEvent.INTERVAL], dtype=jnp.int8),
        "controls": jnp.ones((1, 1)),
        "log_mixture_weights": jnp.zeros((1, 1)),
        "is_limited": jnp.array(True),
        "cut_table": jnp.zeros(4),
        "codes_from_values": jnp.array(False),
        "interval_bounds": jnp.reshape(jnp.stack([lower, upper]), (1, 2)),
    }
    mean, chol, _, loglike, _ = limited_update_from_loop_kwargs(
        kwargs, has_intervals=True
    )
    return jnp.stack([loglike[0], mean[0, 0, 0], chol[0, 0, 0, 0] ** 2])


def _update_references(rows):
    """Log mass, posterior mean and variance for `N(0, 1)` state, from the oracle."""
    references = []
    for lower, upper, location, sigma, loading in rows:
        sd = math.hypot(sigma, loading)
        ref = interval_reference(lower, upper, location, sd)
        references.append(
            [
                ref["log_mass"],
                loading / sd * ref["z_mean"],
                1 - (loading / sd) ** 2 * (1 - ref["z_var"]),
            ]
        )
    return np.asarray(references)


def test_narrow_source_intervals_match_the_oracle_in_the_update() -> None:
    """Collapsed, distorted, adjacent-float, reflected and scaled intervals."""
    rows = [
        (0.0, 1e-17, 1.0, 1.0, 0.0),
        (0.0, 1e-9, -1.0, 1.0, 0.5),
        (-1e-13, 0.0, -3.0, 1.0, 0.0),
        (0.25, math.nextafter(0.25, math.inf), 4.0, 1.0, 0.5),
        (0.0, 2.0**20 * 1e-17, 2.0**20, 2.0**20, 2.0**19),
        (0.0, 0.3, 0.1, 1.0, 0.5),
    ]
    got = np.asarray(jax.jit(jax.vmap(_scan_update))(jnp.asarray(rows)))
    expected = _update_references(rows)
    np.testing.assert_allclose(got[:, 0], expected[:, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(got[:, 1:], expected[:, 1:], rtol=0, atol=1e-8)


@pytest.mark.long_running
def test_source_interval_mutations_values_moments_and_transformed_paths() -> None:
    rows = [
        (lower, upper, location, sigma, loading)
        for lower, upper, location, sigma in source_cases()
        for loading in (0.0, 0.5 * sigma)
    ]
    array = jnp.asarray(rows, dtype=jnp.float64)
    expected = _update_references(rows)
    batched = np.asarray(jax.jit(jax.vmap(_scan_update))(array))
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched[:, 0], expected[:, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(batched[:, 1:], expected[:, 1:], rtol=0, atol=1e-8)
    permutation = np.random.default_rng(106).permutation(len(array))
    permuted = np.asarray(jax.jit(jax.vmap(_scan_update))(array[permutation]))
    np.testing.assert_allclose(permuted, batched[permutation], rtol=0, atol=1e-12)
    np.testing.assert_allclose(_scan_update(array[0]), expected[0], rtol=0, atol=1e-10)


@pytest.mark.long_running
def test_source_interval_scores_match_independent_integral_scores() -> None:
    rows = np.array(
        [
            (0.0, 1e-17, 1.0, 1.0, 0.0),
            (0.0, 1e-9, 2.0, 0.75, 0.4),
            (-1e-17, 0.0, -1.0, 1.0, 0.5),
            (0.25, math.nextafter(0.25, math.inf), 1.0, 1.0, 0.5),
        ]
    )
    gradients = np.asarray(
        jax.jit(jax.vmap(jax.grad(lambda r: _scan_update(r)[0])))(jnp.asarray(rows))
    )
    assert np.isfinite(gradients).all()
    for row, gradient in zip(rows, gradients, strict=True):
        lower, upper, location, sigma, loading = row
        sd = math.hypot(sigma, loading)
        ref = interval_reference(lower, upper, location, sd)
        expected = np.array(
            [
                ref["d_location"],
                ref["d_scale"] * sigma / sd,
                ref["d_scale"] * loading / sd,
            ]
        )
        np.testing.assert_allclose(gradient[2:], expected, rtol=1e-5, atol=1e-8)


@pytest.mark.long_running
def test_public_nonzero_loading_and_split_gradient() -> None:
    inputs, params = _build_inputs(loading=0.5)
    reference = _expected_contributions(0.0, 1e-17, 1.0, 1.0, loading=0.5)
    expected = np.array([r["total_loglike"] for r in reference])
    np.testing.assert_allclose(
        inputs["loglikeobs"](params), expected, rtol=0, atol=1e-10
    )
    post = inputs["debug_loglike"](params)["post_update_states"]
    got = post.loc[(post["measurement"] == "y") & (post["aug_period"] == 0)]
    np.testing.assert_allclose(
        got.sort_values("id")["skills"],
        [r["posterior_mean"] for r in reference],
        rtol=0,
        atol=1e-8,
    )
    split_inputs, split_params = _build_inputs(loading=0.5, split=2)
    _, gradient = inputs["loglike_and_gradient"](params)
    _, split_gradient = split_inputs["loglike_and_gradient"](split_params)
    assert np.isfinite(gradient).all()
    assert np.isfinite(split_gradient).all()
    np.testing.assert_allclose(split_gradient, gradient, rtol=1e-10, atol=1e-10)


@pytest.mark.long_running
def test_source_interval_mixture_weights_remain_finite_and_normalized() -> None:
    prior_means = np.array([-0.25, 0.25])
    prior_weights = np.array([0.3, 0.7])
    loading, sigma, location = 0.5, 1.0, 1.0
    sd = math.hypot(sigma, loading)
    refs = [
        interval_reference(0.0, 1e-17, location + loading * m, sd) for m in prior_means
    ]
    weighted = np.log(prior_weights) + np.array([r["log_mass"] for r in refs])
    maximum = np.max(weighted)
    expected_loglike = maximum + math.log(np.exp(weighted - maximum).sum())
    expected_means = prior_means + loading / sd * np.array([r["z_mean"] for r in refs])
    kwargs = {
        "states": jnp.asarray(prior_means).reshape(1, 2, 1),
        "upper_chols": jnp.ones((1, 2, 1, 1)),
        "loadings": jnp.array([loading]),
        "control_params": jnp.array([location]),
        "meas_sd": jnp.array(sigma),
        "measurements": jnp.array([jnp.nan]),
        "events": jnp.array([ObservationEvent.INTERVAL], dtype=jnp.int8),
        "controls": jnp.ones((1, 1)),
        "log_mixture_weights": jnp.log(jnp.asarray(prior_weights))[None, :],
        "is_limited": jnp.array(True),
        "cut_table": jnp.zeros(4),
        "codes_from_values": jnp.array(False),
        "interval_bounds": jnp.array([[0.0, 1e-17]]),
    }
    state, chol, weights, loglike, _ = jax.jit(
        lambda kw: limited_update_from_loop_kwargs(kw, has_intervals=True)
    )(kwargs)
    assert all(
        np.isfinite(np.asarray(x)).all() for x in (state, chol, weights, loglike)
    )
    np.testing.assert_allclose(loglike[0], expected_loglike, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        weights[0], weighted - expected_loglike, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(state[0, :, 0], expected_means, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.exp(weights[0]).sum(), 1.0, rtol=0, atol=1e-14)


@pytest.mark.long_running
def test_mixed_interval_exact_missing_and_tail_mixture_batch() -> None:
    """Real observation/component axes; inactive rows must not poison gradients."""
    means = np.array([-0.25, 0.25])
    weights = np.array([0.3, 0.7])
    loading, sigma, offset = 0.5, 1.0, 1.0
    sd = math.hypot(sigma, loading)
    events = [
        ObservationEvent.INTERVAL,
        ObservationEvent.DENSITY,
        ObservationEvent.MISSING,
        ObservationEvent.LOWER_TAIL,
    ]
    kwargs = {
        "states": jnp.broadcast_to(jnp.asarray(means)[None, :, None], (4, 2, 1)),
        "upper_chols": jnp.ones((4, 2, 1, 1)),
        "loadings": jnp.array([loading]),
        "control_params": jnp.array([offset]),
        "meas_sd": jnp.array(sigma),
        "measurements": jnp.array([jnp.nan, 0.25, jnp.nan, jnp.nan]),
        "events": jnp.array(events, dtype=jnp.int8),
        "controls": jnp.ones((4, 1)),
        "log_mixture_weights": jnp.broadcast_to(jnp.log(jnp.asarray(weights)), (4, 2)),
        "is_limited": jnp.array(True),
        "cut_table": jnp.array([0.0, -2.0, 2.0, 0.0]),
        "codes_from_values": jnp.array(False),
        "interval_bounds": jnp.array(
            [[0.0, 1e-17], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan], [-jnp.inf, 0.2]]
        ),
    }
    expected_ll, expected_means, expected_vars, expected_weights = [], [], [], []
    expected_score = 0.0
    for event in events:
        component_ll, component_mean, component_var, scores = [], [], [], []
        for mean in means:
            mu = offset + loading * mean
            if event == ObservationEvent.INTERVAL:
                ref = interval_reference(0.0, 1e-17, mu, sd)
                ll, zmean, zvar = ref["log_mass"], ref["z_mean"], ref["z_var"]
                score = ref["d_location"]
            elif event == ObservationEvent.DENSITY:
                zmean = (0.25 - mu) / sd
                ll = -0.5 * zmean * zmean - math.log(sd * math.sqrt(2 * math.pi))
                zvar, score = 0.0, zmean / sd
            elif event == ObservationEvent.LOWER_TAIL:
                beta = (0.2 - mu) / sd
                mass = 0.5 * math.erfc(-beta / math.sqrt(2))
                mills = math.exp(-0.5 * beta * beta) / (math.sqrt(2 * math.pi) * mass)
                ll, zmean = math.log(mass), -mills
                zvar, score = 1 - beta * mills - mills * mills, -mills / sd
            else:
                ll, zmean, zvar, score = 0.0, 0.0, 1.0, 0.0
            component_ll.append(ll)
            component_mean.append(mean + loading / sd * zmean)
            component_var.append(1 - (loading / sd) ** 2 * (1 - zvar))
            scores.append(score)
        weighted = np.log(weights) + np.asarray(component_ll)
        maximum = weighted.max()
        total = maximum + math.log(np.exp(weighted - maximum).sum())
        expected_ll.append(total)
        expected_means.append(component_mean)
        expected_vars.append(component_var)
        expected_weights.append(weighted - total)
        expected_score += float(np.exp(weighted - total) @ np.asarray(scores))
    result = jax.jit(
        lambda kw: limited_update_from_loop_kwargs(kw, has_intervals=True)
    )(kwargs)
    states, chols, logweights, loglikes, _ = result
    assert all(np.isfinite(np.asarray(x)).all() for x in result[:4])
    np.testing.assert_allclose(loglikes, expected_ll, rtol=0, atol=1e-10)
    np.testing.assert_allclose(states[..., 0], expected_means, rtol=0, atol=1e-8)
    np.testing.assert_allclose(chols[..., 0, 0] ** 2, expected_vars, rtol=0, atol=1e-8)
    np.testing.assert_allclose(logweights, expected_weights, rtol=0, atol=1e-10)
    score = jax.jit(
        jax.grad(
            lambda value: limited_update_from_loop_kwargs(
                {**kwargs, "control_params": jnp.reshape(value, (1,))},
                has_intervals=True,
            )[3].sum()
        )
    )(jnp.array(offset))
    np.testing.assert_allclose(score, expected_score, rtol=1e-5, atol=1e-8)
