"""Observed Tobit intervals keep their source geometry under standardization.

An observed interval `lo < y* <= hi` enters the CHS update standardized by the
predicted location `mu` and scale `s`. The update evaluates the original interval,
not the interval between two separately rounded standardized ends:

- rounding `(lo - mu) / s` and `(hi - mu) / s` separately can shrink the width of a
  narrow interval or remove it: at `(lo, hi, mu, s) = (0, 1e-17, 1, 1)` both ends
  round to -1, although the interval has the finite log mass -40.5629;
- the standardized width or its half can underflow: at `(0, 2^-600, 2^500, 2^500)`
  the width `2^-1100` is below the smallest binary64 number, although the log mass
  is the finite -763.8808;
- the difference `lo - mu` can overflow although the standardized interval is
  moderate: at `(-2^1023, -2^1022, 2^1023, 2^1023)` it is `(-2, -1.5]` with log mass
  -3.1223, while `lo - mu` rounds to `-inf`;
- two collapsed standardized ends can belong to an interval whose mass is not a
  narrow-density integral: at `(0, 2^-30, 2^30, 1)` both ends round to `-2^30`,
  while the density changes by the factor `e` across the interval and the log mass
  is `-2^59` (a difference of endpoint log-CDFs gives `log 0`).

Three independent references, all 250-digit decimal integrals in source coordinates
(the binary64 inputs are converted exactly before centering and scaling; none of
them imports the project, JAX, NumPy or SciPy), cover the tested domain:

- `interval_reference`: a global antiderivative series of the normal density, for
  `|standardized end| <= 12` and standardized width `>= 1e-40`;
- `underflow_reference`: the density enclosure `log P = log w - log sqrt(2 pi)
  - c^2 / 2 + e`, `|e| <= |c| w / 2 + w^2 / 8`, for widths `w <= 2^-1000`;
- `source_reference`: the integral over the unit coordinate `z = a + w u` with
  bounded tilt `A = -a w` and curvature `B = w^2`, which covers the far tails.

The oracles, the public reproducers and the mutation cases are adapted from the
external review of observed Tobit intervals (`filter_interval_oracle.py`,
`filter_public_witness.py`, `test_source_intervals.py`,
`source_width_underflow_oracle.py`, `source_width_underflow_witness.py`,
`test_source_width_underflow.py`, `source_geometry_reference.py`,
`source_geometry_witness.py`, `test_source_geometry.py`).
"""

import math
from dataclasses import dataclass
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
from skillmodels.common.normal_terms import source_interval_terms

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
            "d_log_scale": float(z_second - 1),
        }


@cache
def underflow_reference(
    lo: float, hi: float, location: float, scale: float
) -> dict[str, float]:
    """Return log mass, standardized moments and scores of a very narrow `(lo, hi]`.

    The standardized width `w` must lie in `[2^-1600, 2^-1000]` and the standardized
    ends in `[-12, 12]`. The moments are those of the uniform distribution on the
    standardized interval; their errors are below `w`. `d_log_scale` is the
    derivative of the log mass with respect to `log(scale)`.
    """
    values = tuple(float(x) for x in (lo, hi, location, scale))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("The oracle requires finite inputs")
    if not values[0] < values[1] or not values[3] > 0:
        raise ValueError("Require lo < hi and scale > 0")
    with localcontext() as context:
        context.prec = 250
        lower, upper, mean, sd = map(Decimal.from_float, values)
        width = (upper - lower) / sd
        if not Decimal(2) ** -1600 <= width <= Decimal(2) ** -1000:
            raise ValueError("Outside the oracle domain 2^-1600 <= w <= 2^-1000")
        a = (lower - mean) / sd
        if max(abs(a), abs(a + width)) > 12:
            raise ValueError("Outside the oracle domain |z| <= 12")
        centre = a + width / 2
        variance = width * width / 12
        return {
            "log_mass": float(
                width.ln() - (2 * _pi()).sqrt().ln() - centre * centre / 2
            ),
            "z_mean": float(centre),
            "z_var": float(variance),
            "d_location": float(centre / sd),
            "d_log_scale": float(centre * centre + variance - 1),
        }


@cache
def source_reference(
    lo: float, hi: float, location: float, scale: float
) -> dict[str, float]:
    """Return log mass, standardized moments and scores by a unit-interval integral.

    With `z = a + w u`, `u` in `[0, 1]`, `a = (lo - location) / scale` and
    `w = (hi - lo) / scale`, the mass is `phi(a) w I_0` with
    `I_r = int_0^1 u^r exp(A u - B u^2 / 2) du`, `A = -a w`, `B = w^2`. The Taylor
    coefficients of the integrand follow `n c_n = A c_{n-1} - B c_{n-2}`, so
    `I_r = sum_n c_n / (n + r + 1)`; the series stops when the exponential majorant
    of the omitted terms is below `1e-200 exp(-T)`, `T = |A| + B / 2`, a lower bound
    of `I_0`. The domain is `|A| <= 16` and `B <= 16`, which reaches arbitrarily far
    into the tails for narrow enough intervals. `d_log_scale` is the derivative of
    the log mass with respect to `log(scale)`.
    """
    values = tuple(float(x) for x in (lo, hi, location, scale))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("The oracle requires finite inputs")
    if not values[0] < values[1] or not values[3] > 0:
        raise ValueError("Require lo < hi and scale > 0")
    with localcontext() as context:
        context.prec = 250
        lower, upper, mean, sd = map(Decimal.from_float, values)
        width = (upper - lower) / sd
        a = (lower - mean) / sd
        tilt, curvature = -a * width, width * width
        if abs(tilt) > 16 or curvature > 16:
            raise ValueError("Outside the oracle domain |A| <= 16, B <= 16")
        total = abs(tilt) + curvature / 2
        goal = Decimal("1e-200") * (-total).exp()
        integrals = [Decimal(1), Decimal(1) / 2, Decimal(1) / 3]
        previous_two, previous = Decimal(0), Decimal(1)
        majorant = total
        for n in range(1, 4097):
            coefficient = (tilt * previous - curvature * previous_two) / n
            for r in range(3):
                integrals[r] += coefficient / (n + r + 1)
            previous_two, previous = previous, coefficient
            if n % 2 == 0:
                half_degree = n // 2
                majorant *= total / (half_degree + 1)
                ratio = total / (half_degree + 2)
                if ratio < 1 and majorant / (1 - ratio) <= goal:
                    break
        else:
            raise ArithmeticError("Oracle series failed to converge")
        i0, i1, i2 = integrals
        u_mean, u_second = i1 / i0, i2 / i0
        z_mean = a + width * u_mean
        z_second = a * a + 2 * a * width * u_mean + curvature * u_second
        log_mass = -a * a / 2 - (2 * _pi()).sqrt().ln() + width.ln() + i0.ln()
        return {
            "log_mass": float(log_mass),
            "z_mean": float(z_mean),
            "z_var": float(curvature * (u_second - u_mean * u_mean)),
            "d_location": float(z_mean / sd),
            "d_scale": float((z_second - 1) / sd),
            "d_log_scale": float(z_second - 1),
        }


def _reference(lo: float, hi: float, location: float, scale: float) -> dict:
    """Return the oracle whose domain contains the standardized geometry."""
    sd = Decimal(scale)
    width = (Decimal(hi) - Decimal(lo)) / sd
    a = (Decimal(lo) - Decimal(location)) / sd
    if max(abs(a), abs(a + width)) <= 12:
        if width >= Decimal("1e-40"):
            return interval_reference(lo, hi, location, scale)
        if width <= Decimal(2) ** -1000:
            return underflow_reference(lo, hi, location, scale)
    return source_reference(lo, hi, location, scale)


# `(lo, hi, location, scale)` with standardized width `2^-1100`.
UNDERFLOW_WITNESS = (0.0, 2.0**-600, 2.0**500, 2.0**500)
# `lo - location` overflows to -inf; the standardized interval is `(-2, -1.5]`.
AFFINE_OVERFLOW_WITNESS = (-(2.0**1023), -(2.0**1022), 2.0**1023, 2.0**1023)
# Both standardized ends round to `-2^30`; the standardized width times the centre
# is 1, so the interval is not a narrow-density integral. Log mass `-2^59`.
WIDE_COLLAPSE_WITNESS = (0.0, 2.0**-30, 2.0**30, 1.0)


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


def test_standardized_ends_lose_the_location_and_span_of_source_intervals() -> None:
    """The class beyond widths: affine overflow and collapsed ends of a tilted mass."""
    lo, hi, location, scale = AFFINE_OVERFLOW_WITNESS
    assert lo - location == -math.inf
    assert (hi - location) / scale == -1.5
    overflow = interval_reference(*AFFINE_OVERFLOW_WITNESS)
    assert abs(overflow["log_mass"] + 3.122269455465259) < 1e-14
    assert (
        abs(
            overflow["log_mass"]
            - source_reference(*AFFINE_OVERFLOW_WITNESS)["log_mass"]
        )
        < 1e-14
    )
    lo, hi, location, scale = WIDE_COLLAPSE_WITNESS
    lower, upper = (lo - location) / scale, (hi - location) / scale
    assert lower == upper == -(2.0**30)
    assert 0.5 * (hi - lo) / scale * abs(lower) == 0.5
    assert source_reference(*WIDE_COLLAPSE_WITNESS)["log_mass"] == -(2.0**59)


def _build_inputs(
    lo=0.0,
    hi=1e-17,
    location=1.0,
    scale=1.0,
    loading=0.0,
    split=1,
    interval_period=0,
):
    """Two persons, two periods, a normalized anchor and one interval-only Tobit.

    The interval is observed in `interval_period`; `_expected_contributions` assumes
    period 0 unless the loading is 0.
    """
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
            "y_lo": lo if period == interval_period else np.nan,
            "y_hi": hi if period == interval_period else np.nan,
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
        ref = _extreme_reference(lower, upper, location, sd)
        references.append(
            [
                ref["log_mass"],
                loading / sd * ref["z_mean"],
                1 - (loading / sd) ** 2 * (1 - ref["z_var"]),
            ]
        )
    return np.asarray(references)


def _assert_log_masses_close(actual, expected) -> None:
    """Require finite values within `max(1e-10, 4 ulp(expected))` of the reference.

    Moderate log masses keep the absolute budget `1e-10` with `rtol=0`. Far in the
    tails the correctly rounded log mass can be of the order `-2^59`, whose ulp is
    128, so no real-number guarantee below the float spacing exists; four ulps allow
    for the rounding of the leading `-c^2 / 2` term and the final sums. Neither an
    infinite value nor the public clipping floor can pass.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all(), actual
    assert np.isfinite(expected).all(), expected
    tolerance = np.maximum(1e-10, 4 * np.abs(np.spacing(expected)))
    difference = np.abs(actual - expected)
    assert np.all(difference <= tolerance), (actual, expected, difference, tolerance)


def test_source_intervals_match_the_oracle_in_the_update() -> None:
    """One jitted batch over the source-geometry classes.

    Collapsed, distorted, adjacent-float, reflected, scaled, underflowing,
    affine-overflowing and collapsed-wide intervals.
    """
    rows = [
        (0.0, 1e-17, 1.0, 1.0, 0.0),
        (0.0, 1e-9, -1.0, 1.0, 0.5),
        (-1e-13, 0.0, -3.0, 1.0, 0.0),
        (0.25, math.nextafter(0.25, math.inf), 4.0, 1.0, 0.5),
        (0.0, 2.0**20 * 1e-17, 2.0**20, 2.0**20, 2.0**19),
        (0.0, 0.3, 0.1, 1.0, 0.5),
        (*UNDERFLOW_WITNESS, 0.0),
        (-(2.0**-600), 0.0, -(2.0**500), 2.0**500, 2.0**499),
        (*AFFINE_OVERFLOW_WITNESS, 0.0),
        (2.0**1022, 2.0**1023, -(2.0**1023), 2.0**1023, 0.0),
        (*WIDE_COLLAPSE_WITNESS, 0.0),
        (-(2.0**-30), 0.0, -(2.0**30), 1.0, 2.0**-30),
        *[(*witness, 0.0) for witness in EXTREME_WITNESSES],
    ]
    got = np.asarray(jax.jit(jax.vmap(_scan_update))(jnp.asarray(rows)))
    expected = _update_references(rows)
    _assert_log_masses_close(got[:, 0], expected[:, 0])
    np.testing.assert_allclose(got[:, 1:], expected[:, 1:], rtol=0, atol=1e-8)


# `(lo, hi, location, scale)` whose standardized geometry leaves the float range
# while the log mass is a finite float: the span of the whole float range (mass 1);
# `c w` beyond the largest float (mass `Phi(0)`); a standardized end beyond the
# largest float (mass `Phi(0)`).
EXTREME_WITNESSES = (
    (-1e308, 1e308, 0.0, 1.0),
    (-1e200, 0.0, 0.0, 1e-110),
    (-1e300, 0.0, 0.0, 1e-10),
)


def _extreme_rows() -> list[tuple[float, float, float, float, float]]:
    """Source intervals whose standardized ends, centre, width or tilt overflow.

    Half-lines in effect (one end beyond `2^1024` standardized, or the other end
    moderate), the whole line, and moderate standardized intervals at source scales
    from `2^-1000` to `2^1000`; with both reflections and loadings 0 and `sigma / 2`.
    """
    rows = []
    for sigma in (1e-110, 1e-10, 1.0, 1e10, 1e100):
        for end in (-8.0, -1.0, 0.0, 1.0, 8.0):
            for mu in (0.0, 3.0 * sigma):
                hi = mu + end * sigma
                rows.append((-1e308, hi, mu, sigma))
                rows.append((-hi, 1e308, -mu, sigma))
        rows.append((-1e308, 1e308, 0.0, sigma))
        rows.append((-1e308, 1e308, 2.0 * sigma, sigma))
    for exponent in (-1000, -500, 0, 500, 1000):
        unit = math.ldexp(1.0, exponent)
        for lower, upper in ((-2.0, -1.5), (-0.3, 0.2), (-40.0, -39.9), (1.0, 9.0)):
            mu = 3.0 * unit
            rows.append((mu + lower * unit, mu + upper * unit, mu, unit))
            rows.append((-mu - upper * unit, -mu - lower * unit, -mu, unit))
    return [
        (lo, hi, mu, sigma, loading)
        for lo, hi, mu, sigma in rows
        for loading in (0.0, 0.5 * sigma)
    ]


def _extreme_reference(lo: float, hi: float, mu: float, sd: float) -> dict:
    """Return the oracle or the closed form of an interval that is a half-line.

    With exact standardized ends `a < b`: if `a <= -40` the mass below `a` is below
    `2^-53` of any `Phi(b)` with `b >= -8`, so the interval is the half-line below
    `b` (`log Phi(b)`, mean `-r(b)`, variance `1 - r(b) (r(b) + b)`), and the whole
    line if also `b >= 40`. Reflected intervals use the mirrored closed form.
    """
    with localcontext() as context:
        context.prec = 250
        lower, upper, mean, scale = map(Decimal.from_float, (lo, hi, mu, sd))
        a, b = float((lower - mean) / scale), float((upper - mean) / scale)
    if a > -40.0 or b < -8.0:
        if b < 40.0 or a > 8.0:
            return _reference(lo, hi, mu, sd)
        mirrored = _extreme_reference(-hi, -lo, -mu, sd)
        return {
            **mirrored,
            "z_mean": -mirrored["z_mean"],
            "d_location": -mirrored["d_location"],
        }
    if b >= 40.0:
        return {
            "log_mass": 0.0,
            "z_mean": 0.0,
            "z_var": 1.0,
            "d_location": 0.0,
            "d_log_scale": 0.0,
        }
    mass = 0.5 * math.erfc(-b / math.sqrt(2))
    mills = math.exp(-0.5 * b * b) / (math.sqrt(2 * math.pi) * mass)
    return {
        "log_mass": math.log(mass),
        "z_mean": -mills,
        "z_var": 1 - mills * (mills + b),
        "d_location": -mills / sd,
        "d_log_scale": -mills * b,
    }


def _physical_scores(row: jax.Array) -> jax.Array:
    """Scores of the log mass in the source location, `sigma` and the loading."""
    return jax.grad(lambda r: _scan_update(r)[0])(row)[2:]


def _kernel_scores(row: jax.Array) -> jax.Array:
    """Scores of `source_interval_terms` in the location and the log scale."""
    lower, upper, location, sigma, loading = row
    scale = jnp.hypot(sigma, loading)
    scores = jax.grad(
        lambda theta, ell: source_interval_terms(
            lower, upper, theta, scale * jnp.exp(ell)
        )[0],
        argnums=(0, 1),
    )(location, 0.0)
    return jnp.stack(scores)


@pytest.mark.long_running
def test_extreme_source_geometries_keep_finite_values_moments_and_scores() -> None:
    rows = _extreme_rows()
    array = jnp.asarray(rows, dtype=jnp.float64)
    got = np.asarray(jax.jit(jax.vmap(_scan_update))(array))
    kernel_scores = np.asarray(jax.jit(jax.vmap(_kernel_scores))(array))
    scores = np.asarray(jax.jit(jax.vmap(_physical_scores))(array))
    expected, expected_kernel, expected_scores = [], [], []
    for lo, hi, mu, sigma, loading in rows:
        sd = math.hypot(sigma, loading)
        ref = _extreme_reference(lo, hi, mu, sd)
        ratio = loading / sd
        expected.append(
            [ref["log_mass"], ratio * ref["z_mean"], 1 - ratio**2 * (1 - ref["z_var"])]
        )
        expected_kernel.append([ref["d_location"], ref["d_log_scale"]])
        d_scale = ref["d_log_scale"] / sd
        expected_scores.append(
            [ref["d_location"], d_scale * sigma / sd, d_scale * loading / sd]
        )
    expected, expected_scores = np.asarray(expected), np.asarray(expected_scores)
    assert np.isfinite(got).all(), rows
    assert np.isfinite(kernel_scores).all(), rows
    _assert_log_masses_close(got[:, 0], expected[:, 0])
    np.testing.assert_allclose(got[:, 1:], expected[:, 1:], rtol=0, atol=1e-8)
    np.testing.assert_allclose(kernel_scores, expected_kernel, rtol=1e-5, atol=1e-8)
    # The update divides the loading direction by `s` (`w = v / s`), whose default
    # derivative in `s` forms `s^-2`; below `s = 1e-154` that overflows, in every
    # measurement family. The scores through the update are checked above it.
    regular = array[:, 3] >= 1e-154
    assert np.isfinite(scores[regular]).all(), rows
    np.testing.assert_allclose(
        scores[regular], expected_scores[regular], rtol=1e-5, atol=1e-8
    )


def _source_geometry_rows() -> list[tuple[float, float, float, float, float]]:
    """Affine-overflow, collapsed-wide and branch-boundary families with loadings."""
    rows = []
    # The same standardized `(-2, -1.5]` across the overflow boundary of `lo - mu`.
    for power in (0, -1, -2, -10):
        lo, hi, location, sigma = [
            math.ldexp(x, power) for x in AFFINE_OVERFLOW_WITNESS
        ]
        rows.append((lo, hi, location, sigma, 0.0))
        rows.append((-hi, -lo, -location, sigma, 0.0))
    # The far-tail tilt `|c| w` decides between the narrow series and the wide
    # branch at 0.5; at exponent 27 the ends separate for the tilts 2 and 4.
    for exponent in (27, 30, 40):
        location = math.ldexp(1.0, exponent)
        for tilt in (0.25, 0.5 - 2**-10, 0.5, 0.5 + 2**-10, 1.0, 2.0, 4.0):
            hi = math.ldexp(tilt, -exponent)
            rows.append((0.0, hi, location, 1.0, 0.0))
            rows.append((-hi, 0.0, -location, 1.0, 0.0))
        # A loading of `1 / location` maps the standardized mean to a unit shift.
        hi = math.ldexp(1.0, -exponent)
        rows.append((0.0, hi, location, 1.0, 1.0 / location))
        rows.append((-hi, 0.0, -location, 1.0, 1.0 / location))
    # Joint power-of-two rescalings keep the far-tail standardized geometry.
    for exponent in (-8, 100, 400):
        rows.append(
            (
                0.0,
                math.ldexp(1.0, exponent - 30),
                math.ldexp(1.0, exponent + 30),
                math.ldexp(1.0, exponent),
                0.0,
            )
        )
    return rows


@pytest.mark.long_running
def test_source_geometry_family_values_moments_and_transformed_paths() -> None:
    rows = _source_geometry_rows()
    array = jnp.asarray(rows, dtype=jnp.float64)
    expected = _update_references(rows)
    compiled = jax.jit(jax.vmap(_scan_update))
    batched = np.asarray(compiled(array))
    assert batched.dtype == np.float64
    assert np.isfinite(batched).all()
    _assert_log_masses_close(batched[:, 0], expected[:, 0])
    np.testing.assert_allclose(batched[:, 1:], expected[:, 1:], rtol=0, atol=1e-8)
    permutation = np.random.default_rng(107).permutation(len(array))
    permuted = np.asarray(compiled(array[permutation]))
    _assert_log_masses_close(permuted[:, 0], batched[permutation, 0])
    np.testing.assert_allclose(
        permuted[:, 1:], batched[permutation, 1:], rtol=0, atol=1e-12
    )
    for index in (0, 8, 16):
        eager = np.asarray(_scan_update(array[index]))
        _assert_log_masses_close(eager[0], expected[index, 0])
        np.testing.assert_allclose(eager[1:], expected[index, 1:], rtol=0, atol=1e-8)


@pytest.mark.long_running
def test_source_geometry_location_and_log_scale_scores() -> None:
    """Scale-one controls and far collapsed intervals keep informative scores."""
    far, hi = 2.0**30, 2.0**-30
    sources = np.array(
        [
            [-2.0, -1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0],
            [0.0, hi, far, 1.0],
            [-hi, 0.0, far, 1.0],
        ]
    )
    coordinates = np.array([[2.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    sources = np.concatenate(
        [sources, [[lo, hi, 1.0, scale] for lo, hi, _, scale in EXTREME_WITNESSES]]
    )
    coordinates = np.concatenate([coordinates, np.zeros((len(EXTREME_WITNESSES), 2))])
    got = np.asarray(
        jax.jit(jax.vmap(jax.grad(_dimensionless_loglike, argnums=1)))(
            jnp.asarray(sources), jnp.asarray(coordinates)
        )
    )
    expected = []
    for (lo, upper, location_unit, scale_unit), (theta, ell) in zip(
        sources, coordinates, strict=True
    ):
        ref = _extreme_reference(
            lo, upper, theta * location_unit, scale_unit * math.exp(ell)
        )
        expected.append([ref["d_location"] * location_unit, ref["d_log_scale"]])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.long_running
@pytest.mark.parametrize("split", [1, 2], ids=["unsplit", "split_two"])
@pytest.mark.parametrize(
    "witness",
    [AFFINE_OVERFLOW_WITNESS, WIDE_COLLAPSE_WITNESS],
    ids=["affine_overflow", "wide_collapse"],
)
def test_public_terminal_interval_keeps_its_source_geometry(witness, split) -> None:
    """An interval at the last update shows a wrong finite value on its own.

    Raw contributions, every public objective, posterior means and scores.
    """
    lo, hi, location, scale = witness
    inputs, params = _build_inputs(
        lo=lo, hi=hi, location=location, scale=scale, split=split, interval_period=1
    )
    ref = _reference(lo, hi, location, scale)
    anchor = -0.5 * math.log(4 * math.pi) - 0.25
    debug = inputs["debug_loglike"](params)
    raw = (
        debug["all_contributions"]
        .set_index(["aug_period", "measurement", "id"])
        .sort_index()["contribution"]
    )
    _assert_log_masses_close(
        raw,
        [anchor, anchor, 0.0, 0.0, anchor, anchor, ref["log_mass"], ref["log_mass"]],
    )
    person = np.full(2, 2 * anchor + ref["log_mass"])
    for got in (inputs["loglikeobs"](params), debug["contributions"]):
        _assert_log_masses_close(got, person)
    value, gradient = inputs["loglike_and_gradient"](params)
    for got in (inputs["loglike"](params), debug["value"], value):
        _assert_log_masses_close(got, person.sum())
    # The zero loading leaves the anchor posteriors `+/- 1/2` unchanged.
    post = debug["post_update_states"].sort_values(["aug_period", "measurement", "id"])
    np.testing.assert_allclose(post["skills"], [-0.5, 0.5] * 4, rtol=0, atol=1e-10)
    assert np.isfinite(gradient).all()
    if scale == 1.0:
        positions = [
            params.index.get_loc(("controls", 1, "y", "constant")),
            params.index.get_loc(("meas_sds", 1, "y", "-")),
        ]
        np.testing.assert_allclose(
            gradient[positions],
            2 * np.array([ref["d_location"], ref["d_scale"]]),
            rtol=1e-5,
            atol=1e-8,
        )


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


@pytest.mark.long_running
def test_public_terms_keep_an_interval_whose_standardized_width_underflows() -> None:
    """Raw contributions and all public objectives, not the clipping floor."""
    lo, hi, location, scale = UNDERFLOW_WITNESS
    inputs, params = _build_inputs(lo=lo, hi=hi, location=location, scale=scale)
    log_mass = underflow_reference(lo, hi, location, scale)["log_mass"]
    anchor = -0.5 * math.log(4 * math.pi) - 0.25
    debug = inputs["debug_loglike"](params)
    raw = (
        debug["all_contributions"]
        .set_index(["aug_period", "measurement", "id"])
        .sort_index()["contribution"]
    )
    np.testing.assert_allclose(
        raw,
        [anchor, anchor, log_mass, log_mass, anchor, anchor, 0.0, 0.0],
        rtol=0,
        atol=1e-10,
    )
    person = np.full(2, 2 * anchor + log_mass)
    for got in (inputs["loglikeobs"](params), debug["contributions"]):
        np.testing.assert_allclose(got, person, rtol=0, atol=1e-10)
    for got in (inputs["loglike"](params), debug["value"]):
        np.testing.assert_allclose(got, person.sum(), rtol=0, atol=1e-10)


def _underflow_rows() -> list[tuple[float, float, float, float, float]]:
    """Normal inputs at the subnormal boundaries of the standardized (half-)width."""
    sigma = 2.0**500
    rows = []
    for exponent in (-1021, -1022, -1023, -1073, -1074, -1075, -1076, -1100):
        hi = 2.0 ** (exponent + 500)
        rows.append((0.0, hi, sigma, sigma, 0.0))
        rows.append((-hi, 0.0, -sigma, sigma, 0.0))
    for exponent in (400, 450):
        scale = 2.0**exponent
        rows.append((0.0, 2.0 ** (exponent - 1100), scale, scale, 0.0))
    lo, hi, location, scale = UNDERFLOW_WITNESS
    rows.append((lo, hi, location, scale, scale / 2))
    rows.append((-hi, 0.0, -location, scale, scale / 2))
    rows.append((0.0, 1e-200, 1e150, 1e150, 0.0))
    return rows


@pytest.mark.long_running
def test_underflowing_widths_values_moments_and_transformed_paths() -> None:
    rows = _underflow_rows()
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


def _dimensionless_loglike(source: jax.Array, coordinates: jax.Array) -> jax.Array:
    """Log mass at location `theta * location_unit` and scale `scale_unit * exp(ell)`.

    Differentiating in `(theta, ell)` gives O(1) scores, which an absolute tolerance
    cannot pass as zero, unlike the tiny scores in the source location and scale.
    """
    lower, upper, location_unit, scale_unit = source
    theta, ell = coordinates
    row = jnp.stack(
        [lower, upper, theta * location_unit, scale_unit * jnp.exp(ell), jnp.array(0.0)]
    )
    return _scan_update(row)[0]


@pytest.mark.long_running
def test_underflowing_widths_location_and_log_scale_scores() -> None:
    _, hi, _, sigma = UNDERFLOW_WITNESS
    sources = np.array(
        [
            [0.0, hi, sigma, sigma],
            [-hi, 0.0, sigma, sigma],
            [0.0, 2.0**-574, sigma, sigma],
            [0.0, 2.0**-575, sigma, sigma],
        ]
    )
    coordinates = np.array([[0.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])
    got = np.asarray(
        jax.jit(jax.vmap(jax.grad(_dimensionless_loglike, argnums=1)))(
            jnp.asarray(sources), jnp.asarray(coordinates)
        )
    )
    expected = []
    for (lo, upper, base, _), (theta, _) in zip(sources, coordinates, strict=True):
        ref = underflow_reference(lo, upper, theta * base, base)
        expected.append([ref["d_location"] * base, ref["d_log_scale"]])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


MIXTURE_EVENTS = (
    ObservationEvent.INTERVAL,
    ObservationEvent.DENSITY,
    ObservationEvent.MISSING,
    ObservationEvent.LOWER_TAIL,
    ObservationEvent.UPPER_TAIL,
)
MIXTURE_MEANS = (-0.25, 0.25)
MIXTURE_WEIGHTS = (0.3, 0.7)


@dataclass(frozen=True)
class _MixtureGeometry:
    """Source values of a five-event, two-component batch around one interval.

    The loading is `sigma / 2`, the exact value and the lower tail end `sigma / 4`,
    the upper tail end `3 sigma / 4`, so every standardized term is moderate.
    """

    interval: tuple[float, float]
    """The `(lo, hi]` ends of the interval row."""
    sigma: float
    """The measurement-error SD and the unit of the other source values."""
    offset: float
    """The control part of the latent location."""
    score_unit: float
    """Unit by which the batch is rescaled before differentiating. Conditional
    probabilities, posteriors and the score in `theta = offset / sigma` do not
    depend on it. A physical score at `sigma` near `2^1022` is below the smallest
    normal float and XLA's CPU backend flushes it to zero, so that geometry is
    differentiated in units of `sigma`; the underflowing interval keeps physical
    units, since its ends would underflow when divided by `sigma`."""

    @property
    def loading(self) -> float:
        return self.sigma / 2

    @property
    def exact(self) -> float:
        return self.sigma / 4

    @property
    def lower_end(self) -> float:
        return self.sigma / 4

    @property
    def upper_end(self) -> float:
        return 3 * self.sigma / 4


# The standardized width of the interval underflows.
UNDERFLOW_GEOMETRY = _MixtureGeometry(UNDERFLOW_WITNESS[:2], 2.0**500, 2.0**499, 1.0)
# `lo - mu` overflows for both components; the standardized ends are moderate.
OVERFLOW_GEOMETRY = _MixtureGeometry(
    (-(2.0**1023), -(2.0**1022)), 2.0**1022, 2.0**1023, 2.0**1022
)
MIXTURE_GEOMETRIES = [UNDERFLOW_GEOMETRY, OVERFLOW_GEOMETRY]
MIXTURE_IDS = ["underflow", "overflow"]


def _mixture_kwargs(
    geometry: _MixtureGeometry, events: tuple[ObservationEvent, ...]
) -> dict:
    """One row per event, two mixture components, the geometry's interval first."""
    is_interval = events[0] == ObservationEvent.INTERVAL
    first_bounds = list(geometry.interval) if is_interval else [jnp.nan] * 2
    return {
        "states": jnp.broadcast_to(
            jnp.asarray(MIXTURE_MEANS)[None, :, None], (5, 2, 1)
        ),
        "upper_chols": jnp.ones((5, 2, 1, 1)),
        "loadings": jnp.array([geometry.loading]),
        "control_params": jnp.array([geometry.offset]),
        "meas_sd": jnp.array(geometry.sigma),
        "measurements": jnp.array([jnp.nan, geometry.exact, jnp.nan, jnp.nan, jnp.nan]),
        "events": jnp.array(events, dtype=jnp.int8),
        "controls": jnp.array([[1.0], [1.0], [jnp.nan], [1.0], [1.0]]),
        "log_mixture_weights": jnp.broadcast_to(
            jnp.log(jnp.asarray(MIXTURE_WEIGHTS)), (5, 2)
        ),
        "is_limited": jnp.array(True),
        "cut_table": jnp.array([0.0, geometry.lower_end, geometry.upper_end, 0.0]),
        "codes_from_values": jnp.array(False),
        "interval_bounds": jnp.array(
            [
                first_bounds,
                [jnp.nan, jnp.nan],
                [jnp.nan, jnp.nan],
                [-jnp.inf, geometry.lower_end],
                [geometry.upper_end, jnp.inf],
            ]
        ),
    }


def _component_terms(
    geometry: _MixtureGeometry, event: ObservationEvent, mean: float
) -> tuple:
    """Log likelihood, standardized mean and variance, and `theta` score."""
    sigma = geometry.sigma
    sd = math.hypot(sigma, geometry.loading)
    mu = geometry.offset + geometry.loading * mean
    if event == ObservationEvent.INTERVAL:
        ref = _reference(*geometry.interval, mu, sd)
        return ref["log_mass"], ref["z_mean"], ref["z_var"], ref["d_location"] * sigma
    if event == ObservationEvent.DENSITY:
        zmean = (geometry.exact - mu) / sd
        ll = -0.5 * zmean * zmean - math.log(sd * math.sqrt(2 * math.pi))
        return ll, zmean, 0.0, zmean * sigma / sd
    if event in (ObservationEvent.LOWER_TAIL, ObservationEvent.UPPER_TAIL):
        sign = -1 if event == ObservationEvent.LOWER_TAIL else 1
        end = geometry.lower_end if sign == -1 else geometry.upper_end
        beta = sign * (mu - end) / sd
        mass = 0.5 * math.erfc(-beta / math.sqrt(2))
        mills = math.exp(-0.5 * beta * beta) / (math.sqrt(2 * math.pi) * mass)
        return (
            math.log(mass),
            sign * mills,
            1 - mills * (mills + beta),
            sign * mills * sigma / sd,
        )
    return 0.0, 0.0, 1.0, 0.0


def _update(kwargs: dict) -> tuple:
    return limited_update_from_loop_kwargs(kwargs, has_intervals=True)


@pytest.mark.long_running
@pytest.mark.parametrize("geometry", MIXTURE_GEOMETRIES, ids=MIXTURE_IDS)
def test_extreme_interval_in_a_mixed_event_mixture_batch(geometry) -> None:
    """Interval, density, missing and both tails on real person and mixture axes."""
    sigma = geometry.sigma
    ratio = geometry.loading / math.hypot(sigma, geometry.loading)
    expected_ll, expected_means, expected_vars, expected_weights = [], [], [], []
    expected_score = 0.0
    for event in MIXTURE_EVENTS:
        terms = np.array(
            [_component_terms(geometry, event, mean) for mean in MIXTURE_MEANS]
        )
        weighted = np.log(MIXTURE_WEIGHTS) + terms[:, 0]
        maximum = weighted.max()
        total = maximum + math.log(np.exp(weighted - maximum).sum())
        expected_ll.append(total)
        expected_means.append(np.asarray(MIXTURE_MEANS) + ratio * terms[:, 1])
        expected_vars.append(1 - ratio**2 * (1 - terms[:, 2]))
        expected_weights.append(weighted - total)
        expected_score += float(np.exp(weighted - total) @ terms[:, 3])
    kwargs = _mixture_kwargs(geometry, MIXTURE_EVENTS)
    result = jax.jit(_update)(kwargs)
    states, chols, logweights, loglikes, _ = result
    assert all(np.isfinite(np.asarray(x)).all() for x in result[:4])
    np.testing.assert_allclose(loglikes, expected_ll, rtol=0, atol=1e-10)
    np.testing.assert_allclose(states[..., 0], expected_means, rtol=0, atol=1e-8)
    np.testing.assert_allclose(chols[..., 0, 0] ** 2, expected_vars, rtol=0, atol=1e-8)
    np.testing.assert_allclose(logweights, expected_weights, rtol=0, atol=1e-10)
    # Log weights of magnitude ~766 carry rounding errors of ~ulp(766) = 1.1e-13, so
    # their exponentials sum to 1 only to about that precision.
    np.testing.assert_allclose(np.exp(logweights).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    unit = geometry.score_unit
    rescaled = {
        **kwargs,
        "loadings": kwargs["loadings"] / unit,
        "meas_sd": kwargs["meas_sd"] / unit,
        "measurements": kwargs["measurements"] / unit,
        "cut_table": kwargs["cut_table"] / unit,
        "interval_bounds": kwargs["interval_bounds"] / unit,
    }
    score = jax.jit(
        jax.grad(
            lambda theta: _update(
                {**rescaled, "control_params": jnp.reshape(theta * sigma / unit, (1,))}
            )[3].sum()
        )
    )(jnp.array(geometry.offset / sigma))
    assert np.isfinite(np.asarray(score)).all()
    np.testing.assert_allclose(score, expected_score, rtol=1e-5, atol=1e-8)


@pytest.mark.long_running
@pytest.mark.parametrize("geometry", MIXTURE_GEOMETRIES, ids=MIXTURE_IDS)
def test_interval_bounds_leave_other_events_unchanged(geometry) -> None:
    """Without an interval row, the update is identical with and without bounds."""
    inactive = _mixture_kwargs(
        geometry, (ObservationEvent.MISSING, *MIXTURE_EVENTS[1:])
    )
    legacy = {key: value for key, value in inactive.items() if key != "interval_bounds"}
    update = jax.jit(_update)
    for with_bounds, without_bounds in zip(
        update(inactive)[:4], update(legacy)[:4], strict=True
    ):
        np.testing.assert_array_equal(with_bounds, without_bounds)
