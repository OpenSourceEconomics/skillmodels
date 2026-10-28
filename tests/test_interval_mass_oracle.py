"""Independent decimal reference for normal interval probabilities.

`interval_reference` integrates `exp(-z^2 / 2)` with a 220-digit power series at the
exact binary64 endpoints and imports neither the project, JAX, NumPy nor SciPy.
`quantization_case` rounds the endpoint log-CDFs to binary64 to show that a difference
of rounded endpoint log-CDFs cannot represent a narrow interval's mass. Adapted from the
closure review of the limited-measurement CHS work (`interval_mass_oracle.py`).
"""

import math
from decimal import Decimal, localcontext
from functools import cache
from typing import Any

_PRECISION = 220


@cache
def _pi() -> Decimal:
    with localcontext() as context:
        context.prec = _PRECISION + 20

        def atan_inverse(q: Decimal) -> Decimal:
            x = Decimal(1) / q
            term = x
            total = x
            n = 0
            while abs(term) > Decimal(10) ** (-_PRECISION - 10):
                n += 1
                term *= -x * x
                total += term / (2 * n + 1)
            return total

        return 16 * atan_inverse(Decimal(5)) - 4 * atan_inverse(Decimal(239))


def _cdf(x: Decimal) -> Decimal:
    """Normal CDF from the integral of its power series, for |x| <= 12."""
    if abs(x) > 12:
        raise ValueError("This oracle is intentionally restricted to |endpoint| <= 12.")
    term = total = x
    n = 0
    while True:
        n += 1
        term *= -x * x / (2 * n)
        addition = term / (2 * n + 1)
        total += addition
        if n > abs(x * x) and abs(addition) < Decimal(10) ** (-_PRECISION + 20):
            break
    return Decimal("0.5") + total / (2 * _pi()).sqrt()


@cache
def interval_reference(lower: float, upper: float) -> dict[str, float]:
    """Return log mass and endpoint/location derivatives at exact binary64 ends."""
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("Need finite, distinct, increasing binary64 endpoints.")
    with localcontext() as context:
        context.prec = _PRECISION
        a, b = Decimal(float(lower)), Decimal(float(upper))
        mass = _cdf(b) - _cdf(a)
        if mass <= 0:
            raise ArithmeticError("Oracle lost probability mass.")
        norm = (2 * _pi()).sqrt()
        pa = (-a * a / 2).exp() / norm / mass
        pb = (-b * b / 2).exp() / norm / mass
        return {
            "mass": float(mass),
            "log_mass": float(mass.ln()),
            "dlog_da": float(-pa),
            "dlog_db": float(pb),
            "dlog_deta": float(pa - pb),
        }


def quantization_case(width: float) -> dict[str, Any]:
    """Mirror the generic kernel's reflection, then round endpoint log-CDFs."""
    reference = interval_reference(0.0, width)
    with localcontext() as context:
        context.prec = _PRECISION
        # The generic kernel reflects (0, width] to (-width, 0].
        lo = _cdf(-Decimal(width)).ln()
        hi = _cdf(Decimal(0)).ln()
        lo64, hi64 = float(lo), float(hi)
        difference = lo64 - hi64
        output = None if difference == 0 else hi64 + math.log(-math.expm1(difference))
        ulp = math.ulp(hi64)
        smallest_finite = hi64 + math.log(-math.expm1(-ulp))
        return {
            "width": width,
            "oracle_mass": reference["mass"],
            "oracle_log_mass": reference["log_mass"],
            "exact_endpoint_logcdf_difference": float(lo - hi),
            "endpoint_logcdf_ulp": ulp,
            "rounded_logcdf_lower": lo64,
            "rounded_logcdf_upper": hi64,
            "rounded_endpoint_algorithm_log_mass": "-inf" if output is None else output,
            "rounded_endpoint_algorithm_log_error": None
            if output is None
            else output - reference["log_mass"],
            "minimum_nonzero_log_mass_near_log_half": smallest_finite,
            "minimum_nonzero_error": smallest_finite - reference["log_mass"],
        }


def test_rounded_endpoint_log_cdfs_lose_the_narrow_interval_mass() -> None:
    """At `(0, 1e-17]` both rounded endpoint log-CDFs coincide; the mass is finite."""
    witness = quantization_case(1e-17)
    assert witness["rounded_logcdf_lower"] == witness["rounded_logcdf_upper"]
    assert witness["minimum_nonzero_error"] > 2.63
    assert abs(witness["oracle_log_mass"] + 40.06288511410345) < 1e-13
