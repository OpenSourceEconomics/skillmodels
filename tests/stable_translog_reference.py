"""Independent scalar reference for the stable_translog design audit.

No project imports and no emulation of JAX execution. Decimal evaluates the original
real equation, not the normalized expression proposed for production. Fractions give
exact counterexamples to dynamical claims. Running this file tests mathematical
claims only, never native skillmodels/XLA behavior.
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import combinations
from typing import Sequence


def dec(x: object) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def clip(x: object, scale: object = 3, power: int = 4) -> Decimal:
    """Original equation in 90-digit arithmetic; finite, positive scale, even p."""
    x, s = dec(x), dec(scale)
    if not s.is_finite() or s <= 0 or type(power) is not int or power < 2 or power % 2:
        raise ValueError("positive finite scale and even integer power >=2 required")
    if x.is_nan():
        return x
    if x.is_infinite():
        return s.copy_sign(x)
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.Emax = 9999999
        ctx.Emin = -9999999
        return +(x / (1 + abs(x / s) ** power) ** (Decimal(1) / power))


def transition(
    states: Sequence[object],
    params: Sequence[object],
    scale: object = 3,
    power: int = 4,
) -> Decimal:
    n = len(states)
    if len(params) != 2 * n + n * (n - 1) // 2 + 1:
        raise ValueError("parameter count does not match the translog layout")
    with localcontext() as ctx:
        ctx.prec = 90
        x, b = list(map(dec, states)), list(map(dec, params))
        g = [clip(v, scale, power) for v in x]
        ans = b[-1] + sum((x[j] * b[j] for j in range(n)), Decimal(0))
        ans += sum((g[j] ** 2 * b[n + j] for j in range(n)), Decimal(0))
        for k, (a, z) in enumerate(combinations(range(n), 2)):
            ans += b[2 * n + k] * g[a] * g[z]
        return +ans


def clip_derivative(x: object, scale: object = 3, power: int = 4) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 90
        q = abs(dec(x) / dec(scale)) ** power
        return +((1 + q) ** (-Decimal(1) - Decimal(1) / power))


def matvec(a, x):
    return tuple(sum((v * w for v, w in zip(row, x)), Q(0)) for row in a)


def matmul(a, b):
    return tuple(
        tuple(
            sum((a[i][k] * b[k][j] for k in range(len(b))), Q(0))
            for j in range(len(b[0]))
        )
        for i in range(len(a))
    )


def stability_witnesses():
    # Bounded exogenous forcing: x[t+1] = -a*x[t] + z[t].
    inverse_gain = []
    for a in (Q(1, 4), Q(1, 2), Q(3, 4), Q(7, 8)):
        x = -1 / (1 - a)
        amplitude = abs(x)
        for t in range(12):
            x = -a * x + (-1) ** t
            assert abs(x) == amplitude
        assert amplitude > 1 / (1 + a)
        inverse_gain.append(
            {
                "a": str(a),
                "true_amplitude": str(amplitude),
                "claimed_inverse_gain": str(1 / (1 + a)),
            }
        )
    # Each matrix has eigenvalues a,a. Their product has an eigenvalue >1.
    switching = []
    for a, b in ((Q(1, 2), Q(1)), (Q(3, 4), Q(1, 2)), (Q(1, 4), Q(1))):
        aa = ((a, b), (Q(0), a))
        ab = ((a, Q(0)), (b, a))
        product = matmul(ab, aa)
        det_i_minus = (1 - product[0][0]) * (1 - product[1][1]) - product[0][
            1
        ] * product[1][0]
        assert det_i_minus == (1 - a * a) ** 2 - b * b < 0
        x = (Q(1), Q(1))
        for _ in range(50):
            x = matvec(product, x)
        assert max(x) > 1
        switching.append(
            {
                "a": str(a),
                "b": str(b),
                "det(I-product)": str(det_i_minus),
                "max_state_after_100_steps": float(max(x)),
            }
        )
    # Reported entry bounds do not certify rho(A)<1.
    a = tuple(
        tuple(Q(17, 20) if i == j else Q(3, 25) for j in range(4)) for i in range(4)
    )
    assert matvec(a, (Q(1),) * 4) == (Q(121, 100),) * 4
    # Default cf=i-prediction can already change the linear stability condition.
    x, investment, first_stage, kappa = Q(2), Q(0), Q(1), Q(-1)
    base = Q(1, 2) * x
    composed = base + kappa * (investment - first_stage * x)
    assert composed == Q(3, 2) * x
    return {
        "bounded_input_inverse_gain": inverse_gain,
        "switched_schur_family": switching,
        "small_entry_matrix_eigenvalue": "121/100",
        "control_function_effective_own_coefficient": "3/2",
    }


def default_certificate():
    """Prove root uniqueness by a global inequality, not a sign-change grid.

    For z=-x>0, H(z)=f(-z)+z=-d+a*z-b*z^2/sqrt(1+(z/s)^4).
    z/sqrt(1+(z/s)^4)<=s/sqrt(2). Thus H(z)>=m*z-d.
    For z>=1/4 this is positive, while on [0,1/4], H'>=a-2*b/4>0.
    Positive x cannot be fixed. A separate derivative bound proves f increasing,
    hence monotone convergence to the unique fixed point from every finite start.
    """
    with localcontext() as ctx:
        ctx.prec = 90
        a, b, d, s = map(Decimal, ("0.231", "0.103", "0.003", "3"))
        margin = a - b * s / Decimal(2).sqrt()
        cut = Decimal("0.25")
        assert margin * cut > d
        assert a - 2 * b * cut > 0
        zmax = s / Decimal(5) ** Decimal("0.25")
        max_ggprime = zmax / (Decimal("1.2") ** Decimal("1.5"))
        min_fprime = Decimal("0.769") - 2 * b * max_ggprime
        assert min_fprime > 0
        lo, hi = Decimal(0), cut
        for _ in range(240):
            z = (lo + hi) / 2
            h = -d + a * z - b * clip(z) ** 2
            if h < 0:
                lo = z
            else:
                hi = z
        root = -(lo + hi) / 2
        residual = transition([root], ["0.769", "-0.103", "-0.003"]) - root
        assert abs(residual) < Decimal("1e-65")
        x = Decimal(-3)
        maximum = abs(x)
        for _ in range(200):
            next_x = transition([x], ["0.769", "-0.103", "-0.003"])
            assert x <= next_x <= root + Decimal("1e-65")
            x = next_x
            maximum = max(maximum, abs(x))
        fidelity = {
            str(v): str(1 - (clip(v) / v) ** 2) for v in map(Decimal, ("1", "2", "2.5"))
        }
        assert Decimal(fidelity["1"]) <= Decimal(".01")
        assert Decimal(fidelity["2"]) <= Decimal(".1")
        # The same proof covers this deliberate neighborhood around the default.
        for bs in ("0.1025", "0.103", "0.1035"):
            bb = Decimal(bs)
            assert (a - bb * s / Decimal(2).sqrt()) * Decimal("0.5") > d
            assert a - 2 * bb * Decimal("0.5") > 0
        return {
            "fidelity": fidelity,
            "fixed_point": str(root),
            "fixed_point_count": 1,
            "fixed_point_residual": str(residual),
            "sup_0_through_200": str(maximum),
            "state_at_200": str(x),
            "root_margin": str(margin),
            "global_lower_bound_fprime": str(min_fprime),
        }


def integration_diagnostics():
    # An explicitly illustrative 9-dimensional spherical-radial sigma rule:
    # weights 1/(2d), nodes +/-sqrt(d)*e_j, mu=0, P=I. NOT a claim about CHS weights.
    d = 9
    nodes = []
    for j in range(d):
        for sign in (-1, 1):
            x = [Decimal(0)] * d
            x[j] = Decimal(sign) * Decimal(d).sqrt()
            nodes.append(x)
    mean_tl = sum((x[0] ** 2 for x in nodes), Decimal(0)) / len(nodes)
    mean_st = sum((clip(x[0]) ** 2 for x in nodes), Decimal(0)) / len(nodes)
    assert mean_tl == 1
    assert mean_st < Decimal(".71")
    # Changing only the stable_translog default scale to 2.5 is not pinned by
    # the planned small-state/full-output test; it does violate the contract at 2.
    params = [".2", ".1", ".12", ".08", ".04", ".05", ".05", ".03", ".06", ".04"]
    state = [".3", "-.5", ".8"]
    tl = transition(state, params, scale=Decimal("1e20"))
    mutated = transition(state, params, scale="2.5")
    assert abs(mutated - tl) < Decimal(".001")
    bad_fidelity = 1 - (clip(2, "2.5") / 2) ** 2
    assert bad_fidelity > Decimal(".1")
    return {
        "illustrative_sigma_rule": "d=9, +/-sqrt(d)e_j, weights 1/(2d)",
        "sigma_mean_translog_square": str(mean_tl),
        "sigma_mean_stable_square": str(mean_st),
        "default_scale_mutant_2.5_error_at_2": str(bad_fidelity),
        "mutant_small_state_absolute_error": str(abs(mutated - tl)),
        "bypass_loglike_ratio": "0 for any finite nonzero baseline loglike if stable dispatch aliases translog",
    }


def overflow_witness():
    x = Decimal("1e80")
    scale = Decimal(3)
    with localcontext() as ctx:
        ctx.prec = 90
        internal = (x / scale) ** 4
        assert math.isfinite(float(x))
        assert internal > Decimal.from_float(sys.float_info.max)
        expected = transition([x], [0, 1, 0])
        assert abs(expected - 9) < Decimal("1e-70")
    return {
        "finite_float64_input": "1e80",
        "params": [0, 1, 0],
        "internal_fourth_power": str(internal),
        "correct_transition": str(expected),
        "proposed_JAX_outcome": "0 under ordinary IEEE overflow; static prediction, not executed",
        "native_execution": "not_run",
    }


def main():
    result = {
        "stability": stability_witnesses(),
        "default": default_certificate(),
        "finite_overflow": overflow_witness(),
        "integration": integration_diagnostics(),
        "execution_scope": "independent standard-library math only; no JAX/skillmodels imports",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
