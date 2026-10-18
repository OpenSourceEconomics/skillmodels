# `asymmetric_translog`: translog with sign-split squares around an estimated knot

Date: 2026-10-16. Status: design, to be implemented on branch `asymmetric-translog` from
`stable-translog` at 9cd73db.

## 1. Why

In the health-cognition application the estimated translog own-square of physical
capacity is positive (0.07 to 0.10 at ages 74+ with own linear coefficient 0.97). Binned
next-period means of the underlying measures show why: mobility and large-muscle counts
revert strongly at the unhealthy floor (slope 0.6) and are near-linear at the healthy
end (slope 0.83). A single quadratic term has to buy the floor reversion with a slope
above one at the healthy end (own slope ρ + 2γx exceeds 1 for x > 0.15, inside the
data), and extrapolates to an unstable fixed point at +1.5. The mirror pattern (ceiling
at zero depressive symptoms, negative own-square, slope above one at the unhealthy end)
appears for psychological capacity. A symmetric clip (`stable_translog`) cannot separate
the two ends. The remedy is one curvature coefficient per side of a knot.

## 2. Function

For states x (all factors, latent and observed, in the model's factor order) and a
parameter vector:

```
f(x) = c + Σ_j β_j x_j
         + Σ_j γ⁺_j max(x_j − κ_j, 0)²
         + Σ_j γ⁻_j min(x_j − κ_j, 0)²
         + Σ_{a<b} δ_ab x_a x_b
```

- `max(·,0)²` and `min(·,0)²` are C¹ in x and in κ, so the unscented filter, the
  gradient and the optimiser see a smooth function. Implement as
  `jnp.maximum(d, 0.0) ** 2` and `jnp.minimum(d, 0.0) ** 2` with `d = x − κ`; no
  `where`.
- Nesting: γ⁺ = γ⁻ = γ gives `c + Σβx + Σγ(x−κ)² + Σδ x_a x_b`, a translog with the
  linear and constant terms shifted by `−2γκ` and `γκ²`. With κ = 0 it is exactly
  `translog`. Tests pin both.
- The knot κ_j is a parameter per factor, estimated by default. Users fix it through the
  ordinary constraints machinery (profiling over a grid of knots is the intended first
  use).
- No clipping. If a fixed-point check of an estimated law shows a spurious attractor far
  in the convex tail, composing with `_soft_clip` is a later extension, not part of this
  spec.

## 3. Parameter layout and names

`params_asymmetric_translog(factors)` returns, in this order:

```
[f for f in factors]                      # β, linear
+ [f"{f} ** 2 (above)" for f in factors]  # γ⁺
+ [f"{f} ** 2 (below)" for f in factors]  # γ⁻
+ [f"{f} knot" for f in factors]          # κ
+ [f"{a} * {b}" for a, b in combinations(factors, 2)]  # δ
+ ["constant"]
```

so with `nfac = len(states)`: `lin = params[:nfac]`, `above = params[nfac:2*nfac]`,
`below = params[2*nfac:3*nfac]`, `knots = params[3*nfac:4*nfac]`,
`inter = params[4*nfac:-1]`, `constant = params[-1]`.

`identity_constraints_asymmetric_translog(factor, aug_period, all_factors)`: own linear
coefficient 1, every other parameter including the knots 0 (same pattern as
`identity_constraints_translog`).

## 4. Registration

Add `"asymmetric_translog"` wherever `"stable_translog"` is listed:
`_AF_COMPATIBLE_TRANSITIONS` and `_LEAKY_BUILTIN_PRODUCTION` in
`src/skillmodels/af/validate.py`, `builtin_names` in
`src/skillmodels/amn/simulate_and_regress.py`, and the lists in
`docs/reference_guides/transition_functions.md`. The function has no keyword-only
configuration, so `_hide_keyword_only_config` is a no-op for it.

## 5. Tests (`tests/test_asymmetric_translog.py`, `tests/test_asymmetric_translog_estimation.py`)

Kernel, on the 3-factor toy states and parameters of
`tests/test_transition_functions.py`:

1. γ⁺ = γ⁻, κ = 0 equals `translog` to 1e-12 on a grid of states including both signs.
1. γ⁺ = γ⁻ = γ, κ ≠ 0 equals `translog` with linear coefficients β − 2γκ and constant c
   \+ Σγκ² to 1e-12.
1. One-sided curvature: with γ⁺ = 0 and γ⁻ > 0 the own derivative (jax.grad) above the
   knot equals β exactly and below the knot equals β + 2γ⁻(x − κ).
1. C¹ at the knot: jax.grad with respect to x and with respect to κ is continuous across
   the knot (values at κ ± 1e-7 within 1e-6), and the function value is continuous.
1. Parameter companions: `params_asymmetric_translog` names and length
   `4*nfac + nfac*(nfac-1)/2 + 1`; identity constraints as in section 3.
1. JIT, vmap and a 50-step `lax.scan` give the same values as eager evaluation.
1. Fixed-point witness for the application pattern: one-factor own map with β = 0.97, γ⁻
   = 0.1, γ⁺ = 0, κ = −0.5, c = −0.03 iterated 200 steps from x ∈ {−3, −1, 0, 0.7, 2}
   stays finite and converges to the same point from every start (slope ≤ β < 1 above
   the knot, and below it β + 2γ⁻(x − κ) stays in (0, 1) on this range).

Estimation, on the fixture of `tests/test_missing_observed_factors.py` as in
`tests/test_stable_translog_estimation.py`: a registered same-law reference callable
(explicit `(fac1, fac2, ob1, ob2, params)` signature) reproduces `loglikeobs` to 1e-10
and the gradient to rtol 1e-8; with γ⁺ = γ⁻ and knots 0 the contributions equal those of
`"translog"`; with γ⁺ ≠ γ⁻ they differ by more than 1e-8; the endogenous carry-forward
fixture accepts the function with its identity constraints written explicitly for the
reference callable (copy the `_endogenous_inputs` helper). Dispatch: the public model
and the AMN resolver return the asymmetric law (copy
`tests/test_stable_translog_dispatch.py` with one value that distinguishes the two sides
of the knot).

## 6. Documentation

A section in `docs/reference_guides/transition_functions.md` after `stable_translog`:
formula, parameter names, nesting, the knot as a parameter and how to fix it, the
one-sided-tail property, and the recommendation to run a fixed-point check on estimated
laws. `pixi run build-docs` must exit 0.
