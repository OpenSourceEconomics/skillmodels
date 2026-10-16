# `stable_translog`: a translog transition function with saturating curvature

Status: design for review (no code yet). Target repository: skillmodels, branch to be
created from `94aa213`. First consumer: the health-cognition application (HRS, ages
68-93, four latent capacity states, three latent investments, four observed inputs).

## 1. Problem

The translog transition function

    θ_{t+1} = c + Σ_j β_j θ_j + Σ_j γ_j θ_j² + Σ_{a<b} δ_ab θ_a θ_b

is a quadratic map. For the scalar calibrations below there are two real fixed
points, one attracting and one repelling, and sufficiently negative starts diverge.
A negative own-square coefficient alone does not guarantee two real roots or a stable
root: the discriminant and the derivative at each root must be checked. Estimates
from the health-cognition application (run E, age group 68-73, other inputs at zero) give:

| Factor        | c      | own β | own γ  | fixed points (stable, unstable) | unstable point in SDs of the state at 68 |
|---------------|--------|-------|--------|---------------------------------|------------------------------------------|
| psychological | -0.003 | 0.769 | -0.103 | -0.013, -2.23                   | about 3.2 (SD 0.69, observed max 2.7)     |
| cognition     | -0.074 | 0.596 | -0.112 | -0.194, -3.41                   | about 9   (SD 0.39)                       |

The psychological scalar repelling fixed point is inside the stated range of
filtered states. The supplied evidence reports explosive forward predictions and,
for run D, places explosive rows after death or after the last interview. Its two
unrestricted tables give cumulative explosive shares 0.049 and 0.022 at age 92;
they do not establish that every likelihood evaluation or gradient is unaffected.
Long-horizon simulations and exports of the estimated technology therefore need
separate validation of the complete law of motion.

Polynomial fixes do not work: an odd-degree term diverges in at least one tail, an
even-degree term in both or by overshoot. The curvature itself is identified by the lower
tail of the data; what has to go is its extrapolation beyond the support.

## 2. Proposed function

Apply a smooth, bounded squashing map g to each state before it enters any nonlinear
term, and leave the linear part untouched:

    stable_translog(θ) = c + Σ_j β_j θ_j + Σ_j γ_j g(θ_j)² + Σ_{a<b} δ_ab g(θ_a) g(θ_b)

with the algebraic soft clip

    g(x; s, p) = x / (1 + |x/s|^p)^(1/p),        s > 0, p an even integer

Properties of g: smooth (C^∞ for even p), odd, |g(x)| < s, g(x) = x + O(|x/s|^p · x)
near zero, g'(x) ∈ (0, 1], g(x) → ±s as x → ±∞ with g'(x) → 0.

Defaults: s = 3.0, p = 4. Parameter names, parameter count, identity constraints and
start values are those of `translog`; the function is a drop-in replacement selected by
the string `"stable_translog"`.

### 2.1 Why this and not the alternatives discussed

- `θ·tanh(θ/s)` in place of `θ²` only: leaves the interaction terms unbounded; a product
  of two large states diverges as fast as a square.
- softplus hinge `log(1+exp(k(θ-c)))/k`: needs a kink location, changes the meaning of
  every coefficient, does not reduce to the translog anywhere.
- `s·tanh(x/s)` as g: same structure as the proposal, but worse fidelity at equal
  saturation level (relative error of g² versus x² at |x| = 2 is 24 % for s = 3, against
  8.6 % for the algebraic clip with s = 3, p = 4). The algebraic clip is the same idea
  with a flatter approach to the identity.
- CES technology as in Cunha, Heckman and Schennach: smooth and well behaved but requires
  positive inputs; our states are centred latent factors.

### 2.2 What is guaranteed and what is not

Boundedness (fixed-coefficient deterministic theorem). Let x contain the recursively
updated latent coordinates and let z_t be bounded exogenous inputs. Assume the COMPLETE
transition can be written as x_{t+1} = A x_t + b_t(x_t), with a fixed square matrix A,
rho(A) < 1, and sup_{t,x} ||b_t(x)||_infinity <= M. The base clipped nonlinear remainder
has componentwise bounds R_i = s^2 (sum_j |gamma_ij| + sum_{a<b} |delta_iab|);
M must also include constants and the bounded linear forcing B z_t. Then

    ||x_t|| <= ||A^t|| ||x_0|| + M sum_{k=0}^{t-1} ||A^k||,

and sum_{k>=0} ||A^k|| is finite. The gain is NOT generally ||(I-A)^(-1)||:
the forcing is state/time dependent and cannot be pulled through a signed matrix sum.
If a compatible norm has ||A|| <= q < 1, the familiar bound is
q^t ||x_0|| + M(1-q^t)/(1-q). For time-varying A_t, require a uniform product bound
||A_{t-1}...A_k|| <= C q^(t-k), not merely rho(A_t)<1 separately. Periodic/augmented
models may instead be checked over a complete calendar-period product; carry-forward
identity substeps need not individually have spectral radius below one.

This is a conditional theorem, not an estimator constraint or a stability certificate
for the application. Diagonals in [0.6,0.85] and cross coefficients below 0.13 do not
prove stability (a 4-by-4 matrix with diagonal 0.85 and off-diagonal 0.12 has eigenvalue
1.21). Check the full estimated matrix/product, not own coefficients or scalar slices.
The supplied bundle does not contain those full application matrices.

Control-function and endogenous-factor composition must be checked separately.
process_model adds the control-function term AFTER the base production transition.
A linear kappa*(investment - first_stage_prediction) changes the effective A; a
higher-degree correction is not clipped by this function. Unless the COMPLETE law
satisfies the displayed decomposition with bounded remainder, the theorem does not
apply. Do not clip those corrections silently: that changes the econometric model.

The theorem concerns deterministic drift or bounded forcing. Gaussian innovations
have no uniform infinite-horizon pathwise bound even for x_{t+1}=epsilon_t. Moment
stability, filter stability, finite simulation horizons, and absence of NaNs require
their own assumptions and diagnostics. A finite-horizon test is not an asymptotic proof.
The existing robust_translog also bounds deterministic outputs for fixed finite
coefficients by clipping its entire input at 1e12; its drawback is distortion of linear
dynamics and the potentially enormous resulting bound, not mathematical unboundedness.

Uniqueness of the attractor (not guaranteed, application-dependent). The clip removes
divergence but can leave a second stable fixed point far out in the tail if s lies
far enough into the tail: a nearly preserved repelling crossing can coexist with a
new attracting tail crossing. Finite s does not make the soft clip exactly equal to
the translog at a nonzero point. With
the run E psychological coefficients:

| g                | fixed points                                      | rel. error of g² at |x| = 1, 2, 2.5 |
|------------------|---------------------------------------------------|-------------------------------------|
| identity (translog) | -2.23 (unstable), -0.013 (stable)              | 0, 0, 0                              |
| tanh, s = 3      | -0.013 (stable)                                   | 0.070, 0.236, 0.330                  |
| alg, s = 3, p = 4 | -0.013 (stable)                                  | 0.006, 0.086, 0.179                  |
| alg, s = 2.5, p = 8 | -0.013 (stable)                                | 0.000, 0.038, 0.159                  |
| alg, s = 3, p = 8 | -3.92 (stable), -2.29 (unstable), -0.013 (stable) | 0.000, 0.010, 0.051                 |
| alg, s = 4, p = 8 | -7.13 (stable), -2.24 (unstable), -0.013 (stable) | 0.000, 0.001, 0.006                 |

There is a trade-off between fidelity to the translog inside the data region and a
unique attractor: the translog's own unstable fixed point sits at 2.2, inside the data,
and a function that matches the translog EXACTLY on an interval containing its
repelling fixed point inherits that point. Approximate agreement requires a separate
crossing-margin argument; the value of s alone is not a uniqueness test. The default
(s = 3, p = 4) buys the unique attractor at the price of a 9 % deviation of the squared
term at two units from the mean. The library documents this and leaves the choice to the
application; the application checks the fixed points of its estimated equations with a
fifty-line script (planned on the health-cognition side, not in skillmodels).

Nesting (approximate). For every participating coordinate |theta_j| <= 1, each individual squared
or interaction basis term differs by less than 1% at the defaults (absolute zero
error for a vanishing product). This is not a relative-error bound on a cancelling
sum of nonlinear terms or on the likelihood; estimates on data with filtered states mostly
inside ±2 should land close to the translog estimates, and the comparison is a robustness
check rather than a new model. This is a conjecture to be verified on the health-cognition
data, not a property of the function.

Finite precision. Boundedness of the real-valued nonlinear basis does not prevent an
intermediate overflow in a naive formula. Evaluate g with bounded normalized powers,
not `(abs(x/s))**p` on unrestricted x. For example, with m=max(s,|x|), the denominator
`((s/m)**p + (x/m)**p)**(1/p)` is between 1 and 2**(1/p); use numerator x when |x|<=s
and s*(x/m) otherwise. In code select m and the numerator with the SAME <= predicate;
a generic maximum subgradient at |x|=s combined with a one-sided numerator selection
would give an incorrect derivative at the clip knee. All branches need safe operands.
In floating point |g| may round to s, so use <=s with a stated rounding allowance.
The helper preserves NaN and may return +/-s for +/-infinity; the full transition
still propagates nonfinite linear inputs. Linear products/sums or coefficients can
still overflow. No unconditional finite-output or NaN-free likelihood guarantee is made.

### 2.3 Hyperparameters and the string lookup

skillmodels resolves string transition names by `getattr(transition_functions, name)`
and obtains `params_<name>` and `identity_constraints_<name>` by the same convention.
The string `"stable_translog"` uses the documented defaults. scale is a positive finite
Python scalar and power an even Python integer >=2; both are keyword-only configuration.
Close over nondefault configuration before JIT, or explicitly mark scale and power
static with `jax.jit(..., static_argnames=("scale", "power"))`. A supplied Python int
is not automatically static when passed as an ordinary argument to a jitted function.

A one-line `(states, params)` wrapper is NOT a complete alternative registration.
A new module-level string name also needs its parameter/identity companions and all
relevant allow-lists. A public registered callable must instead name its consumed
factors, accept the registered parameter mapping, and reconstruct the ordered vector.
A factory/spec-level option remains out of scope. Document a complete tested recipe
before advertising nondefault configuration through an estimator, rather than an
untested one-line wrapper. Direct calls may use the keyword arguments.

### 2.4 UKF, units, and identification

Saturation changes the model at ALL nonzero inputs, not just an extrapolation rule.
A UKF evaluates sigma points, whose spread depends on covariance, dimension and its
chosen weights; filtered means inside +/-1 do not imply that the points are there.
The pointwise fidelity figures imply no 1% likelihood or coefficient-change bound.
Check actual weighted predicted means/covariances, finite gradients and the fraction
of sigma points near/beyond the clip scale. Compare the same stable law against an
independent sigma-point evaluation; quadrature error against Gaussian integration is
a separate diagnostic, not evidence that the clipped law was incorrectly dispatched.
Do not tune the sigma rule merely to conceal a changed model.

Exact identity/carry-forward transitions survive because their nonlinear coefficients
are zero and their own linear coefficient is one. The shipped constraint machinery
also pins kappa to zero on carry-forward substeps. These identities are not by
themselves an identification proof: preserve measurement location/scale anchors.
The fixed clipping scale is in the factors' specified units, not automatically in
standard deviations. g(a*x; a*s,p)=a*g(x;s,p) for a>0; keeping s fixed while rescaling
factors changes the function family. Translations require transforming the clipping
centre as well and are not generally absorbable into translog coefficients.
Saturated square columns approach s^2 and become nearly collinear with the constant,
so tail-only data may weakly identify curvature. The defaults remain an explicit
heuristic supported by the supplied scalar calibration, not a universal choice or a
reason to drop existing normalization constraints.

## 3. Scope of the skillmodels PR

In scope:

1. `stable_translog`, `params_stable_translog`, `identity_constraints_stable_translog`
   in `src/skillmodels/common/transition_functions.py`.
2. Registration in the three name allow-lists: `_AF_COMPATIBLE_TRANSITIONS` and
   `_LEAKY_BUILTIN_PRODUCTION` in `src/skillmodels/af/validate.py`, `builtin_names` in
   `src/skillmodels/amn/simulate_and_regress.py`.
3. Tests. `tests/test_stable_translog.py` (the round-0 reviewer's acceptance suite RT1 and
   tail generators MT2 against the independent decimal oracle
   `tests/stable_translog_reference.py`): finite tails through 1e80 and ±inf/NaN semantics,
   every coefficient position and factor permutation, the default fidelity budgets at
   |x| = 1 and 2, all 200 iterates and the fixed point of the run E psychological map,
   analytical gradients at zero, both clip knees and finite tails, JIT/vmap/scan with
   static non-default configuration, name and identity companions, loud errors for invalid
   `scale`/`power`, and the float64 overflow threshold neighbourhood.
   `tests/test_stable_translog_dispatch.py` (RT2): the public model and the AMN resolver
   return the clipped law, not translog. `tests/test_stable_translog_estimation.py`: CHS
   likelihood contributions and gradients equal those of an independent same-law
   registered callable on a fixture with missing observed factors and on the endogenous
   carry-forward fixture, and differ from translog by a numerical margin.
4. Documentation section in `docs/reference_guides/transition_functions.md` with the
   formula, the boundedness statement, the uniqueness caveat and the default values.

Out of scope (health-cognition repository): the `stable` model variant, the fixed-point
check of estimated equations, re-running simulations with the new function, and the
pylcm law-of-motion export.

## 4. Questions for the external reviewer

1. Is the boundedness argument in 2.2 correct as stated, including with observed factors
   and with the control-function and endogenous-factor code paths of skillmodels?
2. Is the algebraic soft clip the right g, or does a different bounded, smooth, odd map
   with g'(0) = 1 give a better fidelity/uniqueness trade-off at equal saturation level?
3. Are the defaults s = 3, p = 4 defensible for a general-purpose library whose latent
   states have application-specific scale, or should the function refuse to run without
   an explicit scale?
4. Does replacing θ_j by g(θ_j) in the squares and interactions interact badly with the
   unscented Kalman filter's sigma points (the predict step evaluates the transition at
   sigma points that can lie outside the data region by design)?
5. Does anything in the identification argument for the translog (normalisations via the
   identity transition in the first period, the anchoring of scales) break when the
   nonlinear terms saturate?
6. Is there a reason to prefer bounding the states before the whole function (including
   the linear part), which would also bound the output, over bounding only the nonlinear
   terms, which keeps the linear dynamics exact?
