# Lean proofs for limited-measurement Gaussian projection in a CHS-style filter

This project proves conditional approximation guarantees for the moment-matching
step used to incorporate probit, censored Tobit, and ordered-probit measurements
into a Gaussian-component nonlinear filter. It uses actual probability measures,
Bochner integrals, derivatives, and Gaussian distributions from mathlib.

The central result is a cubic weak-error bound. Replacing a probability law by
the Gaussian with the same mean and variance preserves every quadratic
expectation. For a smooth future measurement, only the third-order Taylor
remainder contributes to the projection error. Additional theorems establish
Gaussian expected-log-score optimality, valid covariance contraction, mixture
weight normalization, Bayes normalization stability, and conditional
finite-horizon log-likelihood bounds.

The guarantee is conditional and quantitative. Its assumptions identify when
the approximation is accurate; the project does not assert a uniform accuracy
claim over all CHS parameters, observations, nonlinear transitions, or horizons.

## Reproduce the checks

**Verified on 2026-10-28:** `lake build` and `lake env lean ProofAudit.lean`
both exited successfully. The audit covers 82 theorem declarations and the
Gaussian probability-measure instance. Their only axiom dependencies are Lean's
standard `propext`, `Classical.choice`, and `Quot.sound`.

The toolchain is pinned to **Lean 4.35.0**. Mathlib is pinned to commit
`56a005855ea17e0b0958f73ec801ae140b0a1679` (tag `v4.35.0`). The manifest also locks
all transitive dependency revisions.

With Lean's usual `elan`/`lake` installation, run from this directory:

```bash
lake exe cache get Mathlib/Probability/Distributions/Gaussian/Real.lean Mathlib/Analysis/Calculus/Taylor.lean Mathlib/Tactic.lean Mathlib/LinearAlgebra/Matrix/PosDef.lean Mathlib/MeasureTheory/Integral/Bochner/Basic.lean
lake build
lake env lean ProofAudit.lean
```

The first command obtains the pinned precompiled mathlib dependencies; internet
access is needed on a fresh installation. The source archive intentionally
excludes the compiler, dependencies, and generated `.lake` build artifacts.
`verification/` records the delivered build, theorem-axiom audit, and source
checksums. There are no project-added axioms, unfinished proofs, or native
decision procedures used to bypass proof checking.

## What is proved

| Module | Principal result | Explicit scope |
| --- | --- | --- |
| `CHS/GaussianProjection.lean` | Actual Gaussian projection preserves mean, variance, second moments, and quadratic expectations; is idempotent; includes arbitrary-state scalar-index pushforwards | Probability laws with a finite second moment; includes zero-variance Dirac laws |
| `CHS/WeakError.lean` | Taylor-derived cubic expectation-error bound, with raw or centered third moments; an actual-Gaussian corollary proves integrability from a finite third moment | Scalar smooth tests with a globally bounded third derivative |
| `CHS/GaussianScore.lean` | Unique minimizer of the actual expected Gaussian negative log score | True variance and candidate Gaussian variance are strictly positive |
| `CHS/UpdateAlgebra.lean` | Covariance contraction, positive lower bound, finite-coordinate Cauchy inequality, rank-one factor identity, independent-residual moment transport | Positive noise variance; selected-index variance range and residual independence are explicit prerequisites |
| `CHS/Mixture.lean` | Weighted error bounds, normalized Bayes weights, preservation of the current mixture score when component scores agree | Nonnegative mixture weights and likelihoods; positive total likelihood |
| `CHS/LimitedApproximation.lean` | One-step log-score and normalized posterior-expectation error bounds from the Taylor theorem | Positive predictive normalizer floor; likelihood and weighted test meet the smoothness bounds |
| `CHS/ErrorPropagation.lean` | Finite-horizon propagation, separate prediction and projection defects, cubic corollary, zero-defect exactness | Transition/Bayes stability and normalizer control must be established for the chosen filter |

## 1. Mapping the limited-variable update to the mathematics

For one incoming Gaussian component, write

\[
\Theta\sim N(m,P),\qquad Y^*=a+\lambda^\top\Theta+\epsilon,
\qquad \epsilon\sim N(0,\sigma^2),\quad \sigma>0.
\]

The measurement error is independent of the state. Define

\[
s^2=\lambda^\top P\lambda+\sigma^2,\quad
Z=\frac{Y^*-a-\lambda^\top m}{s},\quad \beta=\frac{P\lambda}{s}.
\]

A binary outcome, censored outcome, or ordered category identifies an interval
or half-line event `A` for `Z`. Gaussian regression gives the representation

\[
\Theta=m+\beta Z+R,\quad
R\perp Z,\quad E[R]=0,\quad \operatorname{Cov}(R)=P-\beta\beta^\top.
\]

Writing \(\mu_A=E[Z\mid A]\) and \(v_A=\operatorname{Var}(Z\mid A)\), the selected
state's moments are

\[
m^+=m+\beta\mu_A,\qquad
P^+=P-(1-v_A)\beta\beta^\top.
\]

These equations explain the implementation's rank-one mean/covariance update.
`selected_affine_moments` proves the scalar directional moment transport once
the independent-residual representation and its moments are supplied.
`coordinate_update_bounds` proves the covariance inequalities once
\(0\le v_A\le1\) is supplied.

**Formalization boundary:** this project does not formalize the Gaussian
conditioning/regression theorem or derive the truncated-normal CDF formulas.
Those analytical connections are stated here explicitly, rather than encoded
as custom axioms or silently treated as completed Lean proofs. The subsequent
projection, optimality, covariance algebra, and error estimates are proved.

For a finite interval \(A=(l,u]\), the familiar scalar quantities are

\[
p_A=\Phi(u)-\Phi(l),\quad
\mu_A=\frac{\phi(l)-\phi(u)}{p_A},\quad
v_A=1+\frac{l\phi(l)-u\phi(u)}{p_A}-\mu_A^2.
\]

Half-lines use the corresponding endpoint limits. These are exact quantities
for the incoming Gaussian component. The Gaussian approximation arises when
the selected posterior law is replaced by a Gaussian with those moments.
Previous projections and nonlinear prediction can already have made the
incoming component an approximation to the exact-history posterior.

An uncensored Tobit measurement is an ordinary Gaussian measurement; its
Gaussian-prior posterior stays Gaussian. The limited-variable projection issue
arises for censored, binary, and category observations.

## 2. Exact preservation and optimality

`gaussianProjection μ` is the actual mathlib Gaussian with mean
\(\int x\,d\mu\) and variance \(\operatorname{Var}_\mu(X)\). Under a finite second
moment, Lean proves

\[
E_\mu X=E_{\Pi\mu}X,\qquad E_\mu X^2=E_{\Pi\mu}X^2,
\qquad E_\mu q(X)=E_{\Pi\mu}q(X)
\]

for every quadratic \(q\). This also preserves any centered second moment and
the first two moments of any affine scalar transformation. A Gaussian is a
fixed point, and \(\Pi(\Pi\mu)=\Pi\mu\).

The scalar-index interface uses the actual pushforward `μ.map X` for a
square-integrable scalar index on an arbitrary state space. Thus the formal
results are usable for a linear measurement index of a multivariate CHS state;
the package does not claim a completed multivariate Gaussian filtering theorem.

For a true scalar law with mean \(m\) and positive variance \(v\), the expected
Gaussian negative log score, omitting only the constant \(\tfrac12\log(2\pi)\), is

\[
L(a,w)=\frac12\left[\log w+\frac{v+(m-a)^2}{w}\right],\qquad w>0.
\]

Lean identifies this formula with the actual expectation and proves

\[
L(a,w)-L(m,v)=\frac12\left[\log(w/v)+v/w-1+(m-a)^2/w\right]\ge0,
\]

with equality exactly when \(a=m\) and \(w=v\). This provides a precise
optimality criterion for retaining those two moments. Optimality within the
Gaussian family does not, by itself, make the minimum error small. A full KL
projection theorem would also require an explicit entropy/absolute-continuity
bridge, which is not part of this project.

## 3. The quantitative approximation theorem

Suppose \(\mu\) and \(\nu\) have matching first and second moments. For
\(f\in C^3(\mathbb R)\) with \(\sup_z|f'''(z)|\le M\), Lean proves

\[
\left|E_\mu f(b+\varepsilon X)-E_\nu f(b+\varepsilon X)\right|
\le \frac{M|\varepsilon|^3}{6}
\left(E_\mu|X|^3+E_\nu|X|^3\right).
\]

The centered version, `MomentMatch.smooth_scaled_centered_bound`, proves for
any common center \(c\)

\[
\left|E_\mu f(b+\varepsilon X)-E_\nu f(b+\varepsilon X)\right|
\le \frac{M|\varepsilon|^3}{6}
\left(E_\mu|X-c|^3+E_\nu|X-c|^3\right),
\]

with the stated integrability assumptions. Taking \(c\) to be the common mean
usually improves the bound. The raw-moment actual-Gaussian corollary
`gaussianProjection_smooth_scaled_bound` needs only a finite third moment and
the derivative assumptions; it proves the integrability of the evaluated test.

The proof uses mathlib's Lagrange Taylor theorem, including the factor `1/6`.
The constant, linear, and quadratic terms integrate equally. Each remaining
integral is bounded by its cubic envelope, and the triangle inequality completes
the proof. The remainder estimate is derived from calculus, rather than assumed
as the desired approximation conclusion.

Here \(\varepsilon\) is the coefficient in the **evaluated test**, often a future
measurement's standardized latent-index loading. It is not automatically the
loading of the observation that created the selected posterior. A uniform
\(O(|\varepsilon|^3)\) claim requires uniform derivative/moment bounds across the
parameter family. Concentration, through small centered third moments, provides
another accuracy regime even without varying the test loading.

### Application to probit and ordered categories

The following derivative calculations are analytical explanations, **not
additional Lean-checked theorems**. For the standard normal CDF,

\[
\Phi'''(z)=(z^2-1)\phi(z),\qquad
\sup_z|\Phi'''(z)|=1/\sqrt{2\pi}.
\]

For \(|z|\le1\), the normalized absolute expression is at most one. Outside
that interval its maximum is \(2e^{-3/2}<1\), attained at \(|z|=\sqrt3\).
For an ordered-category probability
\(g(z)=\Phi(u-z)-\Phi(l-z)\), the triangle inequality gives
\(\sup|g'''|\le2/\sqrt{2\pi}\). A noise scale \(\sigma\) contributes
\(\sigma^{-3}\) when the index is written in unstandardized units.

Thus the CDF measurement probabilities meet the kind of smoothness condition
used by the formal theorem. A posterior-expectation bound additionally requires
a derivative bound for the likelihood times the chosen test.

## 4. Covariance and mixture properties

Let \(q=\lambda^\top P\lambda\). In every direction, Lean's factor-coordinate
Cauchy inequality proves the algebra needed for

\[
\frac{\sigma^2}{q+\sigma^2}P\preceq P^+\preceq P
\]

when \(P\succeq0\), \(\sigma^2>0\), and \(0\le v_A\le1\). Positive prior
directional variance remains positive. Zero loading/cross covariance leaves the
corresponding update unchanged. The finite-sum rank-one factor identity also
checks the covariance represented by the bounded factor correction.

For nonnegative component probabilities \(w_i\), likelihoods \(L_i\), and
\(Z=\sum_i w_iL_i>0\), the Bayes weights \(w_iL_i/Z\) are nonnegative and sum to
one. Component expectation-error budgets combine as

\[
\left|\sum_iw_i a_i-\sum_iw_i b_i\right|
\le\sum_iw_i|a_i-b_i|.
\]

Keeping component likelihoods unchanged keeps the current mixture likelihood,
current log score, and posterior weights unchanged. This does not determine
future scores after shape information is discarded. The nonnegative-weight
theorems apply to mixture probabilities; unscented sigma-point weights can be
signed and require separate treatment.

## 5. From local error to likelihood error

For positive normalizers with a common lower bound \(p,q\ge\delta>0\), Lean proves

\[
|\log p-\log q|\le |p-q|/\delta.
\]

The floor is a hypothesis about the actual two normalizers. The theorem does
not recommend clipping probabilities or modifying the likelihood. Rare events
can make this amplification large even when the absolute probability error is
small.

If the likelihood and its product with a bounded test have third-derivative
bounds \(M_g,M_{g\psi}\), the proved one-step Bayes estimate is

\[
|B_\mu\psi-B_\nu\psi|
\le\frac{(M_{g\psi}+M_g)|\varepsilon|^3}{6\delta}
\left(E_\mu|X|^3+E_\nu|X|^3\right).
\]

Here the Bayes expectation uses likelihood \(g(b+\varepsilon X)\) and test
\(\psi(b+\varepsilon X)\), with \(|\psi|\le1\). It is a guarantee for that
scaled smooth test class. Arbitrary latent-state tests are not covered merely
by renaming their argument; reparameterization can change the derivative bound.

For a full sequence, the project keeps prediction and projection error separate:

\[
d_t\le\kappa_t e_t+\eta_t,\qquad
e_{t+1}\le B_t d_t+\rho_t.
\]

The proved running envelope is

\[
E_0\ge e_0,\qquad
E_{t+1}=B_t\kappa_tE_t+B_t\eta_t+\rho_t.
\]

If normalizer errors are controlled by \(d_t\), their cumulative log-likelihood
error is bounded by

\[
\frac1\delta\sum_{t<T}(\kappa_tE_t+\eta_t).
\]

`logLikelihood_error_cubic` specializes to zero initial filter error, local
defects \(C|\varepsilon|^3\), and amplification \(L\), using the division-free recurrence
\(G_L(0)=0\), \(G_L(t+1)=LG_L(t)+1\). Its coefficient depends explicitly on the
finite horizon, stability factors, and likelihood floor.

Repeated exact and approximate-history filters are generally not themselves
moment-matched. Apply the local theorem to each preprojection posterior
generated from the approximate incoming law and its own Gaussian projection.
The propagation theorems then compare histories only when their transition,
Bayes, and normalizer-control premises have been established. This project
does not derive universal stability constants or unscented-prediction error
bounds for the nonlinear CHS transition.

## 6. What the certificate establishes about the implementation

The project establishes real-arithmetic mathematical properties of the stated
moment-projection operation. It gives a principled reason for the choice of
retained moments and a quantitative local approximation regime. It does not
certify the floating-point normal-CDF kernels, QR implementation, gradients,
optimizer, statistical consistency of parameter estimates, or numerical
accuracy on a particular dataset. Those are separate obligations from checking
the mathematical projection theorem.

The original CHS identification result is also distinct from this approximation
analysis: identifying a model does not establish accuracy of every numerical
likelihood approximation used to estimate it.

## References

1. Cunha, Heckman, and Schennach (2010), *Estimating the Technology of Cognitive
   and Noncognitive Skill Formation*, Econometrica 78(3), 883–931.
   [Author-hosted published paper](https://jenni.uchicago.edu/papers/Cunha_Heckman_etal_2010_Econometrica_v78_n3.pdf).
2. CHS web appendix, section 6, *Nonlinear Filtering*, especially the update,
   prediction, unscented transform, and implementation subsections.
   [Author-hosted appendix](https://jenni.uchicago.edu/elast-sub/webtab_elast-sub_2010-01-09a_jlt.pdf).
3. Minka (2001), *Expectation Propagation for Approximate Bayesian Inference*,
   section 2 on assumed-density filtering and moment projection.
   [Author-hosted paper](https://tminka.github.io/papers/ep/minka-ep-uai.pdf).
4. [Pinned mathlib source](https://github.com/leanprover-community/mathlib4/tree/56a005855ea17e0b0958f73ec801ae140b0a1679),
   particularly Gaussian.Real, calculus Taylor theorems, and Bochner integration.

The original CHS paper/appendix motivates the filtering setting. The proof
package's local error theorems are the mathematical formalization provided
here; they are not attributed to CHS as claims about this specific software.
