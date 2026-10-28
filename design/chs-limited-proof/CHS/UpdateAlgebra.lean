module

public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
public import Mathlib.Probability.Moments.Variance
public import Mathlib.Tactic

/-!
# Algebra and moment transport for a limited-measurement update

This file proves real-arithmetic statements, not a floating-point implementation
certificate.  In the covariance theorems, `p` is a prior directional variance,
`q` is the loading variance, `cross` is their covariance, and `noise` is the
strictly positive **noise variance**.  The hypothesis `cross ^ 2 ≤ p * q` is
subsequently discharged for finite square-root-factor coordinates.

The affine moment theorems are theorems about actual integrals and variances.
They assume independence explicitly.  They do not themselves identify a
conditional Gaussian distribution or prove truncated-normal moment formulae.
-/

@[expose] public section

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace CHS.UpdateAlgebra

/-- Variance in one direction after a rank-one covariance contraction.
`noise` denotes a variance, rather than a standard deviation. -/
def directionalUpdate (p cross q noise contraction : ℝ) : ℝ :=
  p - contraction * cross ^ 2 / (q + noise)

theorem directionalUpdate_le_prior {p cross q noise contraction : ℝ}
    (hq : 0 ≤ q) (hnoise : 0 < noise) (hd : 0 ≤ contraction) :
    directionalUpdate p cross q noise contraction ≤ p := by
  unfold directionalUpdate
  exact sub_le_self _ (div_nonneg (mul_nonneg hd (sq_nonneg cross)) (by linarith))

/-- Positive observation noise supplies a lower bound independently of the
retained truncated-index variance. -/
theorem directionalUpdate_lower_bound {p cross q noise contraction : ℝ}
    (hq : 0 ≤ q) (hnoise : 0 < noise)
    (hcauchy : cross ^ 2 ≤ p * q) (hd : contraction ≤ 1) :
    noise / (q + noise) * p ≤ directionalUpdate p cross q noise contraction := by
  have hs : 0 < q + noise := by linarith
  have hcontract : contraction * cross ^ 2 ≤ p * q := by
    calc
      contraction * cross ^ 2 ≤ 1 * cross ^ 2 :=
        mul_le_mul_of_nonneg_right hd (sq_nonneg cross)
      _ ≤ p * q := by simpa using hcauchy
  have hdiv : contraction * cross ^ 2 / (q + noise) ≤ p * q / (q + noise) :=
    div_le_div_of_nonneg_right hcontract hs.le
  have hidentity : noise / (q + noise) * p = p - p * q / (q + noise) := by
    field_simp [ne_of_gt hs]
    ring
  rw [hidentity]
  unfold directionalUpdate
  linarith

theorem directionalUpdate_pos {p cross q noise contraction : ℝ}
    (hp : 0 < p) (hq : 0 ≤ q) (hnoise : 0 < noise)
    (hcauchy : cross ^ 2 ≤ p * q) (hd : contraction ≤ 1) :
    0 < directionalUpdate p cross q noise contraction := by
  have hs : 0 < q + noise := by linarith
  exact lt_of_lt_of_le (mul_pos (div_pos hnoise hs) hp)
    (directionalUpdate_lower_bound hq hnoise hcauchy hd)

@[simp]
theorem directionalUpdate_zero_cross (p q noise contraction : ℝ) :
    directionalUpdate p 0 q noise contraction = p := by
  simp [directionalUpdate]

@[simp]
theorem directionalUpdate_zero_contraction (p cross q noise : ℝ) :
    directionalUpdate p cross q noise 0 = p := by
  simp [directionalUpdate]

theorem directionalUpdate_zero_loading_variance {p cross noise contraction : ℝ}
    (hcauchy : cross ^ 2 ≤ p * 0) :
    directionalUpdate p cross 0 noise contraction = p := by
  have hz : cross = 0 := by nlinarith [sq_nonneg cross]
  simp [hz]

/-- The exact independent-residual covariance expression equals the contraction
expression when `retained` is the variance of the selected standardized index. -/
theorem residual_variance_identity (p cross scale retained : ℝ) :
    (p - (cross / scale) ^ 2) + (cross / scale) ^ 2 * retained =
      p - (1 - retained) * (cross / scale) ^ 2 := by
  ring

/-- Algebra behind the bounded square-root coefficient.  The supplied equation
for `c` is its defining nonnegative-square-root equation in the algorithm. -/
theorem bounded_coefficient_identity {alpha contraction c : ℝ}
    (hc : 0 ≤ c) (hsq : c ^ 2 = 1 - contraction * alpha) :
    2 * (contraction / (1 + c)) -
        (contraction / (1 + c)) ^ 2 * alpha = contraction := by
  have hden : 1 + c ≠ 0 := by linarith
  field_simp [hden]
  nlinarith [congrArg (fun x : ℝ => contraction * x) hsq]

/-- Expanding the squared norm after a bounded rank-one factor correction.
Here `alpha = ‖w‖²` and `cross = ⟨w,u⟩` in the factor representation. -/
theorem bounded_factor_quadratic_identity {p cross alpha contraction b : ℝ}
    (hb : 2 * b - b ^ 2 * alpha = contraction) :
    p - 2 * b * cross ^ 2 + b ^ 2 * alpha * cross ^ 2 =
      p - contraction * cross ^ 2 := by
  calc
    p - 2 * b * cross ^ 2 + b ^ 2 * alpha * cross ^ 2 =
        p - (2 * b - b ^ 2 * alpha) * cross ^ 2 := by ring
    _ = p - contraction * cross ^ 2 := by rw [hb]

section FiniteCoordinates

variable {ι : Type*} [Fintype ι]

/-- Squared norm of factor coordinates, such as `U * x`. -/
def squaredNorm (u : ι → ℝ) : ℝ := ∑ i, (u i) ^ 2

def coordinateCross (u w : ι → ℝ) : ℝ := ∑ i, u i * w i

theorem squaredNorm_nonneg (u : ι → ℝ) : 0 ≤ squaredNorm u := by
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- This discharges the covariance Cauchy bound from actual finite sums. -/
theorem coordinate_cauchy (u w : ι → ℝ) :
    coordinateCross u w ^ 2 ≤ squaredNorm u * squaredNorm w := by
  simpa [coordinateCross, squaredNorm] using
    Finset.sum_mul_sq_le_sq_mul_sq Finset.univ u w

/-- The rank-one factor update has exactly the claimed covariance quadratic
form.  For the CHS factor `U`, take `u = U * direction` and
`w = U * loading / predictiveSD`.  This theorem includes the finite-sum
expansion, not just the scalar coefficient identity. -/
theorem squaredNorm_rankOne_factor (u w : ι → ℝ) {b contraction : ℝ}
    (hb : 2 * b - b ^ 2 * squaredNorm w = contraction) :
    squaredNorm (fun i => u i - b * w i * coordinateCross u w) =
      squaredNorm u - contraction * coordinateCross u w ^ 2 := by
  have hterm (i : ι) :
      (u i - b * w i * coordinateCross u w) ^ 2 =
        (u i) ^ 2 - (2 * b * coordinateCross u w) * (u i * w i) +
          (b ^ 2 * coordinateCross u w ^ 2) * (w i) ^ 2 := by
    ring
  change (∑ i, (u i - b * w i * coordinateCross u w) ^ 2) = _
  simp_rw [hterm]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum]
  change squaredNorm u - (2 * b * coordinateCross u w) * coordinateCross u w +
      (b ^ 2 * coordinateCross u w ^ 2) * squaredNorm w = _
  calc
    _ = squaredNorm u - (2 * b - b ^ 2 * squaredNorm w) * coordinateCross u w ^ 2 := by
      ring
    _ = squaredNorm u - contraction * coordinateCross u w ^ 2 := by rw [hb]

theorem coordinate_update_bounds (u w : ι → ℝ) {noise contraction : ℝ}
    (hnoise : 0 < noise) (hd0 : 0 ≤ contraction) (hd1 : contraction ≤ 1) :
    noise / (squaredNorm w + noise) * squaredNorm u ≤
        directionalUpdate (squaredNorm u) (coordinateCross u w)
          (squaredNorm w) noise contraction ∧
      directionalUpdate (squaredNorm u) (coordinateCross u w)
          (squaredNorm w) noise contraction ≤ squaredNorm u := by
  exact ⟨directionalUpdate_lower_bound (squaredNorm_nonneg w) hnoise
      (coordinate_cauchy u w) hd1,
    directionalUpdate_le_prior (squaredNorm_nonneg w) hnoise hd0⟩

theorem coordinate_update_pos (u w : ι → ℝ) {noise contraction : ℝ}
    (hu : 0 < squaredNorm u) (hnoise : 0 < noise) (hd1 : contraction ≤ 1) :
    0 < directionalUpdate (squaredNorm u) (coordinateCross u w)
      (squaredNorm w) noise contraction := by
  exact directionalUpdate_pos hu (squaredNorm_nonneg w) hnoise
    (coordinate_cauchy u w) hd1

/-- Express the contraction directly through a retained scalar variance in
`[0,1]`.  Proving that range for a conditional normal is a separate obligation. -/
theorem coordinate_update_bounds_of_retained_variance (u w : ι → ℝ)
    {noise retained : ℝ} (hnoise : 0 < noise)
    (hv0 : 0 ≤ retained) (hv1 : retained ≤ 1) :
    noise / (squaredNorm w + noise) * squaredNorm u ≤
        directionalUpdate (squaredNorm u) (coordinateCross u w)
          (squaredNorm w) noise (1 - retained) ∧
      directionalUpdate (squaredNorm u) (coordinateCross u w)
          (squaredNorm w) noise (1 - retained) ≤ squaredNorm u := by
  exact coordinate_update_bounds u w hnoise (by linarith) (by linarith)

@[simp]
theorem coordinate_update_zero_loading (u : ι → ℝ) (noise contraction : ℝ) :
    directionalUpdate (squaredNorm u) (coordinateCross u (fun _ => 0))
      (squaredNorm (fun _ : ι => 0)) noise contraction = squaredNorm u := by
  simp [coordinateCross, squaredNorm]

end FiniteCoordinates

section ProbabilityMoments

variable {Ω : Type*} [MeasurableSpace Ω] {ν : Measure Ω}
variable [IsProbabilityMeasure ν] {Z R : Ω → ℝ}

/-- The mean of the affine residual representation; zero residual mean is an
explicit assumption, not an assertion that conditioning has been formalized. -/
theorem affine_residual_mean (hZ : Integrable Z ν) (hR : Integrable R ν)
    (hRmean : (∫ ω, R ω ∂ν) = 0) (m beta : ℝ) :
    (∫ ω, m + beta * Z ω + R ω ∂ν) = m + beta * (∫ ω, Z ω ∂ν) := by
  have hmz : Integrable (fun ω => m + beta * Z ω) ν :=
    (integrable_const m).add (hZ.const_mul beta)
  rw [integral_add hmz hR,
    integral_add (integrable_const m) (hZ.const_mul beta), integral_const_mul]
  simp [hRmean]

/-- Actual variance transport through the independent affine residual
representation.  This is the probabilistic bridge once that representation is
established separately. -/
theorem affine_residual_variance (hZ : MemLp Z 2 ν) (hR : MemLp R 2 ν)
    (hindep : IndepFun Z R ν) (m beta : ℝ) :
    variance (fun ω => m + beta * Z ω + R ω) ν =
      beta ^ 2 * variance Z ν + variance R ν := by
  have h_indep : IndepFun (fun ω => beta * Z ω) R ν :=
    hindep.comp (show Measurable (fun z : ℝ => beta * z) by fun_prop) measurable_id
  have h_repr : (fun ω => m + beta * Z ω + R ω) =
      (fun ω => beta * Z ω + R ω + m) := by
    funext ω
    ring
  have hmeas : AEStronglyMeasurable (fun ω => beta * Z ω + R ω) ν :=
    ((hZ.const_mul beta).add hR).aestronglyMeasurable
  rw [h_repr, variance_add_const hmeas]
  rw [h_indep.variance_fun_add (hZ.const_mul beta) hR, variance_const_mul]

/-- Combine the separately specified scalar-index and residual moments into the
state-update moments.  The hypotheses explicitly expose the independent
residual representation and all scalar moments that a conditioning theorem
must supply; this theorem does not assume the resulting state moments. -/
theorem selected_affine_moments (hZ : MemLp Z 2 ν) (hR : MemLp R 2 ν)
    (hindep : IndepFun Z R ν) (m beta indexMean retained prior : ℝ)
    (hZmean : (∫ ω, Z ω ∂ν) = indexMean)
    (hZvar : variance Z ν = retained)
    (hRmean : (∫ ω, R ω ∂ν) = 0)
    (hRvar : variance R ν = prior - beta ^ 2) :
    (∫ ω, m + beta * Z ω + R ω ∂ν) = m + beta * indexMean ∧
      variance (fun ω => m + beta * Z ω + R ω) ν =
        prior - (1 - retained) * beta ^ 2 := by
  constructor
  · rw [affine_residual_mean (hZ.integrable (by norm_num))
      (hR.integrable (by norm_num)) hRmean, hZmean]
  · rw [affine_residual_variance hZ hR hindep, hZvar, hRvar]
    ring

end ProbabilityMoments

end CHS.UpdateAlgebra
