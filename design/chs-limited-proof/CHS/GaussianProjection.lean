module

public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

/-!
# Continuous Gaussian moment projection

This file uses `MeasureTheory.Measure ℝ` and Bochner integrals. The projection is
the actual Gaussian probability measure with the input law's mean and variance.
Finite second moments are explicit in the preservation theorem; they are not
inferred from Lean's totalized integral or variance definitions.

The conclusions concern moment projection. They do not identify the input law
with a truncated Gaussian posterior, prove a conditional-update formula, or
assert that successive nonlinear filtering projections are exact.
-/

@[expose] public section

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace CHSLimited

/-- Equality of the first two raw moments, with their integrability obligations.
Probability-measure hypotheses are supplied by the theorems that use mass one. -/
structure MomentMatch (μ ν : Measure ℝ) : Prop where
  integrable_id_left : Integrable (fun x : ℝ => x) μ
  integrable_id_right : Integrable (fun x : ℝ => x) ν
  integrable_sq_left : Integrable (fun x : ℝ => x ^ 2) μ
  integrable_sq_right : Integrable (fun x : ℝ => x ^ 2) ν
  mean_eq : (∫ x, x ∂μ) = ∫ x, x ∂ν
  second_eq : (∫ x, x ^ 2 ∂μ) = ∫ x, x ^ 2 ∂ν

/-- An integrable quadratic has the expected expression in terms of raw moments. -/
theorem integral_quadratic (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hid : Integrable (fun x : ℝ => x) μ)
    (hsq : Integrable (fun x : ℝ => x ^ 2) μ) (a b c : ℝ) :
    (∫ x, a + b * x + c * x ^ 2 ∂μ) =
      a + b * (∫ x, x ∂μ) + c * (∫ x, x ^ 2 ∂μ) := by
  rw [integral_add (f := fun x : ℝ => a + b * x) (g := fun x : ℝ => c * x ^ 2)
      ((integrable_const a).add (hid.const_mul b)) (hsq.const_mul c),
    integral_add (f := fun _ : ℝ => a) (g := fun x : ℝ => b * x)
      (integrable_const a) (hid.const_mul b),
    integral_const_mul, integral_const_mul]
  simp

/-- Matching moments of two probability laws preserves every quadratic expectation. -/
theorem MomentMatch.integral_quadratic_eq {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) (a b c : ℝ) :
    (∫ x, a + b * x + c * x ^ 2 ∂μ) =
      ∫ x, a + b * x + c * x ^ 2 ∂ν := by
  rw [integral_quadratic μ h.integrable_id_left h.integrable_sq_left,
    integral_quadratic ν h.integrable_id_right h.integrable_sq_right,
    h.mean_eq, h.second_eq]

/-- Matching raw moments also preserves the second moment about any common center. -/
theorem MomentMatch.integral_centered_sq_eq {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) (m : ℝ) :
    (∫ x, (x - m) ^ 2 ∂μ) = ∫ x, (x - m) ^ 2 ∂ν := by
  have hp : (fun x : ℝ => (x - m) ^ 2) =
      (fun x : ℝ => m ^ 2 + (-2 * m) * x + 1 * x ^ 2) := by
    funext x
    ring
  rw [hp]
  exact h.integral_quadratic_eq (m ^ 2) (-2 * m) 1

/-- The first moment after a scalar affine transformation is preserved. -/
theorem MomentMatch.integral_affine_eq {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) (a b : ℝ) :
    (∫ x, a + b * x ∂μ) = ∫ x, a + b * x ∂ν := by
  simpa using h.integral_quadratic_eq a b 0

/-- The second raw moment after a scalar affine transformation is preserved. -/
theorem MomentMatch.integral_affine_sq_eq {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) (a b : ℝ) :
    (∫ x, (a + b * x) ^ 2 ∂μ) = ∫ x, (a + b * x) ^ 2 ∂ν := by
  have hp : (fun x : ℝ => (a + b * x) ^ 2) =
      (fun x : ℝ => a ^ 2 + (2 * a * b) * x + b ^ 2 * x ^ 2) := by
    funext x
    ring
  rw [hp]
  exact h.integral_quadratic_eq (a ^ 2) (2 * a * b) (b ^ 2)

/-- The Gaussian with the input law's mean and variance.
When the variance is zero this is the corresponding Dirac probability measure.
Moment-preservation claims below require a probability law with a finite second moment. -/
def gaussianProjection (μ : Measure ℝ) : Measure ℝ :=
  gaussianReal (∫ x, x ∂μ)
    ⟨variance (fun x : ℝ => x) μ, variance_nonneg _ _⟩

instance gaussianProjection_isProbabilityMeasure (μ : Measure ℝ) :
    IsProbabilityMeasure (gaussianProjection μ) := by
  unfold gaussianProjection
  exact ProbabilityTheory.instIsProbabilityMeasureGaussianReal _ _

/-- The actual Gaussian measure has the prescribed first moment. -/
@[simp] theorem gaussianProjection_mean (μ : Measure ℝ) :
    (∫ x, x ∂gaussianProjection μ) = ∫ x, x ∂μ := by
  exact integral_id_gaussianReal

/-- The actual Gaussian measure has the prescribed variance. -/
@[simp] theorem gaussianProjection_variance (μ : Measure ℝ) :
    variance (fun x : ℝ => x) (gaussianProjection μ) =
      variance (fun x : ℝ => x) μ := by
  exact variance_fun_id_gaussianReal

/-- The projected Gaussian has a finite second moment, including the degenerate case. -/
theorem gaussianProjection_memLp_two (μ : Measure ℝ) :
    MemLp (fun x : ℝ => x) 2 (gaussianProjection μ) := by
  exact memLp_id_gaussianReal 2

/-- Under a finite-second-moment assumption, the projected law has the same raw second moment. -/
theorem gaussianProjection_second (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hμ : MemLp (fun x : ℝ => x) 2 μ) :
    (∫ x, x ^ 2 ∂gaussianProjection μ) = ∫ x, x ^ 2 ∂μ := by
  have hleft := variance_eq_sub hμ
  have hright := variance_eq_sub (gaussianProjection_memLp_two μ)
  simp only [Pi.pow_apply, gaussianProjection_mean, gaussianProjection_variance] at hright
  simp only [Pi.pow_apply] at hleft
  linarith

/-- The input probability law and its actual Gaussian projection match their first two moments. -/
theorem gaussianProjection_momentMatch (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hμ : MemLp (fun x : ℝ => x) 2 μ) :
    MomentMatch μ (gaussianProjection μ) where
  integrable_id_left := hμ.integrable one_le_two
  integrable_id_right := (gaussianProjection_memLp_two μ).integrable one_le_two
  integrable_sq_left := hμ.integrable_sq
  integrable_sq_right := (gaussianProjection_memLp_two μ).integrable_sq
  mean_eq := (gaussianProjection_mean μ).symm
  second_eq := (gaussianProjection_second μ hμ).symm

/-- Projection preserves the second central moment of a square-integrable probability law. -/
theorem gaussianProjection_centered_second (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hμ : MemLp (fun x : ℝ => x) 2 μ) :
    (∫ x, (x - ∫ y, y ∂μ) ^ 2 ∂gaussianProjection μ) =
      ∫ x, (x - ∫ y, y ∂μ) ^ 2 ∂μ := by
  exact ((gaussianProjection_momentMatch μ hμ).integral_centered_sq_eq _).symm

/-- Every quadratic has exactly the same expectation before and after Gaussian projection. -/
theorem gaussianProjection_integral_quadratic (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hμ : MemLp (fun x : ℝ => x) 2 μ) (a b c : ℝ) :
    (∫ x, a + b * x + c * x ^ 2 ∂gaussianProjection μ) =
      ∫ x, a + b * x + c * x ^ 2 ∂μ := by
  exact ((gaussianProjection_momentMatch μ hμ).integral_quadratic_eq a b c).symm

/-- Already-Gaussian laws are fixed points of moment projection. -/
@[simp] theorem gaussianProjection_gaussianReal (m : ℝ) (v : ℝ≥0) :
    gaussianProjection (gaussianReal m v) = gaussianReal m v := by
  unfold gaussianProjection
  apply congrArg₂ gaussianReal
  · exact integral_id_gaussianReal
  · apply Subtype.ext
    exact variance_fun_id_gaussianReal

/-- Applying Gaussian moment projection twice does not change the resulting law. -/
@[simp] theorem gaussianProjection_idempotent (μ : Measure ℝ) :
    gaussianProjection (gaussianProjection μ) = gaussianProjection μ := by
  exact gaussianProjection_gaussianReal _ _

/-- The actual projected Gaussian has an integrable absolute third power. -/
theorem gaussianProjection_integrable_abs_cube (μ : Measure ℝ) :
    Integrable (fun x : ℝ => |x| ^ 3) (gaussianProjection μ) := by
  have h : MemLp (fun x : ℝ => x) 3 (gaussianProjection μ) :=
    memLp_id_gaussianReal 3
  simpa only [Real.norm_eq_abs] using h.integrable_norm_pow (p := 3) (by decide)

/-- A square-integrable scalar index has a square-integrable pushforward law.
`MemLp` supplies the almost-everywhere measurability needed for the actual map. -/
theorem memLp_id_map {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (X : Ω → ℝ) (hX : MemLp X 2 μ) :
    MemLp (fun x : ℝ => x) 2 (μ.map X) := by
  apply (memLp_map_measure_iff (g := fun x : ℝ => x)
    (by fun_prop) hX.aemeasurable).2
  simpa only [Function.comp_def] using hX

/-- Moment projection applies to the actual law of a scalar index of any state space.
For a CHS linear measurement index, the remaining premise is its square integrability. -/
theorem gaussianProjection_momentMatch_map {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : MemLp X 2 μ) :
    MomentMatch (μ.map X) (gaussianProjection (μ.map X)) := by
  exact gaussianProjection_momentMatch (μ.map X) (memLp_id_map μ X hX)

/-- The mean of the projected index law equals the expectation on the original state space. -/
theorem gaussianProjection_mean_map {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : MemLp X 2 μ) :
    (∫ x, x ∂gaussianProjection (μ.map X)) = ∫ ω, X ω ∂μ := by
  rw [gaussianProjection_mean, integral_map hX.aemeasurable (by fun_prop)]

/-- The second moment of the projected index law is the original index's second moment. -/
theorem gaussianProjection_second_map {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : MemLp X 2 μ) :
    (∫ x, x ^ 2 ∂gaussianProjection (μ.map X)) = ∫ ω, (X ω) ^ 2 ∂μ := by
  rw [gaussianProjection_second (μ.map X) (memLp_id_map μ X hX),
    integral_map hX.aemeasurable (by fun_prop)]

/-- Exact quadratic expectations for a projected scalar measurement index, expressed
directly as an integral over the original (possibly multidimensional) state space. -/
theorem gaussianProjection_integral_quadratic_map {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : MemLp X 2 μ) (a b c : ℝ) :
    (∫ x, a + b * x + c * x ^ 2 ∂gaussianProjection (μ.map X)) =
      ∫ ω, a + b * X ω + c * (X ω) ^ 2 ∂μ := by
  rw [gaussianProjection_integral_quadratic (μ.map X) (memLp_id_map μ X hX) a b c,
    integral_map hX.aemeasurable (by fun_prop)]

end CHSLimited
