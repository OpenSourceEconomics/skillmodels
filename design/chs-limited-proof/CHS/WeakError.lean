module

public import CHS.GaussianProjection
public import Mathlib.Analysis.Calculus.Taylor
public import Mathlib.Tactic

/-!
# A cubic weak-error guarantee for moment projection

This is a theorem about probability measures on ℝ, including continuous laws
and atoms. The analytic input is a globally bounded third derivative. Taylor's
theorem supplies a cubic remainder, and equality of the first two moments
cancels the entire quadratic Taylor polynomial.

The result bounds errors in expectations of smooth tests. It does not identify
weak error with total variation, a likelihood error, or a parameter-estimation
error. The log-likelihood conversion requires the additional hypotheses in
`CHS.ErrorPropagation`.
-/

@[expose] public section

noncomputable section

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal Interval

namespace CHSLimited

/-- An integrable quadratic, used to subtract the moment-matched part. -/
theorem integrable_quadratic {μ : Measure ℝ} [IsProbabilityMeasure μ]
    (hid : Integrable (fun x : ℝ => x) μ)
    (hsq : Integrable (fun x : ℝ => x ^ 2) μ) (a b c : ℝ) :
    Integrable (fun x : ℝ => a + b * x + c * x ^ 2) μ :=
  ((integrable_const a).add (hid.const_mul b)).add (hsq.const_mul c)

/-- Matching two moments cancels every quadratic part; only the remainder contributes. -/
theorem MomentMatch.remainder_bound {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) {f : ℝ → ℝ} (a b c C : ℝ) (weight : ℝ → ℝ)
    (hfμ : Integrable f μ) (hfν : Integrable f ν)
    (h3μ : Integrable weight μ)
    (h3ν : Integrable weight ν)
    (hr : ∀ x, |f x - (a + b * x + c * x ^ 2)| ≤ C * weight x) :
    |(∫ x, f x ∂μ) - ∫ x, f x ∂ν| ≤
      C * ((∫ x, weight x ∂μ) + ∫ x, weight x ∂ν) := by
  let q : ℝ → ℝ := fun x => a + b * x + c * x ^ 2
  have hqμ : Integrable q μ :=
    integrable_quadratic h.integrable_id_left h.integrable_sq_left a b c
  have hqν : Integrable q ν :=
    integrable_quadratic h.integrable_id_right h.integrable_sq_right a b c
  have hq : (∫ x, q x ∂μ) = ∫ x, q x ∂ν := h.integral_quadratic_eq a b c
  have hrμ : |∫ x, f x - q x ∂μ| ≤ C * ∫ x, weight x ∂μ := by
    calc
      _ ≤ ∫ x, C * weight x ∂μ := by
        simpa only [Real.norm_eq_abs] using
          (norm_integral_le_of_norm_le (f := fun x => f x - q x) (h3μ.const_mul C)
            (Filter.Eventually.of_forall (fun x => by
              simpa only [Real.norm_eq_abs] using hr x)))
      _ = _ := integral_const_mul C _
  have hrν : |∫ x, f x - q x ∂ν| ≤ C * ∫ x, weight x ∂ν := by
    calc
      _ ≤ ∫ x, C * weight x ∂ν := by
        simpa only [Real.norm_eq_abs] using
          (norm_integral_le_of_norm_le (f := fun x => f x - q x) (h3ν.const_mul C)
            (Filter.Eventually.of_forall (fun x => by
              simpa only [Real.norm_eq_abs] using hr x)))
      _ = _ := integral_const_mul C _
  rw [integral_sub hfμ hqμ] at hrμ
  rw [integral_sub hfν hqν] at hrν
  calc
    _ = |((∫ x, f x ∂μ) - ∫ x, q x ∂μ) -
        ((∫ x, f x ∂ν) - ∫ x, q x ∂ν)| := by rw [hq]; congr 1; ring
    _ ≤ |(∫ x, f x ∂μ) - ∫ x, q x ∂μ| +
        |(∫ x, f x ∂ν) - ∫ x, q x ∂ν| := abs_sub _ _
    _ ≤ C * (∫ x, weight x ∂μ) + C * (∫ x, weight x ∂ν) :=
      add_le_add hrμ hrν
    _ = _ := by ring

/-- The generic remainder theorem specialized to a cubic envelope. -/
theorem MomentMatch.cubic_remainder_bound {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) {f : ℝ → ℝ} (a b c C : ℝ)
    (hfμ : Integrable f μ) (hfν : Integrable f ν)
    (h3μ : Integrable (fun x : ℝ => |x| ^ 3) μ)
    (h3ν : Integrable (fun x : ℝ => |x| ^ 3) ν)
    (hr : ∀ x, |f x - (a + b * x + c * x ^ 2)| ≤ C * |x| ^ 3) :
    |(∫ x, f x ∂μ) - ∫ x, f x ∂ν| ≤
      C * ((∫ x, |x| ^ 3 ∂μ) + ∫ x, |x| ^ 3 ∂ν) :=
  h.remainder_bound a b c C (fun x => |x| ^ 3) hfμ hfν h3μ h3ν hr

/-- The ordinary quadratic Taylor polynomial, written with actual derivatives. -/
def quadraticTaylor (f : ℝ → ℝ) (b y : ℝ) : ℝ :=
  f b + iteratedDeriv 1 f b * (y - b) +
    (iteratedDeriv 2 f b / 2) * (y - b) ^ 2

/-- The interval Taylor polynomial agrees with the usual derivatives for a C³ function. -/
theorem taylorWithinEval_two_eq {f : ℝ → ℝ} (hf : ContDiff ℝ 3 f)
    {b y : ℝ} (hby : b ≠ y) :
    taylorWithinEval f 2 (uIcc b y) b y = quadraticTaylor f b y := by
  have hd1 : iteratedDerivWithin 1 f (uIcc b y) b = iteratedDeriv 1 f b :=
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hby)
      ((hf.of_le (by norm_num : (1 : WithTop ℕ∞) ≤ 3)).contDiffAt) (left_mem_uIcc)
  have hd2 : iteratedDerivWithin 2 f (uIcc b y) b = iteratedDeriv 2 f b :=
    iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hby)
      ((hf.of_le (by norm_num : (2 : WithTop ℕ∞) ≤ 3)).contDiffAt) (left_mem_uIcc)
  norm_num [show (2 : ℕ) = 1 + 1 from rfl, taylorWithinEval_succ,
    quadraticTaylor, hd1, hd2, smul_eq_mul]
  ring

/-- Taylor's theorem gives the sharp factorial `1/6` in the cubic remainder. -/
theorem quadraticTaylor_remainder_bound {f : ℝ → ℝ} (hf : ContDiff ℝ 3 f)
    {M : ℝ} (_hM : 0 ≤ M) (hthird : ∀ z, |iteratedDeriv 3 f z| ≤ M)
    (b y : ℝ) :
    |f y - quadraticTaylor f b y| ≤ (M / 6) * |y - b| ^ 3 := by
  by_cases hby : b = y
  · subst y
    simp [quadraticTaylor]
  · obtain ⟨z, _, hz⟩ := taylor_mean_remainder_lagrange_iteratedDeriv
      (n := 2) hby hf.contDiffOn
    rw [taylorWithinEval_two_eq hf hby] at hz
    rw [hz]
    norm_num [abs_div, abs_mul, abs_pow]
    nlinarith [mul_le_mul_of_nonneg_right (hthird z) (pow_nonneg (abs_nonneg (y - b)) 3)]

/-- A bounded third derivative makes scaled smooth tests accurate to cubic order. -/
theorem MomentMatch.smooth_scaled_bound {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) {f : ℝ → ℝ} (hf : ContDiff ℝ 3 f)
    {M : ℝ} (hM : 0 ≤ M) (hthird : ∀ z, |iteratedDeriv 3 f z| ≤ M)
    (b ε : ℝ)
    (hfμ : Integrable (fun x => f (b + ε * x)) μ)
    (hfν : Integrable (fun x => f (b + ε * x)) ν)
    (h3μ : Integrable (fun x : ℝ => |x| ^ 3) μ)
    (h3ν : Integrable (fun x : ℝ => |x| ^ 3) ν) :
    |(∫ x, f (b + ε * x) ∂μ) - ∫ x, f (b + ε * x) ∂ν| ≤
      (M / 6) * |ε| ^ 3 * ((∫ x, |x| ^ 3 ∂μ) + ∫ x, |x| ^ 3 ∂ν) := by
  apply h.cubic_remainder_bound (f b) (iteratedDeriv 1 f b * ε)
    (iteratedDeriv 2 f b / 2 * ε ^ 2) ((M / 6) * |ε| ^ 3)
    hfμ hfν h3μ h3ν
  intro x
  have hrem := quadraticTaylor_remainder_bound hf hM hthird b (b + ε * x)
  convert hrem using 1 <;> simp only [quadraticTaylor, add_sub_cancel_left, abs_mul, mul_pow]
  · congr 1
    ring
  · ring

/-- Centered third moments sharpen the same bound without changing the matched law. -/
theorem MomentMatch.smooth_scaled_centered_bound {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) {f : ℝ → ℝ} (hf : ContDiff ℝ 3 f)
    {M : ℝ} (hM : 0 ≤ M) (hthird : ∀ z, |iteratedDeriv 3 f z| ≤ M)
    (b ε center : ℝ)
    (hfμ : Integrable (fun x => f (b + ε * x)) μ)
    (hfν : Integrable (fun x => f (b + ε * x)) ν)
    (h3μ : Integrable (fun x : ℝ => |x - center| ^ 3) μ)
    (h3ν : Integrable (fun x : ℝ => |x - center| ^ 3) ν) :
    |(∫ x, f (b + ε * x) ∂μ) - ∫ x, f (b + ε * x) ∂ν| ≤
      (M / 6) * |ε| ^ 3 *
        ((∫ x, |x - center| ^ 3 ∂μ) + ∫ x, |x - center| ^ 3 ∂ν) := by
  let base := b + ε * center
  let d1 := iteratedDeriv 1 f base
  let d2 := iteratedDeriv 2 f base
  apply h.remainder_bound
    (f base - d1 * ε * center + (d2 / 2) * ε ^ 2 * center ^ 2)
    (d1 * ε - d2 * ε ^ 2 * center) ((d2 / 2) * ε ^ 2)
    ((M / 6) * |ε| ^ 3) (fun x => |x - center| ^ 3) hfμ hfν h3μ h3ν
  intro x
  have hrem := quadraticTaylor_remainder_bound hf hM hthird base (b + ε * x)
  have hshift : b + ε * x - base = ε * (x - center) := by dsimp [base]; ring
  simp only [quadraticTaylor, hshift, abs_mul, mul_pow] at hrem
  convert hrem using 1
  · congr 1
    dsimp [d1, d2]
    ring
  · ring

/-- A measurable function with an integrable quadratic and cubic remainder is integrable. -/
theorem integrable_of_cubic_remainder {μ : Measure ℝ} [IsProbabilityMeasure μ]
    {f : ℝ → ℝ} (hf : AEStronglyMeasurable f μ)
    (hid : Integrable (fun x : ℝ => x) μ)
    (hsq : Integrable (fun x : ℝ => x ^ 2) μ)
    (h3 : Integrable (fun x : ℝ => |x| ^ 3) μ)
    (a b c C : ℝ)
    (hr : ∀ x, |f x - (a + b * x + c * x ^ 2)| ≤ C * |x| ^ 3) :
    Integrable f μ := by
  have hq := integrable_quadratic hid hsq a b c
  apply (hq.norm.add (h3.const_mul C)).mono' hf
  apply Filter.Eventually.of_forall
  intro x
  simp only [Real.norm_eq_abs, Pi.add_apply]
  have hh := abs_add_le (f x - (a + b * x + c * x ^ 2)) (a + b * x + c * x ^ 2)
  have hh' : |f x| ≤ |f x - (a + b * x + c * x ^ 2)| +
      |a + b * x + c * x ^ 2| := by simpa using hh
  linarith [hr x]

/-- C³ tests with bounded third derivative are integrable under a finite-third-moment law. -/
theorem smooth_scaled_integrable {μ : Measure ℝ} [IsProbabilityMeasure μ]
    {f : ℝ → ℝ} (hf : ContDiff ℝ 3 f)
    {M : ℝ} (hM : 0 ≤ M) (hthird : ∀ z, |iteratedDeriv 3 f z| ≤ M)
    (hid : Integrable (fun x : ℝ => x) μ)
    (hsq : Integrable (fun x : ℝ => x ^ 2) μ)
    (h3 : Integrable (fun x : ℝ => |x| ^ 3) μ) (b ε : ℝ) :
    Integrable (fun x => f (b + ε * x)) μ := by
  have hfc : Continuous (fun x : ℝ => f (b + ε * x)) :=
    hf.continuous.comp (continuous_const.add (continuous_const.mul continuous_id))
  apply integrable_of_cubic_remainder (f := fun x => f (b + ε * x))
    hfc.aestronglyMeasurable
    hid hsq h3 (f b) (iteratedDeriv 1 f b * ε)
    (iteratedDeriv 2 f b / 2 * ε ^ 2) ((M / 6) * |ε| ^ 3)
  intro x
  have hrem := quadraticTaylor_remainder_bound hf hM hthird b (b + ε * x)
  convert hrem using 1 <;> simp only [quadraticTaylor, add_sub_cancel_left, abs_mul, mul_pow]
  · congr 1
    ring
  · ring

/-- The cubic bound for the actual Gaussian projection, with integrability proved from L³. -/
theorem gaussianProjection_smooth_scaled_bound (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hμ : MemLp (fun x : ℝ => x) 3 μ)
    {f : ℝ → ℝ} (hf : ContDiff ℝ 3 f)
    {M : ℝ} (hM : 0 ≤ M) (hthird : ∀ z, |iteratedDeriv 3 f z| ≤ M)
    (b ε : ℝ) :
    |(∫ x, f (b + ε * x) ∂μ) - ∫ x, f (b + ε * x) ∂gaussianProjection μ| ≤
      (M / 6) * |ε| ^ 3 *
        ((∫ x, |x| ^ 3 ∂μ) + ∫ x, |x| ^ 3 ∂gaussianProjection μ) := by
  have h2 : MemLp (fun x : ℝ => x) 2 μ := hμ.mono_exponent (by norm_num)
  have h := gaussianProjection_momentMatch μ h2
  have h3μ : Integrable (fun x : ℝ => |x| ^ 3) μ := by
    simpa only [Real.norm_eq_abs] using hμ.integrable_norm_pow (by norm_num : (3 : ℕ) ≠ 0)
  have hν : MemLp (fun x : ℝ => x) 3 (gaussianProjection μ) := memLp_id_gaussianReal 3
  have h3ν : Integrable (fun x : ℝ => |x| ^ 3) (gaussianProjection μ) := by
    simpa only [Real.norm_eq_abs] using hν.integrable_norm_pow (by norm_num : (3 : ℕ) ≠ 0)
  exact h.smooth_scaled_bound hf hM hthird b ε
    (smooth_scaled_integrable hf hM hthird h.integrable_id_left h.integrable_sq_left h3μ b ε)
    (smooth_scaled_integrable hf hM hthird h.integrable_id_right h.integrable_sq_right h3ν b ε)
    h3μ h3ν

end CHSLimited
