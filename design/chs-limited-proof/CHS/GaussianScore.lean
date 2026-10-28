module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Probability.Moments.Variance
public import Mathlib.Tactic

/-!
# Unique optimality of scalar Gaussian moment matching for expected log score

The score is the negative log-density of a Gaussian, with the constant
`(log (2 * π)) / 2` omitted.  `candidateVariance` is a variance, not a standard
deviation.  The theorems identify its actual integral for an arbitrary L² real
random variable and prove unique minimization at the mean and variance when the
true variance is positive.

This file does not define differential entropy or prove a full KL projection
theorem.  No Gaussianity assumption on the true law is needed for the expected
score theorem.  It does not say that the minimized approximation error is small.
-/

@[expose] public section

noncomputable section

open MeasureTheory ProbabilityTheory

namespace CHS.GaussianScore

/-- Gaussian negative log-density with its parameter-independent constant omitted. -/
def pointScore (candidateMean candidateVariance x : ℝ) : ℝ :=
  (Real.log candidateVariance + (x - candidateMean) ^ 2 / candidateVariance) / 2

/-- Expected score expressed in terms of the true mean and variance. -/
def momentScore (mean variance candidateMean candidateVariance : ℝ) : ℝ :=
  (Real.log candidateVariance +
    (variance + (mean - candidateMean) ^ 2) / candidateVariance) / 2

/-- Exact excess expected score, separating a variance mismatch from a mean mismatch. -/
theorem momentScore_excess {mean variance candidateMean candidateVariance : ℝ}
    (hv : 0 < variance) (hw : 0 < candidateVariance) :
    momentScore mean variance candidateMean candidateVariance -
        momentScore mean variance mean variance =
      (Real.log (candidateVariance / variance) + variance / candidateVariance - 1 +
        (mean - candidateMean) ^ 2 / candidateVariance) / 2 := by
  rw [Real.log_div (ne_of_gt hw) (ne_of_gt hv)]
  unfold momentScore
  field_simp [ne_of_gt hv, ne_of_gt hw]
  ring

theorem variance_mismatch_nonneg {variance candidateVariance : ℝ}
    (hv : 0 < variance) (hw : 0 < candidateVariance) :
    0 ≤ Real.log (candidateVariance / variance) + variance / candidateVariance - 1 := by
  have hlog := Real.log_le_sub_one_of_pos (div_pos hv hw)
  rw [Real.log_div (ne_of_gt hv) (ne_of_gt hw)] at hlog
  rw [Real.log_div (ne_of_gt hw) (ne_of_gt hv)]
  linarith

/-- Every nondegenerate scalar Gaussian has expected negative log score at least
that of the Gaussian whose mean and variance match the true distribution. -/
theorem momentScore_minimized {mean variance candidateMean candidateVariance : ℝ}
    (hv : 0 < variance) (hw : 0 < candidateVariance) :
    momentScore mean variance mean variance ≤
      momentScore mean variance candidateMean candidateVariance := by
  have hvar := variance_mismatch_nonneg hv hw
  have hmean : 0 ≤ (mean - candidateMean) ^ 2 / candidateVariance :=
    div_nonneg (sq_nonneg _) hw.le
  have hexcess := momentScore_excess (mean := mean) (candidateMean := candidateMean) hv hw
  linarith

theorem variance_mismatch_pos {variance candidateVariance : ℝ}
    (hv : 0 < variance) (hw : 0 < candidateVariance)
    (hne : candidateVariance ≠ variance) :
    0 < Real.log (candidateVariance / variance) + variance / candidateVariance - 1 := by
  have hratio : variance / candidateVariance ≠ 1 := by
    intro heq
    have := (div_eq_one_iff_eq (ne_of_gt hw)).mp heq
    exact hne this.symm
  have hlog := Real.log_lt_sub_one_of_pos (div_pos hv hw) hratio
  rw [Real.log_div (ne_of_gt hv) (ne_of_gt hw)] at hlog
  rw [Real.log_div (ne_of_gt hw) (ne_of_gt hv)]
  linarith

/-- The minimizer is unique in both mean and positive variance. -/
theorem momentScore_eq_min_iff {mean variance candidateMean candidateVariance : ℝ}
    (hv : 0 < variance) (hw : 0 < candidateVariance) :
    momentScore mean variance candidateMean candidateVariance =
        momentScore mean variance mean variance ↔
      candidateMean = mean ∧ candidateVariance = variance := by
  constructor
  · intro heq
    have hexcess := momentScore_excess (mean := mean) (candidateMean := candidateMean) hv hw
    have hzero : Real.log (candidateVariance / variance) + variance / candidateVariance - 1 +
        (mean - candidateMean) ^ 2 / candidateVariance = 0 := by
      linarith
    have hmean_nonneg : 0 ≤ (mean - candidateMean) ^ 2 / candidateVariance :=
      div_nonneg (sq_nonneg _) hw.le
    have hw_eq : candidateVariance = variance := by
      by_contra hne
      have hstrict := variance_mismatch_pos hv hw hne
      linarith
    have hmean_zero : (mean - candidateMean) ^ 2 = 0 := by
      have hquot : (mean - candidateMean) ^ 2 / candidateVariance = 0 := by
        simpa [hw_eq, ne_of_gt hv] using hzero
      exact ((div_eq_zero_iff).mp hquot).resolve_right (ne_of_gt hw)
    have hm_eq : candidateMean = mean := by
      have := sq_eq_zero_iff.mp hmean_zero
      linarith
    exact ⟨hm_eq, hw_eq⟩
  · rintro ⟨rfl, rfl⟩
    rfl

section ExpectedScore

variable {Ω : Type*} [MeasurableSpace Ω] {ν : Measure Ω}
variable [IsProbabilityMeasure ν] {X : Ω → ℝ}

/-- L² makes every quadratic Gaussian score integrable. -/
theorem pointScore_integrable (hX : MemLp X 2 ν) (a w : ℝ) :
    Integrable (fun ω => pointScore a w (X ω)) ν := by
  unfold pointScore
  exact ((integrable_const _).add ((hX.sub (memLp_const a)).integrable_sq.div_const w)).div_const 2

/-- The expected squared error decomposes into variance plus squared bias. -/
theorem integral_sq_error (hX : MemLp X 2 ν) (a : ℝ) :
    (∫ ω, (X ω - a) ^ 2 ∂ν) =
      variance X ν + ((∫ ω, X ω ∂ν) - a) ^ 2 := by
  have hshift : MemLp (fun ω => X ω - a) 2 ν := hX.sub (memLp_const a)
  have hvar := variance_eq_sub hshift
  rw [variance_sub_const hX.aestronglyMeasurable a] at hvar
  have hmean : (∫ ω, X ω - a ∂ν) = (∫ ω, X ω ∂ν) - a := by
    rw [integral_sub (hX.integrable (by norm_num)) (integrable_const a)]
    simp
  simp only [Pi.pow_apply] at hvar
  rw [hmean] at hvar
  linarith

/-- This identifies `momentScore` with an actual expectation, rather than
assuming the identification in the optimization theorem. -/
theorem integral_pointScore (hX : MemLp X 2 ν) (a w : ℝ) :
    (∫ ω, pointScore a w (X ω) ∂ν) =
      momentScore (∫ ω, X ω ∂ν) (variance X ν) a w := by
  have hquad : Integrable (fun ω => (X ω - a) ^ 2 / w) ν :=
    (hX.sub (memLp_const a)).integrable_sq.div_const w
  unfold pointScore momentScore
  rw [integral_div,
    integral_add (integrable_const _) hquad,
    integral_div, integral_sq_error hX a]
  simp

/-- Unique minimization of the actual expected Gaussian negative log score for
an arbitrary L² law with strictly positive variance. -/
theorem expectedScore_unique_minimizer (hX : MemLp X 2 ν)
    (hv : 0 < variance X ν) {a w : ℝ} (hw : 0 < w) :
    (∫ ω, pointScore (∫ ω, X ω ∂ν) (variance X ν) (X ω) ∂ν) ≤
        (∫ ω, pointScore a w (X ω) ∂ν) ∧
      ((∫ ω, pointScore a w (X ω) ∂ν) =
          (∫ ω, pointScore (∫ ω, X ω ∂ν) (variance X ν) (X ω) ∂ν) ↔
        a = (∫ ω, X ω ∂ν) ∧ w = variance X ν) := by
  simp only [integral_pointScore hX]
  exact ⟨momentScore_minimized hv hw, momentScore_eq_min_iff hv hw⟩

end ExpectedScore

end CHS.GaussianScore
