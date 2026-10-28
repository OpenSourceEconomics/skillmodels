module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Tactic

/-!
# Finite mixture errors and Bayes weights

These are finite-sum theorems over nonnegative mixture weights.  They do not
apply to quadrature rules with negative weights, including some unscented sigma
point rules.  The error theorem concerns supplied real component quantities;
an identification with integrals or probability distances is a separate step.

When prior weights sum to one and component values are likelihoods,
`mixtureLikelihood` is the current mixture likelihood.  The normalization
algebra also works for unnormalized nonnegative prior weights.  Strictly
positive total likelihood is explicit, so a zero-probability event is not
silently given a normalized posterior.
-/

@[expose] public section

noncomputable section

open scoped BigOperators

namespace CHS.Mixture

variable {ι : Type*} [Fintype ι]

/-- Componentwise errors average under nonnegative weights. -/
theorem weighted_sum_error (w a b error : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (he : ∀ i, |a i - b i| ≤ error i) :
    |(∑ i, w i * a i) - ∑ i, w i * b i| ≤ ∑ i, w i * error i := by
  calc
    |(∑ i, w i * a i) - ∑ i, w i * b i| =
        |∑ i, w i * (a i - b i)| := by
      simp_rw [mul_sub]
      rw [Finset.sum_sub_distrib]
    _ ≤ ∑ i, |w i * (a i - b i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, w i * |a i - b i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (hw i)]
    _ ≤ ∑ i, w i * error i := by
      exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (he i) (hw i)

/-- A uniform component error remains a uniform mixture error when the
nonnegative weights sum to one. -/
theorem weighted_sum_error_uniform (w a b error : ι → ℝ) (C : ℝ)
    (hw : ∀ i, 0 ≤ w i) (hsum : (∑ i, w i) = 1)
    (he : ∀ i, |a i - b i| ≤ error i) (hC : ∀ i, error i ≤ C) :
    |(∑ i, w i * a i) - ∑ i, w i * b i| ≤ C := by
  calc
    _ ≤ ∑ i, w i * error i := weighted_sum_error w a b error hw he
    _ ≤ ∑ i, w i * C := by
      exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hC i) (hw i)
    _ = C := by rw [← Finset.sum_mul, hsum, one_mul]

def mixtureLikelihood (w likelihood : ι → ℝ) : ℝ :=
  ∑ i, w i * likelihood i

def posteriorWeight (w likelihood : ι → ℝ) (i : ι) : ℝ :=
  w i * likelihood i / mixtureLikelihood w likelihood

theorem posteriorWeight_nonneg (w likelihood : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hL : ∀ i, 0 ≤ likelihood i)
    (hZ : 0 < mixtureLikelihood w likelihood) (i : ι) :
    0 ≤ posteriorWeight w likelihood i := by
  exact div_nonneg (mul_nonneg (hw i) (hL i)) hZ.le

/-- Positive evidence is sufficient for exact normalization.  Nonnegativity is
supplied separately when these normalized numbers are used as probabilities. -/
theorem posteriorWeight_sum (w likelihood : ι → ℝ)
    (hZ : 0 < mixtureLikelihood w likelihood) :
    (∑ i, posteriorWeight w likelihood i) = 1 := by
  unfold posteriorWeight
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hZ)

theorem posteriorWeights_are_probabilities (w likelihood : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hL : ∀ i, 0 ≤ likelihood i)
    (hZ : 0 < mixtureLikelihood w likelihood) :
    (∀ i, 0 ≤ posteriorWeight w likelihood i) ∧
      (∑ i, posteriorWeight w likelihood i) = 1 := by
  exact ⟨posteriorWeight_nonneg w likelihood hw hL hZ,
    posteriorWeight_sum w likelihood hZ⟩

/-- Apply component error budgets with the actual Bayes-updated weights. -/
theorem posterior_weighted_error (w likelihood a b error : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hL : ∀ i, 0 ≤ likelihood i)
    (hZ : 0 < mixtureLikelihood w likelihood)
    (he : ∀ i, |a i - b i| ≤ error i) :
    |(∑ i, posteriorWeight w likelihood i * a i) -
        ∑ i, posteriorWeight w likelihood i * b i| ≤
      ∑ i, posteriorWeight w likelihood i * error i := by
  exact weighted_sum_error _ a b error
    (posteriorWeight_nonneg w likelihood hw hL hZ) he

/-- Changing component state representations after scoring cannot change the
current likelihood, log score or Bayes weights when the stored component
likelihoods are unchanged.  No equality of future component likelihoods is
claimed.  Positive current evidence makes the log score a valid finite score. -/
theorem current_update_unchanged_of_component_scores_equal
    (w before after : ι → ℝ)
    (hZ : 0 < mixtureLikelihood w before)
    (hscores : ∀ i, before i = after i) :
    0 < mixtureLikelihood w after ∧
      mixtureLikelihood w before = mixtureLikelihood w after ∧
      Real.log (mixtureLikelihood w before) = Real.log (mixtureLikelihood w after) ∧
      (∀ i, posteriorWeight w before i = posteriorWeight w after i) := by
  have hsame : before = after := funext hscores
  subst after
  exact ⟨hZ, rfl, rfl, fun _ => rfl⟩

end CHS.Mixture
