module

public import CHS.WeakError
public import CHS.ErrorPropagation

/-!
# One-step approximation of a smooth limited-measurement likelihood

A probit or ordered-probit measurement is a smooth nonnegative function of a
scalar latent index. These theorems combine actual moment cancellation,
Taylor's theorem, and Bayes normalization. Smoothness and third-derivative
bounds are explicit model obligations. No normal-CDF derivative calculation,
truncation identity, or nonlinear CHS prediction rule is assumed by name.
-/

@[expose] public section

noncomputable section

open MeasureTheory ProbabilityTheory

namespace CHSLimited

/-- The absolute log-score error of a smooth measurement after moment projection. -/
theorem MomentMatch.smooth_log_score_bound {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) {g : ℝ → ℝ} (hg : ContDiff ℝ 3 g)
    {M floor : ℝ} (hM : 0 ≤ M) (hthird : ∀ z, |iteratedDeriv 3 g z| ≤ M)
    (h3μ : Integrable (fun x : ℝ => |x| ^ 3) μ)
    (h3ν : Integrable (fun x : ℝ => |x| ^ 3) ν)
    (b ε : ℝ) (hfloor : 0 < floor)
    (hμfloor : floor ≤ ∫ x, g (b + ε * x) ∂μ)
    (hνfloor : floor ≤ ∫ x, g (b + ε * x) ∂ν) :
    |Real.log (∫ x, g (b + ε * x) ∂μ) - Real.log (∫ x, g (b + ε * x) ∂ν)| ≤
      ((M / 6) * |ε| ^ 3 * ((∫ x, |x| ^ 3 ∂μ) + ∫ x, |x| ^ 3 ∂ν)) / floor := by
  have hweak := h.smooth_scaled_bound hg hM hthird b ε
    (smooth_scaled_integrable hg hM hthird h.integrable_id_left h.integrable_sq_left h3μ b ε)
    (smooth_scaled_integrable hg hM hthird h.integrable_id_right h.integrable_sq_right h3ν b ε)
    h3μ h3ν
  exact (CHS.ErrorPropagation.abs_log_sub_log_le_div hfloor hμfloor hνfloor).trans
    (div_le_div_of_nonneg_right hweak hfloor.le)

/-- Moment projection affects a subsequent normalized Bayes expectation only at
cubic order, when both the likelihood and weighted test have bounded third derivatives. -/
theorem MomentMatch.smooth_bayes_bound {μ ν : Measure ℝ}
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : MomentMatch μ ν) {g test : ℝ → ℝ}
    (hg : ContDiff ℝ 3 g) (hgt : ContDiff ℝ 3 (fun z => g z * test z))
    {Mg Mgt floor : ℝ} (hMg : 0 ≤ Mg) (hMgt : 0 ≤ Mgt)
    (hthirdg : ∀ z, |iteratedDeriv 3 g z| ≤ Mg)
    (hthirdgt : ∀ z, |iteratedDeriv 3 (fun z => g z * test z) z| ≤ Mgt)
    (hgnonneg : ∀ z, 0 ≤ g z) (htest : ∀ z, |test z| ≤ 1)
    (h3μ : Integrable (fun x : ℝ => |x| ^ 3) μ)
    (h3ν : Integrable (fun x : ℝ => |x| ^ 3) ν)
    (b ε : ℝ) (hfloor : 0 < floor)
    (hμfloor : floor ≤ ∫ x, g (b + ε * x) ∂μ)
    (hνfloor : floor ≤ ∫ x, g (b + ε * x) ∂ν) :
    |CHS.ErrorPropagation.bayesExpectation μ (fun x => g (b + ε * x))
        (fun x => test (b + ε * x)) -
      CHS.ErrorPropagation.bayesExpectation ν (fun x => g (b + ε * x))
        (fun x => test (b + ε * x))| ≤
      (((Mgt + Mg) / 6) * |ε| ^ 3 *
        ((∫ x, |x| ^ 3 ∂μ) + ∫ x, |x| ^ 3 ∂ν)) / floor := by
  have hgμ := smooth_scaled_integrable hg hMg hthirdg
    h.integrable_id_left h.integrable_sq_left h3μ b ε
  have hgν := smooth_scaled_integrable hg hMg hthirdg
    h.integrable_id_right h.integrable_sq_right h3ν b ε
  have hgtμ := smooth_scaled_integrable hgt hMgt hthirdgt
    h.integrable_id_left h.integrable_sq_left h3μ b ε
  have hgtν := smooth_scaled_integrable hgt hMgt hthirdgt
    h.integrable_id_right h.integrable_sq_right h3ν b ε
  have hnum := h.smooth_scaled_bound hgt hMgt hthirdgt b ε hgtμ hgtν h3μ h3ν
  have hden := h.smooth_scaled_bound hg hMg hthirdg b ε hgμ hgν h3μ h3ν
  have hb := CHS.ErrorPropagation.bayesExpectation_error μ ν
    (fun x => g (b + ε * x)) (fun x => test (b + ε * x))
    hgμ hgν hgtμ hgtν (fun x => hgnonneg _) (fun x => htest _)
    hfloor hμfloor hνfloor hnum hden
  convert hb using 1; ring

end CHSLimited
