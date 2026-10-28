module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Tactic

/-!
# Conditional finite-horizon error propagation for assumed-density filters

This module does not assume that moment matching makes a nonlinear filter exact.
It proves the algebraic and analytic propagation statements needed after local
projection and prediction errors have been established separately.

The strictly positive likelihood floor is substantive: without it a small
absolute probability error need not give a small log-likelihood error. The
recurrence hypotheses below must be discharged by the chosen transition,
observation, and projection operators before these results apply to CHS.
-/

@[expose] public section

namespace CHS.ErrorPropagation

open scoped BigOperators
open MeasureTheory

/-- A positive common floor makes the real logarithm Lipschitz on the two
normalizers. In particular, neither use of `Real.log` below is at zero. -/
theorem abs_log_sub_log_le_div
    {p q floor : ℝ} (hfloor : 0 < floor)
    (hp : floor ≤ p) (hq : floor ≤ q) :
    |Real.log p - Real.log q| ≤ |p - q| / floor := by
  have upper (a b : ℝ) (ha : floor ≤ a) (hb : floor ≤ b) :
      Real.log a - Real.log b ≤ |a - b| / floor := by
    have ha0 : 0 < a := lt_of_lt_of_le hfloor ha
    have hb0 : 0 < b := lt_of_lt_of_le hfloor hb
    calc
      Real.log a - Real.log b = Real.log (a / b) :=
        (Real.log_div ha0.ne' hb0.ne').symm
      _ ≤ a / b - 1 := Real.log_le_sub_one_of_pos (div_pos ha0 hb0)
      _ = (a - b) / b := by rw [sub_div, div_self hb0.ne']
      _ ≤ |a - b| / b :=
        div_le_div_of_nonneg_right (le_abs_self _) hb0.le
      _ ≤ |a - b| / floor :=
        div_le_div_of_nonneg_left (abs_nonneg _) hfloor hb
  have hpq := upper p q hp hq
  have hqp := upper q p hq hp
  rw [abs_sub_comm q p] at hqp
  exact abs_le.mpr ⟨by linarith, hpq⟩

/-- The sum of predictive log-normalizers, with exactly `horizon` updates. -/
noncomputable def logLikelihood (p : ℕ → ℝ) (horizon : ℕ) : ℝ :=
  ∑ t ∈ Finset.range horizon, Real.log (p t)

/-- Finite-horizon log-likelihood error is bounded by the sum of absolute
normalizer errors divided by a positive common floor. -/
theorem logLikelihood_error
    (p q : ℕ → ℝ) (horizon : ℕ) {floor : ℝ}
    (hfloor : 0 < floor)
    (hp : ∀ t < horizon, floor ≤ p t)
    (hq : ∀ t < horizon, floor ≤ q t) :
    |logLikelihood p horizon - logLikelihood q horizon| ≤
      ∑ t ∈ Finset.range horizon, |p t - q t| / floor := by
  unfold logLikelihood
  rw [← Finset.sum_sub_distrib]
  calc
    |∑ t ∈ Finset.range horizon, (Real.log (p t) - Real.log (q t))| ≤
        ∑ t ∈ Finset.range horizon, |Real.log (p t) - Real.log (q t)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ t ∈ Finset.range horizon, |p t - q t| / floor := by
      apply Finset.sum_le_sum
      intro t ht
      exact abs_log_sub_log_le_div hfloor
        (hp t (Finset.mem_range.mp ht)) (hq t (Finset.mem_range.mp ht))

/-- An explicit score-error budget can be substituted into the log bound. -/
theorem logLikelihood_error_of_score_error
    (p q scoreError : ℕ → ℝ) (horizon : ℕ) {floor : ℝ}
    (hfloor : 0 < floor)
    (hp : ∀ t < horizon, floor ≤ p t)
    (hq : ∀ t < horizon, floor ≤ q t)
    (hscore : ∀ t < horizon, |p t - q t| ≤ scoreError t) :
    |logLikelihood p horizon - logLikelihood q horizon| ≤
      ∑ t ∈ Finset.range horizon, scoreError t / floor := by
  refine (logLikelihood_error p q horizon hfloor hp hq).trans ?_
  apply Finset.sum_le_sum
  intro t ht
  exact div_le_div_of_nonneg_right
    (hscore t (Finset.mem_range.mp ht)) hfloor.le

/-- A recursively defined running mathematical certificate. `localError t`
includes every local error incurred between times `t` and `t+1`. -/
noncomputable def errorEnvelope (amplification : ℕ → ℝ) (initial : ℝ)
    (localError : ℕ → ℝ) : ℕ → ℝ
  | 0 => initial
  | t + 1 => amplification t * errorEnvelope amplification initial localError t +
      localError t

theorem errorEnvelope_nonneg
    (amplification localError : ℕ → ℝ) {initial : ℝ}
    (ha : ∀ t, 0 ≤ amplification t) (hi : 0 ≤ initial)
    (hd : ∀ t, 0 ≤ localError t) :
    ∀ t, 0 ≤ errorEnvelope amplification initial localError t := by
  intro t
  induction t with
  | zero => simpa [errorEnvelope] using hi
  | succ t ih =>
      exact add_nonneg (mul_nonneg (ha t) ih) (hd t)

/-- A one-sided error recurrence is bounded by its running certificate. -/
theorem error_le_envelope
    (error amplification localError : ℕ → ℝ) {initial : ℝ}
    (ha : ∀ t, 0 ≤ amplification t) (hinit : error 0 ≤ initial)
    (hstep : ∀ t, error (t + 1) ≤ amplification t * error t + localError t) :
    ∀ t, error t ≤ errorEnvelope amplification initial localError t := by
  intro t
  induction t with
  | zero => simpa [errorEnvelope] using hinit
  | succ t ih =>
      exact (hstep t).trans
        (add_le_add (mul_le_mul_of_nonneg_left ih (ha t)) (le_refl (localError t)))

/-- Prediction error is explicit, rather than hidden in a projection defect.
`predictedError ≤ transitionFactor * error + predictionDefect` and the Bayes
update followed by projection gives the stated combined recurrence. -/
theorem error_le_envelope_with_prediction
    (error predictedError transitionFactor bayesFactor predictionDefect
      projectionDefect : ℕ → ℝ)
    {initial : ℝ}
    (hk : ∀ t, 0 ≤ transitionFactor t)
    (hb : ∀ t, 0 ≤ bayesFactor t)
    (hinit : error 0 ≤ initial)
    (hpredict : ∀ t, predictedError t ≤
      transitionFactor t * error t + predictionDefect t)
    (hupdate : ∀ t, error (t + 1) ≤
      bayesFactor t * predictedError t + projectionDefect t) :
    ∀ t, error t ≤ errorEnvelope
      (fun s => bayesFactor s * transitionFactor s) initial
      (fun s => bayesFactor s * predictionDefect s + projectionDefect s) t := by
  apply error_le_envelope
  · intro t
    exact mul_nonneg (hb t) (hk t)
  · exact hinit
  · intro t
    calc
      error (t + 1) ≤ bayesFactor t * predictedError t + projectionDefect t :=
        hupdate t
      _ ≤ bayesFactor t *
          (transitionFactor t * error t + predictionDefect t) + projectionDefect t :=
        add_le_add (mul_le_mul_of_nonneg_left (hpredict t) (hb t))
          (le_refl (projectionDefect t))
      _ = (bayesFactor t * transitionFactor t) * error t +
          (bayesFactor t * predictionDefect t + projectionDefect t) := by ring

/-- A full finite-horizon certificate, conditional on explicit prediction,
Bayes/projection, and normalizer-control premises. This theorem does not assert
those premises for an arbitrary nonlinear transition or measurement model. -/
theorem logLikelihood_error_with_prediction
    (p q error predictedError transitionFactor bayesFactor predictionDefect
      projectionDefect : ℕ → ℝ)
    (horizon : ℕ) {initial floor : ℝ}
    (hfloor : 0 < floor)
    (hp : ∀ t < horizon, floor ≤ p t)
    (hq : ∀ t < horizon, floor ≤ q t)
    (hk : ∀ t, 0 ≤ transitionFactor t)
    (hb : ∀ t, 0 ≤ bayesFactor t)
    (hinit : error 0 ≤ initial)
    (hpredict : ∀ t, predictedError t ≤
      transitionFactor t * error t + predictionDefect t)
    (hupdate : ∀ t, error (t + 1) ≤
      bayesFactor t * predictedError t + projectionDefect t)
    (hscore : ∀ t < horizon, |p t - q t| ≤ predictedError t) :
    |logLikelihood p horizon - logLikelihood q horizon| ≤
      ∑ t ∈ Finset.range horizon,
        (transitionFactor t * errorEnvelope
          (fun s => bayesFactor s * transitionFactor s) initial
          (fun s => bayesFactor s * predictionDefect s + projectionDefect s) t +
          predictionDefect t) / floor := by
  apply logLikelihood_error_of_score_error p q _ horizon hfloor hp hq
  intro t ht
  have he := error_le_envelope_with_prediction error predictedError transitionFactor
    bayesFactor predictionDefect projectionDefect hk hb hinit hpredict hupdate t
  exact (hscore t ht).trans ((hpredict t).trans
    (add_le_add (mul_le_mul_of_nonneg_left he (hk t)) (le_refl (predictionDefect t))))

/-- The constant-coefficient geometric accumulation, kept division-free so the
same definition covers amplification equal to one. -/
noncomputable def geometricAccumulation (amplification : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => amplification * geometricAccumulation amplification t + 1

theorem geometricAccumulation_nonneg {amplification : ℝ}
    (ha : 0 ≤ amplification) : ∀ t, 0 ≤ geometricAccumulation amplification t := by
  intro t
  induction t with
  | zero => simp [geometricAccumulation]
  | succ t ih =>
      exact add_nonneg (mul_nonneg ha ih) zero_le_one

/-- Uniform cubic local defects imply a finite-horizon cubic certificate.
Its constant depends on the horizon and amplification; no uniform-in-horizon
claim is made. The cubic local-defect premise is not proved by this recurrence. -/
theorem error_le_cubic_envelope
    (error : ℕ → ℝ) {amplification coefficient loading : ℝ}
    (ha : 0 ≤ amplification) (hinit : error 0 ≤ 0)
    (hstep : ∀ t, error (t + 1) ≤
      amplification * error t + coefficient * |loading| ^ 3) :
    ∀ t, error t ≤ coefficient * |loading| ^ 3 *
      geometricAccumulation amplification t := by
  intro t
  induction t with
  | zero => simpa [geometricAccumulation] using hinit
  | succ t ih =>
      calc
        error (t + 1) ≤ amplification * error t + coefficient * |loading| ^ 3 :=
          hstep t
        _ ≤ amplification * (coefficient * |loading| ^ 3 *
            geometricAccumulation amplification t) + coefficient * |loading| ^ 3 :=
          add_le_add (mul_le_mul_of_nonneg_left ih ha)
            (le_refl (coefficient * |loading| ^ 3))
        _ = coefficient * |loading| ^ 3 *
            geometricAccumulation amplification (t + 1) := by
          simp only [geometricAccumulation]
          ring

/-- A genuinely finite-horizon cubic log-likelihood estimate. The coefficient
is explicit and horizon dependent. In an application the local cubic recurrence
must come from a moment-matching/Taylor theorem, while `hscore` must come from
the chosen observation class and prediction operator. -/
theorem logLikelihood_error_cubic
    (p q error : ℕ → ℝ) (horizon : ℕ)
    {floor amplification coefficient loading scoreFactor : ℝ}
    (hfloor : 0 < floor)
    (hp : ∀ t < horizon, floor ≤ p t)
    (hq : ∀ t < horizon, floor ≤ q t)
    (ha : 0 ≤ amplification) (hk : 0 ≤ scoreFactor)
    (hinit : error 0 ≤ 0)
    (hstep : ∀ t, error (t + 1) ≤
      amplification * error t + coefficient * |loading| ^ 3)
    (hscore : ∀ t < horizon, |p t - q t| ≤ scoreFactor * error t) :
    |logLikelihood p horizon - logLikelihood q horizon| ≤
      (scoreFactor * coefficient * |loading| ^ 3 / floor) *
        ∑ t ∈ Finset.range horizon, geometricAccumulation amplification t := by
  have he := error_le_cubic_envelope error ha hinit hstep
  calc
    |logLikelihood p horizon - logLikelihood q horizon| ≤
        ∑ t ∈ Finset.range horizon,
          (scoreFactor * (coefficient * |loading| ^ 3 *
            geometricAccumulation amplification t)) / floor := by
      apply logLikelihood_error_of_score_error p q _ horizon hfloor hp hq
      intro t ht
      exact (hscore t ht).trans (mul_le_mul_of_nonneg_left (he t) hk)
    _ = (scoreFactor * coefficient * |loading| ^ 3 / floor) *
        ∑ t ∈ Finset.range horizon, geometricAccumulation amplification t := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro t _
      ring

/-- If a nonnegative error starts at zero and there is no local defect, the
recurrence proves exactness at every finite time. -/
theorem error_eq_zero_of_zero_defects
    (error amplification : ℕ → ℝ)
    (hnonneg : ∀ t, 0 ≤ error t) (hinit : error 0 = 0)
    (hstep : ∀ t, error (t + 1) ≤ amplification t * error t) :
    ∀ t, error t = 0 := by
  intro t
  induction t with
  | zero => exact hinit
  | succ t ih =>
      apply le_antisymm _ (hnonneg (t + 1))
      simpa [ih] using hstep t

/-- Equality of controlled normalizers follows from zero filter error. This
algebraic exactness result does not claim that nonlinear moment projection has
zero defect; that hypothesis must be proved for the selected model. -/
theorem logLikelihood_eq_of_zero_defects
    (p q error amplification scoreFactor : ℕ → ℝ) (horizon : ℕ)
    (hnonneg : ∀ t, 0 ≤ error t) (hinit : error 0 = 0)
    (hstep : ∀ t, error (t + 1) ≤ amplification t * error t)
    (hscore : ∀ t < horizon, |p t - q t| ≤ scoreFactor t * error t) :
    logLikelihood p horizon = logLikelihood q horizon := by
  have he := error_eq_zero_of_zero_defects error amplification hnonneg hinit hstep
  unfold logLikelihood
  apply Finset.sum_congr rfl
  intro t ht
  have hzero : |p t - q t| ≤ 0 := by
    simpa [he t] using hscore t (Finset.mem_range.mp ht)
  have hpq : p t = q t := sub_eq_zero.mp (abs_eq_zero.mp (le_antisymm hzero (abs_nonneg _)))
  rw [hpq]

/-- Stability of normalized weighted expectations, proved from numerator and
normalizer differences. `|b| ≤ bound*q` is the bounded-test condition under the
second weighted distribution. -/
theorem normalized_quotient_error
    {a b p q floor numeratorError normalizerError bound : ℝ}
    (hfloor : 0 < floor) (hp : floor ≤ p) (hq : floor ≤ q)
    (hbound : 0 ≤ bound)
    (htest : |b| ≤ bound * q)
    (hnumerator : |a - b| ≤ numeratorError)
    (hnormalizer : |p - q| ≤ normalizerError) :
    |a / p - b / q| ≤ (numeratorError + bound * normalizerError) / floor := by
  have hp0 : 0 < p := lt_of_lt_of_le hfloor hp
  have hq0 : 0 < q := lt_of_lt_of_le hfloor hq
  have hn0 : 0 ≤ numeratorError := (abs_nonneg _).trans hnumerator
  have hd0 : 0 ≤ normalizerError := (abs_nonneg _).trans hnormalizer
  have hratio : |b / q| ≤ bound := by
    rw [abs_div, abs_of_pos hq0]
    exact (div_le_iff₀ hq0).mpr htest
  have hqp : |q - p| ≤ normalizerError := by
    simpa only [abs_sub_comm q p] using hnormalizer
  have hsum : |(a - b) + (b / q) * (q - p)| ≤
      numeratorError + bound * normalizerError := by
    calc
      |(a - b) + (b / q) * (q - p)| ≤
          |a - b| + |(b / q) * (q - p)| := abs_add_le _ _
      _ = |a - b| + |b / q| * |q - p| := by rw [abs_mul]
      _ ≤ numeratorError + bound * normalizerError :=
        add_le_add hnumerator (mul_le_mul hratio hqp (abs_nonneg _) hbound)
  calc
    |a / p - b / q| = |((a - b) + (b / q) * (q - p)) / p| := by
      congr 1
      field_simp [hp0.ne', hq0.ne']
      ring
    _ = |(a - b) + (b / q) * (q - p)| / p := by
      rw [abs_div, abs_of_pos hp0]
    _ ≤ (numeratorError + bound * normalizerError) / p :=
      div_le_div_of_nonneg_right hsum hp0.le
    _ ≤ (numeratorError + bound * normalizerError) / floor :=
      div_le_div_of_nonneg_left (add_nonneg hn0 (mul_nonneg hbound hd0)) hfloor hp

/-- The normalized weighted expectation used by a Bayes update. Positivity and
integrability are premises of the stability theorem below. -/
noncomputable def bayesExpectation {α : Type*} [MeasurableSpace α]
    (μ : Measure α) (likelihood test : α → ℝ) : ℝ :=
  (∫ x, likelihood x * test x ∂μ) / (∫ x, likelihood x ∂μ)

/-- Actual integral-level Bayes normalization stability for a bounded test.
The numerator/normalizer discrepancies must be supplied by a weak-distance or
other independently established estimate. This is not an assumed conclusion
about the normalized posterior discrepancy. -/
theorem bayesExpectation_error {α : Type*} [MeasurableSpace α]
    (μ ν : Measure α) (likelihood test : α → ℝ)
    (_hμ : Integrable likelihood μ)
    (hν : Integrable likelihood ν)
    (_hμTest : Integrable (fun x => likelihood x * test x) μ)
    (_hνTest : Integrable (fun x => likelihood x * test x) ν)
    (hlikelihood : ∀ x, 0 ≤ likelihood x)
    (htest : ∀ x, |test x| ≤ 1)
    {floor numeratorError normalizerError : ℝ}
    (hfloor : 0 < floor)
    (hμfloor : floor ≤ ∫ x, likelihood x ∂μ)
    (hνfloor : floor ≤ ∫ x, likelihood x ∂ν)
    (hnumerator : |(∫ x, likelihood x * test x ∂μ) -
      (∫ x, likelihood x * test x ∂ν)| ≤ numeratorError)
    (hnormalizer : |(∫ x, likelihood x ∂μ) -
      (∫ x, likelihood x ∂ν)| ≤ normalizerError) :
    |bayesExpectation μ likelihood test - bayesExpectation ν likelihood test| ≤
      (numeratorError + normalizerError) / floor := by
  have hb : |∫ x, likelihood x * test x ∂ν| ≤ ∫ x, likelihood x ∂ν := by
    simpa only [Real.norm_eq_abs] using
      (norm_integral_le_of_norm_le (f := fun x => likelihood x * test x)
        hν (Filter.Eventually.of_forall (fun x => by
        simp only [Real.norm_eq_abs, abs_mul, abs_of_nonneg (hlikelihood x)]
        simpa only [mul_one] using
          mul_le_mul_of_nonneg_left (htest x) (hlikelihood x))))
  unfold bayesExpectation
  simpa only [one_mul] using
    (normalized_quotient_error hfloor hμfloor hνfloor zero_le_one
      (by simpa only [one_mul] using hb) hnumerator hnormalizer)

/-- The concrete `9 / floor` normalization factor when a separate weak-test
estimate gives weighted-test error at most `8 * error` and likelihood error at
most `error`. The product-test factor eight is an explicit premise here. -/
theorem bayesExpectation_error_nine {α : Type*} [MeasurableSpace α]
    (μ ν : Measure α) (likelihood test : α → ℝ)
    (hμ : Integrable likelihood μ) (hν : Integrable likelihood ν)
    (hμTest : Integrable (fun x => likelihood x * test x) μ)
    (hνTest : Integrable (fun x => likelihood x * test x) ν)
    (hlikelihood : ∀ x, 0 ≤ likelihood x)
    (htest : ∀ x, |test x| ≤ 1)
    {floor error : ℝ} (hfloor : 0 < floor)
    (hμfloor : floor ≤ ∫ x, likelihood x ∂μ)
    (hνfloor : floor ≤ ∫ x, likelihood x ∂ν)
    (hnumerator : |(∫ x, likelihood x * test x ∂μ) -
      (∫ x, likelihood x * test x ∂ν)| ≤ 8 * error)
    (hnormalizer : |(∫ x, likelihood x ∂μ) -
      (∫ x, likelihood x ∂ν)| ≤ error) :
    |bayesExpectation μ likelihood test - bayesExpectation ν likelihood test| ≤
      9 * error / floor := by
  calc
    |bayesExpectation μ likelihood test - bayesExpectation ν likelihood test| ≤
        (8 * error + error) / floor :=
      bayesExpectation_error μ ν likelihood test hμ hν hμTest hνTest
        hlikelihood htest hfloor hμfloor hνfloor hnumerator hnormalizer
    _ = 9 * error / floor := by ring

end CHS.ErrorPropagation
