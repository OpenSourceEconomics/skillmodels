# Probit and Tobit measurements in the CHS estimator — design brief

Status: implemented on branch `limited-measurements` (PR #105); §5 adds ordered probit
to the scope. The text up to §4 is the original design request. Base: branch
`probit-measurements` (probit and Tobit) on top of the open PR chain

- `main` ← #103 `missing-observed-factors` (integrate out missing observed factors)
- ← #102 `asymmetric-translog`
- ← #104 `frozen-after-last-observation` (head `ccbf9b5`; states are not predicted after
  a person's last observed measurement).

The new work is one more PR (or a short chain) on top of #104.

## 1. Why

The application (health-cognition, HRS, ages 68–98, 17,776 persons, 16 two-year
periods, 5 latent states, 3 investments, about 2,000 free parameters) measures survival by
`alive` ∈ {0, 1}: missing at period 0, 1 while alive, 0 in the period of death (the last
observation of a decedent; 9,467 death rows, 74,153 alive rows). CHS treats `alive` as a
Gaussian linear-probability measurement with a free error SD (on a linear age trend with a
floor of 0.1) and sign-restricted loadings on linear age trends.

Evidence (attached logs, `runi_asym_kmean` = asymmetric translog, `runi_linear` = linear
transitions, same measurement system, both certified local maxima):

- The translog beats the linear model by 6,529 log-likelihood points; 6,434 of them come
  from the `alive` contributions of the alive rows, −408 from the death rows. The vitality
  state's own measurements (grip strength, peak flow, "everything an effort") lose 2,087.
- The `alive` loading on vitality is 0.14–0.36 (linear: 0.02–0.11); the loadings on
  physical health and cognition sit at their sign bound 0 at all ages; the `alive` SD sits
  at its floor; vitality's own-curvature term sits at its bound (−0.5) in 21 of 30 terms.
- Earlier, the two-mixture translog used mental health, then cognition, as the mortality
  factor and collapsed the `alive` SD; floors, sign restrictions and trends were added.

Reading: with a Gaussian density for a 0/1 outcome, every alive row rewards a smaller
residual SD (−log σ), so a flexible transition is spent on sharpening survival
predictions. A probit has no free scale: the reward disappears and survival is fitted as a
probability.

Censored measures appear in the same application (indices with floor/ceiling effects,
counts at zero such as volunteer hours or symptom counts) and are currently Gaussian too.
The Tobit family is already declared in `ModelSpec` and used by AF; CHS should support
both families in one change.

## 2. Current support in the code

- `common/measurement_models.py`: `GaussianMeasurement`, `ProbitMeasurement`,
  `TobitMeasurement`, `MeasurementFamily`, `measurement_loglik` (uses `log_ndtr`).
- `ModelSpec.measurement_models` maps measurement names to families
  (`common/model_spec.py`).
- AF uses the families in its initial period (`af/initial_period.py`); simulation draws
  probit measures (`common/simulate_data.py`).
- CHS rejects non-Gaussian measures (`chs/maximization_inputs.py`,
  `_fail_if_non_gaussian_measurements`); AMN too (`amn/estimate.py`), and CHS seeds from
  AMN by default.
- `params_index.py` / `parse_params.py` / `process_model.py` are family-agnostic: a probit
  measure still gets a `meas_sds` parameter.
- CHS update: `chs/kalman_filters.py::kalman_update`, square-root form via QR of
  `[[σ, 0], [F*, U]]`. Measurements are linear in the states for every transition
  function; the unscented transform is used only in the predict step. One update per
  `(aug_period, measurement)`, all inside one `lax.scan` (`chs/likelihood.py::_scan_body`);
  measurement vs anchoring updates are selected with `lax.cond` on a per-iteration flag.
  Mixtures: per-component likelihoods and weight updates inside `kalman_update`.
- #104: `get_observation_masks` / `predict_if_observed_later` freeze states after the last
  observed measurement; for a decedent that is the `alive = 0` row.
- Debug path: `chs/likelihood_debug.py`, `chs/kalman_filters_debug.py`,
  `chs/process_debug_data.py` (residuals, residual SDs, contributions, filtered states).

## 3. Proposed mathematics

Per person and mixture component the predicted state is θ ~ N(m, P), P = UᵀU. A probit
measurement: y* = a + λᵀθ + ε, ε ~ N(0, 1), y = 1{y* > 0}, a = controls·γ.

- Likelihood contribution (exact under the Gaussian predicted state):
  μ = a + λᵀm, s² = 1 + λᵀPλ, t = (2y − 1) μ / s, ℓ = log Φ(t).
- Update by moment matching (assumed-density filtering): with r = φ(t)/Φ(t),
  m⁺ = m + Pλ (2y − 1) r / s,
  P⁺ = P − Pλλᵀ P · r (r + t) / s².
- Equivalent Gaussian pseudo-measurement, so the existing QR update can be reused:
  κ = r (r + t) / s² ∈ (0, 1/λᵀPλ), pseudo-noise variance σ̃² = 1/κ − λᵀPλ > 0, pseudo-
  residual chosen so that the Kalman mean shift equals (2y − 1) r / s · Pλ. The Gaussian
  log-density of the pseudo-measurement is replaced by ℓ.
- Mixtures: each component gets its own ℓ_k; weights update as for Gaussian measures.
- After `alive = 0` the states are frozen (#104), so the approximate update at the death
  row is never projected. After `alive = 1` the update is small when survival is likely.

Tobit measurement with known bounds L < U (either may be infinite), free error SD σ:
y* = a + λᵀθ + ε, ε ~ N(0, σ²), y = min(max(y*, L), U).

- Interior observation, L < y < U: the ordinary Gaussian Kalman update and density (exact,
  the existing `kalman_update`).
- Censored observation, y = L (or y = U): with s² = σ² + λᵀPλ,
  t = (L − μ)/s (lower) or t = (μ − U)/s (upper), ℓ = log Φ(t) (exact under the Gaussian
  predicted state). The update conditions u = λᵀθ + ε on a half-line; with the sign
  convention of the truncation it is the probit update with 1 replaced by σ² in s² and
  the direction of the half-line in place of (2y − 1): mean shift −(or +) r/s · Pλ,
  covariance P − Pλλᵀ P · r (r + t)/s², r = φ(t)/Φ(t), and the same pseudo-measurement
  construction with σ̃² = 1/κ − λᵀPλ.
- σ enters the censored contributions through s and the interior ones through the
  density; it stays a free parameter (`meas_sds`).

## 4. What has to be decided

The questions this section listed were answered in the review of PR 105 and are not
repeated here.

## 5. Ordered probit (added scope)

An ordered measure y ∈ {0, …, K−1} (K ≥ 2): y* = a + λᵀθ + ε, ε ~ N(0, 1), y = k iff
c_k < y* ≤ c_{k+1}, with c_0 = −∞, c_K = +∞ and c_1 = 0 (the intercept a stays free).

- API (`OrderedProbitMeasurement(n_categories)`, `from_dict` family `"ordered_probit"`):
  data are integer codes 0..K−1 (anything else raises in `process_data`), NaN is missing;
  `meas_sds` is fixed to 1; the parameters `("cutpoints", aug_period, y, "cutpoint_k")`,
  k = 1..K−1, are per period, `cutpoint_1` is fixed to 0 and the cutpoints of a period are
  weakly increasing (optimagic `IncreasingConstraint`; an empty category has zero
  likelihood, so an interior optimum is strictly increasing). Ties across periods are
  the application's equality constraints.
- CHS update: y = 0 and y = K−1 are the probit half-line events below c_1 and above
  c_{K−1}. An interior category conditions on the interval (α, β] =
  ((c_k − μ)/s, (c_{k+1} − μ)/s]: ℓ = log(Φ(β) − Φ(α)), mean shift E[z]·Pλ/s,
  P⁺ = P − Pλλᵀ P (1 − Var[z]) / s², with the moments of the truncated standard normal
  (`normal_interval_terms`: a Hermite series around the midpoint for narrow intervals,
  else the difference of the two half-lines with the tail machinery).
  K = 2 reproduces the probit model bitwise.
- Shared kernel: `measurement_loglik` evaluates log(Φ(c_{k+1} − η) − Φ(c_k − η)) with
  `log_normal_interval_mass` (`skillmodels.common.normal_terms`), which uses the log mass
  of `normal_interval_terms` for two finite ends and log Φ for a half-line. A difference
  of rounded endpoint log-CDFs is not used: for a category (0, 1e-17] both endpoint
  log-CDFs round to the same binary64 value, while the series keeps the finite log mass
  −40.0628851141 and its endpoint derivatives. Rows of other families pass finite
  placeholder ends, so the unused ordered term cannot poison their gradients. AF's
  initial-period likelihood accepts per-measure cut tables. AF estimation itself has no
  cutpoint parameters and rejects ordered measures.
- `simulate_dataset` draws the number of cutpoints below η + N(0, 1).
- Dispatch: in a model with limited measures one `lax.switch` selects the anchoring,
  Gaussian or limited update and the limited branch is rematerialized
  (`jax.checkpoint`), so a Gaussian update does not carry the limited branch's
  zero-filled residuals; all-Gaussian models keep the previous scan.

## 6. Candidate-set error

The profile protocol asks for `eps_candidate`, the error from candidates that a
discretization never represents (omitted grid points, unrepresented choice islands). Its
value for this target is 0 by construction, in both profiles:

- The likelihood has no candidate set. Each update evaluates the probability or density
  of the one observed event (a Gaussian value, a half-line, an interval) under the
  Gaussian predicted state in closed form: `log N`, `log Φ(t)`, or the log interval mass
  of `normal_interval_terms`. No maximum, argmax or selection over represented
  alternatives occurs in the likelihood, the update or the shared kernel.
- The only finite representations are the evaluation of these closed forms (series,
  continued fraction, `log_ndtr`), whose errors are the `update_loglik_error`,
  `interval_moment_error` and `shared_interval_mass_error` of the profile contract, and
  the assumed-density projection of each posterior, whose effect is the reported
  `full_likelihood_gap`.
- The maximization over parameters is done by the user's optimizer on this likelihood;
  it is not part of the target.

Assumptions: the measurement families and update order are those of the model
specification; the bound applies to the CHS likelihood, the shared measurement kernel
and the AF initial-period likelihood with supplied cut tables.

## 7. Tobit measurements with observation-specific intervals

Motivation (health-cognition, HRS): self-respondents report a 27-point cognition score;
proxy respondents report only the Langa–Weir class (dementia / CIND / normal), which is
an interval of the same score. A shared intercept, loadings and error SD for both kinds
of report put them on one scale.

### API

`TobitMeasurement(lower, upper, interval_columns=(lo_column, hi_column))`; in
`ModelSpec.from_dict`: `{"family": "tobit", "lower": ..., "upper": ...,
"interval_columns": [lo_column, hi_column]}`.

- The measurement's own column holds the exact values; an exact value is censored at
  a fixed bound as before.
- The two named columns hold, for a person whose value is missing, the ends
  `lo < y* <= hi` of the latent interval; NaN where no interval is observed; `-inf` /
  `inf` for an open end. `lower` and `upper` may both be `None` with interval columns.
- Invalid rows raise in `process_data`: a value together with an interval, one NaN end,
  `lo >= hi`, or two infinite ends.
- An integer score treated as continuous maps the observed scores `k_lo, ..., k_hi` to
  the latent interval `(k_lo - 0.5, k_hi + 0.5]`, with an infinite end where the range
  reaches a censoring bound. This is the application's choice, documented on the class.

### Mathematics

The latent response is `y* = a + λ'θ + ε`, `ε ~ N(0, σ²)`, `σ` free. Under the Gaussian
predicted state `N(m, P)`, with `μ = a + λ'm`, `s² = σ² + λ'Pλ`:

- interval with two finite ends: `ℓ = log(Φ(β) − Φ(α))`, `α = (lo − μ)/s`,
  `β = (hi − μ)/s`; posterior mean `m + Pλ E[z]/s` and covariance
  `P − Pλλ'P (1 − Var[z])/s²` with the moments of `z ~ N(0,1)` truncated to `(α, β]`
  (`normal_interval_terms`, narrow-safe);
- `lo = −∞` / `hi = ∞`: the half-line update of §3 with threshold `hi` / `lo`.
- The interval is evaluated from its source geometry (`source_interval_terms`), not
  from `α` and `β` as two separately rounded floats. Standardizing the ends one by
  one loses information in four ways, each reachable from accepted finite data:
  rounding collapses or distorts a narrow interval (`(0, 1e-17]` with `μ = s = 1`
  gives `α = β = −1` while `ℓ = −40.5629`); the standardized width or its half
  underflows (`(0, 2^-600]` with `μ = s = 2^500` has width `2^-1100`,
  `ℓ = −763.8808`); the difference `lo − μ` overflows although the standardized
  interval is moderate (`(−2^1023, −2^1022]` with `μ = s = 2^1023` is `(−2, −1.5]`,
  `ℓ = −3.1223`, while `lo − μ` rounds to `−∞`); and two ends that round to the same
  float enclose a mass that is not a narrow-density integral (`(0, 2^-30]` with
  `μ = 2^30`, `s = 1`: both ends round to `−2^30`, the density changes by the
  factor `e` across the interval, `ℓ = −2^59`, and a difference of endpoint
  log-CDFs gives `log 0`).
- The representation is the triple `(α, β, log w)` with `w = (hi − lo)/s`, each
  element formed from the source values with one rounding per operation and no
  intermediate that can overflow or underflow: `α = (lo − μ)/s` and `β` halve both
  operands of the subtraction (exactly) when one of them has magnitude `≥ 2^1023`
  and double the quotient; `log w = log(hi − lo) − log s` with the same halving of
  the span, and the standardized width itself is never formed. The centre
  `c = (α + β)/2` and half-width `h = w/2` select the branch (`h max(1, |c|) ≤ 1/4`):
  the narrow Hermite series around `c` uses `log w` in `ℓ`; the wide branch
  evaluates `ℓ = log Φ(β) + log(1 − q)` with `log q = log Φ(α) − log Φ(β)` written
  as `c w + log r(β) − log r(α)`, `r` the Mills ratio. The leading term
  `(β² − α²)/2 = c w` comes from the centre and the log width, and `log r` varies
  slowly in its argument (`d log r / dx = −(x + r)`, of order `1/|x|` in the tail),
  so `log q` keeps the span of the source interval when the two rounded ends
  coincide; a difference of their log-CDFs would be 0 there. Both branches derive
  from the same triple, and agree to rounding at their boundary.
- Why this closes the class on the public finite domain (finite `lo < hi`, finite
  `μ`, finite `s > 0`): every intermediate is either a correctly rounded function
  of the source values (the triple), or a slowly varying function of a rounded
  standardized end (`log Φ`, `r`, `V`), or formed from `c` and `w` without
  cancellation (`c w`, `−c²/2`, the Hermite series). The remaining errors are the
  relative rounding of `α`, `β`, `c` (`2^-53`, which enters `ℓ` as `2^-53 c²`, the
  spacing of the leading term `−c²/2`, hence "a few ulps" where `|ℓ|` is huge and
  below `1e-15` where it is moderate), and the absolute rounding of `log w` (at most
  about `1500 · 2^-53 ≈ 2e-13`, entering `ℓ` directly in the narrow branch and
  through `q/(1 − q) · |c w| · 2e-13 ≲ 1e-13` in the wide branch, since the wide
  branch needs `|c w| ≥ 1/2` or `w ≥ 1/2` and `q ≤ e^{-1/2}`). No tolerance, cap or
  `nextafter` is used. Where the correctly rounded `ℓ` itself is not a finite float
  (`|c| ≳ 2^512`, `ℓ < −2^1023`), `ℓ` is `−∞`, as for a Gaussian density with such
  an innovation. Rows without an interval get constant finite placeholder
  arguments before any arithmetic, so their discarded values and derivatives are
  finite. Models without interval columns keep the difference of the standardized
  cutpoints (`normal_interval_terms`), literally unchanged.
- Standardized intermediates beyond the float range with a finite `ℓ`: `α` or `β`
  beyond `2^1023` (e.g. `lo = −10^300`, `s = 10^-10`), `w` beyond `e^709` (a span of
  the whole float range), or `c w` beyond the largest float. The division that
  would overflow is not performed: the end is replaced by the infinity of its sign
  before the division (`_scaled_difference`), which also keeps the derivative of
  the division finite. The branch is then
  chosen from `log(w·max(1,|c|))`, which is finite or `+∞`: at most `log(1/2)` the
  narrow series; above `log 760` — or with an infinite end — `q = Φ(α)/Φ(β)` is 0
  in float64 (if `|c| ≥ 1`, `c w < −760`; else `α < −380`), and the interval is
  evaluated as the half-line below `β`: `log q = −∞` exactly, and the arguments of
  the tilt, the exponential of the log width and the lower-end tail terms are
  replaced by finite placeholders before those operations, so that neither a value
  nor the derivative of a discarded branch contains `0·∞`. The far branch changes
  no value: on its rows the moderate formula gives `q = 0` as well. Where the
  nearer end `β` is itself `−∞` (both ends beyond `2^1023` on one side) the
  interval holds no representable mass: `ℓ = −∞`, `E[z] = β`, `Var[z] = 0`, and
  the tail terms receive a placeholder argument.
- Intermediates of the narrow series and of the derivatives beyond the float
  range with a finite result: the Hermite series around `c` multiplies `He_n(c)`,
  which overflows for `|c| ≳ 10^20` at `n = 16`, by `h^n`, which underflows for
  tiny `h`, although their product is bounded by `(h max(1,|c|))^n ≤ 4^-n`. The
  recursion carries the product `t_n = He_n(c) h^n / n!` as one quantity
  (`t_n = (c h) t_{n−1}/n − h² t_{n−2}/n`), so no term leaves the float range; and
  `−c²/2` is formed as `(−c/2)·c`, which overflows only where `ℓ` is not a finite
  float. The derivative in the scale is routed through `log s`: `kalman_update_limited`
  passes `log s` beside `s`, and the only operation in which `s` enters, the
  division `(lo − μ)/s`, has the custom rule `d(num/s) = d num / s − (num/s)·d log s`
  with the tangent of `s` itself ignored (`s` and `log s` are one quantity). The
  derivative of `ℓ` in `log s` is about `c²`, twice the log mass, and is therefore
  a finite float whenever `ℓ` is, except where `ℓ < −2^1022`; the derivative in `s`
  itself, `c²/s`, and the default rule's `s^-2` are not. Values and derivatives are
  therefore finite floats wherever `ℓ` and the true derivatives are, and `ℓ = −∞`
  exactly where the correctly rounded log mass is not a finite float (checked on a
  seeded log-uniform sample of the public domain against an exact classification
  of each row); the one remaining non-finite derivative in the *update* is shared
  by every measurement family: `w = v/s` in `kalman_update_limited` differentiates
  in `s` through `s^-2`, which overflows for `s < 10^-154`.
- Supported domain for the accuracy of the posterior moments: prior state SD,
  loading and measurement SD within `[2^-10, 2^10]` in absolute value, and
  standardized ends within `2^10` of the predicted mean. Values, derivatives and
  finiteness as above hold on the whole public domain. Outside the supported domain
  `E[z]` and `Var[z]` may lose the source width: where the two standardized ends
  round to the same float but the wide branch applies (a tail distance near
  `2^30`, as in the reviewer's witness with prior SD `2^40`), `Var[z]` is the
  half-line value. Inside it, 2,000 seeded log-uniform cases against an
  independent 800-digit integral of the source interval give errors of the
  posterior covariance and mean, divided by the prior SD, of at most `2.4e-11` and
  `2.5e-13`, and of `log Z` of at most `1.2e-10` (4 ulps of `|log Z| ≈ 1.5e5`).
  Applications use a few predictive SDs (health-cognition: a 0–27 score with
  class cuts at 6.5 and 11.5).

This is the ordered-probit interval update of §5 with a free `σ` in `s` and the ends
taken from the data instead of the cutpoint parameters; the square-root update
(`kalman_update_limited`) already takes `σ` and per-observation ends. Likelihood exact
given the Gaussian predicted state; the filtered state is the moment-matched Gaussian.

### Data flow

- `process_data` reads the two columns of each update row of such a measurement into
  `interval_bounds`, float64, shape `(n_updates, n_obs, 2)`, NaN elsewhere, and
  classifies the events (`classify_observations(..., interval_bounds=...)`). The
  columns are carried through the endogenous-factor augmentation, required by
  `_check_data`, and set to NaN with the measurements when controls are missing.
- The scan receives `interval_bounds` as one more per-update input
  (`LIMITED_LOOP_KEYS`); `get_event_thresholds` returns the interval ends for the
  observations that have them and the cut-table thresholds otherwise.
- Observed-ness for the freeze after the last observation (`get_observation_masks`) is
  `events != MISSING` in models with limited measurements, so a person whose last
  report is an interval is still predicted up to it. For all other observations this
  equals the previous `isfinite(measurements)`.
- Models without interval columns pass no `interval_bounds`; their scan inputs and
  results are unchanged (all-Gaussian models run the untouched Gaussian scan).
- `interval_bounds` costs two float64 per update row and person; rows of other
  measurements hold NaN. A compact per-measurement table with a row map is the upgrade
  if memory ever binds.
- AF rejects interval columns (`NotImplementedError`); AMN rejects all non-Gaussian
  measurements; the Gaussian twin used for start values ignores the interval columns
  (interval-only persons count as missing there); `simulate_dataset` draws the censored
  value of every observation.

### Alternatives considered and rejected

- A new `IntervalMeasurement` class and family code: duplicates the Tobit handling of
  exact and censored values, and every family switch (kernel, simulation, events,
  update tables, AF/AMN guards) grows a branch. Interval censoring is the general case
  of Tobit censoring, so one optional field on `TobitMeasurement` is the smaller
  interface.
- Ordered probit with fixed known cutpoints and a free SD plus a class-code column:
  limits the intervals to one finite set per measurement and needs a second column
  anyway; per-observation ends are as simple and more general.
- The two-column `intreg` convention (equal ends = exact value) in place of the
  measurement column: breaks the rule that a measurement's name is its data column,
  which update tables, debug residuals, filtered-state masks and the Gaussian twin
  rely on.
- A finite placeholder in the measurement column for interval observations: code that
  reads the column (start values, residuals, plots) would see invented values.
