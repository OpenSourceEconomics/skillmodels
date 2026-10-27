# Binary (probit) measurements in the CHS estimator — design brief

Status: design request. No production code changed yet. Base: branch
`probit-measurements` on top of the open PR chain

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

@pro: Verify the formulas in §3 (likelihood, posterior mean and covariance, the κ bounds and the pseudo-measurement equivalence including the residual), give a numerically stable evaluation of r = φ(t)/Φ(t) and r(r + t) for t → −∞ and t → +∞ in float64 with JAX (values and first derivatives), and state whether reverse-mode AD through the QR with σ̃ = (1/κ − λᵀPλ)^{1/2} is stable or whether a direct rank-one Cholesky downdate is preferable.

@pro: The update is an assumed-density approximation of the exact filter. State precisely what is exact (the per-update likelihood given the Gaussian predicted state) and what is approximate, how this compares to the approximation CHS already makes in the unscented predict step, and give a test that bounds the approximation error of the full likelihood on a model small enough for exact integration (e.g. one latent factor, linear transition, three periods, quadrature over the latent path).

## 4. What has to be decided

@pro: Lay out the exact implementation path on top of #104 as an ordered list of commits, each with the files and functions it touches, the tests written first (red) and what they assert, and the acceptance criterion. Cover at least: family information in `ProcessedModel`/`update_info`; the parameter index and parsing for probit measures (drop `meas_sds` from the index, or keep and fix it — which interacts less with optimagic constraints, `params_template`, `fixed_constraint.py` and downstream code that indexes `meas_sds`); dispatch inside the `lax.scan` (per-iteration flag with `lax.cond`, or a branch-free pseudo-measurement computed with `jnp.where` on per-update scalars — which keeps compile time and gradients clean); `kalman_update` and the anchoring update; mixtures; the debug path and `process_debug_data` (what are "residual" and "residual SD" for a probit update); `get_filtered_states`; `decompose_measurement_variance`; diagnostic plots; the error in `maximization_inputs.py` (lift it for probit only, keep it for Tobit); AMN seeding (AMN rejects probit — seed from a Gaussian twin of the model, or exclude probit measures from the AMN moment map).

@pro: Identification and normalizations: with ε's SD fixed at 1, the probit intercept and loadings are on the latent-index scale. State the restrictions this imposes: a probit measure cannot be a factor's normalizing measurement; what changes for time-varying loadings, constraints that tie a probit measure's parameters across periods, and anchoring. Say whether any existing constraint generator in `common/constraints.py` produces invalid constraints when a measure is probit.

@pro: Interplay with the open PRs: #104's observation masks (does a probit observation count as an observed measurement for the freeze — it should); #103's handling of missing observed factors (do observed factors ever enter a measurement update, so that the pseudo-measurement has to account for their variance); #102's asymmetric translog (no expected interaction — confirm). Name any place where the chain's code assumes Gaussian measurements.

@pro: Test plan, written as tests: independent oracles for one update (Gauss–Hermite or adaptive quadrature over θ for the likelihood and posterior moments, any number of factors up to 3, both y, extreme t), the pseudo-measurement equivalence, mixtures, missing values, AD gradients vs. finite differences, JIT vs. eager, a synthetic DGP from `simulate_dataset` with a probit measure whose parameters are recovered, and the small-model full-likelihood bound from §3. Give tolerances.

@pro: Cost: the extra operations per update and whether compile time or memory change measurably for a model like the application (16 periods, about 50 measurements per period, 5 latent states plus 3 investments, 17,776 persons, one A100, float64). Name any choice that would make the Gaussian path slower.

@pro: Out of scope unless it falls out for free: ordered probit, Tobit in CHS, probit in AMN. Say whether the family dispatch should be designed so that these can be added later without touching the probit code, but do not design them.
