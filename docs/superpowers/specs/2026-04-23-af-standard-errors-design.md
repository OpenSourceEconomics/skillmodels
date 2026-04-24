# Standard errors for the AF estimator

## Problem

`estimate_af` returns point estimates only. The AF estimator is a sequential
M-estimator (period-by-period MLE, where each period conditions on previously
estimated parameters through a plug-in `prev_distribution`). We need an
asymptotic covariance estimator that propagates estimation uncertainty from
earlier periods into later-period standard errors.

The AF paper (Antweiler-Freyberger 2025) suggests a score bootstrap. The
companion MATLAB code does not actually ship a bootstrap routine, and its
reported SEs come from Monte-Carlo across simulations, not within-sample.
We implement the econometrically-equivalent closed-form sandwich that the
score bootstrap approximates — Newey-McFadden (1994, §6.2) for sequential
M-estimators.

## Target formula

Let `theta = (theta_0, theta_1, ..., theta_{T-1})` be the stacked parameter
vector, and let `g_{ti}(theta) = d log L_{it} / d theta_t` be individual
`i`'s period-`t` own-parameter score. Stack per-individual scores into
`g_i(theta) in R^{P_total}`. Then

- `Omega = (1/n) sum_i g_i g_i^T` — outer product of stacked scores
  (captures within-individual correlation across periods)
- `A_{ts} = (1/n) sum_i d g_{ti} / d theta_s` for `s <= t`, `0` for `s > t`
  (block lower triangular)
- `V_hat = A^{-1} Omega A^{-T} / n`
- `SE(theta_k) = sqrt(V_hat[k, k])`

This is the standard sandwich for a sequential two-step estimator. The key
observation is that period `t`'s likelihood depends on `theta_s` (s<t) both
directly (through measurement/transition parameters from the previous
period, reused as fixed inputs) and indirectly (through the
`prev_distribution` object — conditional mixture means and Cholesky
factors that were estimated at period `s`). Both channels contribute to
`d g_{ti} / d theta_s` and must be retained under autodiff.

## Computation plan

For each period `t`:

1. Build a JAX-differentiable function
   `period_t_loglike_per_obs(free_params) -> Array of shape (n_obs,)`
   that runs the same per-observation likelihood used during estimation
   but as a pure function of the free-parameter vector.
2. For `t >= 1`, inside this function, re-derive `prev_distribution`
   from the subset of `free_params` belonging to period `< t`, by
   replaying the deterministic chain
   `initial params -> cond_dist_0 -> transition params_1 + data ->
   cond_dist_1 -> ... -> cond_dist_{t-1}`.
3. Compute `S_t = jax.jacrev(period_t_loglike_per_obs)(free_params_hat)`,
   a dense `(n_obs, P_free)` matrix. Columns corresponding to `theta_{>t}`
   are zero by construction but we keep the dense matrix to simplify
   indexing.
4. Assemble per-individual stacked score `G in R^{n x P_free}`:
   for each `t`, `G[:, idx_t] = S_t[:, idx_t]` (own-period block only).
   Then `Omega = G^T G / n`.
5. Assemble `A`: for each `t`, the `t`-th row-block of `A` equals
   `jax.jacfwd(lambda p: jax.vmap(grad_own)(...))` — or equivalently the
   Hessian-by-free-params of the mean own-period loglike. Row-block `t`
   has shape `(P_t, P_free)` with zeros for `theta_{>t}`.
6. Solve `V = solve(A, Omega) @ inv(A).T / n` (use `jax.scipy.linalg.solve`
   twice to avoid explicit inverse when possible; since `A` is square
   `P_free x P_free`, a direct `inv(A)` is acceptable for the parameter
   counts we deal with — typically ~50-200).
7. Map `V` and `SE = sqrt(diag(V))` back onto the full params MultiIndex.
   Fixed parameters (pinned via `FixedConstraint`) receive `SE = 0` and
   zero rows/cols in `vcov`.

## API

Add a module `skillmodels/af/inference.py` exposing:

```python
@dataclass(frozen=True)
class AFInferenceResult:
    standard_errors: pd.Series
    """SE for every entry in all_params (fixed entries = 0)."""

    vcov: pd.DataFrame
    """Full variance-covariance matrix, indexed both rows and cols
    matching all_params.index. Fixed rows/cols are zero."""

    stacked_scores: jax.Array
    """Per-individual stacked score matrix, shape (n_obs, P_free).
    Retained so users can compute score-based tests without re-running."""

    information_matrix_A: jax.Array
    """Block-lower-triangular A matrix, shape (P_free, P_free)."""

    score_outer_product_Omega: jax.Array
    """Omega = G.T @ G / n, shape (P_free, P_free)."""


def compute_af_standard_errors(
    result: AFEstimationResult,
    data: pd.DataFrame,
    af_options: AFEstimationOptions | None = None,
) -> AFInferenceResult: ...
```

No change to `estimate_af`. Standard errors are opt-in and computed after
the fact.

## Scope

### Phase 1 (this PR): Block-diagonal sandwich

Ship the block-diagonal version of the sequential sandwich:

- For each period `t` independently, compute
  `V_t = A_tt^{-1} Omega_tt A_tt^{-T} / n` using the own-period scores
  and own-period Hessian. This is the Newey-McFadden formula restricted
  to its diagonal blocks.
- Correct handling of `fixed_params` (zero SE, zero covariance rows).
- Return the per-period Jacobian matrices (`S_t`, one per period, with
  columns only for `theta_t`). These are exactly the raw ingredients for
  the block-diagonal version; Phase 2 only adds cross-period Jacobian
  columns to them. No wasted work.
- Prominently document that period-`t` SEs for `t >= 1` are a lower
  bound on the true asymptotic SE, because they do not propagate
  plug-in uncertainty from `theta_{<t}`.

### Phase 2 (follow-up): Full cross-period chain

Required to get asymptotically-correct SEs for later periods. Needs:

- JAX-pure reconstruction of `prev_distribution` from `flat_params` via
  a differentiable chain: `theta_0 -> cond_dist_0 -> ... -> cond_dist_{t-1}`.
  This means mirroring `_extract_conditional_distribution` and
  `_update_conditional_distribution` as pure functions of flat arrays
  (no pandas `.loc` access).
- JAX-pure reconstruction of `prev_meas_info` (control params, loadings,
  SDs from period `t-1`).
- Cross-period score columns in `S_t` (non-zero for `theta_s`, `s < t`)
  and cross-period Hessian columns in `A_tt` row-block.

### Out (not planned)

- Armstrong-Bertanha-Hong style score-bootstrap — same asymptotics,
  heavier machinery.
- Anchored / delta-method SEs for transformed quantities — straightforward
  once `vcov` is available, left as follow-up.
- Unbalanced panel — current implementation assumes each period has the
  same number of observations, aligned by individual. Extend to NaN masking
  if needed.

## Verification

- Unit tests on shapes and structure (SE length matches params,
  fixed-param entries are exactly zero, `vcov` is symmetric PSD up to
  floating point).
- Integration: simulate a linear DGP with known parameters, fit, compute
  SEs; verify that as `n` doubles, SEs shrink by roughly `sqrt(2)`.
- Cross-check: on a model with no `prev_distribution` dependence
  (period 0 only, or identity transitions that strip the chain), the
  sequential sandwich should reduce to the standard single-step sandwich.
