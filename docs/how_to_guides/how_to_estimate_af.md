# Estimate a Model with AF (sequential Halton MLE)

The Antweiler & Freyberger (2025) estimator (`skillmodels.af.estimate_af`) fits
each period in sequence: period 0 jointly estimates the initial-period
measurement system and the latent mixture; each subsequent period takes the
estimated conditional state distribution and runs a period-specific MLE over a
joint Halton design.

## Minimal example

```python
import pandas as pd

from skillmodels import ModelSpec, FactorSpec, Normalizations
from skillmodels.af import AFEstimationOptions, estimate_af

model = ModelSpec(
    factors={
        "skill": FactorSpec(
            measurements=(("y1", "y2", "y3"),) * 3,
            normalizations=Normalizations(
                loadings=({"y1": 1},) * 3,
                intercepts=({"y1": 0},) * 3,
            ),
            transition_function="linear",
        ),
    },
)
data: pd.DataFrame  # long-format, indexed by (caseid, period)

af_options = AFEstimationOptions(
    n_halton_points=200,        # main quadrature
    n_halton_points_shock=50,   # production-shock integration
)
result = estimate_af(model, data, af_options)

result.params                  # canonical skillmodels params DataFrame
result.period_results[0]       # per-period AFPeriodResult
```

For score-bootstrap standard errors, pass the same `data` and
`af_options` to `compute_af_standard_errors`:

```python
from skillmodels.af import compute_af_standard_errors

inference = compute_af_standard_errors(
    result, data, af_options, n_boot=10_000, seed=0
)
inference.standard_errors
```

`compute_af_standard_errors` implements the propagated influence-function
score bootstrap of Antweiler & Freyberger (2025) §4.2 (after Armstrong,
Bertanha & Hong 2014). It builds a single per-observation influence matrix
once at the optimum — each period block carries the earlier periods'
estimation uncertainty through the cross-period blocks of the full-chain
Hessian — then resamples its caseid rows with one shared index per
replicate. Because the same index is used across periods, the resulting
$t \geq 1$ standard errors are consistent and the cross-period covariances
are non-zero. No per-replicate re-estimation is involved, so 10 000
replicates run in seconds. The result exposes `standard_errors`, `vcov`,
and `replicate_params`.

## Optimizer

Each period's MLE runs through `optimagic.minimize` with the algorithm in
`AFEstimationOptions.optimizer_algorithm` (default `"fides"`; pass
`"scipy_lbfgsb"` for Monte Carlo sweeps where a deterministic stopping
rule matters). The parameter vector crosses host↔device once per
iteration:

1. optimagic hands a pandas DataFrame to the user-supplied `fun` / `fun_and_jac`.
1. The wrapper extracts the `"value"` column, pushes it to device, runs the
   jitted log-likelihood, and copies the scalar + gradient back to numpy.

Pass scipy_lbfgsb stopping options through `optimizer_options`:

```python
af_options = AFEstimationOptions(
    n_halton_points=200,
    n_halton_points_shock=50,
    optimizer_algorithm="scipy_lbfgsb",
    optimizer_options={
        "algo_options": {
            "convergence_gtol_abs": 1e-5,
            "convergence_ftol_rel": 2.22e-9,
            "stopping_maxiter": 15_000,
        },
    },
)
```

All optimagic constraint kinds are supported: `FixedConstraintWithValue`
(from normalisations / `fixed_params`), `ProbabilityConstraint` (from
`log_ces` `gamma` simplex), and `EqualityConstraint` (within-step and
cross-period equalities passed through `estimate_af(constraints=...)`).

## Start-values strategy

`AFEstimationOptions.start_params_strategy` controls how the per-period
parameter templates are seeded:

- `"amn"` (default) — run the full AMN three-stage estimator upfront and use
  its parameter estimates as start values. Most accurate, slowest.
- `"spearman"` — moment-based seeds from Spearman cross-covariances and
  Bartlett-style residual variances. Fast; good enough for most diagnostics.
- `"constant"` — legacy 0.5 / data-scaled defaults; useful for regression
  testing and reproducing pre-fix results.
- `"none"` — accepted for cross-estimator symmetry; behaves identically to
  `"constant"` (AF always needs concrete per-period starts).

`compute_af_standard_errors` does not re-run the optimizer per replicate, so
the choice of `start_params_strategy` does not enter the inference path:
the score bootstrap reuses the point estimate and only resamples the
precomputed influence matrix.

## Production transition functions

For an AF production function, use the AF-specific transition functions
`translog_af` (eq. 6: linear terms + pairwise interactions, NO square terms)
or `log_ces_af` (eq. 7: CES over the production factors only). They enumerate
parameters over the production factors (skill + investment) so observed
factors such as income do not leak in as free production coefficients.

The general-library transitions (`linear`, `translog`, `robust_translog`,
`linear_and_squares`, `log_ces`, `log_ces_with_constant`, `log_ces_general`)
enumerate parameters over *all* factors, including observed ones. Using one of
these for a (non-endogenous) production factor while observed factors are
present makes income enter the production function with its own free
coefficients, which changes the AF estimand (income should affect skills only
through the investment equation). `validate_af_model` emits a `UserWarning` in
that case; either switch to `translog_af` / `log_ces_af`, or pin every
observed-factor transition coefficient to 0 via `fixed_params`.

## Anchoring and endogenous factors

Anchoring is currently only supported by the CHS path
(`get_maximization_inputs`). The AF path estimates the latent-factor scale
implied by the measurement-system normalisations; anchoring outcomes can be
added back into the model spec for downstream visualisation but do not enter
the AF likelihood.

Endogenous factors (investment in period $t$ measured by inv-measures in
period $t$) are supported. See `tests/test_af_estimate.py` for a worked
example with `is_endogenous=True`.

The AF likelihood implements only the exogenous-investment case
($\kappa_t = 0$): production and investment shocks are integrated as
independent draws. The endogenous-investment control function is not part of
the AF estimator. If the model declares a `CorrectionSpec` (via
`FactorSpec.correction`), `validate_af_model` raises `NotImplementedError`;
strip it with `ModelSpec.without_correction()` to run AF, or use
`estimate_chs` to estimate the correction. (Supplying `kappa` / `kappa_t`
parameters directly via `start_params` / `fixed_params` likewise raises
`NotImplementedError`.) See
[Endogeneity Corrections](../reference_guides/endogeneity_corrections.md) for
the full control-function interface and
[How to estimate AMN](how_to_estimate_amn.md) for the AMN route.
