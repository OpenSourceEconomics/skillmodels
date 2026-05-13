# Estimate a Model with AF (sequential Halton MLE)

The Attanasio & Freyberger 2025 estimator (`skillmodels.af.estimate_af`) fits
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
    n_mixture_components=2,
)
result = estimate_af(model, data, af_options=af_options)

result.all_params              # canonical skillmodels params DataFrame
result.period_results[0]       # per-period AFPeriodResult
```

For cluster-bootstrap standard errors, pass the same `model`, `data`, and
`af_options` to `compute_af_standard_errors`:

```python
from skillmodels.af import compute_af_standard_errors

inference = compute_af_standard_errors(
    result, data, af_options, n_boot=200, seed=0
)
inference.standard_errors
```

## Optimizer backends

By default each period's MLE runs through `optimagic.minimize`. The parameter
vector crosses host↔device once per iteration:

1. optimagic hands a pandas DataFrame to the user-supplied `fun`/`fun_and_jac`.
1. The wrapper extracts the `"value"` column, pushes it to device, runs the
   jitted log-likelihood, and copies the scalar + gradient back to numpy.

For models without probability or equality constraints, an on-device backend
can replace this. Set `optimizer_backend="jaxopt"` to run `jaxopt.LBFGSB`
directly on the device-resident parameter vector:

```python
af_options = AFEstimationOptions(
    n_halton_points=200,
    n_halton_points_shock=50,
    optimizer_backend="jaxopt",
    optimizer_options={"maxiter": 500, "tol": 1e-7, "history_size": 10},
)
```

The trade-offs:

- **Supported**: pinned values from normalisations, user-supplied
  `fixed_params`, and parameter bounds.
- **Not supported**: probability constraints (raised by `log_ces` transitions
  whose `gamma` weights live on a simplex) and equality constraints (within-
  step and cross-period equalities passed through `estimate_af(constraints=)`).
  The jaxopt backend raises `NotImplementedError` with a clear hint to fall
  back to `optimizer_backend="optimagic"`.
- The `optimizer_algorithm` field is ignored; jaxopt always uses L-BFGS-B.

When to pick which:

| Situation                                              | Backend       |
| ------------------------------------------------------ | ------------- |
| Any `log_ces` transition                               | `"optimagic"` |
| Within-step / cross-period equality constraints        | `"optimagic"` |
| Linear / translog model with many iterations on GPU    | `"jaxopt"`    |
| Small CPU run, debuggability matters                   | `"optimagic"` |

## Initialization strategy

`AFEstimationOptions.initialization_strategy` controls how the per-period
parameter templates are seeded:

- `"amn"` (default) — run the full AMN three-stage estimator upfront and use
  its parameter estimates as start values. Most accurate, slowest.
- `"spearman"` — moment-based seeds from Spearman cross-covariances and
  Bartlett-style residual variances. Fast; good enough for most diagnostics.
- `"constant"` — legacy 0.5 / data-scaled defaults; useful for regression
  testing and reproducing pre-fix results.

The bootstrap inference path internally re-runs the optimizer with
`initialization_strategy="constant"` on each replicate so that the AMN seeds
are computed only once.

## Anchoring and endogenous factors

Anchoring is currently only supported by the CHS path
(`get_maximization_inputs`). The AF path estimates the latent-factor scale
implied by the measurement-system normalisations; anchoring outcomes can be
added back into the model spec for downstream visualisation but do not enter
the AF likelihood.

Endogenous factors (investment in period $t$ measured by inv-measures in
period $t$) are supported. See `tests/test_af_estimate.py` for a worked
example with `is_endogenous=True`.
