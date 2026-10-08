# Estimate a Model with AMN (three-stage mixture-of-normals)

The Attanasio-Meghir-Nix 2020 estimator (`skillmodels.amn.estimate_amn`) runs
three stages:

1. **Mixture EM.** Fit a Gaussian mixture
   $F_{M, X} = \sum_k \tau_k \, \mathcal{N}(\Pi_k, \Psi_k)$ to the augmented
   measure vector $[M_{1:T}, X]$ (controls / instruments stacked alongside
   measurements as zero-error rows).
2. **Minimum distance.** Recover structural parameters
   $(\Lambda, A, \Sigma, \mu, \Omega)$ from the reduced-form
   $(\Pi_k, \Psi_k)$ subject to factor-measurement assignment, scale
   normalisations, and the mean-zero mixture restriction.
3. **Simulate and regress.** Draw a large synthetic latent-factor panel
   from the fitted mixture and estimate the production function by
   regression on the synthetic data.

AMN shines when the latent factor distribution is non-Gaussian. CHS supports a
finite mixture of Gaussian initial states via `ModelSpec.n_mixtures`, but filters
each component with a Gaussian (square-root Kalman) recursion; setting
`n_mixtures=1` is the deliberately restricted single-Gaussian benchmark. AF also
supports multiple mixture components but fits them jointly with the
period-specific optimizer. AMN cleanly separates the mixture (Stage-1 EM) from
the structural recovery, so it models the non-Gaussianity explicitly.

## Minimal example

```python
import pandas as pd

from skillmodels import ModelSpec, FactorSpec, Normalizations
from skillmodels.amn import AMNEstimationOptions, estimate_amn

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
    n_mixtures=2,
)
data: pd.DataFrame  # long-format, indexed by (caseid, period)

amn_options = AMNEstimationOptions(
    em_max_iter=500,
    n_simulation_draws=100_000,
    seed=0,
)
result = estimate_amn(model, data, amn_options)

result.params                  # canonical skillmodels params DataFrame
result.stages.mixture          # Stage 1: reduced-form Pi, Psi, tau
result.stages.structural       # Stage 2: Lambda, A, Sigma, mu, Omega
result.stages.production       # Stage 3: production-function regression
result.success                 # AND across stage convergence flags
```

## When AMN beats CHS: a synthetic 2-mixture DGP

The smallest example that lets AMN's non-Gaussian fit show its advantage is a
1-factor / 3-period model where the latent skill is drawn from a non-trivial
mixture-of-normals. CHS with a single Gaussian component (`n_mixtures=1`)
produces biased production-function estimates on this DGP; AMN's Stage 1 EM
recovers the
mixture and the structural step undoes the bias.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 4000
n_periods = 3

# Two-mixture latent factor: 60% drawn from N(-0.8, 0.7^2), 40% from N(1.2, 0.4^2).
mixture_component = rng.choice([0, 1], size=n, p=[0.6, 0.4])
f0 = np.where(
    mixture_component == 0,
    rng.normal(-0.8, 0.7, size=n),
    rng.normal(1.2, 0.4, size=n),
)

# Linear transition with a known slope and small shock.
factors = [f0]
slope = 0.7
shock_sd = 0.15
for _ in range(n_periods - 1):
    factors.append(slope * factors[-1] + rng.normal(0, shock_sd, size=n))

# Three noisy measurements per period; "y1" is the reference (loading = 1).
rows = []
for caseid in range(n):
    for period in range(n_periods):
        f = factors[period][caseid]
        rows.append(
            {
                "caseid": caseid,
                "period": period,
                "y1": f + rng.normal(0, 0.30, size=1)[0],
                "y2": 0.9 * f + rng.normal(0, 0.35, size=1)[0],
                "y3": 1.1 * f + rng.normal(0, 0.40, size=1)[0],
            }
        )
data = pd.DataFrame(rows).set_index(["caseid", "period"])
```

Estimate AMN with two mixture components on this DGP:

```python
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
    n_mixtures=2,
)

result = estimate_amn(
    model,
    data,
    AMNEstimationOptions(
        em_max_iter=500,
        n_simulation_draws=50_000,
        seed=0,
    ),
)

result.params.loc[("transition", 0, "skill", "skill"), "value"]
# Should be close to 0.7 (the true slope).

result.stages.mixture.weights      # tau, should be near (0.6, 0.4) up to label switching
result.stages.mixture.means        # Pi_k for the augmented measure vector
```

Compare against a CHS fit of the same model with `n_mixtures=1` (the deliberately
restricted single-Gaussian benchmark) and verify that the slope estimate from CHS
is biased downward — that's the signal AMN was designed to capture.

## Tuning knobs

### Number of mixture components

`ModelSpec.n_mixtures` (not an `AMNEstimationOptions` field) controls the
flexibility of the Stage-1 EM fit; AMN reads it as the number of mixture
components $K$. The paper fixes $K = 2$; in practice values from 2 to 4 are
reasonable. Higher $K$ adds free parameters to the reduced-form fit but does
not change the structural model — the minimum-distance step constrains them.

### Stage-1 EM stability

Stage 1 uses `sklearn.mixture.GaussianMixture` under the hood. The defaults
(`em_n_init=5`, `em_reg_covar=1e-6`) reliably converge on well-identified
models; if the EM warns about degenerate covariances, bump `em_reg_covar` to
`1e-4` first. The fit is initialised from a Spearman-moment guess for the
loadings, then projected back to the augmented-measure space; that
data-driven start beats random init by a wide margin.

### Stage-1 missing data

`mixture_em_method` selects how Stage 1 handles incomplete measurement rows:

- `"complete_case"` (default) fits `sklearn.mixture.GaussianMixture` on
  listwise-complete rows and raises `InsufficientCompleteCasesError` when fewer
  than `n_mixtures` rows are complete.
- `"missing_data"` fits an EM that marginalises over each row's missing entries,
  valid under an ignorable (MAR) missingness assumption even when no row is
  complete.

A column that is never observed in any sampled row has unidentified moments; the
missing-data EM raises unless `allow_never_observed_measurements=True` is set
(intended only for seeding-style uses where such a column is tolerated).
`mixture_em_max_rows` optionally subsamples the EM input for speed.

### Stage-2 weighting

`minimum_distance_weighting="identity"` (the paper's default, and currently the
only implemented option) is fast and robust: it is an unweighted identity-metric
criterion over per-component means and the full covariance matrices. The
`"optimal"` value is reserved for a future Avar-weighted criterion and currently
raises `NotImplementedError`.

### Stage-3 simulation size

`n_simulation_draws` controls Monte-Carlo error in the production-function
regression. The paper notes "the larger the data we draw the lower the
simulation error" (p. 2522); 100 000 is overkill for $n \approx 2000$. Drop
to 50 000 for iterating, then bump back to 100 000 for the final fit. The
RNG is fully reproducible via `seed`.

## Inference

Inference is a cluster bootstrap that re-runs all three stages on each
replicate. Each replicate draws a fresh seed, so the Stage-1 EM initialisation
and Stage-3 simulation vary across replicates; replicates that fail to converge
are excluded from the distribution and reported via a warning. Wall-clock is
dominated by Stage 1 EM ($\approx$ seconds for $n \approx 2000$), so 1000
replicates run in $\approx$ 10-30 minutes on a single machine.

```python
from skillmodels.amn import compute_amn_standard_errors

inference = compute_amn_standard_errors(
    result, data, amn_options, n_boot=1000, seed=0
)
inference.standard_errors
inference.replicate_params  # (n_boot, n_params); failed replicates are NaN
```

The paper itself uses 100 replicates (Tables 5-6); 1000 gives smoother CIs
without changing the qualitative picture.

## Endogenous investment (control-function correction)

The AMN (2020) eq. 7-8 / AF Sec. 3.5 control-function correction is configured
declaratively, not via an estimation option: attach a `CorrectionSpec` to the
endogenous investment `FactorSpec` (which must set `is_endogenous=True`). Its
mere presence triggers the correction in Stage 3 — there is no
`investment_endogeneity` flag. See
[Endogeneity Corrections](../reference_guides/endogeneity_corrections.md) for
the full interface; the same `CorrectionSpec` is also read by `estimate_chs`.

```python
from skillmodels import CorrectionSpec

model = base_model.with_correction(
    "investment",
    CorrectionSpec(
        state_predictors=("skills",),
        instruments=("income",),
        targets=("skills",),
    ),
)
result = estimate_amn(model, data, amn_options)
```

Per period, a first-stage investment equation `ln I_t ~ theta_t (+ observed
instruments Y_t)` is OLS-fit on the simulated panel; its residual
`eta_{I,t} = ln I_t - E[ln I_t | theta_t, Y_t]` is added as an additive `cf`
covariate (coefficient `kappa_t`, period- and output-specific) to each target
factor's production regression. Under the correction:

- the instruments act as excluded regressors: they enter the first-stage
  equation only, never a target's production function;
- at least one instrument is REQUIRED — `CorrectionSpec.__post_init__` raises
  `ValueError` otherwise, because the residual would be collinear with the
  production inputs and `kappa` would be unidentified;
- the first-stage coefficients and shock SD are returned under the
  `investment_eq` / `investment_sds` categories on
  `result.stages.production.investment_params`, and the production shock SD
  (`shock_sds`) is the corrected SD(eps_C);
- AMN implements only the **linear** `cf` term. A higher-order `kappa_terms`
  request (anything other than `("cf",)` per target, e.g. a `kappa_degree=2`
  translog basis) raises `NotImplementedError` in `simulate_and_regress`; use
  `estimate_chs` for the full polynomial basis.

## What AMN does not (yet) do

- **Anchoring** is not wired through the AMN stages. The model spec's
  `AnchoringSpec` is accepted (so the spec stays compatible with CHS), but
  the AMN result reports unanchored factor scales.
- **`start_params` and `constraints`.** `estimate_amn` does not honour either
  and raises `NotImplementedError` if you pass them — the three-stage pipeline
  has no single parameter vector to seed or constrain.
- **`fixed_params`** are honoured only for the categories each stage estimates,
  and pinned inside that stage rather than post-hoc: Stage 2 honours `loadings`,
  measurement intercepts, and measurement SDs; Stage 3 honours `transition`.
  Passing any other category raises `NotImplementedError`. (The generic Stage-3
  NLS path supports pinning for most transitions, but `log_ces` /
  `log_ces_with_constant` reject a non-empty fixed set.)

See [How to compare estimators](how_to_compare_estimators.md) for an
overlay of CHS, AF, and AMN on the same data with confidence intervals.
