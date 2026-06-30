# Compare CHS, AF, and AMN with Confidence Intervals

The getting-started tutorial shows the same `ModelSpec` estimated by all three
estimators on CNLSY data. This guide picks up where the tutorial leaves off and
quantifies the uncertainty around each estimator's point estimates:

1. **CHS**: analytic OPG / inverse-score standard errors from
   `estimagic.estimate_ml`.
2. **AF**: propagated influence-function score bootstrap
   (`compute_af_standard_errors`).
3. **AMN**: nonparametric cluster bootstrap (`compute_amn_standard_errors`).

The end of the guide overlays the three posterior factor trajectories on a
single panel so you can read off whether the estimators agree on the latent
factor path, not just on the parameter estimates.

The guide assumes the three estimation results from the tutorial are in scope:
`chs_result`, `af_result`, and `amn_result`. The corresponding model and data
fixtures (`model`, `data`) are the same across all three.

## Why each estimator gets a different inference

Each estimator computes the same point estimate of the same model, but the
sampling-distribution machinery differs:

| Estimator | Inference                                            | Why this and not bootstrap (CHS) / not OPG (AF, AMN) |
| --------- | ---------------------------------------------------- | --------------------------------------------------------- |
| CHS       | Analytic OPG / inverse-score (information equality)  | Closed-form is valid under correct likelihood specification; bootstrap is just slower. |
| AF        | Propagated influence-function score bootstrap        | The closed-form variance ignores estimation error in period-$t-1$ nuisance params, biasing every period-$t \geq 1$ SE down. The influence-function score bootstrap propagates that earlier-period uncertainty, so cross-period covariances are non-zero. |
| AMN       | Full re-estimation cluster bootstrap                 | The three-stage estimator has no clean sandwich form; each stage's residual variance compounds.  |

## CHS: analytic standard errors

`estimate_chs` runs CHS through `estimagic.estimate_ml`, so its result carries
ML inference directly. By default `estimate_chs` sets `hessian=False` (the
numerical Hessian costs $O(\text{n\_params}^2)$ Kalman passes), so the reported
covariance is the **OPG / inverse-score** form — the inverse of the
outer-product-of-gradients information. This is valid under correct likelihood
specification (the information-equality assumption) but is **not** the
misspecification-robust sandwich, which would also need the Hessian.

```python
chs_result.likelihood_result.summary()  # CHS via estimate_chs
chs_result.likelihood_result.se()
chs_result.likelihood_result.cov()
```

To get the Hessian-based sandwich covariance instead, override the estimagic
keyword arguments via `CHSEstimationOptions.estimate_ml_options`:

```python
from skillmodels import CHSEstimationOptions, estimate_chs

chs_result = estimate_chs(
    model,
    data,
    CHSEstimationOptions(estimate_ml_options={"hessian": True}),
)
```

If you drive the optimiser yourself via `get_maximization_inputs`, you can call
`estimagic.estimate_ml` directly on `max_inputs["loglikeobs"]` and the
`max_inputs["constraints"]`, choosing `hessian=True`/`False` to pick the
sandwich or the OPG/inverse-score covariance.

## AF: propagated influence-function score bootstrap

`compute_af_standard_errors` implements the Antweiler & Freyberger (2025)
§4.2 score bootstrap (after Armstrong, Bertanha & Hong 2014) in its
sequential-estimator influence-function form. It builds a single
per-observation influence matrix once at the optimum: each period block is a
one-step Newton update of that period's full-chain score that also carries
the earlier periods' influence via the cross-period (Hessian) blocks. For
each of `n_boot` replicates it draws *one shared* caseid index, resamples the
rows of the influence matrix with it, and shifts the estimate by the negated
resample mean. The shared index propagates earlier-period estimation
uncertainty, so the $t \geq 1$ standard errors are consistent and the
cross-period covariances are non-zero (unlike an own-block, independent-period
resample). No per-replicate re-estimation, so 10 000 replicates run in
seconds.

```python
from skillmodels.af import compute_af_standard_errors

af_inference = compute_af_standard_errors(
    af_result,
    data,
    af_options,
    n_boot=10_000,
    seed=0,
)
af_inference.standard_errors.head()
af_inference.vcov  # (n_params, n_params) DataFrame indexed by the params MultiIndex
af_inference.replicate_params  # (n_boot, n_params)
```

The `replicate_params` DataFrame is the right object for plotting 95%
intervals: take the 2.5%/97.5% empirical quantiles per parameter rather than
$\hat{\theta} \pm 1.96 \cdot \mathrm{SE}$, since the one-step shifts can be
visibly skewed.

## AMN: cluster bootstrap

AMN's three-stage pipeline (EM → minimum distance → simulate-and-regress) has
no analytic sandwich, so inference is a full cluster bootstrap: resample
caseids with replacement, re-run all three stages, repeat. Per-replicate cost
is dominated by the Stage 1 EM (~seconds for $n \approx 2000$, $K = 2$,
$\approx 40$ augmented measures). Each replicate draws a fresh per-replicate
seed (so the Stage-1 EM initialisation and the Stage-3 simulation vary across
replicates), and any replicate that fails to converge is excluded from the
bootstrap distribution and reported via a `RuntimeWarning`; its row in
`replicate_params` is `NaN`.

```python
from skillmodels.amn import compute_amn_standard_errors

amn_inference = compute_amn_standard_errors(
    amn_result,
    data,
    amn_options,
    n_boot=200,
    seed=0,
)
amn_inference.standard_errors.head()
amn_inference.replicate_params  # (n_boot, n_params) -- includes failed replicates as NaN rows
```

Bumping `n_boot` to 1000 is reasonable on a multi-core machine; the paper's
original AMN application uses 100.

## Overlaying CES production-function CIs

Side-by-side $\phi$ estimates with 95% CIs:

```python
import pandas as pd

def _ci(replicate_params, param_loc, q=0.025):
    samples = replicate_params[param_loc].dropna()
    return samples.quantile(q), samples.quantile(1 - q)

rows = []
for period in (0, 1):
    phi_loc = ("transition", period, "skills", "phi")
    rows.append({
        "period": period,
        "estimator": "CHS",
        "estimate": chs_result.params.loc[phi_loc, "value"],
        "lower": chs_result.likelihood_result.summary().loc[phi_loc, "ci_lower"],
        "upper": chs_result.likelihood_result.summary().loc[phi_loc, "ci_upper"],
    })
    rows.append({
        "period": period,
        "estimator": "AF",
        "estimate": af_result.params.loc[phi_loc, "value"],
        "lower": _ci(af_inference.replicate_params, phi_loc)[0],
        "upper": _ci(af_inference.replicate_params, phi_loc)[1],
    })
    rows.append({
        "period": period,
        "estimator": "AMN",
        "estimate": amn_result.params.loc[phi_loc, "value"],
        "lower": _ci(amn_inference.replicate_params, phi_loc)[0],
        "upper": _ci(amn_inference.replicate_params, phi_loc)[1],
    })

phi_comparison = pd.DataFrame(rows)
```

## Posterior factor trajectories

The three estimators produce different posterior beliefs about the latent
factor paths. `chs_states`, `af_states`, `amn_states` (built in the tutorial
via `get_individual_states`, `get_af_posterior_states`,
`get_amn_posterior_states`) all share a `period` column and one column per
factor, so a single melt + facet plot covers the comparison:

```python
import plotly.express as px

states = pd.concat(
    [
        chs_states.assign(estimator="CHS"),
        af_states.assign(estimator="AF"),
        amn_states.assign(estimator="AMN"),
    ]
)
trajectories = states.groupby(["estimator", "period"])["skills"].mean().reset_index()

fig = px.line(
    trajectories,
    x="period",
    y="skills",
    color="estimator",
    title="Mean posterior skill across estimators",
    template="plotly_white",
)
fig.show()
```

For a stronger visual comparison, plot the cross-individual variance band
($q_{0.1}$, $q_{0.5}$, $q_{0.9}$) per estimator side-by-side; agreement on the
median path with disagreement on the band is a useful diagnostic about how
the estimator treats the tail of the latent distribution.

## When the estimators disagree

If CHS, AF, and AMN disagree by more than the bootstrap CIs predict, the
candidate explanations are:

- **Non-Gaussian latent factors.** CHS assumes Gaussian-mixture latents; AF and
  AMN are more flexible about the mixture. Run `decompose_measurement_variance`
  on each (the tutorial does this) and check whether the signal fractions
  diverge — that's the leading indicator.
- **Misspecified transition function.** `log_ces` enforces a CES form via the
  simplex constraint on the $\gamma$ weights; if the data prefers a linear
  technology with a free constant, the CHS optimum can land in a different
  basin than the AF/AMN sequential estimates that escape the constraint via
  their integration weights.
- **Endogenous investment misalignment.** If `investment` is meant to be
  endogenous (`is_endogenous=True`), CHS uses augmented periods internally
  while AF treats it as a regular state per calendar period. The two answers
  should still agree, but the augmented-period plumbing has historically been
  the source of subtle bugs — start the diagnosis here if the disagreement is
  concentrated around investment.

See [How to estimate AF](how_to_estimate_af.md) and
[How to estimate AMN](how_to_estimate_amn.md) for the estimator-specific tuning
that matters when the headline disagreement turns out to be numerical, not
substantive.
