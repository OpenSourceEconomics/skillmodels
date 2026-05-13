# skillmodels

Skillmodels is a Python toolbox for estimating nonlinear dynamic latent factor
models. It started as a Kalman-filter implementation of Cunha, Heckman & Schennach
([Econometrica 2010](http://onlinelibrary.wiley.com/doi/10.3982/ECTA6551/abstract))
and has since grown to host three estimators side by side, all sharing the same
`ModelSpec` and the same parameter index.

## Overview

Skillmodels was developed for skill-formation research but works for any dynamic
nonlinear latent-factor model. Key features:

- **Three estimators with one model spec**:
  - `chs` — Kalman MLE (CHS 2010), the historical core.
  - `af` — sequential Halton-quadrature MLE (Attanasio & Freyberger 2025),
    period-by-period.
  - `amn` — three-stage mixture-of-normals (Attanasio, Meghir & Nix 2020):
    EM, minimum distance, simulated regression.
- **Strongly-typed, immutable model spec**: frozen dataclasses with
  `MappingProxyType` containers throughout.
- **JAX everywhere**: jitted likelihoods, autodiff gradients, optional GPU.
- **Optional on-device optimizer for AF**: `optimizer_backend="jaxopt"` runs
  `jaxopt.LBFGSB` on the device-resident params vector, eliminating the
  host↔device transfer that `optimagic` incurs once per likelihood call.

## Public API

The top-level `skillmodels` package re-exports the four model-spec dataclasses
that every estimator consumes:

- `ModelSpec`
- `FactorSpec`
- `AnchoringSpec`
- `Normalizations`

Estimator-specific entry points live in their own subpackages so the scope of
each call is explicit at the import site:

```python
from skillmodels.chs import (
    CHSEstimationOptions,
    get_maximization_inputs,   # likelihood + gradients + constraints for optimagic
    get_filtered_states,
)
from skillmodels.af import (
    AFEstimationOptions,
    estimate_af,
    compute_af_standard_errors,
)
from skillmodels.amn import (
    AMNEstimationOptions,
    estimate_amn,
    compute_amn_standard_errors,
)
```

Estimator-agnostic helpers live under `skillmodels.common`:

```python
from skillmodels.common.simulate_data import simulate_dataset, simulate_policy_effect
from skillmodels.common.variance_decomposition import (
    decompose_measurement_variance,
    summarize_measurement_reliability,
)
from skillmodels.common.diagnostic_plots import (
    plot_likelihood_contributions,
    plot_residual_boxplots,
)
from skillmodels.common.state_ranges import create_state_ranges
```

The estimator-agnostic diagnostic and variance-decomposition helpers take
pre-computed DataFrames (`residuals`, `contributions`, `filtered_states`); the
caller produces them via the estimator they ran. See the how-to guides for
worked examples.

## Implementation Notes

The CHS estimator differs from the original
[replication files](https://tinyurl.com/yyuq2sa4) in two ways:

1. Uses normalizations that account for the
   [critique](https://tinyurl.com/y3wl43kz) of Wiswall and Agostinelli.
2. Uses robust square-root implementations of the Kalman filters.

The AF and AMN estimators are independent rewrites of the algorithms in their
respective papers and share only the `ModelSpec` and parameter-index machinery
with CHS; they do not call the Kalman filter.

## Citation

If you find skillmodels helpful for research, please cite it. See the
[GitHub repository](https://github.com/OpenSourceEconomics/skillmodels) for
citation information.

## Feedback

If you hit a problem or have a suggestion, please open an issue on
[GitHub](https://github.com/OpenSourceEconomics/skillmodels/issues).
