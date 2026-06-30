# Names and Concepts

This section explains key concepts and variable names used throughout skillmodels.
Understanding these is helpful if you want to understand the implementation or extend
the package.

## Dimensions

The `Dimensions` dataclass contains integer values for model dimensions:

- **n_latent_factors**: Number of latent factors (states) in the model
- **n_observed_factors**: Number of observed factors
- **n_periods**: Number of periods (one more than transition equations)
- **n_aug_periods**: Number of augmented periods (includes sub-periods for endogenous
  factors)
- **n_mixtures**: Number of elements in the finite mixture of normals distribution
- **n_controls**: Number of control variables (always >= 1 due to constant)

## Labels

The `Labels` dataclass contains names and indices:

- **latent_factors**: Tuple of latent factor names
- **observed_factors**: Tuple of observed factor names
- **controls**: Tuple of control variable names (first is always "constant")
- **periods**: Tuple of period indices (0, 1, 2, ...)
- **aug_periods**: Tuple of augmented period indices
- **stagemap**: Tuple mapping periods to stages
- **stages**: Tuple of stage indices

## Development Stages vs Periods

A **development stage** is a group of consecutive periods where the skill formation
technology (transition function parameters) remains constant. Stages are just equality
constraints on parameters.

Example: With 5 periods, you can estimate at most 4 different transition functions.
The stagemap `[0, 0, 1, 1]` means:
- Periods 0→1 and 1→2 share the same parameters (stage 0)
- Periods 2→3 and 3→4 share the same parameters (stage 1)

## Augmented Periods

When models include endogenous factors (factors that depend on other factors in the
same period), skillmodels internally expands periods into "augmented periods" to handle
the sequential updating. Each regular period may contain multiple augmented periods.

## Anchoring

Anchoring links latent factors to observable outcomes, allowing identification and
interpretation of the factor scale. The `Anchoring` dataclass contains:

- **outcomes**: Which factors are anchored to which outcome variables
- **free_controls**: Whether anchoring equations have their own control coefficients
- **free_constant**: Whether anchoring equations have a free constant
- **free_loadings**: Whether anchoring loadings are estimated (vs fixed to 1)
- **ignore_constant_when_anchoring**: Skip constant in anchoring transformation

## Update Info

A DataFrame specifying each Kalman update step:

- One row per measurement equation evaluation
- Columns indicate which factors have free loadings for each measurement
- Used internally to structure the Kalman filter passes

## Normalizations

Settings for identifying the model scale and location:

- **loadings**: Fixed factor loading values (cannot be zero)
- **intercepts**: Fixed intercept values for measurement equations

Without normalizations, latent factor models are not identified (the scale and location
of factors are arbitrary).

## Estimation Options

Each estimator has its own options dataclass, passed at call time rather than
embedded in `ModelSpec`. The three classes share no fields — what counts as a
tuning knob differs between estimators.

`CHSEstimationOptions` (from `skillmodels.chs`) controls the Kalman MLE:

- **robust_bounds**: Tightens parameter bounds to avoid numerical issues
- **bounds_distance**: How much stricter to make bounds (zeroed if robust_bounds is
  false)
- **sigma_points_scale**: Controls spread of sigma points in unscented Kalman filter
- **clipping_\***: Parameters for soft-clipping the log-likelihood to prevent
  infinities
- **start_params_strategy**: How to seed the `params_template`. `"amn"` (default)
  runs the full AMN three-stage estimator and uses its parameters as the start;
  `"spearman"` uses moment-based start values; `"none"` leaves entries as NaN
  for the caller to fill in.

`AFEstimationOptions` (from `skillmodels.af`) controls the sequential MLE:

- **n_halton_points**, **n_halton_points_shock**: quadrature counts.
- **n_mixture_components**: number of components in the latent-factor mixture.
- **optimizer_algorithm**: the optimagic algorithm name passed to
  `optimagic.minimize(algorithm=...)` (default `"fides"`; use
  `"scipy_lbfgsb"` for MC sweeps).
- **initialization_strategy**: `"amn"`, `"spearman"`, or `"constant"`. Same
  meaning as in CHS.

`AMNEstimationOptions` (from `skillmodels.amn`) controls the three-stage
pipeline:

- **n_mixture_components**: Stage-1 EM components.
- **em_max_iter**, **em_tol**, **em_n_init**, **em_reg_covar**: Stage-1 EM
  numerical knobs.
- **n_simulation_draws**: Stage-3 synthetic-panel size.
- **minimum_distance_weighting**: Stage-2 weighting. `"identity"` (default) is
  the paper's unweighted identity-metric criterion over per-component means and
  full covariance matrices, and is currently the only implemented option.
  `"optimal"` is reserved for a future Avar-weighted criterion and raises
  `NotImplementedError`.
- **investment_endogeneity**: apply the AMN (2020) eq. 7-8 / AF Sec. 3.5
  investment control-function correction in Stage 3. Defaults to `False`. When
  `True` and the model has an endogenous (investment) factor, a first-stage
  investment equation is OLS-fit per period and its residual is added as an
  additive `cf` covariate (coefficient `kappa_t`) to each state factor's
  production regression; observed factors are then excluded from the production
  function and act as instruments (at least one observed instrument is
  required). The default stays `False` because `estimate_af` calls
  `estimate_amn` for start values and the AF likelihood implements only
  `kappa=0`; opt into the correction at the application call site. A no-op for
  models without endogenous factors.

The shared structural field — number of mixture components in the latent
distribution — lives directly on `ModelSpec.n_mixtures`, since it changes the
model itself rather than the optimizer.
