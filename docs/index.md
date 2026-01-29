# skillmodels

Welcome to skillmodels, a Python implementation of estimators for nonlinear dynamic
latent factor models. The package implements the Kalman filter-based maximum likelihood
estimator proposed by Cunha, Heckman and Schennach
([Econometrica 2010](http://onlinelibrary.wiley.com/doi/10.3982/ECTA6551/abstract)).

## Overview

Skillmodels was developed for skill formation models but can be applied to any dynamic
nonlinear latent factor model. Key features:

- **Kalman filter estimation**: Uses square-root implementations for numerical stability
- **Flexible model specification**: Define models using Python dataclasses or dictionaries
- **JAX-powered**: Automatic differentiation and JIT compilation for fast optimization
- **GPU support**: Optional CUDA acceleration

## Public API

The main package exports three functions:

- `get_maximization_inputs()`: Prepare optimization problem for parameter estimation
- `get_filtered_states()`: Extract filtered latent factor estimates
- `simulate_dataset()`: Generate synthetic data from model specification

And dataclasses for model specification:

- `ModelSpec`: Main model specification container
- `FactorSpec`: Specification for individual factors
- `AnchoringSpec`: Anchoring settings
- `EstimationOptionsSpec`: Options for estimation
- `Normalizations`: Normalization settings for loadings and intercepts

## Implementation Notes

The CHS estimator implemented here differs from the original
[replication files](https://tinyurl.com/yyuq2sa4) in two ways:

1. Uses different normalizations that account for the
   [critique](https://tinyurl.com/y3wl43kz) of Wiswall and Agostinelli
2. Uses robust square-root implementations of the Kalman filters

## Citation

If you find skillmodels helpful for research, please cite it. See the
[GitHub repository](https://github.com/OpenSourceEconomics/skillmodels) for citation
information.

## Feedback

If you encounter any problems or have suggestions, please open an issue on
[GitHub](https://github.com/OpenSourceEconomics/skillmodels/issues).
