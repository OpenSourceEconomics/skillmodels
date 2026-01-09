# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

skillmodels is a Python implementation of estimators for nonlinear dynamic latent factor
models, primarily used for skill formation research in economics. It implements Kalman
filter-based maximum likelihood estimation following Cunha, Heckman, Schennach (2010).

## Development Commands

```bash
# Run tests
pixi run tests

# Run tests with coverage
pixi run tests-with-cov

# Run a single test file
pixi run -e test-cpu pytest tests/test_kalman_filters.py

# Run a single test
pixi run -e test-cpu pytest tests/test_kalman_filters.py::test_function_name

# Type checking
pixi run ty

# Install pre-commit hooks (required before committing)
pre-commit install

# Build documentation (from docs/ directory)
make html
```

## Architecture

### Core Pipeline Flow

```
Model Dict + Data
       ↓
process_model() → Validates/extends model specification
       ↓
process_data() → Transforms data to estimation format
       ↓
get_maximization_inputs() → Creates optimization problem (likelihood, gradients, constraints)
       ↓
[optimagic optimization]
       ↓
get_filtered_states() → Extract estimated latent factors
```

### Key Modules

- **process_model.py**: Model specification validation and preprocessing. Handles
  dimensions, labels, stagemap, anchoring, and endogenous factors.
- **kalman_filters.py**: Core Kalman filter implementation (predict/update steps). Uses
  square-root form for numerical stability.
- **likelihood_function.py**: Log-likelihood computation using Kalman filtering.
  Includes soft clipping for numerical stability.
- **constraints.py**: Generates parameter constraints (bounds, equalities from stagemap,
  fixed values) for optimization.
- **parse_params.py**: Converts flat parameter vectors to structured model parameters.
- **transition_functions.py**: Pre-built transition equations (`linear`, `log_ces`,
  `constant`). Custom functions can be added.

### JAX Usage

All computation-heavy code uses JAX for automatic differentiation and JIT compilation.
The codebase uses:

- `jax.vmap` for vectorization across observations
- `jax.jit` for compilation
- JAX arrays throughout the estimation pipeline
- Optional GPU support via CUDA

### Public API

The main package exports three functions:

- `get_maximization_inputs()`: Prepare optimization problem for parameter estimation
- `get_filtered_states()`: Extract filtered latent factor estimates
- `simulate_dataset()`: Generate synthetic data from model specification

## Code Style

- Require Python 3.14
- Uses Ruff for linting (target: Python 3.14, line length: 88)
- Google-style docstrings
- Pre-commit hooks enforce formatting and linting
- Type checking via `ty` with strict rules
- Do not use `from __future__ import annotations`

## Testing

- pytest with markers: `wip`, `unit`, `integration`, `end_to_end`
- Test files mirror source structure in `tests/`
- Memory profiling available via pytest-memray (Unix only)
