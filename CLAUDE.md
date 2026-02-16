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
pixi run -e test-cpu tests

# Run tests with coverage
pixi run -e test-cpu tests-with-cov

# Run a single test file
pixi run -e test-cpu pytest tests/test_kalman_filters.py

# Run a single test
pixi run -e test-cpu pytest tests/test_kalman_filters.py::test_function_name

# Type checking
pixi run ty

# Quality checks (linting, formatting)
prek run --all-files

# Build documentation (mystmd, from docs/ directory)
myst build
```

## Command Rules

Always use these command mappings:

- **Python**: Use `pixi run python` instead of `python` or `python3`
- **Type checker**: Use `pixi run ty` instead of running ty/mypy/pyright directly
- **Tests**: Use `pixi run -e test-cpu tests` instead of `pytest` directly
- **Linting/formatting**: Use `prek run --all-files` instead of `ruff` directly
- **All quality checks**: Use `prek run --all-files`

Before finishing any task that modifies code, always run:

1. `pixi run ty` (type checker)
1. `pixi run -e test-cpu tests` (tests)
1. `prek run --all-files` (quality checks)

## Architecture

### Core Pipeline Flow

```
ModelSpec + Data
       ↓
process_model() → Validates/extends model specification → ProcessedModel
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

- **model_spec.py**: User-facing frozen dataclasses for model specification
  (`ModelSpec`, `FactorSpec`, `AnchoringSpec`, `EstimationOptions`, `Normalizations`).
- **types.py**: Internal frozen dataclasses (`ProcessedModel`, `Labels`, `Dimensions`,
  `Anchoring`, `ParsingInfo`, `ParsedParams`, etc.) and immutability utilities.
- **process_model.py**: Model specification validation and preprocessing. Converts
  `ModelSpec` into `ProcessedModel`.
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

The main package exports model specification classes and core functions:

- `ModelSpec`, `FactorSpec`, `AnchoringSpec`, `EstimationOptions`, `Normalizations`:
  Frozen dataclasses for defining models
- `get_maximization_inputs()`: Prepare optimization problem for parameter estimation
- `get_filtered_states()`: Extract filtered latent factor estimates
- `simulate_dataset()`: Generate synthetic data from model specification (accepts
  optional `seed` parameter for reproducibility)

## Code Style

- Require Python 3.14
- Uses Ruff for linting (target: Python 3.14, line length: 88)
- Google-style docstrings with imperative mood ("Return" not "Returns")
- Use MyST syntax in docstrings (single backticks `like this`), not reStructuredText (no
  double backticks, no `:ref:`, `:func:`, etc.)
- Dataclass attributes use inline docstrings (docstring on the line after the field):
  ```python
  name: str
  """Description of name."""
  ```
- Pre-commit hooks enforce formatting and linting
- Type checking via `ty` with strict rules
- Do not use `from __future__ import annotations`
- Use modern numpy random API: `rng = np.random.default_rng(seed)` instead of
  `np.random.seed()` or legacy functions like `np.random.randn()`

### Immutability Conventions

- All model configuration and internal data structures use frozen dataclasses
- Dict fields on internal dataclasses use `MappingProxyType` (not `Mapping`); wrap at
  the call site with `MappingProxyType(...)`
- Dict fields on user-facing dataclasses (`AnchoringSpec`, `Normalizations`) use
  `Mapping` with `__post_init__` conversion via `ensure_containers_are_immutable()`
- List fields use `tuple`, set fields use `frozenset`
- `ensure_containers_are_immutable()` recursively converts dict→MappingProxyType,
  list→tuple, set→frozenset

## Testing

- pytest with markers: `wip`, `unit`, `integration`, `end_to_end`
- Test files mirror source structure in `tests/`
- Memory profiling available via pytest-memray (Unix only)
