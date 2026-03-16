@.ai-instructions/profiles/tier-a.md @.ai-instructions/modules/jax.md
@.ai-instructions/modules/optimagic.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

skillmodels is a Python implementation of estimators for nonlinear dynamic latent factor
models, primarily used for skill formation research in economics. It implements Kalman
filter-based maximum likelihood estimation following Cunha, Heckman, Schennach (2010).

Used as the core estimation engine by sibling application projects (`skane-struct-bw`,
`health-cognition`) in the parent workspace.

## Development Commands

```bash
# Run tests
pixi run -e tests-cpu tests

# Run tests with coverage
pixi run -e tests-cpu tests-with-cov

# Run a single test file
pixi run -e tests-cpu pytest tests/test_kalman_filters.py

# Run a single test
pixi run -e tests-cpu pytest tests/test_kalman_filters.py::test_function_name

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
- **Tests**: Use `pixi run -e tests-cpu tests` instead of `pytest` directly
- **Linting/formatting**: Use `prek run --all-files` instead of `ruff` directly
- **All quality checks**: Use `prek run --all-files`

Before finishing any task that modifies code, always run:

1. `pixi run ty` (type checker)
1. `pixi run -e tests-cpu tests` (tests)
1. `prek run --all-files` (quality checks)

## Architecture

### Core Pipeline Flow

```
ModelSpec + Data
       ↓
process_model() → Validates/extends model specification → ProcessedModel
       ↓
get_maximization_inputs() → Creates optimization problem (likelihood, gradients,
                            constraints, params_template)
       ↓
[optimagic maximize / estimagic estimate_ml with fides algorithm]
       ↓
get_filtered_states() → Extract estimated latent factors
simulate_dataset() → Simulate states (with optional policy effects)
```

### Key Modules

- **model_spec.py**: User-facing frozen dataclasses (`ModelSpec`, `FactorSpec`,
  `AnchoringSpec`). Re-exports `EstimationOptions` and `Normalizations` from `types.py`.
  `ModelSpec` supports construction via `__init__` or `ModelSpec.from_dict()`, and
  fluent builder methods: `with_transition_functions()`, `with_added_factor()`,
  `with_added_observed_factors()`, `with_estimation_options()`, `with_anchoring()`,
  `with_controls()`, `with_stagemap()`.
- **types.py**: Internal frozen dataclasses (`ProcessedModel`, `Labels`, `Dimensions`,
  `Anchoring`, `ParsingInfo`, `ParsedParams`, `EndogenousFactorsInfo`, etc.),
  `EstimationOptions`, `Normalizations`, and immutability utilities.
- **process_model.py**: Model specification validation and preprocessing. Converts
  `ModelSpec` into `ProcessedModel`.
- **kalman_filters.py**: Core Kalman filter implementation (predict/update steps). Uses
  square-root form for numerical stability.
- **likelihood_function.py** / **likelihood_function_debug.py**: Log-likelihood
  computation using Kalman filtering. The debug variant is not jitted and returns
  intermediate results (residuals, contributions, filtered states).
- **constraints.py**: Generates parameter constraints (bounds, equalities from stagemap,
  fixed values) for optimization. Exports `get_constraints()`,
  `enforce_fixed_constraints()`, `add_bounds()`, `FixedConstraintWithValue`.
- **parse_params.py**: Converts flat parameter vectors to structured model parameters.
  Exports `create_parsing_info()` and `parse_params()`.
- **params_index.py**: Builds the `pd.MultiIndex` for the params DataFrame via
  `get_params_index()`.
- **transition_functions.py**: Pre-built transition equations: `linear`, `translog`,
  `robust_translog`, `linear_and_squares`, `log_ces`, `log_ces_general`, `constant`.
- **decorators.py**: `register_params` decorator for custom transition functions. Tags a
  callable with `__registered_params__` so skillmodels knows its parameter names.
- **process_data.py**: `process_data()` for internal estimation format,
  `pre_process_data()` for reshaping data to long format with period indexing.
- **simulate_data.py**: `simulate_dataset()` and `simulate_policy_effect()`.
- **diagnostic_plots.py**: `plot_likelihood_contributions()` and
  `plot_residual_boxplots()` (Plotly-based).
- **variance_decomposition.py**: `decompose_measurement_variance()` and
  `summarize_measurement_reliability()`.
- **process_debug_data.py**: `create_state_ranges()` and `process_debug_data()` for
  converting raw debug output into DataFrames.
- **utilities.py**: Model manipulation helpers (`extract_factors`, `remove_factors`,
  `update_parameter_values`, `switch_translog_to_linear`, etc.).
- **Visualization modules** (not in `__all__`, imported by module path):
  `correlation_heatmap.py`, `visualize_factor_distributions.py`,
  `visualize_transition_equations.py`, `utils_plotting.py`.

### `get_maximization_inputs()` Return Dict

Returns a dict with 6 keys:

- `"loglike"`: `(params: pd.DataFrame) -> float` — jitted scalar log-likelihood
- `"loglikeobs"`: `(params: pd.DataFrame) -> NDArray` — jitted per-observation
  log-likelihood
- `"debug_loglike"`: `(params: pd.DataFrame) -> dict` — non-jitted, returns dict with
  keys `value`, `contributions`, `residuals`, `residual_sds`, `filtered_states`,
  `state_ranges`, etc.
- `"loglike_and_gradient"`: `(params: pd.DataFrame) -> tuple[float, NDArray]`
- `"constraints"`: list of optimagic constraint objects
- `"params_template"`: `pd.DataFrame` with correct MultiIndex and bounds; fixed
  constraints pre-applied

### ProcessedModel Key Attributes

Applications frequently access these after calling `process_model(model_spec)`:

- `processed.labels` — `.latent_factors`, `.observed_factors`, `.all_factors`,
  `.controls`, `.stagemap`, `.stages`, `.aug_periods_to_periods`
- `processed.dimensions` — `.n_periods`, `.n_latent_factors`
- `processed.update_info` — DataFrame indexed by `(aug_period, variable)` with factor
  columns and a `purpose` column
- `processed.endogenous_factors_info` — `.has_endogenous_factors`,
  `.aug_periods_from_period(period)`, `.factor_info`
- `processed.normalizations` — dict of factor name to `Normalizations`
- `processed.transition_info` — `.func` (vectorized), `.individual_functions[factor]`

### JAX Usage

All computation-heavy code uses JAX for automatic differentiation and JIT compilation.
The codebase uses:

- `jax.vmap` for vectorization across observations
- `jax.jit` for compilation
- JAX arrays throughout the estimation pipeline
- Optional GPU support via CUDA or Metal

### Public API (`__init__.py`)

**Model specification classes:**

- `ModelSpec`, `FactorSpec`, `AnchoringSpec`, `EstimationOptions`, `Normalizations`

**Core estimation:**

- `get_maximization_inputs(model_spec, data, split_dataset=1)` — prepare optimization
  problem
- `get_filtered_states(model_spec, data, params)` — returns nested dict with
  `"anchored_states"` and `"unanchored_states"`, each containing `"states"` (DataFrame)
  and `"state_ranges"`

**Simulation:**

- `simulate_dataset(model_spec, params, n_obs=None, data=None, policies=None, seed=None)`
  — returns dict with `"unanchored_states"`, `"anchored_states"`
- `simulate_policy_effect(model_spec, params, data, policies, seed=None)` — returns
  DataFrame of factor mean differences between policy and baseline

**Diagnostics and visualization:**

- `plot_likelihood_contributions(model_spec, data, params, period=None)`
- `plot_residual_boxplots(model_spec, data, params, period=None)`
- `decompose_measurement_variance(model_spec, params, data)` — returns DataFrame indexed
  by `(period, measurement, factor)` with signal/noise columns
- `summarize_measurement_reliability(variance_decomposition)`
- `create_state_ranges(filtered_states, factors, quantile_cutoff=None)`

### Frequently Used Internal APIs

These are not in `__all__` but are imported directly by application projects:

- `skillmodels.process_model.process_model` — central to all application code
- `skillmodels.types.ProcessedModel`, `EndogenousFactorsInfo`
- `skillmodels.decorators.register_params` — essential for custom transition functions
- `skillmodels.constraints.get_constraints`, `enforce_fixed_constraints`,
  `FixedConstraintWithValue`, `select_by_loc`
- `skillmodels.utilities.extract_factors`, `update_parameter_values`
- `skillmodels.process_data.pre_process_data`
- `skillmodels.correlation_heatmap.get_measurements_corr`, `get_quasi_scores_corr`,
  `get_scores_corr`, `plot_correlation_heatmap`
- `skillmodels.visualize_factor_distributions.univariate_densities`,
  `bivariate_density_contours`, `combine_distribution_plots`
- `skillmodels.visualize_transition_equations.get_transition_plots`,
  `combine_transition_plots`
- `skillmodels.parse_params.create_parsing_info`, `parse_params`
- `skillmodels.params_index.get_params_index`

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

### Period vs Aug_period

Models with endogenous factors split each calendar period into multiple **augmented
periods** (`aug_period`). The public API uses `period` (user-facing); `aug_period` is
strictly internal. All public functions now return `period`:

- `ModelSpec` — clean, no `aug_period` exposure.
- `get_transition_plots()` — clean, accepts `period`/`periods`.
- `get_filtered_states()` — clean, returns `period` column.
- `simulate_dataset()` — clean, returns `period` in states DataFrames.
- `plot_residual_boxplots()` / `plot_likelihood_contributions()` — clean, accept and
  return `period`.
- `decompose_measurement_variance()` — clean, indexed by
  `(period, measurement, factor)`.
- `simulate_policy_effect()` / `simulate_dataset()` policies — accept `"period"` key.
- `ProcessedModel.labels` — exposes `aug_periods_to_periods` mapping (acceptable for
  internal/advanced use).

When writing new public-facing code, always accept and return `period`. Convert to
`aug_period` internally using `ProcessedModel.labels.aug_periods_to_periods`.

## Testing

- pytest with markers: `wip`, `unit`, `integration`, `end_to_end`
- Test files mirror source structure in `tests/`
- Memory profiling available via pytest-memray (Unix only)
