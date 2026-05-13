# Model Specifications

Models are specified using Python dataclasses.

## Defining a Model

```python
from skillmodels import (
    AnchoringSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)

# Define factors
fac1 = FactorSpec(
    measurements=(
        ("y1", "y2", "y3"),  # period 0
        ("y1", "y2", "y3"),  # period 1
        # ...
    ),
    normalizations=Normalizations(
        loadings=(
            {"y1": 1.0},  # fix loading of y1 to 1 in period 0
            {},
        ),
        intercepts=({}, {}),
    ),
    transition_function="log_ces",
)

# Create model
model = ModelSpec(
    factors={"fac1": fac1, "fac2": fac2, "fac3": fac3},
    anchoring=AnchoringSpec(
        outcomes={"fac1": "Q1"},
        free_loadings=True,
    ),
    controls=("x1", "x2"),
    stagemap=(0, 0, 1, 1, 2, 2, 3),
    n_mixtures=2,
)
```

The `ModelSpec` is purely structural -- it describes the model, not how to
estimate it. Estimator-specific tuning (number of Halton draws, mixture
components in CHS Kalman, sigma-point scale, ...) lives on the relevant
options class and is passed at the call site:

```python
from skillmodels.chs import CHSEstimationOptions, get_maximization_inputs

max_inputs = get_maximization_inputs(
    model_spec=model,
    data=data,
    chs_options=CHSEstimationOptions(bounds_distance=1e-4),
)
```

See the [AF how-to](how_to_estimate_af.md) for the corresponding pattern with
`estimate_af(model, data, af_options=...)`.

## Factor Specification

Each factor requires:

- **measurements**: A nested tuple with measurement variable names for each period.
  Empty tuples indicate no measurements in that period.
- **transition_function**: Name of a transition function (`linear`, `log_ces`,
  `constant`, `translog`) or a custom function.
- **normalizations** (optional): Fixed values for loadings and intercepts to identify
  the model.
- **is_endogenous** (optional): Whether this factor is endogenous (default: false).
  See [Endogeneity Corrections](../reference_guides/endogeneity_corrections.md).
- **is_correction** (optional): Whether this is a correction factor (default: false).
  Must also be endogenous.

## Anchoring

Anchoring links latent factors to observable outcomes. Options:

- **outcomes**: Dictionary mapping factor names to anchoring outcome variables
- **free_controls**: Whether to estimate control coefficients in anchoring equations
  (default: false)
- **free_constant**: Whether to estimate a constant in anchoring equations
  (default: false)
- **free_loadings**: Whether to estimate loadings in anchoring equations
  (default: false)
- **ignore_constant_when_anchoring**: Skip constant when anchoring (default: false)

## Controls

A tuple of variable names used as control variables in measurement equations. A constant
is always included automatically.

## Stagemap

Maps periods to development stages. Has one entry less than the number of periods.
Parameters are constrained to be equal within a stage.

Example: `(0, 0, 1, 1)` means periods 0-1 share stage 0 parameters, and periods 2-3
share stage 1 parameters.

## Observed Factors

Variables in the dataset that represent observed (not latent) factors. These don't need
transition equations or multiple measurements.

```python
model = ModelSpec(
    factors={...},
    observed_factors=("income", "treatment"),
)
```

## Estimation Options

Fine-tune the estimation:

- **robust_bounds**: Make bounds stricter to avoid numerical issues (default: true)
- **bounds_distance**: How much stricter to make bounds (default: 0.001)
- **n_mixtures**: Number of mixture components (default: 1)
- **sigma_points_scale**: Scaling for Julier sigma points (default: 2)
- **clipping_lower_bound**: Clip log-likelihood from below (default: -1e30)
- **clipping_upper_bound**: Clip log-likelihood from above (default: null)
- **clipping_lower_hardness**: Hardness of lower clipping (default: 1)
- **clipping_upper_hardness**: Hardness of upper clipping (default: 1)

## Custom Transition Functions

Define custom transition equations using the `@register_params` decorator:

```python
from skillmodels.common.decorators import register_params

@register_params(params=["lincoeff"])
def my_linear(fac, params):
    return params["lincoeff"] * fac
```

Custom functions must:
- Accept `params` as a required argument (dictionary with registered parameters)
- Accept factor values as floats or use `states` for a JAX array of all states
- Return a float
- Be JAX jit and vmap compatible
