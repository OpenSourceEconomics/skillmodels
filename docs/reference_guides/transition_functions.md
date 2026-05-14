# Transition Functions

Transition functions describe how latent factors evolve over time. skillmodels provides
several pre-built functions and supports custom functions.

The same transition functions work for all three estimators (CHS, AF, AMN) — they live
in `skillmodels.common.transition_functions` and are dispatched by name through each
estimator's pipeline. AMN's Stage 3 currently supports the pre-built set listed below;
custom `@register_params` transitions work with CHS and AF but not yet with AMN.

## Pre-built Transition Functions

### linear

Linear transition function with a constant term:

$$
f_{t+1} = \sum_j \beta_j \cdot s_j + c
$$

where $s_j$ are the state values and $c$ is a constant.

**Parameters**: One coefficient per factor plus a constant.

### translog

Linear-in-parameters function with squares and interaction terms:

$$
f_{t+1} = \sum_j \beta_j s_j + \sum_j \gamma_j s_j^2 + \sum_{j < k} \delta_{jk} s_j s_k + c
$$

Despite the name (convention in skill formation literature), this is not a true
translog function.

**Parameters**: Linear terms, squared terms, interaction terms, and constant.

### robust_translog

Same as `translog` but clips state values at ±10^12 before computation. Use this when
states might grow very large and cause numerical overflow.

### linear_and_squares

Like `translog` but without interaction terms:

$$
f_{t+1} = \sum_j \beta_j s_j + \sum_j \gamma_j s_j^2 + c
$$

### log_ces

Log CES (Constant Elasticity of Substitution) in the Known Location and Scale version:

$$
f_{t+1} = \frac{1}{\phi} \ln\left(\sum_j \gamma_j e^{\phi \cdot s_j}\right)
$$

This is a KLS function—see [Notes on Factor Scales](../explanations/notes_on_factor_scales.md)
for implications.

**Parameters**: One weight $\gamma_j$ per factor (constrained to sum to 1) plus $\phi$.

### log_ces_general

Generalized log CES without known location and scale:

$$
f_{t+1} = \text{tfp} \cdot \ln\left(\sum_j \gamma_j e^{\sigma_j \cdot s_j}\right)
$$

**Parameters**: Weights $\gamma_j$, factor-specific elasticities $\sigma_j$, and total
factor productivity.

### constant

The factor value does not change:

$$
f_{t+1} = f_t
$$

**Parameters**: None.

## Custom Transition Functions

Define custom functions using the `@register_params` decorator:

```python
from skillmodels.decorators import register_params

@register_params(params=["alpha", "beta"])
def my_transition(fac1, fac2, params):
    return params["alpha"] * fac1 + params["beta"] * fac2**2
```

### Requirements

Custom transition functions must:

1. Accept `params` as a mandatory argument (dictionary with registered parameter names)
2. Accept factor values as floats or use `states` for a JAX array of all factors
3. Return a float (or scalar JAX array)
4. Be JAX jit and vmap compatible (no Python control flow on state values)

### Using Custom Functions

```python
from skillmodels import FactorSpec

factor = FactorSpec(
    measurements=[...],
    transition_function=my_transition,  # Pass the function object
)
```

Or with a dictionary-based model:

```python
model["factors"]["fac1"]["transition_function"] = my_transition
```

### Advanced: Accessing All States

If your transition function needs access to all states at once:

```python
@register_params(params=["weights"])
def weighted_sum(states, params):
    return jnp.dot(states, params["weights"])
```

The `states` argument is a 1D JAX array with all factor values in order.
