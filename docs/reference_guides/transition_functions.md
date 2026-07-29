# Transition Functions

Transition functions describe how latent factors evolve over time. skillmodels provides
several pre-built functions and supports custom functions.

The same transition functions work for all three estimators (CHS, AF, AMN) — they live
in `skillmodels.common.transition_functions` and are dispatched by name through each
estimator's pipeline. CHS and AF support both the pre-built set and custom
`@register_params` transitions. AMN also supports custom callables through its Stage-3
generic nonlinear-least-squares path, but with narrower correction- and fixed-parameter
support than CHS (for example, `log_ces` / `log_ces_with_constant` reject fixed
parameters in that path).

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

Despite the name (convention in skill formation literature), this is not a true translog
function.

**Parameters**: Linear terms, squared terms, interaction terms, and constant.

This is the general-library specification: parameters are enumerated over **all**
factors (latent and observed). An observed factor (e.g. income) therefore enters the
production function with its own free linear, square and interaction coefficients. This
is by design for the CHS estimator. In an AF production function, observed factors must
affect skills only through the investment equation, so use `translog_af` (below)
instead.

### translog_af

The AF (2020) production translog from equation (6): linear terms plus pairwise
interactions only, with **no** squared-factor terms:

$$
f_{t+1} = \sum_j \beta_j s_j + \sum_{j < k} \delta_{jk} s_j s_k + c
$$

For the canonical (skill, investment) pair this matches AF eq. (6),
$a_t + g_1 \ln\theta + g_2 \ln I + g_3 \ln\theta \ln I$.

Unlike the general `translog`, this function enumerates parameters over only the
production factors you pass to it. Use it for AF production so that observed factors
(e.g. income) cannot leak in as free production coefficients — pass only the production
factors (skill + investment) and keep observed factors out of the production function.
See [When to use the AF variants](#when-to-use-the-af-variants) below.

**Parameters**: Linear terms, interaction terms, and constant (no squares).

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

This is a KLS function—see
[Notes on Factor Scales](../explanations/notes_on_factor_scales.md) for implications.

**Parameters**: One weight $\gamma_j$ per factor (constrained to sum to 1) plus $\phi$.

This is the general-library specification: the CES weights $\gamma_j$ are enumerated
over **all** factors (latent and observed), so an observed factor (e.g. income) receives
a share of the probability simplex and enters the production aggregate. This is by
design for the CHS estimator. In an AF production function, use `log_ces_af` (below) so
the CES runs over the production factors only.

### log_ces_af

The AF (2020) production CES from equation (7): a log CES over the production factors
only. The math is identical to `log_ces`,

$$
f_{t+1} = \frac{1}{\phi} \ln\left(\sum_j \gamma_j e^{\phi \cdot s_j}\right)
$$

but parameters are enumerated over only the production factors you pass to it, not over
observed factors. Use it for AF production (skill + investment) so that observed factors
cannot leak in as free CES weights — pass only the production factors. See
[When to use the AF variants](#when-to-use-the-af-variants) below.

**Parameters**: One weight $\gamma_j$ per production factor (constrained to sum to 1)
plus $\phi$ — the same set as `log_ces`.

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

## When to use the AF variants

The general `translog` and `log_ces` (and the other built-in production functions:
`linear`, `robust_translog`, `linear_and_squares`, `log_ces_with_constant`,
`log_ces_general`) enumerate parameters over **all** factors, including observed
factors. For the CHS estimator this is the intended behaviour: observed factors are
allowed to enter the production function with free coefficients.

For an AF production function this is usually wrong. The AF model assumes that observed
factors (e.g. income) affect skills **only** through the investment equation, not
directly through production. Using a general built-in production transition would give
income its own free production coefficients, silently changing the AF estimand. The
`translog_af` and `log_ces_af` variants exist for exactly this case: they take only the
production factors (skill + investment) and match AF equations (6) and (7) respectively.

To make the leakage visible, `validate_af_model` emits a loud `UserWarning` when a
built-in production transition (`linear`, `linear_and_squares`, `translog`,
`robust_translog`, `log_ces`, `log_ces_with_constant`, `log_ces_general`) is used on a
non-endogenous production factor while observed factors are present. The warning is not
an error — intentionally-leaky models still run — but it flags the wrong-estimand risk.
Switch to `translog_af` / `log_ces_af`, or pin every observed-factor transition
coefficient to `0.0` via `fixed_params`, to remove the leakage. (The endogenous
investment equation legitimately uses observed factors, so endogenous factors do not
trigger the warning.)

## Custom Transition Functions

Define custom functions using the `@register_params` decorator:

```python
from skillmodels.common.decorators import register_params


@register_params(params=["alpha", "beta"])
def my_transition(fac1, fac2, params):
    return params["alpha"] * fac1 + params["beta"] * fac2**2
```

### Requirements

Custom transition functions must:

1. Accept `params` as a mandatory argument (dictionary with registered parameter names)
1. Accept factor values as floats or use `states` for a JAX array of all factors
1. Return a float (or scalar JAX array)
1. Be JAX jit and vmap compatible (no Python control flow on state values)

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
