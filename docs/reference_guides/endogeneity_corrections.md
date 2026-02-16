# Endogeneity Corrections

When investment decisions depend on concurrent latent factor values, investments are
endogenous. skillmodels implements an endogeneity correction via the control function
approach, using period augmentation in the Kalman filter.

## Using Endogenous Factors

Mark a factor as endogenous by setting `is_endogenous=True` on its `FactorSpec`:

```python
from skillmodels import FactorSpec, ModelSpec

model = ModelSpec(
    factors={
        "cognitive": FactorSpec(
            measurements=(("y1", "y2"), ("y1", "y2")),
            # ... normalizations ...
            transition_function="linear",
        ),
        "investment": FactorSpec(
            measurements=(("inv1", "inv2"), ("inv1", "inv2")),
            # ... normalizations ...
            is_endogenous=True,
            transition_function="linear",
        ),
    },
)
```

## How It Works: Period Augmentation

When the model contains endogenous factors, skillmodels internally doubles the number of
periods. Each original period is split into two augmented periods:

1. **Even augmented periods** (0, 2, 4, ...): State factor measurements are updated.
2. **Odd augmented periods** (1, 3, 5, ...): Endogenous factor measurements are updated
   using the predicted state values.

This sequential updating ensures that endogenous factors are measured conditional on
current state information, which is essential for the control function approach.

## Correction Factors

For a full control function correction, add a correction factor with
`is_correction=True`. A correction factor must also be endogenous:

```python
model = ModelSpec(
    factors={
        "cognitive": FactorSpec(
            measurements=(("y1", "y2"), ("y1", "y2")),
            # ...
        ),
        "investment": FactorSpec(
            measurements=(("inv1", "inv2"), ("inv1", "inv2")),
            is_endogenous=True,
            # ...
        ),
        "investment_pred": FactorSpec(
            measurements=(("inv1", "inv2"), ("inv1", "inv2")),
            is_endogenous=True,
            is_correction=True,
            # ...
        ),
    },
)
```

The correction factor typically shares the same measurements as the endogenous factor it
corrects. In the transition function, the difference between the actual and predicted
values (the control function residual) captures the endogeneity:

```python
@register_params(params=["investment", "investment_pred", "cf", "constant"])
def f_cognitive(investment, investment_pred, params):
    cf = investment - investment_pred
    return (
        params["constant"]
        + params["investment"] * investment
        + params["cf"] * cf
    )
```

The estimated `cf` coefficient captures the effect of the unobserved component of
investment on the outcome.

## Background: CHS Methods

Cunha, Heckman, and Schennach (2010) propose two alternative endogeneity correction
methods that rely on stronger assumptions about factor scales:

### Time-Invariant Heterogeneity (Section 4.2.4)

Adds a time-invariant individual fixed effect. Requires constant factor scales across
all periods (highly unlikely with KLS transition functions), age-invariant normalization
measurements, and three adult outcomes.

### Time-Varying Heterogeneity (Section 4.2.5)

Uses heterogeneity following an AR(1) process. Requires constant factor scales, a
time-invariant investment equation, and exclusion restrictions (e.g., income affects
investment but not skill transitions).

These methods are not implemented in skillmodels. If your dataset meets their
requirements, consider the original
[CHS Fortran code](https://tinyurl.com/yyuq2sa4).
