# Endogeneity Corrections

This page discusses endogeneity correction methods from the CHS paper and their
limitations. Note that skillmodels does not currently implement these methods—this is
background for users considering extensions.

## CHS Methods

CHS use two endogeneity correction methods, both requiring strong assumptions about
factor scales.

### Time-Invariant Heterogeneity (Section 4.2.4)

This method adds a time-invariant individual fixed effect. The assumption of time
invariance is only valid if factor scales remain constant throughout the model.

**Requirements:**
- Age-invariant measurements for normalization in all periods for all factors
- Three adult outcomes
- Constant factor scales (highly unlikely with KLS transition functions)

If your dataset meets these requirements, consider using the original CHS Fortran code.

### Time-Varying Heterogeneity (Section 4.2.5)

This method uses heterogeneity that follows an AR(1) process. It also relies on:
- Constant factor scales
- A time-invariant investment equation
- Exclusion restrictions (e.g., income affects investment but not skill transitions)

To adapt this for models with changing factor scales, you would need:
- A linear transition function with period-specific parameters (instead of AR(1))
- Period-specific investment functions

Identification of such a model is an open question.

## Wiswall-Agostinelli Approach

Wiswall and Agostinelli propose a simpler endogeneity model (Section 6.1.2 of their
[paper](https://tinyurl.com/y5ezloh2)) that could work with both the CHS and WA
estimators.

## Implementation Status

None of these correction methods are currently implemented in skillmodels. Users
interested in endogeneity corrections should consider:

1. The Wiswall-Agostinelli approach as a starting point
2. The original CHS Fortran code for their specific methods
3. Contributing an implementation to skillmodels
