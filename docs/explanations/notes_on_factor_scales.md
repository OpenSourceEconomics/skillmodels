# Notes on Scales and Normalizations

This section discusses factor scales and normalization, building on the
[critique by Wiswall and Agostinelli](https://tinyurl.com/y3wl43kz) of the original
CHS estimator.

Wiswall and Agostinelli define a class of transition functions with Known Location and
Scale (KLS) that require fewer normalizations. Their critique potentially invalidates
certain empirical estimates from CHS, but not the general estimation approach.

To reduce the risk of renormalization issues, you can either:
1. Use fewer normalizations with KLS transition functions, or
2. Use non-KLS transition functions with one normalization per period and factor

skillmodels supports both schemes, but supporting a scheme is not the same as
guaranteeing identification: the model checker performs syntactic checks on the
normalizations, not a transition-specific rank or scale-invariance analysis.
Whether a given scheme identifies the model still depends on the transition
functions and data; see the cautions below.

As there is no natural scale for skills, neither approach is inherently better.
However, we prefer using flexible non-KLS transition functions with explicit
normalizations because:
1. They are more compatible with development stages spanning multiple periods
2. Suitable normalizations can give latent factors a more meaningful interpretation

## Why KLS Functions Don't Keep Scales Constant

After reading the Wiswall-Agostinelli critique, one might think that using KLS
transition functions identifies some sort of "natural" scale. This is not the case.

Consider a simple model of financial investments with two latent factors:
- **w**: wealth (stock variable)
- **i**: investment (flow variable)

Suppose periods are one year and the annual interest rate is 10%. The most intuitive
representation measures everything in dollars:

$$
w_{t+1} = 1.1 w_t + i_t
$$

However, we could measure w in period t in dollars, i in 1000 dollars, and w in period
t+1 in cents. The transition equation becomes:

$$
w_{t+1} = 110 w_t + 100000 i_t
$$

This describes the exact same system in different scales. Any linear function could
describe this system—just with different scale combinations.

The CES function is KLS and contains all linear functions (without intercept) whose
parameters sum to 1. If we set both factor scales to dollars initially, the CES
function would choose:

$$
w_{t+1} = \frac{1}{2.1}(1.1 w_t + i_t) \approx 0.524 w_t + 0.476 i_t
$$

This means wealth in period t+1 is measured in approximately 0.476 dollars—an
arbitrary choice made by the functional form, not something "natural."

## Why CES and log_CES Functions are Problematic

The KLS definition refers only to the scale of the output. But CES and log_CES
functions may also impose restrictions on input scales.

Simulations suggest that with log_CES:
- You need initial location normalizations for all factors
- You only need to normalize the scale of one factor initially

However, we don't have formal identification results for this. **We advise caution**
when using CES or log_CES functions—think carefully about your normalizations rather
than relying on automatic generation.

## Normalizations and Development Stages

When using development stages (periods with identical transition parameters), the
normalization requirements change.

The key insight: you can identify scale from the first period of a stage, so no later
normalizations are needed until the next stage begins.

**Recommendations:**
- Normalize only in the first period of each stage
- For the initial stage, normalize the first two periods
- Use automatic normalizations when working with stages to avoid confusion

This reveals another type of over-normalization in the original CHS paper.
