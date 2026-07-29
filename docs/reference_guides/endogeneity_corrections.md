# Endogeneity Corrections

When investment decisions depend on the concurrent latent state, investment is
endogenous: its shock is correlated with the production shock of the factors it feeds.
The **control-function** approach corrects for this by adding the unobserved component
of investment — a first-stage residual `cf` — to the production equations of the
affected factors.

skillmodels exposes this as a single, declarative object: a `CorrectionSpec` attached to
the endogenous investment factor. The same specification is read by both estimators that
implement the correction (CHS and AMN), so the control function is configured in exactly
one place regardless of which estimator runs.

## The `CorrectionSpec` interface

`CorrectionSpec` lives in `skillmodels.common.model_spec` and is re-exported from the
top-level package (`from skillmodels import CorrectionSpec`). You declare it via the
`correction=` field of the endogenous investment `FactorSpec` (the factor must still set
`is_endogenous=True`):

```python
from skillmodels import CorrectionSpec, FactorSpec, ModelSpec, Normalizations

model = ModelSpec(
    factors={
        "skills": FactorSpec(
            measurements=(("y1", "y2"), ("y1", "y2")),
            normalizations=Normalizations(
                loadings=({"y1": 1}, {"y1": 1}),
                intercepts=({"y1": 0}, {}),
            ),
            transition_function="linear",
        ),
        "investment": FactorSpec(
            measurements=(("i1", "i2"), ("i1", "i2")),
            normalizations=Normalizations(
                loadings=({"i1": 1}, {"i1": 1}),
                intercepts=({"i1": 0}, {}),
            ),
            transition_function="linear",
            is_endogenous=True,
            correction=CorrectionSpec(
                state_predictors=("skills",),
                instruments=("income",),
                targets=("skills",),
            ),
        ),
    },
    observed_factors=("income",),
)
```

This is the model used in `tests/test_amn_simulate_and_regress.py` (`_cf_model`); see
also `tests/test_cf_recovery.py` for the end-to-end CHS recovery test.

### Fields

`CorrectionSpec` has the following fields (see `src/skillmodels/common/model_spec.py`):

- `instruments: tuple[str, ...]` — **required**, at least one. Excluded observed factors
  that enter the first-stage investment equation **only** (never a production equation),
  and so identify the correction coefficient `kappa`. With no excluded instrument the
  residual would be collinear with the production inputs and `kappa` would be
  unidentified (the `__post_init__` raises `ValueError`). Instruments must be observed
  factors on the `ModelSpec`; `ModelSpec.with_correction` registers them for you (see
  below).
- `state_predictors: tuple[str, ...] = ()` — state factors entering the first-stage
  equation. Empty means **all** state factors.
- `targets: tuple[str, ...] = ()` — state factors whose production equation receives the
  additive `kappa * cf` term. Empty means **all** state factors.
- `kappa_degree: int | None = None` — degree of the `cf`-interaction polynomial applied
  to every target. `1` is a linear `cf` term; `2` is the translog basis. `None` resolves
  to degree `1`. Mutually exclusive with `kappa_terms`.
- `kappa_terms: Mapping[str, tuple[str, ...]] | None = None` — per-target override of
  the `cf` regressor names, e.g. `{"skills": ("cf", "cf ** 2", "cf * skills")}`.
  Mutually exclusive with `kappa_degree`. A target omitted from the mapping defaults to
  `("cf",)`.

`kappa_degree` and `kappa_terms` are mutually exclusive; supplying both raises
`ValueError`.

### How the `cf` regressors are resolved

Internally (in `process_model._resolve_control_function`) each target's regressor list
is resolved as follows:

- if `kappa_terms` is set, target `t` uses `kappa_terms.get(t, ("cf",))`;
- otherwise the degree (defaulting to `1`) is expanded over the state factors via
  `generate_kappa_terms`, and every target shares that basis.

## Building the basis with `generate_kappa_terms`

`generate_kappa_terms` (top-level: `from skillmodels import generate_kappa_terms`)
builds the `cf`-interaction monomial basis you can pass as a target's `kappa_terms`:

```python
from skillmodels import generate_kappa_terms

generate_kappa_terms(("skills", "health"), max_degree=1)
# ("cf",)

generate_kappa_terms(("skills", "health"), max_degree=2)
# ("cf", "cf * skills", "cf * health", "cf ** 2")
```

Its signature is
`generate_kappa_terms(factors, max_degree, max_cf_power=None) -> tuple[str, ...]`: every
monomial `cf ** a * prod_i factor_i ** b_i` with `a >= 1` and total degree
`a + sum_i b_i <= max_degree` (optionally capping the `cf` power at `max_cf_power`).
Pass the result as a target's `kappa_terms` and pin any unwanted coefficients to zero
with an optimagic constraint.

## Builder methods on `ModelSpec`

Two fluent builders make it easy to add or remove the correction without rewriting the
factor dict:

- `ModelSpec.with_correction(factor_name, correction)` — attach `correction` to
  `factor_name` **and** auto-register its instruments as observed factors (deduped
  against existing ones), so instruments are declared exactly once:

  ```python
  model = base_model.with_correction(
      "investment",
      CorrectionSpec(
          state_predictors=("skills",),
          instruments=("income",),
          targets=("skills",),
      ),
  )
  # "income" is now in model.observed_factors automatically.
  ```

- `ModelSpec.without_correction()` — return a copy with every `FactorSpec.correction`
  stripped. Useful for running an estimator that does not implement the correction (AF)
  on a spec authored for CHS.

You can equivalently set `correction=` directly on the `FactorSpec` (as in the first
example), but then you must list the instruments in `observed_factors` yourself.

## Estimator support

The correction is read from the model spec by all three estimators, but only two
implement it:

| Estimator                                            | Control-function support                                                                                                                                                           |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CHS** (`estimate_chs` / `get_maximization_inputs`) | Full polynomial `kappa` basis. Any `kappa_degree` / `kappa_terms` is honoured; the `cf` nodes are grafted into the per-period transition DAG.                                      |
| **AMN** (`estimate_amn`)                             | **Linear `cf` term only.** A higher-order `kappa_terms` request (anything other than `("cf",)` per target) raises `NotImplementedError` in Stage 3.                                |
| **AF** (`estimate_af`)                               | **Not implemented.** If any `FactorSpec.correction` is set, `validate_af_model` raises `NotImplementedError`, directing you to `estimate_chs` or `ModelSpec.without_correction()`. |

The AMN gate lives in `src/skillmodels/amn/simulate_and_regress.py`: if any target's
`kappa_terms` is not exactly `("cf",)`, it raises `NotImplementedError` ("AMN implements
only a linear control function (kappa * cf) ... use estimate_chs for the full polynomial
basis").

The AF gate lives in `src/skillmodels/af/validate.py`: AF covers only the `kappa = 0`
(exogenous-investment) special case, so any declared correction raises
`NotImplementedError`. Strip it with `ModelSpec.without_correction()` to run AF.

## How it works internally

The CHS estimator does not require you to write a custom transition function for the
correction. `process_model` injects three kinds of synthetic node into the per-period
transition DAG (see `src/skillmodels/common/control_function.py`):

1. a deterministic, contemporaneous first-stage prediction `E[ln I_t | theta_t, Y_t]`
   for the endogenous investment factor, fitted over the `state_predictors` and
   `instruments`;
1. the residual `cf_t = ln I_t - E[ln I_t | theta_t, Y_t]`; and
1. for each target factor, the additive `sum_k kappa_k * cf_term_k` grafted onto the
   factor's base transition output.

The first-stage coefficients appear in the params DataFrame under the `investment_eq`
category and the correction coefficients under the `kappa` category. The corrected
production-shock SD is recovered in `shock_sds`.

Models with endogenous factors split each calendar period into augmented periods
internally (`aug_period`); this is what lets the investment residual be measured
conditional on the current state before it enters the next factor's production equation.
Augmented periods are strictly internal — every public function accepts and returns the
user-facing `period`.

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
requirements, consider the original [CHS Fortran code](https://tinyurl.com/yyuq2sa4).

## Legacy / migration

Earlier versions of skillmodels exposed the control function through a much more manual
interface. If you are migrating an old model spec, note:

- There is **no separate correction factor.** A dedicated `is_correction=True` factor
  paired with the endogenous factor is gone; `is_correction` is no longer a `FactorSpec`
  field. Replace the pair with a single endogenous investment factor carrying a
  `CorrectionSpec`.
- You no longer write the control function by hand. The old pattern — a custom
  transition decorated with `@register_params` that computed
  `cf = investment - investment_pred` inside the function body — is obsolete. The
  library now forms `cf` and injects `kappa * cf` for you from the `CorrectionSpec`.
- Period augmentation is unchanged but is now an internal detail (see "How it works
  internally" above); you do not configure it directly.
