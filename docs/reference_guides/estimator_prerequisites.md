# Estimator Prerequisites — When to Use Which

skillmodels ships three estimators for the same family of nonlinear dynamic latent
factor models — CHS, AF, and AMN. They accept the same structural `ModelSpec`, but they
differ in what data features and model constructs they actually support. Pick an
estimator by checking your model against the prerequisites below **before** estimating;
a feature one estimator handles natively may be rejected, ignored, or silently
restricted by another.

A statement that a model uses, say, `ModelSpec.measurement_models` does **not** imply
uniform support: measurement-family handling in particular differs sharply across the
three estimators.

## Prerequisites matrix

| Dimension             | CHS                                                                                                                      | AF                                                                                                                                                                                           | AMN                                                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Estimator             | Joint maximum likelihood with square-root Gaussian-component filtering.                                                  | Sequential period-by-period likelihood with Halton integration.                                                                                                                              | Three stages: Gaussian-mixture EM → minimum distance → simulate-and-regress.                                                                                                              |
| Latent distribution   | Finite mixture of Gaussian initial states via `ModelSpec.n_mixtures`; Gaussian component filtering thereafter.           | Initial finite mixture plus sequentially carried distributions; Halton nodes approximate the integrals.                                                                                      | Stage-1 Gaussian mixture over the augmented measurement vector; `n_mixtures` sets the component count.                                                                                    |
| Measurement families  | **Gaussian only** (standard Kalman update). Probit/Tobit measurements are not consumed by the CHS path.                  | **Initial period only**: probit/Tobit measurement families are honoured at the period-0 measurement system; transition periods ($t \geq 1$) fall back to an all-Gaussian measurement kernel. | **Gaussian only**: `estimate_amn` raises `NotImplementedError` if `ModelSpec.measurement_models` declares any probit/Tobit measurement.                                                   |
| Missing data          | Gaussian measurement updates skip individually missing measurements.                                                     | Measurement masks skip missing rows in the per-step likelihood contributions.                                                                                                                | `mixture_em_method="complete_case"` (default) or `"missing_data"` (marginalises over missing entries under MAR); never-observed columns require `allow_never_observed_measurements=True`. |
| Endogenous investment | Reconstructed endogenous factors plus a full `CorrectionSpec` control-function basis.                                    | Reconstructed endogenous investment with independent shocks (source/destination calendar adapter); rejects `CorrectionSpec` / nonzero `kappa`.                                               | Endogenous investment in Stage 3; correction support is a linearised control-function term, narrower than CHS.                                                                            |
| Corrections           | Full `CorrectionSpec` / `kappa` polynomial basis in the processed transition DAG.                                        | Not implemented — the validator raises if a `CorrectionSpec` is attached.                                                                                                                    | Linear `cf` term only; a higher-order `kappa_terms` request raises.                                                                                                                       |
| Custom transitions    | Built-ins and `@register_params` callables.                                                                              | Built-ins and `@register_params` callables.                                                                                                                                                  | Built-ins and `@register_params` callables via the Stage-3 generic NLS path, but `log_ces` / `log_ces_with_constant` reject fixed parameters there.                                       |
| `fixed_params`        | Honoured through the shared parameter index.                                                                             | Honoured (with public `constraints=` limited to `select_by_loc` equality groups).                                                                                                            | Honoured only for the categories each stage estimates (Stage-2 loadings/intercepts/SDs, Stage-3 `transition`); other categories raise. `start_params` and `constraints` raise.            |
| Normalization         | Shared `Normalizations` / `fixed_params` / equality constraints. The checker is a precheck, not an identification proof. | Same public `ModelSpec`, with the AF-specific source/destination calendar and `af_state_role` metadata.                                                                                      | Stage-2 imposes its own structural moment restrictions and a mean-zero mixture convention.                                                                                                |
| Anchoring             | Supported through `get_maximization_inputs` / the CHS path.                                                              | Not part of the AF likelihood; downstream visualization only.                                                                                                                                | Not wired through the AMN stages (the result reports unanchored scales).                                                                                                                  |
| Cost / scaling        | Potentially expensive joint ML; the JAX square-root filter helps numerical stability.                                    | Sequential but quadrature-heavy; cost grows with node count and state dimension.                                                                                                             | Fast when the stages are well behaved; the cluster bootstrap is expensive because it re-estimates every stage.                                                                            |
| Use it for            | Likelihood benchmark, anchoring, correction-heavy models, Gaussian measurement systems.                                  | Sequential AF-style models, initial-period limited measurements, period-by-period diagnostics.                                                                                               | Mixture-heavy Gaussian-measurement models, fast start values for CHS, Stage-1/2 structural diagnostics.                                                                                   |

## Reading the matrix

- **Measurement families** are the sharpest divide. If any measurement is probit or
  Tobit, AMN rejects the model outright, CHS ignores the family and treats it as
  Gaussian, and AF honours it only at the initial period. Only models with
  initial-period-only limited measurements have genuine (partial) support, via AF.
- **Endogeneity corrections** flow through `FactorSpec.correction` (a `CorrectionSpec`).
  CHS reads the full polynomial `kappa` basis; AMN keeps only the linear term; AF
  rejects the correction entirely (its endogenous-investment handling is the
  reconstructed-factor calendar adapter, not a control function).
- **`start_params` / `constraints`** are CHS/AF concepts; AMN raises on both.

See [How to compare estimators](../how_to_guides/how_to_compare_estimators.md) for an
overlay of all three on the same data with confidence intervals, and
[Endogeneity Corrections](endogeneity_corrections.md) for the `CorrectionSpec`
interface.
