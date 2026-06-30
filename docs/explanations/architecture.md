# Package Architecture

Skillmodels hosts three estimators under one model specification. The package
layout reflects that:

```
src/skillmodels/
├── common/                  Estimator-agnostic machinery
│   ├── model_spec.py          ModelSpec, FactorSpec, AnchoringSpec, Normalizations
│   ├── types.py               ProcessedModel, Dimensions, Labels,
│   │                          EndogenousFactorsInfo, ParsingInfo, ...
│   ├── process_model.py       process_model(spec) -> ProcessedModel
│   ├── process_data.py        long-format data -> internal arrays
│   ├── params_index.py        4-level MultiIndex used by all estimators
│   ├── parse_params.py        flat vector <-> structured params
│   ├── constraints.py         get_constraints, FixedConstraintWithValue,
│   │                          collect_fixed_locs, project_to_probability_constraints
│   ├── selector.py            select_by_loc, align_index_names
│   ├── transition_functions.py  linear / translog / log_ces / translog_af /
│   │                          log_ces_af / ...
│   ├── transitions.py         apply_anchored_transition (sigma-points-agnostic)
│   ├── anchoring.py           anchor / unanchor states
│   ├── state_ranges.py        create_state_ranges
│   ├── simulate_data.py       simulate_dataset, simulate_policy_effect
│   ├── variance_decomposition.py  signal/noise decomposition
│   └── diagnostic_plots.py    plot_residual_boxplots, plot_likelihood_contributions
├── chs/                     Cunha-Heckman-Schennach Kalman MLE
│   ├── options.py             CHSEstimationOptions
│   ├── kalman_filters.py      square-root unscented Kalman filter
│   ├── likelihood.py          jitted log-likelihood
│   ├── likelihood_debug.py    non-jitted variant with debug arrays
│   ├── maximization_inputs.py get_maximization_inputs(...)
│   ├── filtered_states.py     get_filtered_states(...)
│   └── process_debug_data.py  post-process Kalman debug arrays
├── af/                      Antweiler-Freyberger sequential Halton MLE
│   ├── types.py               AFEstimationOptions, AFEstimationResult, ...
│   ├── estimate.py            estimate_af(...) -- top-level orchestration
│   ├── validate.py            validate_af_model, kappa-scope / leakage checks
│   ├── params.py              per-period optimagic params + constraints
│   ├── initial_period.py      period-0 mixture + measurement system MLE
│   ├── transition_period.py   period-t transition + measurement-system MLE
│   ├── likelihood.py          jitted period-specific log-likelihoods
│   ├── halton.py              quadrature nodes / weights
│   ├── batching.py            obs-batching for the autodiff chunking
│   ├── posterior_states.py    posterior means from the chained sample
│   └── inference.py           compute_af_standard_errors (propagated
│                              influence-function score bootstrap)
└── amn/                     Attanasio-Meghir-Nix 2020 (three-stage)
    ├── types.py               AMNEstimationOptions, ...
    ├── estimate.py            estimate_amn(...) -- top-level orchestration
    ├── mixture_em.py          Stage 1: EM on the augmented mixture
    ├── minimum_distance.py    Stage 2: structural recovery
    ├── simulate_and_regress.py Stage 3: synthetic-panel regression
    │                          (optional endogenous-investment control function)
    ├── moments.py             Spearman + Bartlett start-values
    ├── start_values.py        get_spearman_start_params, pool_equality_groups
    ├── posterior_states.py    simulate factor paths from fitted mixture
    └── inference.py           compute_amn_standard_errors (cluster bootstrap,
                               per-replicate seeds + nonconvergence filtering)
```

## How the layers interact

Every estimator reads the same `ModelSpec` and produces the same canonical
params DataFrame (4-level MultiIndex
`(category, period, name1, name2)`). The differences live entirely below the
spec:

- **CHS** consumes `process_model(spec) -> ProcessedModel`, then plugs that
  into the Kalman recursion. `CHSEstimationOptions` is passed at call time
  to `get_maximization_inputs(spec, data, chs_options=...)`.
- **AF** also calls `process_model`, but uses `ProcessedModel` only for the
  parameter index, labels, and transition info. The Kalman filter is not
  invoked; period-specific Halton designs replace the predict step.
- **AMN** likewise calls `process_model` for the index/labels, then runs its
  three-stage pipeline. The result re-uses the same params DataFrame format
  so the AMN output can seed CHS or AF estimation when desired.

`process_model` itself is structural: it takes only the spec and produces
shapes, labels, transition info, and an `EndogenousFactorsInfo`. It does not
carry any estimator-specific tuning. Each estimator's options class
(`CHSEstimationOptions`, `AFEstimationOptions`, `AMNEstimationOptions`) is
passed in at call time.

## Why this split

The package grew organically: CHS was the original codebase; AF and AMN were
later additions. Earlier iterations stored CHS-only options on `ModelSpec`,
which made the spec leak CHS assumptions into a notionally agnostic container.
The split into `common/`, `chs/`, `af/`, `amn/` makes the scope of each piece
explicit at the import site:

- `from skillmodels import ModelSpec` — pure structural description.
- `from skillmodels.chs import CHSEstimationOptions, get_maximization_inputs`
  — CHS-specific.
- `from skillmodels.af import estimate_af, AFEstimationOptions` — AF-specific.
- `from skillmodels.common.variance_decomposition import decompose_measurement_variance`
  — works for any estimator, given pre-computed filtered states.

The architectural principle: a function lives in `common/` iff it does not
import from `chs/`, `af/`, or `amn/`. Anything that does belongs in the
relevant subpackage. There is one practical exception:
`CHSEstimationOptions` is defined in `chs/options.py` but the
`process_model` orchestration in `common/` doesn't read it (it reads the
structural `ModelSpec.n_mixtures` field instead), so the layering is clean.
