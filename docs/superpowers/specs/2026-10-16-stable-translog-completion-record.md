# Claude Code completion record

Evidence for one fresh compact closure audit of the `stable_translog` change.

- Workflow class: `computational-method`
- Audit round / repair attempt: `1` / `0`
- Audited baseline commit: `94aa213d64955044c4afd436dd27af5e0f8852b1`
- Final commit: `HEAD` of branch `stable-translog` (feature commits `2ccfebb`, `be82adc`,
  `ab9bd82`, `671f596`; one further commit adds this record, the native closure log and
  the resource report). PR: https://github.com/OpenSourceEconomics/skillmodels/pull/101
- Model used for implementation: `Opus 5.5` (subagent, medium effort), coordinated and
  reviewed by `Fable 5.1`
- Reviewer patch IDs applied/adapted/rejected: `P1` applied in full to the design, plan and
  contract (commit `2ccfebb`); the plan's prose test placeholders were then replaced by the
  reviewer's own test files (see next line)
- Oracle/test/mutation/demo IDs retained: `OR1` (`tests/stable_translog_reference.py`),
  `RT1` + `MT2` (`tests/test_stable_translog.py`), `RT2`
  (`tests/test_stable_translog_dispatch.py`), `MT1` (run from the reply package, see
  execution record; not added to the project suite because it is pure mathematics with no
  project import)

## Root-cause class disposition

| Root cause | Defect class | Strategy | Class closure evidence | Status |
| --- | --- | --- | --- | --- |
| R1 | full-law-stability-scope | `text_change` | P1 wording in design 2.2 and 2.4, plan goal/docstring/docs section/PR body; MT1 stability witnesses pass (false gain, switched Schur family, small-entry matrix eigenvalue 1.21, control-function effective coefficient 3/2) | `closed` |
| R2 | finite-tail-soft-clip-overflow | `local_patch` | `_soft_clip` evaluates normalised powers with one shared `<=` predicate; RT1 finite tails through 1e80 and MT2 overflow-threshold neighbourhood for three (scale, power) pairs pass against OR1 | `closed` |
| R3 | non-discriminating-acceptance-oracles | `local_patch` | RT1 default-budget tests, RT2 dispatch tests, CHS same-law tests; scale-2.5 and translog-alias mutants fail numerically | `closed` |

R1 invariant: every boundedness statement in the shipped text refers to the complete
recursively updated law, with a fixed Schur-stable matrix and uniformly bounded forcing, gain
written as the sum of the norms of the matrix powers, and switched, control-function and
stochastic cases explicitly excluded from the certificate. Counterexample class covered: the
text no longer makes any of the four overclaims MT1 refutes, and the design documents each of
them as a separately checked condition. The schema warning that F1 "lacks a runnable
witness/boundary test" is settled by MT1 (`audit_tests/test_reference_mutations.py`, run
below) together with `stability_witnesses()` in OR1, which is executed on every run of the
oracle; a text finding has no project test by construction.

R2 invariant: g is evaluated as `numerator / (unit_s**p + unit_x**p)**(1/p)` with
`m = scale` inside `|x| <= scale` and `m = |x|` outside, numerator `x` inside and
`scale * x/m` outside, both selected by the same predicate; all operands of unselected
branches are finite; non-finite inputs are replaced by 0 before the arithmetic and restored to
`+/-scale` (inf) or NaN afterwards. Counterexample class covered: every finite float64 input
has powered operands of magnitude at most one, so no intermediate can overflow for any
(scale, power); MT2 samples the naive overflow threshold `scale * max**(1/p)` and its
neighbouring floats for (0.5, 4), (3, 4), (9, 8) and checks values, derivatives and
finiteness against OR1; RT1 pins the knee derivative at `nextafter(3, -inf)`, `3`,
`nextafter(3, inf)` and the tails at `+/-50`, `+/-1e76`, `+/-1e80`.

R3 invariant: tests compare against an independent same-law oracle at informative inputs,
pin the public defaults directly, and test the dispatched callable's output rather than
source strings. Counterexample class covered: RT1 `test_default_profile_fidelity_for_squares_and_interactions`
asserts the 1 % / 10 % budgets at |x| = 1, 2 on the public default (a scale of 2.5 gives
15.8 % at 2 and fails); RT2 asserts 9/sqrt(2) at x = 3 from both `process_model` output and
the AMN resolver (translog gives 9); the CHS tests compare per-observation likelihood
contributions and gradients with a registered custom callable that evaluates the original
formula and share no code with `stable_translog`; a temporary alias of `stable_translog` to
`translog` made all three numerical CHS tests fail (max abs differences 1.11 and 1135.7
on contributions; endogenous difference exactly 0.0, failing the `> 1e-8` assertion).

## Finding disposition

| Finding | Root cause | Final status | Evidence command/result |
| --- | --- | --- | --- |
| F1 | R1 | `verified_fixed` | P1 applied; `git diff 94aa213 HEAD -- docs/superpowers/specs/2026-10-16-stable-translog-design.md` shows sections 2.2 and 2.4; MT1: 3 tests OK |
| F2 | R2 | `verified_fixed` | `tests/test_stable_translog.py`: 26 passed, including `test_finite_tail_witness_and_limits` (1e80 -> 9.0 at rtol 3e-14) and `test_finite_power_overflow_neighborhood[3.0-4]` |
| F3 | R3 | `verified_fixed` | `tests/test_stable_translog_dispatch.py`: 3 passed; `tests/test_stable_translog_estimation.py`: 4 passed; alias mutant fails as recorded above |

## Exact execution record

All native runs on host `hmg-home`, Linux, CPU only, Python 3.14.6, JAX 0.10.2, float64
(`jax_enable_x64` set by skillmodels at import), pixi environment `tests-cpu`, branch
`stable-translog`. Commands were run by the implementing subagent or by the coordinator;
observed output is quoted from the terminal or from the saved log.

- [x] `pixi run -e tests-cpu python docs/superpowers/specs/stable-translog-audit/reference.py`
  (coordinator, 2026-10-16, before implementation) — observed: JSON with
  `fidelity {"1": 0.00611..., "2": 0.08618...}`, `fixed_point -0.01306310127854210308...`,
  `fixed_point_count 1`, `sup_0_through_200 3`, `correct_transition 9.000...`.
- [x] `pixi run -e tests-cpu python tests/stable_translog_reference.py` (subagent, Task 6
  Step 1) — observed: same certificate, saved in
  `docs/superpowers/specs/2026-10-16-stable-translog-native-closure.log`
  (`"fixed_point_count": 1`).
- [x] RT1 integrated as `tests/test_stable_translog.py` (together with MT2):
  `zsh -ic "cap pixi run -e tests-cpu pytest tests/test_stable_translog.py tests/test_transition_functions.py -v -p no:cacheprovider"`
  — observed: `51 passed, 2 warnings in 15.29s` (26 new + 25 baseline). Before the
  implementation the same command gave `3 failed, 23 errors` (fixture assertion
  `callable(getattr(tf, 'stable_translog', None))`).
- [x] RT2 integrated as `tests/test_stable_translog_dispatch.py`:
  `pytest tests/test_transition_functions.py tests/test_stable_translog_dispatch.py tests/test_af_validate_production_leakage.py -v`
  — observed: `33 passed`. Before the `process_model` helper (see Project integration)
  `test_public_model_uses_clipped_values` failed with dags
  `InvalidFunctionArgumentsError: missing required arguments: scale, states`.
- [x] MT1: `pixi run --manifest-path pyproject.toml -e tests-cpu python audit_tests/test_reference_mutations.py`
  from the reply package directory (coordinator, 2026-10-16) — observed: `Ran 3 tests in
  0.050s`, `OK` (`test_clip_symmetry_monotonicity_scaling_and_adjacent_boundary`,
  `test_default_budget_mutation_is_detected`, `test_stability_families_and_false_gain`).
- [x] MT2 integrated into `tests/test_stable_translog.py` (`test_finite_power_overflow_neighborhood[0.5-4|3.0-4|9.0-8]`)
  — observed: passed within the 26.
- [x] CHS same-law tests: `pytest tests/test_stable_translog_estimation.py -v` — observed:
  `4 passed in 20.0s`. Endogenous fixture at nonlinear = 0.1: stable `-8.835967036515843`,
  translog `-8.803603893696966`, linear `-9.236432508580393`; at nonlinear = 0 stable
  equals linear to 1e-10.
- [x] Closure log (all four modules together, Task 6 Step 1) — observed:
  `59 passed, 26 warnings in 33.65s`.
- [x] Full suite (Task 5): `zsh -ic "cap /usr/bin/time -v pixi run -e tests-cpu pytest tests -m 'not long_running' -v -p no:cacheprovider -n 2 --durations=20"`
  — observed: `871 passed, 282 warnings in 1182.07s`, wall 1185 s, max RSS 6.24 GiB.
- [x] `pixi run prek run --all-files` — observed: all hooks pass (ruff, ty, yaml, docs
  checks). `pixi run build-docs` — observed: exit 0, no warning for
  `docs/reference_guides/transition_functions.md`.

## Class-level acceptance evidence

R2 (the only code-level blocking class):

- Historical witness reproduced before repair: the planned naive expression
  `x / (1 + |x/3|**4)**(1/4)` at x = 1e80 gives 0 in IEEE float64 (static prediction by the
  reviewer, OR1 `overflow_witness`); the repaired helper gives `9.0` for params [0, 1, 0]
  at rtol 3e-14 (`test_finite_tail_witness_and_limits`).
- Independent oracle after repair: OR1 (90-digit decimal, original formula). Maximum
  observed discrepancy within the test tolerances: values rtol 4e-13 / atol 4e-13 on random
  states in [-6, 6] for n = 1..4, rtol 3e-14 on tails and defaults, rtol 8e-14 on the
  overflow-threshold neighbourhood.
- Boundary cases: 0, -0, `+/-nextafter(3)`, `+/-3`, `+/-50`, `+/-1e76`, `+/-1e80`,
  `+/-inf`, NaN, overflow threshold and its neighbours, isolated unit coefficient vectors,
  factor permutations.
- Generated mutation/neighbourhood suite: MT2, 3 (scale, power) pairs x 13 magnitudes x 2
  signs, values + gradients + finiteness + monotonicity; all pass.
- Transformations: `jit`, `vmap`, `lax.scan` (200 steps), `grad` (states and params),
  `static_argnames=("scale", "power")` for non-default configuration, scaling identity
  g(a x; a s) = a g(x; s).
- Level effect after repair: none inside the data region (the helper is algebraically the
  original formula); in the tail the nonlinear term saturates at scale**2 instead of
  silently vanishing.
- Decision/estimand effect: none beyond the documented model change of the function
  itself; CHS likelihood contributions and gradients equal the same-law oracle to 1e-10 /
  1e-8.
- Fail-loud behaviour: `ValueError` for non-positive or non-finite `scale` and for
  `power` not an even integer >= 2; beartype rejects a float `power` before the body runs
  (tested).

R3: covered by the dispositions above (default-budget pin, dispatch pin, same-law CHS
comparison, two mutants demonstrated).

R1: text only; MT1 and OR1 `stability_witnesses()` pass; no project code claims the
refuted bound.

## Project integration

- Targeted tests: 59 passed (closure log).
- Relevant project suite: 871 passed, 0 failed (full `not long_running` suite).
- Linters/type checks/docs: `prek run --all-files` clean; `build-docs` exit 0.
- Performance comparison: unit module base 5.27 s / 0.497 GiB versus head 5.24 s /
  0.498 GiB (fresh process, zero warmups, single repetition), inside the 30 s / 4 GiB
  profile budget and inside the 10 % regression triggers. Full suite 1182 s versus the
  contract's 900 s `ci.max_wall_seconds`, and the two CHS reference tests take about 10 s
  each versus the contract note of under 5 s for new tests. **Maintainer decision
  (Hans-Martin von Gaudecker, 2026-10-16): both breaches are waived.** Rationale: the full
  suite is dominated by three pre-existing tests in `tests/test_chs_estimate.py` at
  roughly 250 s, 233 s and 233 s, which the branch does not touch; no base full-suite run
  was made. Peak memory 6.24 GiB is inside the 8 GiB budget. A follow-up PR will shard
  the CI run (the three long tests each on their own runner plus the rest).

Scope extension beyond the plan's "no estimator code" constraint, declared here: commit
`be82adc` adds `_hide_keyword_only_config` (15 lines) to
`src/skillmodels/common/process_model.py`. dags exposes keyword-only arguments of a
string-named built-in transition as inputs of the transition graph, so `"stable_translog"`
could not be processed at all. The helper binds keyword-only arguments at their defaults
for string built-ins only; `translog`, `robust_translog`, the constraints and every
custom-callable path are untouched. RT2 and the CHS tests exercise it.

Other deviations from the plan, all mechanical: `power` hint stays `int` and the invalid-
power test also accepts `BeartypeCallHintParamViolation`; the oracle and the copied
reviewer sources are lint-exempted via `pyproject.toml` per-file ignores and a ty exclude
instead of in-file `noqa`; `.pre-commit-config.yaml` excludes `docs/superpowers/` so the
audit artefacts are committed byte-identical; the plan's same-law reference callable uses
explicit factor arguments because dags cannot feed `**factors`; the endogenous fixture is
rebuilt with `dataclasses.replace` because `with_transition_functions` rejects a
single-factor mapping.

## Final changeset

The bundle's `project/.git` carries the exact diff: `git diff 94aa213...HEAD` on branch
`stable-translog`. 21 files changed, 2722 insertions, 8 deletions before this record was
added. Source files: `src/skillmodels/common/transition_functions.py` (+92),
`src/skillmodels/common/process_model.py` (+19 -2), `src/skillmodels/af/validate.py` (+2),
`src/skillmodels/amn/simulate_and_regress.py` (+9 -2), `pyproject.toml` (+7),
`.pre-commit-config.yaml`; tests: `tests/test_stable_translog.py`,
`tests/test_stable_translog_dispatch.py`, `tests/test_stable_translog_estimation.py`,
`tests/stable_translog_reference.py`, `tests/test_transition_functions.py` (+10); docs:
`docs/reference_guides/transition_functions.md`, `docs/superpowers/**`.

## Remaining blocker

`none`
