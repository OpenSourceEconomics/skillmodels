# `stable_translog` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a translog transition function whose squares and interactions are evaluated on smoothly clipped states, so that the base nonlinear remainder is bounded and the stated conditional drift theorem applies, and register it everywhere skillmodels enumerates built-in transition names.

**Architecture:** One new function plus its `params_` and `identity_constraints_` companions in `src/skillmodels/common/transition_functions.py`, following the `robust_translog` pattern of delegating names and constraints to the translog helpers. Three allow-lists learn the new name. Tests pin equality with the translog near zero, boundedness far from zero, and a 200-period iteration with real estimated coefficients. One docs section.

**Tech Stack:** Python 3.14, JAX (`jax.numpy`), pytest, pixi environment `tests-cpu`. All commands run from the skillmodels repository root `/home/hmg/econ/skillmodels-applications/skillmodels` and go through pixi (never bare `python` or `pytest`). Use `zsh -ic "cap pixi run -e tests-cpu pytest ... -v"` for test runs on the local Linux boxes.

**Spec:** `docs/superpowers/specs/2026-10-16-stable-translog-design.md` (same repository). Read sections 2 and 3 before starting; section 2.2 states exactly what the function guarantees.

**Audit:** round-0 Pro audit (`~/sciebo/pro-audits/skillmodels-stable-translog/bundle-round0/result/`) returned `serious_gap`: F1 theorem scope (text, applied via P1), F2 finite-tail overflow of the naive clip (fixed in Task 1 Step 4), F3 non-discriminating tests (replaced by the reviewer suites RT1/RT2/MT2 in Tasks 1-3). Reviewer test sources and the decimal oracle live in `docs/superpowers/specs/stable-translog-audit/`. Task 6 produces the closure evidence.

## Global Constraints

- Function signature must be `(states: Array, params: Array) -> Array`; extra arguments only as keywords with defaults (`scale: float = 3.0`, `power: int = 4`). String lookup via `getattr(transition_functions, "stable_translog")` must work.
- Parameter layout identical to `translog`: `[linear (nfac), squares (nfac), interactions (nfac choose 2 in `itertools.combinations` order), constant]`.
- No new dependencies. No changes to `translog`, `robust_translog`, or any estimator code.
- Code style: the repo runs `prek` (ruff, ty). Type hints with `Array` from `jax` as the file actually does. Docstrings in the file's existing style.
- Commits on a new branch `stable-translog` created from `94aa213` (branch `missing-observed-factors` is the current checkout; do not commit there). Each task ends with a commit.

## Review Focus

Inputs or conditions the spec implies but which are easy to get wrong; each has a pinning test in the task named.

1. `power` odd or non-integer: `|x/s|^p` is used, so odd p still yields an odd g, but p = 1 is not C-infinity at zero (although g itself has a first derivative there). Pin: Task 1 test that `power=4` default is used and that the docstring states even p; reject values outside the documented even-integer domain; also validate positive finite scale (Task 1).
2. `states` containing `inf` or `nan` (e.g. from an upstream overflow): g(inf) must be ±s, not nan. `x / (1 + |x/s|^p)^(1/p)` is inf/inf = nan at x = inf. Pin: Task 1 test with `jnp.inf` input expecting a finite nonlinear contribution; implementation uses `jnp.where(jnp.isfinite(x), ..., s * jnp.sign(x))`.
3. Gradients: the even-power mathematical clip is smooth. Verify the implemented derivative independently at zero, around both clip knees, and in finite tails, including identity coefficients; do not rely on a convention for the derivative of abs at zero (Task 1).
4. Mixed latent and observed factors: observed factors (inputs) are part of `states` and are also clipped inside nonlinear terms. For an input with |value| > scale (e.g. unscaled wealth) this silently changes the estimand. Pin: Task 4 docs state it; Task 1 test shows an observed input at 50 is clipped to below scale in the interaction term, so users see the behaviour.
5. Identity period: the first-period identity transition must be exact. Fixing squares and interactions at 0 and the own linear coefficient at 1 makes the output equal to the state regardless of g. Pin: Task 1 test that with identity-constraint values the function returns the own state exactly for states far beyond scale.

---

### Task 1: The function, its companions, and unit tests

**Files:**
- Modify: `src/skillmodels/common/transition_functions.py` (append after `identity_constraints_robust_translog`, line 383)
- Test: `tests/test_stable_translog.py` (new; from `docs/superpowers/specs/stable-translog-audit/test_kernel.py` + `test_tail_mutations.py`)
- Create: `tests/stable_translog_reference.py` (copy of `docs/superpowers/specs/stable-translog-audit/reference.py`, independent decimal oracle)

**Interfaces:**
- Consumes: `translog`, `params_translog`, `identity_constraints_translog`, `FixedConstraintWithValue`, `combinations`, `jnp`, `Array` (all already imported in the module).
- Produces: `stable_translog(states, params, scale=3.0, power=4) -> Array`, `params_stable_translog(factors) -> list[str]`, `identity_constraints_stable_translog(factor, aug_period, all_factors) -> list[FixedConstraintWithValue]`, `_soft_clip(x, scale, power) -> Array`.

- [ ] **Step 1: Create the branch**

```bash
cd /home/hmg/econ/skillmodels-applications/skillmodels
git stash list   # note: .ai-instructions is modified in the working tree; leave it alone
git switch -c stable-translog 94aa213
```

- [ ] **Step 2: Write the failing tests**

The tests are the reviewer's native acceptance suite (RT1) and tail generators (MT2) from
the round-0 audit, checked against an independent 90-digit `decimal` oracle. Copy them in
rather than retyping:

```bash
cd /home/hmg/econ/skillmodels-applications/skillmodels
cp docs/superpowers/specs/stable-translog-audit/reference.py tests/stable_translog_reference.py
{ cat docs/superpowers/specs/stable-translog-audit/test_kernel.py
  sed -n '/^@pytest.mark.parametrize/,$p' docs/superpowers/specs/stable-translog-audit/test_tail_mutations.py
} > tests/test_stable_translog.py
```

Then edit `tests/test_stable_translog.py` by hand:

1. Delete the two `if sys.version_info[:2] != (3, 14): raise RuntimeError(...)` lines and the
   `sys.path.insert(0, ...)` line (pytest prepends `tests/` to `sys.path` because the
   directory has no `__init__.py`).
2. Replace `from audit_oracles.reference import clip, clip_derivative, transition` with
   `from stable_translog_reference import clip, clip_derivative, transition`.
3. Replace the module docstring with
   `"""stable_translog against an independent decimal oracle (round-0 audit RT1 + MT2)."""`.
4. Append this test (Review Focus 4, observed input beyond scale):

```python
def test_observed_input_beyond_scale_is_clipped_in_interaction(api) -> None:
    f = api[0]
    # one latent at 1.0, one observed input at 50: only the interaction coefficient is on
    params = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    value = float(f(jnp.array([1.0, 50.0]), params))
    assert_allclose(value, float(clip(1.0) * clip(50.0)), rtol=3e-14)
    assert value < 3.0  # translog would give 50
```

`tests/stable_translog_reference.py` is stdlib only; keep its `main()` so
`pixi run -e tests-cpu python tests/stable_translog_reference.py` reprints the certificate
JSON (fixed-point count 1, fidelity 0.0061 / 0.0862, overflow witness). Add
`# ruff: noqa` as its second line; it is reviewer-authored reference code, not house style.

The suite covers: finite tails through 1e80 and ±inf/NaN semantics; every coefficient
position in isolation for n = 1..4 and simultaneous factor/coefficient permutation; the
default fidelity budgets at |x| = 1 and 2 (kills a `scale=2.5` mutant); all 200 iterates
and the fixed point −0.013063101278542104 of the run E psychological map; analytical
gradients at zero, both clip knees and their neighbouring floats, and finite tails;
JIT/vmap/scan with static non-default `scale`/`power`; the scaling identity
g(a·x; a·s) = a·g(x; s); name and identity-constraint companions; loud `ValueError` for
invalid `scale` and `power`; and the float64 overflow threshold s·max^(1/p) with its
neighbouring floats for three (scale, power) pairs.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `zsh -ic "cap pixi run -e tests-cpu pytest tests/test_stable_translog.py -v -p no:cacheprovider"`
Expected: collection error `ImportError: cannot import name 'stable_translog'`.

- [ ] **Step 4: Implement**

Add the standard-library `import math`, then append to `src/skillmodels/common/transition_functions.py` after `identity_constraints_robust_translog`:

```python
def _soft_clip(x: Array, scale: float, power: int) -> Array:
    """Smooth odd map; in real arithmetic |g(x)| < scale for finite x.

    Floating-point results may round to +/-scale.

    g(x) = x / (1 + |x / scale| ** power) ** (1 / power). The helper maps +/-infinity to +/-scale and preserves NaN.
    The full transition does not repair nonfinite linear inputs.
    """
    # Validate configuration in stable_translog; every evaluated branch is safe.
    safe_x = jnp.where(jnp.isfinite(x), x, 0.0)
    magnitude = jnp.abs(safe_x)
    inside = magnitude <= scale
    # Match this tie branch to the numerator branch for the correct derivative at |x|=scale.
    m = jnp.where(inside, scale, magnitude)
    unit_x = safe_x / m
    unit_s = scale / m
    denominator = (unit_s**power + unit_x**power) ** (1.0 / power)
    # Keeping x itself in the small branch avoids losing subnormals in x / scale.
    numerator = jnp.where(inside, safe_x, scale * unit_x)
    clipped = numerator / denominator
    return jnp.where(jnp.isfinite(x), clipped, scale * jnp.sign(x))


def stable_translog(
    states: Array,
    params: Array,
    *,
    scale: float = 3.0,
    power: int = 4,
) -> Array:
    """Translog whose squares and interactions are evaluated on soft-clipped states.

    f = c + Σ_j β_j s_j + Σ_j γ_j g(s_j)² + Σ_{a<b} δ_ab g(s_a) g(s_b), with
    g(x) = x / (1 + |x/scale|**power)**(1/power).

    The linear part is untouched and each nonlinear basis term is bounded by scale².
    The deterministic fixed-coefficient law has bounded iterates when the complete
    linear matrix is Schur-stable and all remaining forcing is uniformly bounded.
    This does not certify switched dynamics, control-function corrections, Gaussian
    sample paths or UKF stability; see design section 2.2. Each nonlinear basis term
    has <1% relative error for coordinates within +/-1 at the defaults, not necessarily
    a cancelling sum or a likelihood. Observed factors are clipped in nonlinear terms.

    Unlike robust_translog, this leaves the linear dynamics exact. A saturation
    `scale` beyond the translog's own unstable fixed point can leave a second, spurious
    stable fixed point in the tail; check the fixed points of estimated equations
    before using the function as a long-horizon law of motion. `power` must be an even
    integer ≥ 2 for a smooth g.

    Parameter layout is that of `translog`.

    """
    if type(power) is not int or power < 2 or power % 2:
        msg = f"power must be an even integer >= 2, got {power}."
        raise ValueError(msg)
    if not math.isfinite(scale) or scale <= 0:
        msg = f"scale must be positive and finite, got {scale}."
        raise ValueError(msg)
    nfac = len(states)
    constant = params[-1]
    lin_beta = params[:nfac]
    square_beta = params[nfac : 2 * nfac]
    inter_beta = params[2 * nfac : -1]

    g = _soft_clip(states, scale=scale, power=power)
    res = jnp.dot(states, lin_beta)
    res += jnp.dot(g**2, square_beta)
    for p, (a, b) in zip(inter_beta, combinations(range(nfac), 2), strict=False):
        res += p * g[a] * g[b]
    res += constant
    return res


def params_stable_translog(factors: tuple[str, ...]) -> list[str]:
    """Parameter names for stable_translog (same as translog)."""
    return params_translog(factors)


def identity_constraints_stable_translog(
    factor: str,
    aug_period: int,
    all_factors: tuple[str, ...],
) -> list[FixedConstraintWithValue]:
    """Identity constraints for stable_translog (same as translog)."""
    return identity_constraints_translog(
        factor=factor, aug_period=aug_period, all_factors=all_factors
    )
```

Configuration must be closed over or explicitly static under JIT. Default string lookup is compatible; an explicitly passed Python int is not automatically static. Test `jax.jit(stable_translog, static_argnames=("scale", "power"))` for nondefaults.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `zsh -ic "cap pixi run -e tests-cpu pytest tests/test_stable_translog.py tests/test_transition_functions.py -v -p no:cacheprovider"`
Expected: baseline tests and all new tests pass. Integrate the reviewer kernel
acceptance suite (RT1) before treating this step as green: actual default basis errors
at 1 and 2; direct oracle values rather than subtracting a huge linear term; finite
tails through 1e80; oddness/monotonicity; every coefficient position; simultaneous
factor permutations; every state in the 200-step path; zero/knee/tail analytical
gradients; JIT, vmap, scan, and static nondefault settings. A strict real-arithmetic
`|g|<scale` is not a finite-precision saturation test. Check the default root count
using the independent structural proof (OR1), not only this single trajectory.
Kill a changed-default-scale=2.5 mutant: its nonlinear error at 2 is about 15.773%,
while the originally proposed small-state test still passes. Do not weaken a failing
assertion or convert absence of the new implementation into a skipped test.

- [ ] **Step 6: Lint and type-check**

Run: `pixi run prek run --files src/skillmodels/common/transition_functions.py tests/test_stable_translog.py tests/stable_translog_reference.py`
Expected: clean. If ruff flags `PLR0913` for the four-argument signature, add `# noqa: PLR0913` on the `def` line (the repo already does this elsewhere; check with `grep -n PLR0913 src -r | head -3`).

- [ ] **Step 7: Commit**

```bash
git add src/skillmodels/common/transition_functions.py tests/test_stable_translog.py tests/stable_translog_reference.py
git commit -m "Add stable_translog: translog with soft-clipped nonlinear terms

Squares and interactions are evaluated on g(x) = x / (1 + |x/s|^p)^(1/p),
so the base nonlinear remainder is bounded; full-law conditions are documented.
Parameter layout, names and identity constraints are those of translog.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017psXwkMuDeVmTu5WgoM26k"
```

---

### Task 2: Register the name in the three allow-lists

**Files:**
- Modify: `src/skillmodels/af/validate.py:15-45` (`_AF_COMPATIBLE_TRANSITIONS`, `_LEAKY_BUILTIN_PRODUCTION`)
- Modify: `src/skillmodels/amn/simulate_and_regress.py:254-262` (`builtin_names`) and the docstring lines 9, 359, 473 that list built-in names
- Test: `tests/test_transition_functions.py`
- Create: `tests/test_stable_translog_dispatch.py` (from `docs/superpowers/specs/stable-translog-audit/test_dispatch.py`)
**Interfaces:**
- Consumes: the name string `"stable_translog"` from Task 1.
- Produces: nothing new; AF validation accepts the name, AF leakage warning covers it, AMN stage 3 resolves it to the callable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transition_functions.py`:

```python
def test_stable_translog_is_registered_in_allow_lists() -> None:
    from skillmodels.af.validate import (
        _AF_COMPATIBLE_TRANSITIONS,
        _LEAKY_BUILTIN_PRODUCTION,
    )

    assert "stable_translog" in _AF_COMPATIBLE_TRANSITIONS
    assert "stable_translog" in _LEAKY_BUILTIN_PRODUCTION
```

Create `tests/test_stable_translog_dispatch.py` from the reviewer's RT2:

```bash
cp docs/superpowers/specs/stable-translog-audit/test_dispatch.py tests/test_stable_translog_dispatch.py
```

and delete its `if sys.version_info[:2]!=(3,14): raise RuntimeError(...)` lines and the
`import sys`. It builds a real one-factor `ModelSpec`, runs `process_model`, and asserts
that `processed.transition_info.func` and `_resolve_transition_callable("stable_translog", ...)`
return the clipped law (9/√2 at x = 3 with params [0, 1, 0]; translog would give 9), and
that `validate_af_model` accepts the name. Source-string inspection is not an oracle.

- [ ] **Step 2: Run them to verify they fail**

Run: `zsh -ic "cap pixi run -e tests-cpu pytest tests/test_transition_functions.py tests/test_stable_translog_dispatch.py -v -p no:cacheprovider -k 'registered or stable_translog_dispatch or clipped or resolver or validation'"`
Expected: missing registration or an incorrect dispatched law fails the relevant test.

- [ ] **Step 3: Edit the allow-lists**

In `src/skillmodels/af/validate.py`, add `"stable_translog",` directly after each `"robust_translog",` line (lines 20 and 42). In `src/skillmodels/amn/simulate_and_regress.py`, add `"stable_translog",` after `"robust_translog",` in `builtin_names` (line 257) and append `stable_translog` to the three docstring enumerations at lines 9, 359 and 473 (e.g. `translog, robust_translog, stable_translog, linear_and_squares, ...`).

- [ ] **Step 4: Run the tests**

Run: `zsh -ic "cap pixi run -e tests-cpu pytest tests/test_transition_functions.py tests/test_stable_translog_dispatch.py tests/test_af_validate_production_leakage.py -v -p no:cacheprovider"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/skillmodels/af/validate.py src/skillmodels/amn/simulate_and_regress.py tests/test_transition_functions.py tests/test_stable_translog_dispatch.py
git commit -m "Register stable_translog in AF and AMN built-in name lists

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017psXwkMuDeVmTu5WgoM26k"
```

---

### Task 3: End-to-end smoke test through the CHS estimator

**Files:**
- Test: `tests/test_stable_translog_estimation.py` (new)

**Interfaces:**
- Consumes: the CHS fixture of `tests/test_missing_observed_factors.py` (`_model`, `_data`, `_params`, `_loglikeobs`, `_endogenous_model`, `_endogenous_data`, `MOMENTS`, `OPTIONS`; importable as a sibling module because `tests/` has no `__init__.py`), `get_maximization_inputs` (keys `loglikeobs`, `loglike_and_gradient`, `params_template`), `register_params`, `params_translog`.
- Produces: proof that the string resolves through `process_model`, the params index, the constraints, the unscented predict step with the fixture's actual sigma points and weights (including the missing-observed-factor integration path), and the likelihood and its gradient, and that the dispatched law is the clipped one.

Same-law oracle: a `@register_params` callable that evaluates the original formula
x / (1 + |x/3|⁴)^(1/4) with `jnp` on the bounded fixture states. It shares no code with
`stable_translog` or `_soft_clip` and goes through the custom-callable interface, so a
dispatcher that aliases translog fails by a numerical margin rather than passing with zero
discrepancy. No stable-versus-translog likelihood budget is asserted anywhere.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stable_translog_estimation.py`:

```python
"""stable_translog through the CHS likelihood, against an independent same-law oracle.

Fixture: the two-latent, two-observed, three-period panel of
tests/test_missing_observed_factors.py (with missing observed-factor cells, so the
sigma-point integration over N(m, v) fills is exercised too) and its endogenous
two-period model with carry-forward substeps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_missing_observed_factors import (
    MOMENTS,
    OPTIONS,
    _data,
    _endogenous_data,
    _endogenous_model,
    _loglikeobs,
    _model,
    _params,
)

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.common.decorators import register_params
from skillmodels.common.transition_functions import params_translog

jax.config.update("jax_enable_x64", True)


def _g(x):
    return x / (1.0 + jnp.abs(x / 3.0) ** 4) ** 0.25


def _reference_law_for(all_factors: tuple[str, ...]):
    """Original-formula translog-on-clipped-states as a registered custom callable."""
    names = params_translog(all_factors)

    @register_params(params=names)
    def reference_law(params, **factors):
        x = [factors[f] for f in all_factors]
        g = [_g(v) for v in x]
        out = params["constant"]
        for f, v in zip(all_factors, x, strict=True):
            out = out + params[f] * v
        for f, gv in zip(all_factors, g, strict=True):
            out = out + params[f"{f} ** 2"] * gv**2
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                out = out + params[f"{all_factors[i]} * {all_factors[j]}"] * g[i] * g[j]
        return out

    return reference_law


REFERENCE = _reference_law_for(("fac1", "fac2", "ob1", "ob2"))


def _loglikeobs_and_grad(transition, data, nonlinear):
    inputs = get_maximization_inputs(_model(2, transition, MOMENTS), data, chs_options=OPTIONS)
    params = _params(inputs["params_template"], nonlinear=nonlinear)
    _, grad = inputs["loglike_and_gradient"](params)
    return np.asarray(inputs["loglikeobs"](params)), np.asarray(grad)


def test_zero_nonlinear_params_reproduce_linear_filter() -> None:
    data = _data()
    linear = _loglikeobs(_model(2, "linear", MOMENTS), data)
    stable = _loglikeobs(_model(2, "stable_translog", MOMENTS), data)
    assert_allclose(stable, linear, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("ob_scale", [1.0, 10.0], ids=["inputs_inside_scale", "inputs_beyond_scale"])
def test_contributions_and_gradient_match_same_law_reference(ob_scale) -> None:
    data = _data()
    data[["ob1", "ob2"]] *= ob_scale
    stable, g_stable = _loglikeobs_and_grad("stable_translog", data, nonlinear=0.05)
    reference, g_reference = _loglikeobs_and_grad(REFERENCE, data, nonlinear=0.05)
    translog, _ = _loglikeobs_and_grad("translog", data, nonlinear=0.05)
    assert np.all(np.isfinite(stable)) and np.all(np.isfinite(g_stable))
    assert_allclose(stable, reference, rtol=1e-10, atol=1e-10)
    assert_allclose(g_stable, g_reference, rtol=1e-8, atol=1e-10)
    # The sigma points reach the nonlinear region: an alias to translog must fail here.
    assert np.max(np.abs(stable - translog)) > 1e-6


def _endogenous_loglike(transition, nonlinear: float) -> float:
    model = _endogenous_model(None)
    model = model.with_transition_functions({"s": transition})
    inputs = get_maximization_inputs(model, _endogenous_data(), chs_options=OPTIONS)
    params = inputs["params_template"].copy()
    cat = params.index.get_level_values("category")
    name2 = params.index.get_level_values(3).astype(str)
    params["value"] = np.where(cat.isin(["meas_sds", "shock_sds", "loadings"]), 0.5, 0.1)
    params.loc[(cat == "transition") & name2.str.contains(r"\*"), "value"] = nonlinear
    for idx in params.query("category == 'initial_cholcovs'").index:
        row, col = idx[3].split("-")
        params.loc[idx, "value"] = float(row == col)
    return float(inputs["loglike"](params))


def test_endogenous_carry_forward_substeps_accept_stable_translog() -> None:
    """Carry-forward substeps keep their identity/zero-kappa constraints; the law is finite."""
    linear = _endogenous_loglike("linear", nonlinear=0.0)
    assert_allclose(_endogenous_loglike("stable_translog", nonlinear=0.0), linear, rtol=1e-10)
    stable = _endogenous_loglike("stable_translog", nonlinear=0.1)
    assert np.isfinite(stable)
    assert abs(stable - _endogenous_loglike("translog", nonlinear=0.1)) > 1e-8
```

If `ModelSpec.with_transition_functions` rejects a mapping for a single factor, use
`dataclasses.replace(model.factors["s"], transition_function=transition)` and rebuild the
`factors` dict as `tests/test_constraints.py::test_get_constraints_skips_custom_target_transition`
does. If the endogenous model's identity-constraint pass fixes nonlinear parameters to
zero on carry-forward substeps so that `stable` and `translog` coincide to 1e-8 in the last
assertion, lower the threshold is NOT the fix: print both values; if they are identical,
the nonlinear coefficient on `s` never enters the law and the fixture needs a measurement
in period 1 for `inv` so the transition is estimated, not pinned.
- [ ] **Step 2: Run it to verify it fails**

Run: `zsh -ic "cap pixi run -e tests-cpu pytest tests/test_stable_translog_estimation.py -v -p no:cacheprovider"`
Expected: after implementing the real fixture, a missing/wrong dispatcher and a translog-alias mutation fail for numerical reasons, not for fictitious imports.

- [ ] **Step 3: Run it to verify it passes**

Same command. Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_stable_translog_estimation.py
git commit -m "Smoke-test stable_translog through the CHS likelihood

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017psXwkMuDeVmTu5WgoM26k"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/reference_guides/transition_functions.md` (insert after the `### robust_translog` section, line 69-72; also add the name to the two enumerations at lines 143 and 159)

- [ ] **Step 1: Insert the section**

```markdown
### stable_translog

Like `translog`, but every square and interaction is evaluated on a smoothly clipped
state, while the linear part is untouched:

$$
f_{t+1} = \sum_j \beta_j s_j + \sum_j \gamma_j g(s_j)^2 + \sum_{a<b} \delta_{ab} g(s_a) g(s_b) + c,
\qquad g(x) = \frac{x}{\left(1 + |x/\text{scale}|^{\text{power}}\right)^{1/\text{power}}}
$$

with defaults `scale=3.0` and `power=4`. Near zero $g(x) \approx x$ and the function
agrees with `translog` (within 1 % of the nonlinear terms at $|x| \le 1$). Every
nonlinear basis term is bounded by $\text{scale}^2$. For a deterministic fixed-matrix
law with a Schur-stable COMPLETE linear part and uniformly bounded remaining forcing,
iterates are bounded. This is not a certificate for changing matrices, control-function
corrections, Gaussian sample paths or the UKF. The scalar application examples have a
repelling translog fixed point, but a negative square coefficient alone is not enough
to establish that root structure.

Use `stable_translog` when the estimated technology is iterated for many periods
(simulations, laws of motion in life-cycle models), subject to those conditions.
`robust_translog` clips the whole input and hence also bounds deterministic output for
fixed coefficients, but can settle at enormous artificial levels and alters linear
dynamics. Neither name certifies the complete estimator or prevents all arithmetic overflow.

Two caveats. Observed factors in the state vector are clipped too; keep their scale
comparable to the latent factors or fix their nonlinear coefficients. And a `scale`
beyond the translog's own unstable fixed point can leave a second stable fixed point in
the tail; check the fixed points of your estimated equations before simulating. Other
values of `scale` and `power` work in direct calls. Estimator registration requires
complete parameter/identity companions and allow-lists for a new module-level string,
or a correctly registered factor-argument callable; a one-line vector wrapper alone
is insufficient. The string lookup uses defaults. Scale is measured in the declared
factor units, not automatically in standard deviations. Preserving a change of units
requires transforming the clipping scale too. Sigma-point and identification caveats
are stated in design section 2.4.

**Parameters**: identical to `translog`.
```

Then add `stable_translog` to the lists at the two places that enumerate built-in production functions (`linear`, `robust_translog`, ...), and to `_LEAKY_BUILTIN_PRODUCTION`'s description.

- [ ] **Step 2: Build the docs locally**

Run: `pixi run build-docs`
Expected: builds without a new warning for `transition_functions.md`. Open `docs/_build/html/reference_guides/transition_functions.html` and check the formula renders.

- [ ] **Step 3: Commit**

```bash
git add docs/reference_guides/transition_functions.md
git commit -m "Document stable_translog

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017psXwkMuDeVmTu5WgoM26k"
```

---

### Task 5: Full suite, push, PR

- [ ] **Step 1: Run the full non-long-running suite**

Check for neighbours first: `systemctl --user status cap.slice`. Then
Run: `zsh -ic "cap pixi run -e tests-cpu pytest tests -m 'not long_running' -v -p no:cacheprovider -n 2"`
Expected: everything passes; preserve the actual command, revision, Python/JAX
versions, dtype, total session time, pytest call durations and Linux ru_maxrss. The
profile's JAX environment must agree with pyproject.toml (>=0.9). Native evidence is
not supplied by this review. Measure base versus implemented head on the same machine:
unit module <=30 seconds and <=4 GiB, new tests <5 seconds, full CI <=900 seconds and
<=8 GiB; warm/RSS repair triggers are 10%. Clarify the contract's ambiguous "after the
first call" wording versus zero warmups before certifying a warm-runtime comparison.
Do not label the supplied 2.47-second baseline pytest session as measured warm-kernel
time or infer missing peak memory. The normalized helper adds bounded O(n) arithmetic;
keep high-precision or exhaustive references out of the production hot path.

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin stable-translog
gh pr create --title "Add stable_translog transition function" --body "$(cat <<'EOF'
Translog whose squares and interactions are evaluated on soft-clipped states
g(x) = x / (1 + |x/s|^p)^(1/p), linear part untouched. Iterated transitions stay
bounded under the fixed-matrix, bounded-forcing theorem in the revised design; this
is not a general certificate for a composed stochastic model.

Design note: docs/superpowers/specs/2026-10-16-stable-translog-design.md
Plan: docs/superpowers/plans/2026-10-16-stable-translog.md

Motivation: in the health-cognition application the estimated psychological-capacity
translog has an unstable fixed point at -2.2, inside the data range; forward
predictions without measurements can diverge. The supplied diagnostic claims do
not establish that every likelihood value or gradient is unaffected; estimation and
simulation with the changed law must be checked separately.

Tests: unit tests for clipping, boundedness, identity, JIT/grad, a 200-period
iteration with the estimated coefficients, allow-list registration, and a CHS
likelihood smoke test.

https://claude.ai/code/session_017psXwkMuDeVmTu5WgoM26k
EOF
)"
```

---

### Task 6: Closure evidence for the audit

**Files:**
- Create: `docs/superpowers/specs/2026-10-16-stable-translog-native-closure.log`
- Create: `docs/superpowers/specs/2026-10-16-stable-translog-native-resource-report.json`
- Create: `docs/superpowers/specs/2026-10-16-stable-translog-completion-record.md` (from `~/sciebo/pro-audits/skillmodels-stable-translog/bundle-round0/result/CLAUDE-COMPLETION-RECORD.template.md`)

- [ ] **Step 1: Native closure log**

```bash
cd /home/hmg/econ/skillmodels-applications/skillmodels
{ git rev-parse HEAD; pixi run -e tests-cpu python -c "import sys, jax; print(sys.version); print('jax', jax.__version__)"
  pixi run -e tests-cpu python tests/stable_translog_reference.py
  zsh -ic "cap pixi run -e tests-cpu pytest tests/test_stable_translog.py tests/test_stable_translog_dispatch.py tests/test_stable_translog_estimation.py tests/test_transition_functions.py -v -p no:cacheprovider --durations=0"
} 2>&1 | tee docs/superpowers/specs/2026-10-16-stable-translog-native-closure.log
```

Expected: every test passes; the oracle JSON reports `fixed_point_count: 1`.

- [ ] **Step 2: Resource report, base versus head**

Run the unit module on `94aa213` (base) and on the branch head in fresh processes and
record wall time and `ru_maxrss`:

```bash
for ref in 94aa213 HEAD; do
  git stash -q 2>/dev/null; git switch -q --detach $ref
  pixi run -e tests-cpu python - <<'PY'
import json, resource, subprocess, sys, time
t = time.perf_counter()
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_transition_functions.py", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True)
print(json.dumps({"ref": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
                  "wall_seconds": round(time.perf_counter() - t, 2),
                  "ru_maxrss_children_gib": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 3),
                  "returncode": r.returncode}))
PY
done | tee docs/superpowers/specs/2026-10-16-stable-translog-native-resource-report.json
git switch -q stable-translog
```

(`pytest -q` inside the subprocess is the reviewer's measurement convention, not an
interactive run; the hook only guards the interactive command line.) Add a third object
for the full `-m 'not long_running' -n 2` suite from Task 5 with its total wall time.
Budgets from the contract: unit module <= 30 s and <= 4 GiB; new tests < 5 s; full CI
<= 900 s and <= 8 GiB; warm-runtime and RSS regression triggers 10 %. Record "warm" as
"fresh process, zero warmups, single repetition" so the contract's wording is pinned.

- [ ] **Step 3: Completion record and closure bundle**

Fill the template section by section (strategy, witness, oracle agreement, mutation
results, eager/JIT/vmap/scan paths, dispositions, exact commands, final diff) and build
the closure round:

```bash
python3 ~/.claude/skills/pro-comp-method-audit/scripts/prepare_prompt.py \
  --text docs/superpowers/specs/2026-10-16-stable-translog-design.md \
  --text docs/superpowers/plans/2026-10-16-stable-translog.md \
  --code src/skillmodels/common/transition_functions.py --code tests/test_stable_translog.py \
  --code tests/stable_translog_reference.py --code tests/test_stable_translog_dispatch.py \
  --code tests/test_stable_translog_estimation.py --code src/skillmodels/af/validate.py \
  --code src/skillmodels/amn/simulate_and_regress.py \
  --project-root . --workflow-class computational-method --output-detail compact \
  --scope-mode delta --profile default \
  --protocol-extra ~/.claude/skills/pro-comp-method-audit/references/numerical-defect-index.md \
  --profile-contract docs/superpowers/specs/2026-10-16-stable-translog-contract.yaml \
  --mode closure --round 1 \
  --prior ~/sciebo/pro-audits/skillmodels-stable-translog/bundle-round0/result/AUDIT-STATE.json \
  --evidence docs/superpowers/specs/2026-10-16-stable-translog-completion-record.md \
  --evidence docs/superpowers/specs/2026-10-16-stable-translog-native-closure.log \
  --evidence docs/superpowers/specs/2026-10-16-stable-translog-native-resource-report.json \
  --base-ref 94aa213 \
  --target "stable_translog design: translog transition function with soft-clipped nonlinear terms; verify the boundedness argument, the fidelity/uniqueness trade-off of the clip and its defaults (scale 3, power 4), the interaction with the unscented Kalman filter and identification, and the implementation plan, before code is written" \
  --out ~/sciebo/pro-audits/skillmodels-stable-translog
```

Verify with `verify_bundle.py`, upload to a fresh Pro chat, save the reply under
`bundle-round1/reply/`, then `ingest_report.py --merge` and `audit_gate.py` as in the
skill. Gate 0 closes the audit; the PR of Task 5 may be opened before that, but say so in
its description.

---

## Follow-ups outside this plan (health-cognition repository)

- Add a `stable` model variant (`transition_function="stable_translog"` for the four capacity states) and compare transition coefficients with run G's translog.
- A fifty-line script that, for each estimated equation and age group, solves for the fixed points of the own-state map (other states at zero and at ±1) and reports the number of stable fixed points; fail loudly if more than one.
- Re-run section 4.4 simulations with the stable variant; the state clip in the simulation code becomes unnecessary.
- pylcm law-of-motion export uses the stable variant.
