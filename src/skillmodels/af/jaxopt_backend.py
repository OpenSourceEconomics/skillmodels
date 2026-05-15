"""JAX-native optimizer backend for AF estimation via `jaxopt.LBFGSB`.

Runs L-BFGS-B directly on device, so the params vector never leaves
the accelerator between iterations. This avoids the host->device->host
roundtrip that occurs once per iteration when AF's likelihood is
called from optimagic.

The backend supports `FixedConstraintWithValue` (pinned-value rows)
plus parameter bounds. It does NOT support `ProbabilityConstraint`
or `EqualityConstraint`: those require constraint folding that
optimagic provides and that jaxopt's LBFGS-B does not. Models that
include log_ces transitions or cross-section equalities should use
`optimizer_backend="optimagic"`.
"""

import os

# Belt-and-suspenders for callers that import this module directly without
# going through `skillmodels/__init__.py`. Two things must be set before
# `import jax` / `from jaxopt import LBFGSB`:
#
# 1. `JAX_ENABLE_X64=1` — the AF pipeline assumes float64 throughout.
# 2. `XLA_FLAGS=--xla_disable_hlo_passes=permutation_sort_simplifier` —
#    works around a JAX 0.10 bug where the `argsort` inside
#    `LBFGSB.update` emits an s32 reduction accumulator into an s64
#    scatter operand, and XLA's `permutation_sort_simplifier` pass
#    rejects the mismatch. See `skillmodels/__init__.py` for the full
#    explanation.
os.environ.setdefault("JAX_ENABLE_X64", "1")

_xla_pass_disable = "--xla_disable_hlo_passes=permutation_sort_simplifier"  # noqa: S105
_existing_xla_flags = os.environ.get("XLA_FLAGS", "")
if _xla_pass_disable not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = f"{_existing_xla_flags} {_xla_pass_disable}".strip()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, cast  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optimagic as om  # noqa: E402
import pandas as pd  # noqa: E402
from jax import Array  # noqa: E402
from jaxopt import LBFGSB  # noqa: E402

from skillmodels.common.constraints import FixedConstraintWithValue  # noqa: E402


@dataclass(frozen=True)
class JaxoptResult:
    """Minimal stand-in for `optimagic.OptimizeResult`.

    Only carries the fields downstream AF code reads from the
    optimagic result: ``params`` (DataFrame with the optimized
    ``"value"`` column), ``fun`` (the minimized negative
    log-likelihood, matching the attribute name on
    `optimagic.OptimizeResult`), and ``success``.
    """

    params: pd.DataFrame
    fun: float
    success: bool
    n_iter: int


def _check_constraints_supported(
    constraints: list[om.constraints.Constraint],
) -> None:
    for c in constraints:
        if not isinstance(c, FixedConstraintWithValue):
            msg = (
                f"jaxopt backend supports only `FixedConstraintWithValue`; "
                f"got `{type(c).__name__}`. Use "
                f'`optimizer_backend="optimagic"` for models with '
                f"probability or equality constraints (e.g. log_ces "
                f"transitions or within-step / cross-period equalities)."
            )
            raise NotImplementedError(msg)


def minimize_with_jaxopt(
    loglike_and_grad: Callable[[Array], tuple[Array, Array]],
    full_params_df: pd.DataFrame,
    constraints: list[om.constraints.Constraint],
    optimizer_options: dict[str, Any] | None = None,
) -> JaxoptResult:
    """Minimize the negative log-likelihood with `jaxopt.LBFGSB`.

    The optimization is performed over the un-pinned coordinates only;
    pinned values from `FixedConstraintWithValue` rows are spliced
    back into the full parameter vector before each likelihood
    evaluation. Bounds on the free coordinates are taken from
    `full_params_df["lower_bound"]` and `full_params_df["upper_bound"]`.

    Args:
        loglike_and_grad: Jitted function mapping the full parameter
            vector (length `n_params`) to ``(neg_loglike, gradient)``.
        full_params_df: DataFrame indexed by the AF params MultiIndex
            with columns ``value``, ``lower_bound``, ``upper_bound``.
        constraints: Must contain only `FixedConstraintWithValue`
            objects. Other constraint kinds raise `NotImplementedError`.
        optimizer_options: Forwarded to `LBFGSB` (e.g. `maxiter`,
            `tol`, `history_size`).

    Return:
        `JaxoptResult` carrying the final params DataFrame and the
        attained criterion value.

    """
    _check_constraints_supported(constraints)
    options = dict(optimizer_options or {})

    # Pre-compute which positions are pinned + their target values.
    is_pinned = np.zeros(len(full_params_df), dtype=bool)
    pinned_values = full_params_df["value"].to_numpy().astype(np.float64).copy()
    for c in constraints:
        # `_check_constraints_supported` guarantees every entry is
        # `FixedConstraintWithValue`; the cast lets ty see `loc`/`value`.
        fc = cast("FixedConstraintWithValue", c)
        idx = full_params_df.index.get_loc(fc.loc)
        is_pinned[idx] = True
        pinned_values[idx] = float(fc.value)  # ty: ignore[invalid-argument-type]

    free_idx_np = np.where(~is_pinned)[0]
    free_idx = jnp.array(free_idx_np)
    full_template = jnp.array(pinned_values)
    free_initial = jnp.array(pinned_values[free_idx_np])

    raw_lower = full_params_df["lower_bound"].to_numpy()[free_idx_np]
    raw_upper = full_params_df["upper_bound"].to_numpy()[free_idx_np]
    # jaxopt.LBFGSB rejects non-finite bounds, so clip infinities to a
    # very wide finite range. The clip is large enough that LBFGSB will
    # never run into it for sensibly-scaled likelihoods.
    free_lower = jnp.array(np.where(np.isfinite(raw_lower), raw_lower, -1e30))
    free_upper = jnp.array(np.where(np.isfinite(raw_upper), raw_upper, 1e30))

    def objective_and_grad(free_vec: Array) -> tuple[Array, Array]:
        full_vec = full_template.at[free_idx].set(free_vec)  # noqa: PD008
        val, grad = loglike_and_grad(full_vec)
        return val, grad[free_idx]

    # Match scipy_lbfgsb's stopping rule: stop when EITHER
    #   * max|projected_grad| < gtol_abs  ("gtol channel"), OR
    #   * (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) < ftol_rel
    #     ("ftol channel"; this is the criterion that typically fires in
    #      practice for skill-formation likelihoods that go locally flat
    #      before the gradient does).
    # Accept the canonical scipy keys so the same `optimizer_options`
    # dict works for both backends; fall back to historical jaxopt
    # names for compatibility.
    gtol_abs = float(options.pop("convergence_gtol_abs", options.pop("tol", 1e-5)))
    ftol_rel = float(options.pop("convergence_ftol_rel", 2.22e-9))
    maxiter = int(options.pop("stopping_maxiter", options.pop("maxiter", 15_000)))
    history_size = int(options.pop("history_size", 10))

    solver = LBFGSB(
        fun=objective_and_grad,
        value_and_grad=True,
        # `maxiter` here is jaxopt's *internal* fail-safe cap; the outer
        # Python loop below drives stopping. Set huge so jaxopt never
        # interrupts us mid-iteration.
        maxiter=maxiter,
        tol=gtol_abs,
        history_size=history_size,
        **options,
    )

    bounds = (free_lower, free_upper)
    state = solver.init_state(free_initial, bounds=bounds)
    params = free_initial
    prev_val = jnp.inf
    stopped_on = "maxiter"
    n_iter = 0
    # fallback if `maxiter == 0` and the loop body never executes.
    for n_iter in range(1, maxiter + 1):  # noqa: B007
        params, state = solver.update(params, state, bounds=bounds)
        cur_val = state.value
        # gtol channel
        if bool(state.error < gtol_abs):
            stopped_on = "gtol"
            break
        # ftol channel (skip first iteration where prev_val == inf)
        denom = jnp.maximum(
            jnp.maximum(jnp.abs(prev_val), jnp.abs(cur_val)),
            1.0,
        )
        rel_drop = jnp.abs(prev_val - cur_val) / denom
        if bool(jnp.isfinite(prev_val)) and bool(rel_drop < ftol_rel):
            stopped_on = "ftol"
            break
        prev_val = cur_val

    final_full = full_template.at[free_idx].set(params)  # noqa: PD008
    result_df = full_params_df.copy()
    result_df["value"] = np.asarray(jax.device_get(final_full))

    return JaxoptResult(
        params=result_df,
        fun=float(jax.device_get(state.value)),
        success=stopped_on != "maxiter",
        n_iter=n_iter,
    )
