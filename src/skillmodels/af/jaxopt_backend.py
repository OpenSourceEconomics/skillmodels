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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
from jax import Array
from jaxopt import LBFGSB

from skillmodels.common.constraints import FixedConstraintWithValue


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

    solver = LBFGSB(
        fun=objective_and_grad,
        value_and_grad=True,
        maxiter=int(options.pop("maxiter", 500)),
        tol=float(options.pop("tol", 1e-6)),
        history_size=int(options.pop("history_size", 10)),
        **options,
    )
    opt_step = solver.run(free_initial, bounds=(free_lower, free_upper))

    final_full = full_template.at[free_idx].set(opt_step.params)  # noqa: PD008
    result_df = full_params_df.copy()
    result_df["value"] = np.asarray(jax.device_get(final_full))

    n_iter = int(opt_step.state.iter_num)
    return JaxoptResult(
        params=result_df,
        fun=float(jax.device_get(opt_step.state.value)),
        success=n_iter < solver.maxiter,
        n_iter=n_iter,
    )
