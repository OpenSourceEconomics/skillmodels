"""Bootstrap inference for the AMN estimator.

Cluster (caseid-level) nonparametric bootstrap that re-runs all three
estimation stages on each replicate, mirroring AMN 2020 p. 2523:

    "To estimate confidence intervals and obtain critical values for
    test statistics, we use the non-parametric bootstrap over all three
    steps."

Each bootstrap replicate:

1. Resamples caseids with replacement (size = n_clusters).
2. Calls `estimate_amn` on the resampled panel with the same options.
3. Stores the resulting `all_params` row.

After `n_boot` replicates, the standard errors are the column-wise std
across replicate parameter vectors, and the covariance is the
column-wise covariance. The first replicate inherits the original
fit's params (resampling is i.i.d.; no need to recompute the point
estimate).
"""

import dataclasses
import warnings

import numpy as np
import pandas as pd
from beartype import beartype

from skillmodels._beartype_conf import INFERENCE_CONF
from skillmodels.amn.estimate import estimate_amn
from skillmodels.amn.types import (
    AMNEstimationOptions,
    AMNEstimationResult,
    AMNInferenceResult,
)


def _resample_by_caseid(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw a caseid bootstrap sample with replacement."""
    case_level = str(data.index.names[0])
    caseids = data.index.get_level_values(case_level).unique()
    n = len(caseids)
    sampled = caseids[rng.integers(0, n, size=n)]
    # Rebuild the panel with fresh sequential caseids so duplicates from
    # the bootstrap survive the (caseid, period) uniqueness assumed by
    # build_augmented_measure_matrix.
    pieces = []
    for new_id, original_id in enumerate(sampled):
        block = data.xs(original_id, level=case_level, drop_level=False).copy()
        old_periods = block.index.get_level_values(1)
        block.index = pd.MultiIndex.from_arrays(
            [np.full(len(block), new_id), old_periods],
            names=data.index.names,
        )
        pieces.append(block)
    return pd.concat(pieces)


@beartype(conf=INFERENCE_CONF)
def compute_amn_standard_errors(
    result: AMNEstimationResult,
    data: pd.DataFrame,
    amn_options: AMNEstimationOptions | None = None,
    *,
    n_boot: int = 1_000,
    seed: int = 0,
) -> AMNInferenceResult:
    """Cluster-bootstrap standard errors for AMN parameter estimates.

    Args:
        result: A fitted `AMNEstimationResult` (used to determine the
            parameter index and as a fallback when a replicate fails).
        data: Panel dataset used for the original fit.
        amn_options: AMN options for replicate estimation. If None,
            uses defaults (same as `estimate_amn`).
        n_boot: Number of bootstrap replicates.
        seed: RNG seed.

    Return:
        AMNInferenceResult with replicate-level params, std errors, and
        covariance.

    """
    if amn_options is None:
        amn_options = AMNEstimationOptions()

    rng = np.random.default_rng(seed)
    case_level = str(data.index.names[0])
    caseids = data.index.get_level_values(case_level).unique()
    n_clusters = len(caseids)

    base_index = result.all_params.index
    replicate_rows: list[pd.Series] = []
    n_failed = 0
    for b in range(n_boot):
        replicate_seed = int(rng.integers(0, 2**32 - 1))
        boot_data = _resample_by_caseid(data, rng)
        boot_options = dataclasses.replace(amn_options, seed=replicate_seed)
        try:
            boot_result = estimate_amn(
                result.model_spec,
                boot_data,
                boot_options,
            )
            if not boot_result.success:
                n_failed += 1
                warnings.warn(
                    f"AMN bootstrap replicate {b} did not converge; "
                    "excluding it from the bootstrap distribution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                row = pd.Series(np.nan, index=base_index)
            else:
                row = boot_result.all_params.reindex(base_index)["value"]
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            n_failed += 1
            warnings.warn(
                f"AMN bootstrap replicate {b} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            row = pd.Series(np.nan, index=base_index)
        replicate_rows.append(row)

    replicate_df = pd.DataFrame(replicate_rows).reset_index(drop=True)
    replicate_df.columns = base_index
    standard_errors = replicate_df.std(axis=0, ddof=1)
    vcov = replicate_df.cov(ddof=1)

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_boot} AMN bootstrap replicates failed; "
            "standard errors may be biased.",
            RuntimeWarning,
            stacklevel=2,
        )

    return AMNInferenceResult(
        standard_errors=standard_errors,
        vcov=vcov,
        replicate_params=replicate_df,
        n_clusters=n_clusters,
        n_boot=n_boot,
    )
