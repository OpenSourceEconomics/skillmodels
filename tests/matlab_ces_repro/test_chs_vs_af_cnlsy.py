r"""GPU-only comparison of AF and CHS estimators on the CNLSY data.

The hold-last-value imputation in ``load_cnlsy.py`` lets CHS's
``process_data`` consume the same long-format frame as AF (CHS otherwise
rejects the NaN ``log_income`` at period 2). We fit a linear-transitions
variant of the MATLAB CES model with both estimators and emit a
side-by-side table of their measurement-system estimates plus the
linear transition coefficients.

Run via::

    pixi run -e tests-cuda12 pytest \\
        tests/matlab_ces_repro/test_chs_vs_af_cnlsy.py -m long_running -s
"""

from pathlib import Path

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels import get_maximization_inputs
from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.constraints import FixedConstraintWithValue
from skillmodels.model_spec import FactorSpec, ModelSpec
from skillmodels.types import EstimationOptions

from .load_cnlsy import (
    INCOME_MEASURE,
    INV_MEASURES,
    MC_MEASURES,
    MN_MEASURES,
    SKILL_MEASURES,
    load_measurements,
)
from .model_specs import _common_fixed_rows, _measurements, _normalizations

_DATA_PATH = Path(__file__).parent / "data" / "complete_7_9_11.xls"

pytestmark = pytest.mark.skipif(
    not _DATA_PATH.exists(),
    reason=f"CNLSY reference data not available at {_DATA_PATH}",
)

_N_PERIODS = 3


def _build_af_model() -> tuple[ModelSpec, pd.DataFrame]:
    """AF-flavoured model with investment as the endogenous factor."""
    factors: dict[str, FactorSpec] = {
        "skills": FactorSpec(
            measurements=_measurements(SKILL_MEASURES),
            normalizations=_normalizations(SKILL_MEASURES, normalize_periods=(0,)),
            transition_function="linear",
        ),
        "MC": FactorSpec(
            measurements=_measurements(MC_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MC_MEASURES, active_periods=(0,)),
            transition_function="linear",
            has_production_shock=False,
        ),
        "MN": FactorSpec(
            measurements=_measurements(MN_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MN_MEASURES, active_periods=(0,)),
            transition_function="linear",
            has_production_shock=False,
        ),
        "investment": FactorSpec(
            measurements=_measurements(INV_MEASURES, active_periods=(1,)),
            normalizations=_normalizations(
                INV_MEASURES, active_periods=(1,), normalize_periods=()
            ),
            transition_function="linear",
            is_endogenous=True,
            has_initial_distribution=False,
        ),
    }
    rows = _common_fixed_rows()
    fixed_idx = pd.MultiIndex.from_tuples(
        [r[0] for r in rows], names=["category", "period", "name1", "name2"]
    )
    fixed_params = pd.DataFrame({"value": [r[1] for r in rows]}, index=fixed_idx)
    model = ModelSpec(
        factors=factors,
        observed_factors=(INCOME_MEASURE,),
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )
    return model, fixed_params


def _build_chs_model() -> ModelSpec:
    """CHS-flavoured model: investment is a regular latent factor.

    AF treats investment as ``is_endogenous=True`` (it is reconstructed
    from a deterministic equation). CHS does not have that concept; here
    we treat investment as a regular latent factor with linear transition
    and its three measurements at period 1 (the only period the CNLSY
    file ships investment data for).
    """
    factors: dict[str, FactorSpec] = {
        "skills": FactorSpec(
            measurements=_measurements(SKILL_MEASURES),
            normalizations=_normalizations(SKILL_MEASURES, normalize_periods=(0,)),
            transition_function="linear",
        ),
        "MC": FactorSpec(
            measurements=_measurements(MC_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MC_MEASURES, active_periods=(0,)),
            transition_function="linear",
            has_production_shock=False,
        ),
        "MN": FactorSpec(
            measurements=_measurements(MN_MEASURES, active_periods=(0,)),
            normalizations=_normalizations(MN_MEASURES, active_periods=(0,)),
            transition_function="linear",
            has_production_shock=False,
        ),
        "investment": FactorSpec(
            measurements=_measurements(INV_MEASURES, active_periods=(1,)),
            normalizations=_normalizations(
                INV_MEASURES, active_periods=(1,), normalize_periods=()
            ),
            transition_function="linear",
        ),
    }
    return ModelSpec(
        factors=factors,
        observed_factors=(INCOME_MEASURE,),
        estimation_options=EstimationOptions(
            robust_bounds=True,
            bounds_distance=0.001,
            n_mixtures=1,
        ),
    )


def _build_chs_fixed_rows(
    model: ModelSpec,
    template_index: pd.MultiIndex,
) -> list[tuple[tuple[str, int, str, str], float]]:
    """Pin MC and MN identity transitions at every CHS aug_period.

    CHS's params index is ``aug_period``-keyed: each calendar period may
    span multiple aug_periods (one per endogenous factor). MC and MN are
    time-invariant, so we pin their self-coefficient to 1 and every
    other coefficient (including ``log_income`` and ``investment``) to 0
    for every aug-transition that the template actually contains.
    """
    del model  # only the template index is needed here
    rows: list[tuple[tuple[str, int, str, str], float]] = []
    transition_locs = [loc for loc in template_index if loc[0] == "transition"]
    for loc in transition_locs:
        _, _aug_period, name1, name2 = loc
        if name1 not in ("MC", "MN"):
            continue
        value = 1.0 if name2 == name1 else 0.0
        rows.append((loc, value))
    return rows


def _run_chs(
    model: ModelSpec,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Run CHS estimation, pinning MC/MN identity transitions per aug_period."""
    inputs = get_maximization_inputs(model, data)
    params = inputs["params_template"].copy()

    free = params["lower_bound"] != params["upper_bound"]
    cat = params.index.get_level_values("category")
    params.loc[free, "value"] = 0.5
    params.loc[free & (cat == "loadings"), "value"] = 1.0
    params.loc[free & (cat == "controls"), "value"] = 0.0
    params.loc[free & (cat == "initial_states"), "value"] = 0.0
    for constr in inputs["constraints"]:
        if isinstance(constr, om.ProbabilityConstraint):
            prob_idx = constr.selector(params[["value"]]).index
            params.loc[prob_idx, "value"] = 1.0 / len(prob_idx)

    fixed_rows = _build_chs_fixed_rows(model, params.index)
    extra_constraints: list[om.constraints.Constraint] = []
    for loc, value in fixed_rows:
        params.loc[loc, "value"] = value
        # FixedConstraintWithValue handles the pin; relax finite bounds
        # so optimagic does not also see lower==upper.
        params.loc[loc, "lower_bound"] = -np.inf
        params.loc[loc, "upper_bound"] = np.inf
        extra_constraints.append(FixedConstraintWithValue(loc=loc, value=value))

    def fun_and_jac(p: pd.DataFrame) -> tuple[float, np.ndarray]:
        val, grad = inputs["loglike_and_gradient"](p)
        return -float(val), -np.array(grad)

    res = om.minimize(
        fun=lambda p: -inputs["loglike"](p),
        params=params[["value"]],
        algorithm="scipy_lbfgsb",
        bounds=om.Bounds(lower=params["lower_bound"], upper=params["upper_bound"]),
        constraints=list(inputs["constraints"]) + extra_constraints,
        fun_and_jac=fun_and_jac,
    )
    return res.params, -float(res.fun)


def _run_af(
    model: ModelSpec,
    data: pd.DataFrame,
    fixed_params: pd.DataFrame,
):
    # 20_000 Halton nodes match the MATLAB reproduction. Needs a GPU with
    # enough memory for the (n_obs x n_halton) matmul at the transition
    # step; smaller cards can hit cuBLAS autotune failures.
    opts = AFEstimationOptions(
        n_halton_points=20_000,
        n_halton_points_shock=20_000,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
        two_stage_measurement=False,
    )
    res = estimate_af(
        model_spec=model,
        data=data,
        af_options=opts,
        fixed_params=fixed_params,
    )
    return res, opts


def _format_comparison(
    chs_params: pd.DataFrame,
    af_params: pd.DataFrame,
    af_se: pd.Series,
) -> pd.DataFrame:
    common = chs_params.index.intersection(af_params.index)
    rows = []
    for loc in common:
        rows.append(
            {
                "category": loc[0],
                "period": loc[1],
                "name1": loc[2],
                "name2": loc[3],
                "chs": float(chs_params.loc[loc, "value"]),
                "af": float(af_params.loc[loc, "value"]),
                "af_se": float(af_se.loc[loc]),
                "diff": float(af_params.loc[loc, "value"])
                - float(chs_params.loc[loc, "value"]),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.end_to_end
@pytest.mark.long_running
def test_chs_vs_af_linear_cnlsy() -> None:
    """Run AF and CHS on CNLSY with linear transitions and emit side-by-side."""
    data = load_measurements(_DATA_PATH)
    af_model, af_fixed = _build_af_model()
    chs_model = _build_chs_model()

    print("Fitting CHS...", flush=True)
    chs_params, chs_loglike = _run_chs(chs_model, data)
    print(f"  CHS log-likelihood: {chs_loglike:.4f}", flush=True)

    print("Fitting AF (5k Halton nodes, GPU)...", flush=True)
    af_res, _opts = _run_af(af_model, data, af_fixed)
    af_total_ll = sum(pr.loglikelihood for pr in af_res.period_results)
    print(
        f"  AF log-likelihood (sum of period contributions): {af_total_ll:.4f}",
        flush=True,
    )

    # SEs via the Phase-2 sandwich need O(n_params x n_obs) GPU memory and
    # OOM/segfault at the AF MATLAB scale. Report point estimates only;
    # SEs can be obtained per-period via method="block_diagonal" once the
    # Hessian path uses forward-over-forward batched HVPs.
    se_series = pd.Series(np.nan, index=af_res.all_params.index, name="se")
    table = _format_comparison(chs_params, af_res.all_params, se_series)

    print("\nSide-by-side estimates:")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        160,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(table.to_string(index=False))

    if len(table) > 0:
        diff = table["diff"].abs()
        print(
            f"\nAcross {len(table)} shared params: "
            f"max |diff| = {diff.max():.4f}, "
            f"median |diff| = {diff.median():.4f}, "
            f"mean |diff| = {diff.mean():.4f}"
        )

    assert np.all(np.isfinite(chs_params["value"].to_numpy()))
    assert np.all(np.isfinite(af_res.all_params["value"].to_numpy()))
