"""Compare skillmodels' AF loglike to MATLAB's loglike on the CNLSY CES model.

Runs skillmodels AF estimation to convergence and also evaluates skillmodels'
AF likelihood at MATLAB's converged ``est_0`` parameters. Prints both values
so we can see whether MATLAB's optimum is higher or lower than ours under our
own likelihood.

Scoped to the initial period here (period 0). The transition-period
translation is more involved (CES reparameterisation, investment equation
mapping) and would go in a follow-up.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from skillmodels.af import AFEstimationOptions, estimate_af
from skillmodels.af.params import (
    create_af_params_template,
    get_initial_period_params_index,
    get_measurements_per_factor,
    get_normalizations_for_period,
)

from .evaluate import evaluate_af_initial_loglike
from .load_cnlsy import INCOME_MEASURE, load_measurements
from .matlab_mapping import (
    MatlabResults,
    fill_initial_params_from_matlab,
    load_matlab_results,
)
from .model_specs import build_ces_model

_REF_DIR = Path("/home/hmg/sciebo/Skill estimation")
_DATA_PATH = _REF_DIR / "complete_7_9_11.xls"
_CES_RESULTS = _REF_DIR / "Results" / "Results_AF_One_Normal_CES.mat"


pytestmark = pytest.mark.skipif(
    not (_DATA_PATH.exists() and _CES_RESULTS.exists()),
    reason=f"MATLAB reference not available at {_REF_DIR}",
)


def _extract_period_0_arrays(
    data: pd.DataFrame, model_spec, controls_names: tuple[str, ...]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build period-0 ``(measurements, controls, observed_factor_values)`` arrays."""
    measurements_p0 = get_measurements_per_factor(model_spec.factors, period=0)
    period_df = data.xs(0, level="period")
    seen: set[str] = set()
    ordered: list[str] = []
    for cols in measurements_p0.values():
        for m in cols:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
    meas = jnp.array(period_df[ordered].to_numpy(dtype=np.float64, na_value=np.nan))
    ctrl_cols = []
    for ctrl in controls_names:
        if ctrl == "constant":
            ctrl_cols.append(np.ones(len(period_df)))
        elif ctrl in period_df.columns:
            ctrl_cols.append(period_df[ctrl].to_numpy(dtype=np.float64))
        else:
            ctrl_cols.append(np.zeros(len(period_df)))
    ctrls = jnp.array(np.column_stack(ctrl_cols))
    obs_fac = jnp.array(
        period_df[INCOME_MEASURE].to_numpy(dtype=np.float64).reshape(-1, 1)
    )
    return meas, ctrls, obs_fac


@pytest.mark.end_to_end
@pytest.mark.long_running
def test_initial_period_loglike_ours_vs_matlab(capsys) -> None:
    """Report skillmodels' initial-period loglike and the loglike at MATLAB's est_0.

    Passes if both are finite; the interesting output is printed.
    """
    built = build_ces_model()
    data = load_measurements(_DATA_PATH)
    matlab: MatlabResults = load_matlab_results(_CES_RESULTS, variant="ces")

    af_options = AFEstimationOptions(
        n_halton_points=20_000,
        n_halton_points_shock=20_000,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
    )

    # ----- our own estimate on period 0 -----
    result = estimate_af(
        model_spec=built.model_spec,
        data=data,
        af_options=af_options,
        fixed_params=built.fixed_params,
    )
    skm_loglike_p0 = float(result.period_results[0].loglikelihood)

    # ----- MATLAB params, scored under our likelihood -----
    processed_factors = ("skills", "MC", "MN", "investment")
    measurements_p0 = get_measurements_per_factor(built.model_spec.factors, period=0)
    params_index = get_initial_period_params_index(
        n_mixture_components=1,
        latent_factors=processed_factors,
        measurements_period_0=measurements_p0,
        controls=("constant",),
        observed_factors=(INCOME_MEASURE,),
    )
    normalizations = get_normalizations_for_period(built.model_spec.factors, period=0)
    params_template = create_af_params_template(params_index, normalizations, period=0)
    # Seed defaults from skillmodels' own period-0 result to fill entries that
    # don't have a MATLAB analogue (investment measurement model, investment's
    # row/col in the initial distribution).
    seeded = result.period_results[0].params.copy()
    params_template.loc[params_template.index, "value"] = seeded.loc[
        params_template.index, "value"
    ]
    params_with_matlab = fill_initial_params_from_matlab(
        params_template, matlab.initial
    )

    meas, ctrls, obs_fac = _extract_period_0_arrays(
        data, built.model_spec, controls_names=("constant",)
    )
    matlab_loglike_p0 = evaluate_af_initial_loglike(
        model_spec=built.model_spec,
        measurements=meas,
        controls=ctrls,
        params_df=params_with_matlab,
        af_options=af_options,
        observed_factors=(INCOME_MEASURE,),
        observed_factor_values=obs_fac,
    )

    print("\n=== initial-period log-likelihood ===")
    print(f"  skillmodels AF converged loglike = {skm_loglike_p0:+.6f}")
    print(f"  skillmodels likelihood at MATLAB's est_0 = {matlab_loglike_p0:+.6f}")
    diff = skm_loglike_p0 - matlab_loglike_p0
    better = "skillmodels higher" if diff >= 0 else "MATLAB higher"
    print(f"  difference = {diff:+.6f} ({better})")

    assert np.isfinite(skm_loglike_p0)
    assert np.isfinite(matlab_loglike_p0)
