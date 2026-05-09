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
    get_transition_period_params_index,
)
from skillmodels.process_model import process_model

from .evaluate import (
    evaluate_af_initial_loglike,
    evaluate_af_transition_loglike,
)
from .load_cnlsy import INCOME_MEASURE, load_measurements
from .matlab_mapping import (
    MatlabResults,
    fill_initial_params_from_matlab,
    fill_transition_params_from_matlab,
    load_matlab_results,
)
from .model_specs import build_ces_model, build_translog_model

_REF_DIR = Path("/home/hmg/sciebo/Skill estimation/Application")
_DATA_PATH = Path(__file__).parent / "data" / "complete_7_9_11.xls"
_CES_RESULTS = _REF_DIR / "Results" / "Results_AF_One_Normal_CES.mat"
_TRANSLOG_RESULTS = _REF_DIR / "Results" / "Results_AF_One_Normal_Translog.mat"


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
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("ces", id="ces_matlab_norm"),
        pytest.param("translog", id="translog"),
    ],
)
def test_total_loglike_ours_vs_matlab(variant: str, capsys) -> None:
    """Sum all three period log-likelihoods under skillmodels' AF and compare.

    Under skillmodels' own likelihood:
      - ours = sum over periods of the converged log-likelihoods.
      - matlab = same sum evaluated at MATLAB's translated parameters.

    Prints both, asserts both are finite; the arithmetic of the total
    answers "does MATLAB produce a higher likelihood than our solution?".

    For ``variant="ces"`` we use ``match_matlab_normalisation=True`` so the
    parameter values are directly comparable to MATLAB. For
    ``variant="translog"`` MATLAB's identification matches skillmodels'
    default already.
    """
    if variant == "ces":
        built = build_ces_model(match_matlab_normalisation=True)
        results_path = _CES_RESULTS
    else:
        built = build_translog_model()
        results_path = _TRANSLOG_RESULTS
    if not results_path.exists():
        pytest.skip(f"MATLAB reference {results_path} not available")
    data = load_measurements(_DATA_PATH)
    matlab: MatlabResults = load_matlab_results(results_path, variant=variant)

    af_options = AFEstimationOptions(
        n_halton_points=20_000,
        n_halton_points_shock=20_000,
        n_mixture_components=1,
        optimizer_algorithm="scipy_lbfgsb",
        two_stage_measurement=False,
    )

    # ----- our own estimate (all periods) -----
    result = estimate_af(
        model_spec=built.model_spec,
        data=data,
        af_options=af_options,
        fixed_params=built.fixed_params,
    )
    skm_ll_by_period = [float(pr.loglikelihood) for pr in result.period_results]
    total_skm_ll = sum(skm_ll_by_period)

    # ----- MATLAB params, scored under our likelihood -----
    period_ll_matlab, matlab_params_by_period = _score_matlab_under_our_lik(
        built=built,
        data=data,
        matlab=matlab,
        af_options=af_options,
        our_result=result,
        match_matlab_normalisation=variant == "ces",
    )
    total_matlab_ll = sum(period_ll_matlab)

    print("\n=== log-likelihood comparison ===")
    for t, (skm, matlab_val) in enumerate(
        zip(skm_ll_by_period, period_ll_matlab, strict=True)
    ):
        tag = "initial" if t == 0 else f"trans {t - 1}->{t}"
        print(f"  period {t} ({tag}):  ours={skm:+.6f}  matlab={matlab_val:+.6f}")
    print(f"  TOTAL: ours={total_skm_ll:+.6f}  matlab={total_matlab_ll:+.6f}")
    diff = total_skm_ll - total_matlab_ll
    better = "skillmodels higher" if diff >= 0 else "MATLAB higher"
    print(f"  difference = {diff:+.6f} ({better})")

    assert np.isfinite(total_skm_ll)
    assert np.isfinite(total_matlab_ll)

    _print_param_comparison(
        our_params=[pr.params for pr in result.period_results],
        matlab_params=matlab_params_by_period,
    )

    _reoptimize_from_matlab_start(
        built=built,
        data=data,
        af_options=af_options,
        skm_ll_by_period=skm_ll_by_period,
        total_skm_ll=total_skm_ll,
        matlab_params_by_period=matlab_params_by_period,
    )


def _score_matlab_under_our_lik(
    *,
    built,
    data: pd.DataFrame,
    matlab: MatlabResults,
    af_options: AFEstimationOptions,
    our_result,
    match_matlab_normalisation: bool = False,
) -> tuple[list[float], list[pd.DataFrame]]:
    """Evaluate the AF log-likelihood at MATLAB's translated parameters.

    Uses our own conditional distribution at each period as the prior for
    the next period's transition evaluation; MATLAB-translated parameters
    are substituted only in the current-period transition and measurement
    blocks. Returns per-period log-likelihoods and the per-period
    MATLAB-filled parameter DataFrames.
    """
    processed_model = process_model(built.model_spec)
    factors = processed_model.labels.latent_factors
    controls_names = processed_model.labels.controls
    state_factors = tuple(
        f
        for f in factors
        if not processed_model.endogenous_factors_info.factor_info[f].is_endogenous
    )
    endogenous_factors = tuple(
        f
        for f in factors
        if processed_model.endogenous_factors_info.factor_info[f].is_endogenous
    )
    shock_factors = tuple(
        f for f in state_factors if built.model_spec.factors[f].has_production_shock
    )
    transition_info = processed_model.transition_info
    meas_p0, ctrls_p0, obs_fac_p0 = _extract_period_0_arrays(
        data, built.model_spec, controls_names=controls_names
    )

    measurements_p0 = get_measurements_per_factor(built.model_spec.factors, period=0)
    reconstructed_factors = tuple(
        f for f in factors if not built.model_spec.factors[f].has_initial_distribution
    )
    initial_index = get_initial_period_params_index(
        n_mixture_components=1,
        latent_factors=factors,
        measurements_period_0=measurements_p0,
        controls=controls_names,
        observed_factors=(INCOME_MEASURE,),
        reconstructed_factors=reconstructed_factors,
    )
    initial_norms = get_normalizations_for_period(built.model_spec.factors, period=0)
    initial_template = create_af_params_template(initial_index, initial_norms, period=0)
    initial_with_matlab = fill_initial_params_from_matlab(
        initial_template,
        matlab.initial,
        match_matlab_normalisation=match_matlab_normalisation,
    )
    # Apply built.fixed_params on top so initial_states pins survive.
    for idx, val in built.fixed_params["value"].items():
        if idx in initial_with_matlab.index:
            initial_with_matlab.loc[idx, "value"] = val
            initial_with_matlab.loc[idx, "lower_bound"] = val
            initial_with_matlab.loc[idx, "upper_bound"] = val
    matlab_ll_p0 = evaluate_af_initial_loglike(
        model_spec=built.model_spec,
        measurements=meas_p0,
        controls=ctrls_p0,
        params_df=initial_with_matlab,
        af_options=af_options,
        observed_factors=(INCOME_MEASURE,),
        observed_factor_values=obs_fac_p0,
    )

    period_ll_matlab = [matlab_ll_p0]
    matlab_params_by_period: list[pd.DataFrame] = [initial_with_matlab]
    for skillmodels_period in (1, 2):
        measurements_pt = get_measurements_per_factor(
            built.model_spec.factors, period=skillmodels_period
        )
        t_index = get_transition_period_params_index(
            period=skillmodels_period,
            latent_factors=state_factors,
            transition_info=transition_info,
            measurements_at_period=measurements_pt,
            controls=controls_names,
            endogenous_factors=endogenous_factors,
            observed_factors=(INCOME_MEASURE,),
            shock_factors=shock_factors,
        )
        t_norms = get_normalizations_for_period(
            built.model_spec.factors, period=skillmodels_period
        )
        t_template = create_af_params_template(
            t_index, t_norms, period=skillmodels_period
        )
        # Seed from our own converged values for any slot the translator
        # won't touch (currently none, but safe default).
        t_template.loc[t_template.index, "value"] = our_result.period_results[
            skillmodels_period
        ].params.loc[t_template.index, "value"]
        t_with_matlab = fill_transition_params_from_matlab(
            t_template, matlab, skillmodels_period=skillmodels_period
        )
        matlab_params_by_period.append(t_with_matlab)

        meas_t, ctrls_t, obs_fac_t = _extract_period_arrays(
            data,
            built.model_spec,
            period=skillmodels_period,
            controls_names=controls_names,
        )
        prev_meas, prev_ctrls, _ = _extract_period_arrays(
            data,
            built.model_spec,
            period=skillmodels_period - 1,
            controls_names=controls_names,
        )
        matlab_ll_t = evaluate_af_transition_loglike(
            model_spec=built.model_spec,
            period=skillmodels_period,
            measurements=meas_t,
            controls=ctrls_t,
            prev_measurements=prev_meas,
            prev_controls=prev_ctrls,
            prev_period_params=our_result.period_results[skillmodels_period - 1].params,
            prev_distribution=our_result.conditional_distributions[
                skillmodels_period - 1
            ],
            params_df=t_with_matlab,
            af_options=af_options,
            endogenous_factors=endogenous_factors,
            observed_factors=(INCOME_MEASURE,),
            observed_factor_data=obs_fac_t,
        )
        period_ll_matlab.append(matlab_ll_t)

    return period_ll_matlab, matlab_params_by_period


def _reoptimize_from_matlab_start(
    *,
    built,
    data: pd.DataFrame,
    af_options: AFEstimationOptions,
    skm_ll_by_period: list[float],
    total_skm_ll: float,
    matlab_params_by_period: list[pd.DataFrame],
) -> None:
    """Run a second full AF estimation starting from MATLAB's translated values.

    If our default-start optimum is a strict improvement over MATLAB's
    basin, starting from MATLAB's params should converge back to our
    optimum (or very close). If they converge to different
    log-likelihoods, there are genuinely multiple local maxima.
    """
    matlab_start_params = pd.concat(matlab_params_by_period)[["value"]].dropna()
    result_from_matlab = estimate_af(
        model_spec=built.model_spec,
        data=data,
        af_options=af_options,
        start_params=matlab_start_params,
        fixed_params=built.fixed_params,
    )
    from_matlab_ll_by_period = [
        float(pr.loglikelihood) for pr in result_from_matlab.period_results
    ]
    total_from_matlab_ll = sum(from_matlab_ll_by_period)

    print("\n=== re-optimization from MATLAB start ===")
    for t, (skm, fm) in enumerate(
        zip(skm_ll_by_period, from_matlab_ll_by_period, strict=True)
    ):
        tag = "initial" if t == 0 else f"trans {t - 1}->{t}"
        print(
            f"  period {t} ({tag}):  default_start={skm:+.6f}  "
            f"matlab_start={fm:+.6f}  delta={skm - fm:+.6f}"
        )
    print(
        f"  TOTAL: default_start={total_skm_ll:+.6f}  "
        f"matlab_start={total_from_matlab_ll:+.6f}  "
        f"delta={total_skm_ll - total_from_matlab_ll:+.6f}"
    )


def _print_param_comparison(
    our_params: list[pd.DataFrame],
    matlab_params: list[pd.DataFrame],
) -> None:
    """Print a side-by-side comparison of estimates by parameter category.

    Excludes parameters whose ``lower_bound == upper_bound`` (normalisations
    and other pinned rows) and rows MATLAB did not translate (``NaN``).
    """
    print("\n=== parameter comparison (ours vs MATLAB, under our spec) ===")
    for t, (ours_t, matlab_t) in enumerate(zip(our_params, matlab_params, strict=True)):
        tag = "initial" if t == 0 else f"trans {t - 1}->{t}"
        merged = pd.DataFrame(
            {
                "ours": ours_t["value"],
                "matlab": matlab_t["value"],
            }
        )
        free = ours_t["lower_bound"] != ours_t["upper_bound"]
        merged = merged.loc[free & merged["matlab"].notna()]
        merged["abs_diff"] = merged["ours"] - merged["matlab"]
        denom = merged["matlab"].abs().clip(lower=1e-6)
        merged["rel_diff"] = merged["abs_diff"] / denom

        print(f"\n--- period {t} ({tag}) ---")
        categories = merged.index.get_level_values("category").unique()
        for cat in categories:
            sub = merged.xs(cat, level="category", drop_level=False)
            label_lens = [len(f"{idx[2]}:{idx[3]}") for idx in sub.index]
            wlabel = max(18, *label_lens) if label_lens else 18
            print(f"  [{cat}]")
            for idx, row in sub.iterrows():
                label = f"{idx[2]}:{idx[3]}"
                print(
                    f"    {label:<{wlabel}} "
                    f"ours={row['ours']:+10.4f}  "
                    f"matlab={row['matlab']:+10.4f}  "
                    f"delta={row['abs_diff']:+10.4f}  "
                    f"rel={row['rel_diff']:+7.2%}"
                )


def _extract_period_arrays(
    data: pd.DataFrame,
    model_spec,
    *,
    period: int,
    controls_names: tuple[str, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(measurements, controls, observed_factor_values)`` for a period."""
    measurements_pt = get_measurements_per_factor(model_spec.factors, period=period)
    period_df = data.xs(period, level="period")
    seen: set[str] = set()
    ordered: list[str] = []
    for cols in measurements_pt.values():
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
    obs_col = INCOME_MEASURE
    if obs_col in period_df.columns and period_df[obs_col].notna().any():
        obs_fac = jnp.array(
            period_df[obs_col].fillna(0.0).to_numpy(dtype=np.float64).reshape(-1, 1)
        )
    else:
        obs_fac = jnp.zeros((len(period_df), 1))
    return meas, ctrls, obs_fac
