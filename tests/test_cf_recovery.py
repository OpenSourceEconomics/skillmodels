"""End-to-end recovery test for the CHS control-function correction.

Simulate a panel from a known endogenous DGP with PERIOD-SPECIFIC kappa_t, then
estimate and confirm the control function recovers the known kappa_t, the
first-stage coefficients, and the corrected production-shock SD. This is the
proof that the whole CHS control-function machinery (DAG graft, investment_eq /
kappa params, carry-forward kappa zeroing, forced unscented path) works.
"""

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pytest

from skillmodels import CorrectionSpec, FactorSpec, ModelSpec, Normalizations
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.params_index import get_params_index
from skillmodels.common.parse_params import create_parsing_info, parse_params
from skillmodels.common.process_data import process_data
from skillmodels.common.process_model import process_model
from skillmodels.common.simulate_data import _simulate_dataset

N_OBS = 8000
SEED = 20240615
TRUE_FIRST_STAGE = {"fac1": 0.5, "fac2": 0.3, "z1": 0.7, "constant": 0.1}
# Period-specific kappa lives on the FREE odd aug_periods (1 -> period 0,
# 3 -> period 1). The even aug_periods are pinned to 0 by carry-forward zeroing.
TRUE_KAPPA = {1: 0.6, 3: -0.4}
TRUE_SHOCK_SD = 0.5
TRUE_MEAS_SD = 0.3


def _build_model() -> ModelSpec:
    state_norm = Normalizations(
        loadings=({"y1": 1}, {"y1": 1}, {"y1": 1}), intercepts=({}, {}, {})
    )
    fac2_norm = Normalizations(
        loadings=({"w1": 1}, {"w1": 1}, {"w1": 1}), intercepts=({}, {}, {})
    )
    inv_norm = Normalizations(
        loadings=({"yi1": 1}, {"yi1": 1}, {}), intercepts=({}, {}, {})
    )
    factors = {
        "fac1": FactorSpec(
            measurements=(("y1", "y2", "y3"),) * 3,
            normalizations=state_norm,
            transition_function="linear",
        ),
        "fac2": FactorSpec(
            measurements=(("w1", "w2", "w3"),) * 3,
            normalizations=fac2_norm,
            transition_function="linear",
        ),
        "inv": FactorSpec(
            # Endogenous factor must NOT be measured in the last period.
            measurements=(("yi1", "yi2", "yi3"), ("yi1", "yi2", "yi3"), ()),
            normalizations=inv_norm,
            is_endogenous=True,
            transition_function="linear",
            correction=CorrectionSpec(
                state_predictors=("fac1", "fac2"),
                instruments=("z1",),
                targets=("fac1", "fac2"),
            ),
        ),
    }
    return ModelSpec(factors=factors, observed_factors=("z1",))


_MEAS_COLS = ["y1", "y2", "y3", "w1", "w2", "w3", "yi1", "yi2", "yi3"]


def _z1_panel(rng: np.random.Generator):
    import pandas as pd  # noqa: PLC0415

    index = pd.MultiIndex.from_product(
        [np.arange(N_OBS), [0, 1, 2]], names=["caseid", "period"]
    )
    panel = pd.DataFrame(index=index)
    panel["z1"] = rng.normal(size=len(index))
    for col in _MEAS_COLS:
        panel[col] = np.nan
    return panel


def _fill_true_params(template, all_factors):
    """Fill every free template entry with its true DGP value."""
    p = template.copy()
    free = p["value"].isna()
    cat = p.index.get_level_values("category")
    name1 = p.index.get_level_values("name1")
    name2 = p.index.get_level_values("name2")
    aug_period = p.index.get_level_values("aug_period")

    p.loc[free & (cat == "initial_states"), "value"] = 0.0
    p.loc[free & (cat == "mixture_weights"), "value"] = 1.0
    p.loc[free & (cat == "loadings"), "value"] = 1.0
    p.loc[free & (cat == "meas_sds"), "value"] = TRUE_MEAS_SD
    p.loc[free & (cat == "shock_sds"), "value"] = TRUE_SHOCK_SD
    p.loc[free & (cat == "controls"), "value"] = 0.0

    # initial cholcovs: identity (diagonal 1, off-diagonal 0).
    diag = np.array([str(n).split("-")[0] == str(n).split("-")[-1] for n in name2])
    p.loc[free & (cat == "initial_cholcovs") & diag, "value"] = 1.0
    p.loc[free & (cat == "initial_cholcovs") & ~diag, "value"] = 0.0

    # transition: self-persistence 0.7, constant 0, other cross terms 0.1.
    is_trans = free & (cat == "transition")
    p.loc[is_trans, "value"] = 0.1
    p.loc[is_trans & (name2 == name1), "value"] = 0.7
    p.loc[is_trans & (name2 == "constant"), "value"] = 0.0

    # first-stage investment equation.
    for predictor, value in TRUE_FIRST_STAGE.items():
        p.loc[free & (cat == "investment_eq") & (name2 == predictor), "value"] = value

    # period-specific kappa on the free odd aug_periods.
    for ap, value in TRUE_KAPPA.items():
        p.loc[free & (cat == "kappa") & (aug_period == ap), "value"] = value

    assert not p["value"].isna().any(), "some params left unset"
    return p


@pytest.mark.end_to_end
@pytest.mark.long_running
def test_control_function_recovers_period_specific_kappa() -> None:

    model = _build_model()
    pm = process_model(model)
    all_factors = pm.labels.all_factors

    cf = pm.endogenous_factors_info.control_function
    assert cf is not None
    assert cf.investment_factor == "inv"
    assert cf.targets == ("fac1", "fac2")

    # --- known params on the template ---
    rng = np.random.default_rng(SEED)
    z1_panel = _z1_panel(rng)
    template = get_maximization_inputs(
        model,
        z1_panel,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )["params_template"]
    true_params = _fill_true_params(template, all_factors)

    # carry-forward kappa zeroing must have pinned the even aug_periods to 0.
    kappa_even = true_params.xs("kappa", level="category").reset_index()
    pinned = kappa_even[kappa_even["aug_period"].isin([0, 2])]
    assert (pinned["value"] == 0.0).all()

    # --- simulate measurements from the known params ---
    pidx = get_params_index(
        update_info=pm.update_info,
        labels=pm.labels,
        dimensions=pm.dimensions,
        transition_info=pm.transition_info,
        endogenous_factors_info=pm.endogenous_factors_info,
    )
    parsing = create_parsing_info(
        params_index=pidx,
        update_info=pm.update_info,
        labels=pm.labels,
        anchoring=pm.anchoring,
        has_endogenous_factors=True,
    )
    pds = process_data(
        df=z1_panel,
        has_endogenous_factors=True,
        labels=pm.labels,
        update_info=pm.update_info,
        anchoring_info=pm.anchoring,
        purpose="simulation",
    )
    states, covs, log_weights, parsed = parse_params(
        jnp.array(true_params.reindex(pidx)["value"].to_numpy()),
        parsing,
        pm.dimensions,
        pm.labels,
        n_obs=N_OBS,
    )
    obs_meas, _ = _simulate_dataset(
        latent_states=states,
        covs=covs,
        log_weights=log_weights,
        parsed_params=parsed,
        labels=pm.labels,
        dimensions=pm.dimensions,
        n_obs=N_OBS,
        has_endogenous_factors=True,
        update_info=pm.update_info,
        control_data=pds["controls"],
        observed_factors=pds["observed_factors"],
        policies=None,
        transition_info=pm.transition_info,
        rng=np.random.default_rng(SEED + 1),
    )

    # --- assemble estimable panel ---
    obs_meas = obs_meas.copy()
    obs_meas["period"] = obs_meas["aug_period"].map(pm.labels.aug_periods_to_periods)
    meas_cols = [c for c in _MEAS_COLS if c in obs_meas.columns]
    collapsed = obs_meas.groupby(["id", "period"])[meas_cols].first().reset_index()
    z1_long = z1_panel.reset_index()[["caseid", "period", "z1"]].rename(
        columns={"caseid": "id"}
    )
    est_data = collapsed.merge(z1_long, on=["id", "period"]).set_index(["id", "period"])

    # --- estimate ---
    mi = get_maximization_inputs(
        model,
        est_data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    start = mi["params_template"].copy()
    free = start["value"].isna()
    start.loc[~free, "value"] = true_params.loc[~free, "value"]
    noise = np.random.default_rng(SEED + 2).normal(scale=0.05, size=int(free.sum()))
    start.loc[free, "value"] = true_params.loc[free, "value"].to_numpy() + noise

    import pandas as pd  # noqa: PLC0415

    def _neg_loglike_and_gradient(q: pd.DataFrame) -> tuple[float, np.ndarray]:
        value, gradient = mi["loglike_and_gradient"](q)
        return -value, -np.asarray(gradient)

    res = om.minimize(
        fun=lambda q: -mi["loglike"](q),
        params=start[["value"]],
        algorithm="scipy_lbfgsb",
        bounds=om.Bounds(lower=start["lower_bound"], upper=start["upper_bound"]),
        constraints=mi["constraints"],
        fun_and_jac=_neg_loglike_and_gradient,
    )
    assert res.success
    est = res.params["value"]

    # --- assert recovery ---
    for ap, true_kappa in TRUE_KAPPA.items():
        for target in ("fac1", "fac2"):
            got = est.loc[("kappa", ap, target, "cf")]
            assert got == pytest.approx(true_kappa, abs=0.1)
    # even aug_periods stay pinned at 0.
    for ap in (0, 2):
        for target in ("fac1", "fac2"):
            assert est.loc[("kappa", ap, target, "cf")] == pytest.approx(0.0, abs=1e-8)
    # first-stage betas.
    for ap in (0, 1, 2, 3):
        for predictor, value in TRUE_FIRST_STAGE.items():
            got = est.loc[("investment_eq", ap, "inv", predictor)]
            assert got == pytest.approx(value, abs=0.1)
