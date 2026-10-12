"""Mixture-component-specific measurement-error SDs in the CHS estimator."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from skillmodels.af.estimate import estimate_af
from skillmodels.amn.estimate import estimate_amn
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.params_index import get_params_index
from skillmodels.common.parse_params import create_parsing_info, parse_params
from skillmodels.common.process_model import process_model
from skillmodels.common.simulate_data import _simulate_dataset

jax.config.update("jax_enable_x64", True)

MEASUREMENTS = ("y1", "y2", "y3")
SPECIFIC = ("y1", "y3")
NONE_OPTIONS = CHSEstimationOptions(start_params_strategy="none")


def _model(mixture_specific: tuple[str, ...] = (), n_mixtures: int = 2) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(MEASUREMENTS,) * 2,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 2,
                    intercepts=({"y1": 0},) * 2,
                ),
                transition_function="linear",
            ),
        },
        n_mixtures=n_mixtures,
        mixture_specific_meas_sds=mixture_specific,
    )


def _data(n_obs: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        {
            "caseid": i,
            "period": t,
            **dict(zip(MEASUREMENTS, rng.normal(size=3) * 1.5, strict=True)),
        }
        for i in range(n_obs)
        for t in (0, 1)
    ]
    # `process_data` stores measurements as float32; pre-round so the NumPy
    # reference sees exactly the same numbers.
    return (
        pd.DataFrame(rows)
        .set_index(["caseid", "period"])
        .astype(np.float32)
        .astype(np.float64)
    )


# Hand-picked parameter values; the per-component SDs differ clearly.
INTERCEPTS = {(0, "y2"): 0.3, (0, "y3"): -0.2, (1, "y2"): 0.1, (1, "y3"): 0.4}
LOADINGS = {(0, "y2"): 0.8, (0, "y3"): 1.2, (1, "y2"): 0.9, (1, "y3"): 1.1}
POOLED_SDS = {
    (0, "y1"): 0.5,
    (0, "y2"): 0.6,
    (0, "y3"): 0.7,
    (1, "y1"): 0.55,
    (1, "y2"): 0.65,
    (1, "y3"): 0.75,
}
COMPONENT_SDS = {
    (0, "y1"): (0.3, 1.1),
    (0, "y3"): (0.9, 0.4),
    (1, "y1"): (0.45, 1.3),
    (1, "y3"): (1.0, 0.35),
}
MEANS = (-0.5, 0.7)
CHOLS = (0.8, 1.3)
WEIGHTS = (0.3, 0.7)
TRANS_COEF, TRANS_CONST, SHOCK_SD = 0.9, 0.1, 0.6


def _params(  # noqa: C901
    index: pd.MultiIndex,
    component_sds: dict,
    *,
    means: tuple = MEANS,
    chols: tuple = CHOLS,
    weights: tuple = WEIGHTS,
) -> pd.DataFrame:
    values = []
    for cat, period, name1, name2 in index:
        if cat == "controls":
            values.append(INTERCEPTS.get((period, name1), 0.0))
        elif cat == "loadings":
            values.append(LOADINGS.get((period, name1), 1.0))
        elif cat == "meas_sds" and name2 == "-":
            values.append(POOLED_SDS[(period, name1)])
        elif cat == "meas_sds":
            values.append(component_sds[(period, name1)][int(name2[-1])])
        elif cat == "shock_sds":
            values.append(SHOCK_SD)
        elif cat == "initial_states":
            values.append(means[int(name1[-1])])
        elif cat == "mixture_weights":
            values.append(weights[int(name1[-1])])
        elif cat == "initial_cholcovs":
            values.append(chols[int(name1[-1])])
        elif cat == "transition":
            values.append(TRANS_COEF if name2 == "skills" else TRANS_CONST)
        else:
            raise AssertionError(cat)
    return pd.DataFrame({"value": values}, index=index)


def _brute_force_loglike(data: pd.DataFrame, component_sds: dict) -> float:
    """Exact mixture-of-Gaussians likelihood of the linear 2-period model."""
    y = np.stack(
        [
            data.xs(i, level="caseid").loc[[0, 1], list(MEASUREMENTS)].to_numpy()
            for i in data.index.get_level_values("caseid").unique()
        ]
    ).reshape(-1, 6)  # (n_obs, [t0 y1..y3, t1 y1..y3])
    lam = np.array(
        [[1.0 if m == "y1" else LOADINGS[(t, m)] for m in MEASUREMENTS] for t in (0, 1)]
    )
    icept = np.array(
        [[INTERCEPTS.get((t, m), 0.0) for m in MEASUREMENTS] for t in (0, 1)]
    )
    comp_logpdfs = []
    for k in range(2):
        v0 = CHOLS[k] ** 2
        mean_x = np.array([MEANS[k], TRANS_CONST + TRANS_COEF * MEANS[k]])
        cov_x = np.array(
            [
                [v0, TRANS_COEF * v0],
                [TRANS_COEF * v0, TRANS_COEF**2 * v0 + SHOCK_SD**2],
            ]
        )
        big_lam = np.zeros((6, 2))
        big_lam[:3, 0] = lam[0]
        big_lam[3:, 1] = lam[1]
        sds = [
            component_sds[(t, m)][k] if (t, m) in component_sds else POOLED_SDS[(t, m)]
            for t in (0, 1)
            for m in MEASUREMENTS
        ]
        mean = icept.ravel() + big_lam @ mean_x
        cov = big_lam @ cov_x @ big_lam.T + np.diag(np.square(sds))
        comp_logpdfs.append(
            np.log(WEIGHTS[k]) + multivariate_normal(mean, cov).logpdf(y)
        )
    return float(logsumexp(np.stack(comp_logpdfs), axis=0).sum())


@pytest.fixture(scope="module")
def specific_inputs():
    return get_maximization_inputs(_model(SPECIFIC), _data(), chs_options=NONE_OPTIONS)


def test_params_index_has_one_meas_sd_row_per_component(specific_inputs) -> None:
    index = specific_inputs["params_template"].index
    sd_rows = [tup for tup in index if tup[0] == "meas_sds"]
    assert sd_rows == [
        ("meas_sds", 0, "y1", "mixture_0"),
        ("meas_sds", 0, "y1", "mixture_1"),
        ("meas_sds", 0, "y2", "-"),
        ("meas_sds", 0, "y3", "mixture_0"),
        ("meas_sds", 0, "y3", "mixture_1"),
        ("meas_sds", 1, "y1", "mixture_0"),
        ("meas_sds", 1, "y1", "mixture_1"),
        ("meas_sds", 1, "y2", "-"),
        ("meas_sds", 1, "y3", "mixture_0"),
        ("meas_sds", 1, "y3", "mixture_1"),
    ]
    lower = specific_inputs["params_template"].loc["meas_sds", "lower_bound"]
    assert (lower > 0).all()


def test_default_params_index_unchanged() -> None:
    inputs = get_maximization_inputs(_model(), _data(), chs_options=NONE_OPTIONS)
    sd_rows = [t for t in inputs["params_template"].index if t[0] == "meas_sds"]
    assert sd_rows == [("meas_sds", t, m, "-") for t in (0, 1) for m in MEASUREMENTS]


def test_equal_component_sds_reproduce_pooled_loglike(specific_inputs) -> None:
    pooled = get_maximization_inputs(_model(), _data(), chs_options=NONE_OPTIONS)
    pooled_ll = pooled["loglike"](_params(pooled["params_template"].index, {}))
    equal_sds = {key: (POOLED_SDS[key],) * 2 for key in COMPONENT_SDS}
    specific_params = _params(specific_inputs["params_template"].index, equal_sds)
    assert specific_inputs["loglike"](specific_params) == pytest.approx(
        pooled_ll, abs=1e-10, rel=0
    )
    debug_ll = specific_inputs["debug_loglike"](specific_params)["value"]
    assert debug_ll == pytest.approx(pooled_ll, abs=1e-10, rel=0)


def test_loglike_matches_brute_force_mixture_of_gaussians(specific_inputs) -> None:
    params = _params(specific_inputs["params_template"].index, COMPONENT_SDS)
    expected = _brute_force_loglike(_data(), COMPONENT_SDS)
    assert specific_inputs["loglike"](params) == pytest.approx(
        expected, abs=1e-10, rel=0
    )
    assert specific_inputs["debug_loglike"](params)["value"] == pytest.approx(
        expected, abs=1e-10, rel=0
    )
    # Sanity: the component-specific SDs actually matter.
    assert abs(expected - _brute_force_loglike(_data(), {})) > 1e-2


def test_gradient_of_component_sds_matches_finite_differences(
    specific_inputs,
) -> None:
    params = _params(specific_inputs["params_template"].index, COMPONENT_SDS)
    _, grad = specific_inputs["loglike_and_gradient"](params)
    grad = pd.Series(grad, index=params.index)
    rows = [t for t in params.index if t[0] == "meas_sds" and t[3] != "-"]
    h = 1e-6
    for row in rows:
        up, down = params.copy(), params.copy()
        up.loc[row, "value"] += h
        down.loc[row, "value"] -= h
        fd = (specific_inputs["loglike"](up) - specific_inputs["loglike"](down)) / (
            2 * h
        )
        assert grad[row] == pytest.approx(fd, rel=1e-6, abs=1e-6), row


def _simulate(model: ModelSpec, component_sds: dict, n_obs: int, seed: int, **kwargs):
    """Simulate measurements (and latent states) through `_simulate_dataset`."""
    pm = process_model(model)
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
        has_endogenous_factors=False,
    )
    states, covs, log_weights, parsed = parse_params(
        jnp.array(_params(pidx, component_sds, **kwargs)["value"].to_numpy()),
        parsing,
        pm.dimensions,
        pm.labels,
        n_obs=n_obs,
    )
    meas, latent = _simulate_dataset(
        latent_states=states,
        covs=covs,
        log_weights=log_weights,
        parsed_params=parsed,
        labels=pm.labels,
        dimensions=pm.dimensions,
        n_obs=n_obs,
        has_endogenous_factors=False,
        update_info=pm.update_info,
        control_data=jnp.ones((2, n_obs, 1)),
        observed_factors=jnp.zeros((2, n_obs, 0)),
        policies=None,
        transition_info=pm.transition_info,
        rng=np.random.default_rng(seed),
    )
    meas = meas.rename(columns={"id": "caseid", "aug_period": "period"})
    latent = latent.rename(columns={"id": "caseid", "aug_period": "period"})
    return (
        meas.set_index(["caseid", "period"]).sort_index(),
        latent.set_index(["caseid", "period"]).sort_index(),
    )


def test_simulation_draws_component_specific_noise() -> None:
    """Identical components except y3's SD (0.1 vs 2): residuals are a scale mix."""
    sds = {**COMPONENT_SDS, (0, "y3"): (0.1, 2.0)}
    meas, latent = _simulate(
        _model(SPECIFIC),
        sds,
        n_obs=20_000,
        seed=3,
        means=(0.0, 0.0),
        chols=(1.0, 1.0),
        weights=(0.5, 0.5),
    )
    resid = (
        meas.xs(0, level="period")["y3"]
        - INTERCEPTS[(0, "y3")]
        - LOADINGS[(0, "y3")] * latent.xs(0, level="period")["skills"]
    ).to_numpy()
    assert resid.var() == pytest.approx(0.5 * (0.1**2 + 2.0**2), rel=0.05)
    # A pooled normal with that variance has ~17% of mass in |r| < 0.3; the scale
    # mixture has ~0.5 * 1 + 0.5 * 0.12 = 56%.
    assert np.mean(np.abs(resid) < 0.3) > 0.5


def test_amn_seeding_fills_every_component_sd() -> None:
    data, _ = _simulate(_model(SPECIFIC), COMPONENT_SDS, n_obs=500, seed=1)
    inputs = get_maximization_inputs(
        _model(SPECIFIC),
        data,
        chs_options=CHSEstimationOptions(start_params_strategy="amn"),
    )
    template = inputs["params_template"]
    for t in (0, 1):
        for m in SPECIFIC:
            seeds = template.loc[
                [("meas_sds", t, m, "mixture_0"), ("meas_sds", t, m, "mixture_1")],
                "value",
            ]
            assert np.isfinite(seeds).all()
            assert seeds.iloc[0] == seeds.iloc[1]
            assert seeds.iloc[0] != 0.5  # data-derived, not the neutral default


def test_ctor_rejects_single_mixture() -> None:
    with pytest.raises(ValueError, match="n_mixtures >= 2"):
        _model(SPECIFIC, n_mixtures=1)


def test_ctor_rejects_unknown_measurement() -> None:
    with pytest.raises(ValueError, match="not a measurement"):
        _model(("nope",))


def test_from_dict_reads_option() -> None:
    spec = ModelSpec.from_dict(
        {
            "factors": {"skills": {"measurements": [list(MEASUREMENTS)] * 2}},
            "n_mixtures": 2,
            "mixture_specific_meas_sds": ["y1"],
        }
    )
    assert spec.mixture_specific_meas_sds == ("y1",)
    assert spec._replace(controls=()).mixture_specific_meas_sds == ("y1",)


def test_amn_rejects_option() -> None:
    with pytest.raises(NotImplementedError, match="mixture_specific_meas_sds"):
        estimate_amn(_model(SPECIFIC), _data())


def test_af_rejects_option() -> None:
    with pytest.raises(NotImplementedError, match="mixture_specific_meas_sds"):
        estimate_af(_model(SPECIFIC), _data())
