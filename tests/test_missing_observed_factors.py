"""Tests for integrating out missing observed factors in the CHS Kalman filter.

Where an observed factor `x` is missing for a person-period, it is treated as
`x ~ N(m, v)`, independent of the latent states, with `(m, v)` taken from data
columns named in `ModelSpec.missing_observed_factor_moments`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from skillmodels.chs.kalman_filters import (
    calculate_sigma_scaling_factor_and_weights,
    kalman_predict,
    linear_kalman_predict,
)
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.model_spec import FactorSpec, ModelSpec

jax.config.update("jax_enable_x64", True)

LATENT = ("fac1", "fac2")
OBSERVED = ("ob1", "ob2")
N_PERIODS = 3
N_PERSONS = 6
MOMENTS = {"ob1": ("ob1_mean", "ob1_var"), "ob2": ("ob2_mean", "ob2_var")}
OPTIONS = CHSEstimationOptions(start_params_strategy="none")

# (person, period) cells where each observed factor is missing. Person 3 misses
# both in period 1 (cross-equation covariance), person 5 misses ob1 only in the
# last period (never used by a transition), person 4 misses nothing.
MISSING_CELLS = {
    "ob1": ((0, 0), (1, 0), (1, 1), (3, 1), (5, 2)),
    "ob2": ((1, 0), (2, 1), (3, 1)),
}

# Log-likelihood of `_model(n_mixtures=2, transition=...)` on `_data()` without
# missings at `_params(..., nonlinear=0.05)`: reference values pinning the
# complete-data filter, which supports observed factors without fill moments.
BASELINE_LOGLIKE = {
    "linear": -198.6803696097199,
    "translog": -206.7069287270014,
}


def _model(
    n_mixtures: int = 1,
    transition: str = "linear",
    moments: dict | None = None,
) -> ModelSpec:
    factors = {
        f: FactorSpec(
            measurements=((f"{f}_m1", f"{f}_m2"),) * N_PERIODS,
            transition_function=transition,
        )
        for f in LATENT
    }
    return ModelSpec(
        factors=factors,
        observed_factors=OBSERVED,
        n_mixtures=n_mixtures,
        missing_observed_factor_moments=moments,
    )


def _round(arr: np.ndarray) -> np.ndarray:
    """Round to a float32-representable grid so dtype casts are exact."""
    return np.round(arr * 64) / 64


def _data(*, with_missings: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    idx = pd.MultiIndex.from_product(
        [range(N_PERSONS), range(N_PERIODS)], names=["id", "period"]
    )
    n_rows = len(idx)
    meas = [f"{f}_m{j}" for f in LATENT for j in (1, 2)]
    df = pd.DataFrame(_round(rng.normal(size=(n_rows, 4))), index=idx, columns=meas)
    df.loc[(2, 1), "fac1_m1"] = np.nan
    for ob, (mean_col, var_col) in MOMENTS.items():
        df[ob] = _round(rng.normal(size=n_rows))
        df[mean_col] = _round(rng.normal(scale=0.5, size=n_rows))
        df[var_col] = _round(rng.uniform(0.3, 1.2, size=n_rows))
    if with_missings:
        for ob, cells in MISSING_CELLS.items():
            for cell in cells:
                df.loc[cell, ob] = np.nan
    return df


def _params(template: pd.DataFrame, nonlinear: float = 0.0) -> pd.DataFrame:
    """Fill a CHS params template with fixed, well-conditioned values."""
    params = template.copy()
    lin = {
        "fac1": {"fac1": 0.7, "fac2": 0.2, "ob1": 0.9, "ob2": -0.6, "constant": 0.1},
        "fac2": {"fac1": -0.3, "fac2": 0.6, "ob1": 0.5, "ob2": 0.8, "constant": -0.2},
    }
    values = []
    for category, period, name1, name2 in params.index:
        if category == "controls":
            val = 0.1
        elif category == "loadings":
            val = 1.0 if name1.endswith("m1") else 0.8
        elif category == "meas_sds":
            val = 0.5
        elif category == "shock_sds":
            val = 0.4 + 0.1 * period
        elif category == "initial_states":
            val = 0.0 if name1 == "mixture_0" else 1.0
            val += 0.3 if name2 == "fac2" else 0.0
        elif category == "mixture_weights":
            val = 0.4 if name1 == "mixture_0" else 0.6
        elif category == "initial_cholcovs":
            a, b = name2.split("-")
            val = (1.0 if name1 == "mixture_0" else 0.7) if a == b else 0.3
        elif category == "transition":
            val = nonlinear if "*" in name2 else lin[name1][name2] * (1 + 0.1 * period)
        else:
            raise ValueError(category)
        values.append(val)
    params["value"] = values
    return params


def _loglikeobs(model: ModelSpec, data: pd.DataFrame, nonlinear: float = 0.0):
    inputs = get_maximization_inputs(model, data, chs_options=OPTIONS)
    params = _params(inputs["params_template"], nonlinear=nonlinear)
    return inputs["loglikeobs"](params)


def _monte_carlo_likelihood(
    n_mixtures: int, n_draws: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average the per-person likelihood over draws of the missing cells.

    Each draw fills every missing observed-factor cell with `N(m, v)` and runs the
    ordinary filter (all observed factors present). The draws are stacked as
    replicas of the panel so one likelihood evaluation covers all of them.

    Returns:
        Per-person Monte-Carlo likelihood and its standard error.

    """
    rng = np.random.default_rng(seed)
    base = _data().reset_index()
    rep = pd.concat([base] * n_draws, ignore_index=True)
    replica = np.repeat(np.arange(n_draws), len(base))
    rep["id"] = np.tile(base["id"].to_numpy(), n_draws) + N_PERSONS * replica
    for ob, (mean_col, var_col) in MOMENTS.items():
        values = rep[ob].to_numpy(copy=True)
        miss = np.isnan(values)
        mean = rep[mean_col].to_numpy()[miss]
        sd = np.sqrt(rep[var_col].to_numpy()[miss])
        values[miss] = rng.normal(mean, sd)
        rep[ob] = values
    rep = rep.set_index(["id", "period"])
    loglikes = _loglikeobs(_model(n_mixtures), rep).reshape(n_draws, N_PERSONS)
    likes = np.exp(loglikes)
    return likes.mean(axis=0), likes.std(axis=0, ddof=1) / np.sqrt(n_draws)


@pytest.fixture(scope="module", params=[1, 2], ids=["1mix", "2mix"])
def mc_reference(request):
    n_mixtures = request.param
    mc_like, mc_se = _monte_carlo_likelihood(n_mixtures, n_draws=40_000, seed=7)
    return n_mixtures, mc_like, mc_se


def test_linear_filter_likelihood_matches_monte_carlo_integral(mc_reference) -> None:
    """The exact marginal likelihood equals the MC average of per-draw likelihoods."""
    n_mixtures, mc_like, mc_se = mc_reference
    model = _model(n_mixtures, moments=MOMENTS)
    exact = np.exp(_loglikeobs(model, _data()))
    assert np.all(np.abs(exact - mc_like) <= 4 * mc_se + 1e-12 * mc_like)


def test_monte_carlo_reference_rejects_mean_imputation(mc_reference) -> None:
    """Plugging in `m` without its variance is far outside the MC error band.

    This shows the MC comparison has power to detect a missing variance term.
    """
    n_mixtures, mc_like, mc_se = mc_reference
    data = _data()
    for ob, (mean_col, _) in MOMENTS.items():
        data[ob] = data[ob].fillna(data[mean_col])
    naive = np.exp(_loglikeobs(_model(n_mixtures), data))
    affected = mc_se > 0
    assert np.all(np.abs(naive - mc_like)[affected] > 10 * mc_se[affected])


@pytest.mark.parametrize("n_mixtures", [1, 2])
def test_unscented_path_matches_linear_path_for_linear_translog(n_mixtures) -> None:
    """A translog with all squares/interactions at 0 reproduces the linear filter."""
    data = _data()
    linear = _loglikeobs(_model(n_mixtures, "linear", MOMENTS), data)
    unscented = _loglikeobs(_model(n_mixtures, "translog", MOMENTS), data)
    assert_allclose(unscented, linear, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("transition", ["linear", "translog"])
def test_declared_moments_without_missings_leave_loglike_unchanged(
    transition,
) -> None:
    """Declaring moments for fully observed factors does not change the filter."""
    data = _data(with_missings=False)
    plain = _loglikeobs(_model(2, transition), data, nonlinear=0.05)
    declared = _loglikeobs(_model(2, transition, MOMENTS), data, nonlinear=0.05)
    assert np.array_equal(plain, declared)


@pytest.mark.parametrize("transition", ["linear", "translog"])
def test_loglike_without_missings_matches_baseline(transition) -> None:
    """Models with complete observed factors keep their pre-feature likelihood."""
    data = _data(with_missings=False)
    loglike = _loglikeobs(_model(2, transition), data, nonlinear=0.05).sum()
    assert_allclose(loglike, BASELINE_LOGLIKE[transition], rtol=1e-12)


def _endogenous_model(moments: dict | None) -> ModelSpec:
    """Endogenous investment `inv` and state `s` share the observed inputs."""
    return ModelSpec(
        factors={
            "s": FactorSpec(
                measurements=(("y",), ("y",)), transition_function="linear"
            ),
            "inv": FactorSpec(
                measurements=(("i",), ()),
                is_endogenous=True,
                transition_function="linear",
            ),
        },
        observed_factors=("x", "z"),
        missing_observed_factor_moments=moments,
    )


def _endogenous_data(variance: float | None = None) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=["id", "period"])
    data = pd.DataFrame(
        {"y": [-1.0, 0.0, 1.0, 2.0], "i": np.nan, "x": 0.0, "z": 0.0}, index=idx
    )
    data[["m", "v"]] = 0.0
    if variance is not None:
        data.loc[(0, 0), ["x", "v"]] = [np.nan, variance]
    return data


def _endogenous_loglike(model: ModelSpec, data: pd.DataFrame) -> float:
    inputs = get_maximization_inputs(model, data, chs_options=OPTIONS)
    params = inputs["params_template"].copy()
    params["value"] = np.where(
        params.index.get_level_values("category").isin(
            ["meas_sds", "shock_sds", "loadings"]
        ),
        0.5,
        0.1,
    )
    for idx in params.query("category == 'initial_cholcovs'").index:
        row, col = idx[3].split("-")
        params.loc[idx, "value"] = float(row == col)
    return float(inputs["loglike"](params))


@pytest.mark.parametrize(
    "moments",
    [
        {"x": ("m", "v"), "z": ("m", "v")},
        {"x": ("z", "v"), "z": ("m", "v")},
    ],
    ids=["shared_pair", "moment_is_observed_factor"],
)
def test_moment_columns_shared_or_aliased_leave_loglike_unchanged(moments) -> None:
    """Moment columns may be shared across factors or alias a selected column."""
    data = _endogenous_data()
    plain = _endogenous_loglike(_endogenous_model(None), data)
    declared = _endogenous_loglike(_endogenous_model(moments), data)
    assert_allclose(declared, plain, rtol=0, atol=1e-12)


def test_endogenous_model_rejects_uncertain_missing_input() -> None:
    """Both half-steps would need the input's joint posterior, which is not carried."""
    model = _endogenous_model({"x": ("m", "v")})
    with pytest.raises(NotImplementedError, match="joint missing-observed-factor"):
        get_maximization_inputs(model, _endogenous_data(1.0), chs_options=OPTIONS)


def test_endogenous_model_plugs_in_zero_variance_fill() -> None:
    """A zero-variance fill acts like observing the fill mean."""
    model = _endogenous_model({"x": ("m", "v")})
    filled = _endogenous_loglike(model, _endogenous_data(0.0))
    plugged_in = _endogenous_loglike(model, _endogenous_data())
    assert_allclose(filled, plugged_in, rtol=0, atol=1e-12)


def test_moments_where_factor_is_observed_are_ignored() -> None:
    """Fill moments only matter in cells where the observed factor is missing."""
    data = _data()
    garbled = data.copy()
    for ob, (mean_col, var_col) in MOMENTS.items():
        observed = garbled[ob].notna()
        garbled.loc[observed, mean_col] = np.nan
        garbled.loc[observed, var_col] = -1.0
    model = _model(1, moments=MOMENTS)
    assert np.array_equal(_loglikeobs(model, data), _loglikeobs(model, garbled))


@pytest.mark.parametrize("bad_col", ["ob1_mean", "ob1_var"])
def test_missing_factor_without_moments_raises(bad_col) -> None:
    data = _data()
    data.loc[(1, 1), bad_col] = np.nan
    with pytest.raises(ValueError, match=r"ob1[\s\S]*fill moments") as excinfo:
        get_maximization_inputs(_model(1, moments=MOMENTS), data, chs_options=OPTIONS)
    assert "1" in str(excinfo.value)


def test_negative_fill_variance_raises() -> None:
    data = _data()
    data.loc[(1, 1), "ob1_var"] = -0.5
    with pytest.raises(ValueError, match=r"ob1[\s\S]*negative fill variance"):
        get_maximization_inputs(_model(1, moments=MOMENTS), data, chs_options=OPTIONS)


def test_missing_moment_column_raises() -> None:
    data = _data().drop(columns=["ob2_var"])
    with pytest.raises(ValueError, match=r"ob2_var[\s\S]*Variable is missing"):
        get_maximization_inputs(_model(1, moments=MOMENTS), data, chs_options=OPTIONS)


def test_missing_factor_without_declared_moments_raises() -> None:
    moments = {"ob1": MOMENTS["ob1"]}
    with pytest.raises(ValueError, match=r"ob2[\s\S]*Variable has missings"):
        get_maximization_inputs(
            _model(1, moments=moments), _data(), chs_options=OPTIONS
        )


def test_model_spec_rejects_moments_for_unknown_observed_factor() -> None:
    with pytest.raises(ValueError, match="not_an_observed_factor"):
        _model(1, moments={"not_an_observed_factor": ("a", "b")})


def test_model_spec_from_dict_reads_moments() -> None:
    spec = ModelSpec.from_dict(
        {
            "factors": {"fac1": {"measurements": [["y1"], ["y1"]]}},
            "observed_factors": ["ob1"],
            "missing_observed_factor_moments": {"ob1": ["ob1_mean", "ob1_var"]},
        }
    )
    assert dict(spec.missing_observed_factor_moments) == {
        "ob1": ("ob1_mean", "ob1_var")
    }


def _predict_inputs():
    """Inputs for one predict step: 3 persons, 2 mixtures, 2 latent, 2 observed.

    Person 0 misses ob1, person 1 misses both, person 2 misses none.
    """
    rng = np.random.default_rng(0)
    n_obs, n_mix, n_lat = 3, 2, 2
    states = rng.normal(size=(n_obs, n_mix, n_lat))
    covs = np.empty((n_obs, n_mix, n_lat, n_lat))
    chols = np.empty_like(covs)
    for i in range(n_obs):
        for k in range(n_mix):
            a = rng.normal(size=(n_lat, n_lat))
            covs[i, k] = a @ a.T + np.eye(n_lat)
            chols[i, k] = np.linalg.cholesky(covs[i, k]).T
    # Means are already substituted where the factor is missing.
    observed = rng.normal(size=(n_obs, 2))
    # float32-representable variances with irrational roots.
    obs_vars = np.array([[0.75, 0.0], [0.375, 1.25], [0.0, 0.0]])
    f_obs = np.array([[0.9, -0.6], [0.5, 0.8]])
    f_lat = np.array([[0.7, 0.2], [-0.3, 0.6]])
    const = np.array([0.1, -0.2])
    shock_sds = np.array([0.4, 0.3])
    scaling = np.array([[1.5, 0.8, 1.0, 1.0], [1.2, 0.9, 1.0, 1.0]])
    constants = np.array([[0.3, -0.1, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]])
    return {
        "states": states,
        "covs": covs,
        "chols": chols,
        "observed": observed,
        "obs_vars": obs_vars,
        "f_lat": f_lat,
        "f_obs": f_obs,
        "const": const,
        "shock_sds": shock_sds,
        "scaling": scaling,
        "constants": constants,
    }


def _analytic_prediction(inp):
    """Predicted mean and covariance with `x ~ N(m, diag(v))` integrated out."""
    s_in, s_out = inp["scaling"][0, :2], inp["scaling"][1, :2]
    c_in, c_out = inp["constants"][0, :2], inp["constants"][1, :2]
    n_obs, n_mix, _ = inp["states"].shape
    means = np.empty_like(inp["states"])
    covs = np.empty_like(inp["covs"])
    for i in range(n_obs):
        for k in range(n_mix):
            anchored = inp["states"][i, k] * s_in + c_in
            pred = inp["f_lat"] @ anchored + inp["f_obs"] @ inp["observed"][i]
            means[i, k] = (pred + inp["const"] - c_out) / s_out
            g_mat = np.diag(1 / s_out) @ inp["f_lat"] @ np.diag(s_in)
            b_mat = np.diag(1 / s_out) @ inp["f_obs"]
            covs[i, k] = (
                g_mat @ inp["covs"][i, k] @ g_mat.T
                + np.diag((inp["shock_sds"] / s_out) ** 2)
                + b_mat @ np.diag(inp["obs_vars"][i]) @ b_mat.T
            )
    return means, covs


def _linear_trans_coeffs(inp):
    return {
        f: jnp.array([*inp["f_lat"][j], *inp["f_obs"][j], inp["const"][j]])
        for j, f in enumerate(LATENT)
    }


@pytest.mark.parametrize("variance_dtype", [jnp.float64, jnp.float32])
def test_linear_predict_integrates_out_missing_observed_factors(
    variance_dtype,
) -> None:
    inp = _predict_inputs()
    exp_means, exp_covs = _analytic_prediction(inp)
    means, chols = linear_kalman_predict(
        None,
        jnp.array(inp["states"]),
        jnp.array(inp["chols"]),
        1.0,
        jnp.ones(1),
        _linear_trans_coeffs(inp),
        jnp.array(inp["shock_sds"]),
        jnp.array(inp["scaling"]),
        jnp.array(inp["constants"]),
        jnp.array(inp["observed"]),
        observed_factor_vars=jnp.array(inp["obs_vars"], dtype=variance_dtype),
        latent_factors=LATENT,
        constant_factor_indices=frozenset(),
        n_all_factors=4,
        missing_observed_indices=(0, 1),
    )
    covs = np.einsum("nkja,nkjb->nkab", chols, chols)
    assert_allclose(means, exp_means, rtol=1e-12)
    assert_allclose(covs, exp_covs, rtol=1e-12)


def _linear_transition(params, states):
    return jnp.column_stack([states @ params[f][:-1] + params[f][-1] for f in LATENT])


@pytest.mark.parametrize("variance_dtype", [jnp.float64, jnp.float32])
def test_unscented_predict_integrates_out_missing_observed_factors(
    variance_dtype,
) -> None:
    inp = _predict_inputs()
    exp_means, exp_covs = _analytic_prediction(inp)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(4, 2)
    # The unscented predict adds shocks in the unanchored output scale.
    shock_sds = inp["shock_sds"] / inp["scaling"][1, :2]
    means, chols = kalman_predict(
        _linear_transition,
        jnp.array(inp["states"]),
        jnp.array(inp["chols"]),
        scaling_factor,
        weights,
        _linear_trans_coeffs(inp),
        jnp.array(shock_sds),
        jnp.array(inp["scaling"]),
        jnp.array(inp["constants"]),
        jnp.array(inp["observed"]),
        observed_factor_vars=jnp.array(inp["obs_vars"], dtype=variance_dtype),
        missing_observed_indices=(0, 1),
    )
    covs = np.einsum("nkja,nkjb->nkab", chols, chols)
    assert_allclose(means, exp_means, rtol=1e-10)
    assert_allclose(covs, exp_covs, rtol=1e-10)


def _nonlinear_transition(params, states):
    lat = states[:, :2]
    out = _linear_transition(params, states)
    return out + 0.2 * lat**2 - 0.1 * lat[:, ::-1] * states[:, 2:3]


def test_unscented_predict_augmented_by_observed_cells_equals_larger_kappa() -> None:
    """With zero fill variance, the augmented sigma points act like `kappa + n_aug`.

    The latent marginal of the augmented sigma-point set is the plain set built
    with `kappa` raised by the number of augmented dimensions.
    """
    inp = _predict_inputs()
    common = (
        _nonlinear_transition,
        jnp.array(inp["states"]),
        jnp.array(inp["chols"]),
    )
    rest = (
        _linear_trans_coeffs(inp),
        jnp.array(inp["shock_sds"]),
        jnp.array(inp["scaling"]),
        jnp.array(inp["constants"]),
        jnp.array(inp["observed"]),
    )
    aug_scaling, aug_weights = calculate_sigma_scaling_factor_and_weights(4, 2)
    aug_means, aug_chols = kalman_predict(
        *common,
        aug_scaling,
        aug_weights,
        *rest,
        observed_factor_vars=jnp.zeros((3, 2)),
        missing_observed_indices=(0, 1),
    )
    plain_scaling, plain_weights = calculate_sigma_scaling_factor_and_weights(2, 4)
    plain_means, plain_chols = kalman_predict(
        *common, plain_scaling, plain_weights, *rest
    )
    aug_covs = np.einsum("nkja,nkjb->nkab", aug_chols, aug_chols)
    plain_covs = np.einsum("nkja,nkjb->nkab", plain_chols, plain_chols)
    assert_allclose(aug_means, plain_means, rtol=1e-12)
    assert_allclose(aug_covs, plain_covs, rtol=1e-12)
