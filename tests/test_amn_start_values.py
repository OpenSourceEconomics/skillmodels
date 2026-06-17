"""Tests for `skillmodels.amn.start_values.get_spearman_start_params`.

These tests exercise the Spearman + Bartlett-OLS start-value pipeline
(the legacy default, now opt-in via `start_params_strategy="spearman"`).
The new default `"amn"` runs the full Attanasio-Meghir-Nix estimator
upfront and is tested in `test_amn_estimate.py` and via
`test_maximization_inputs.py`.
"""

import functools

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.amn.start_values import (
    _apply_neutral_defaults,
    get_spearman_start_params,
    pool_equality_groups,
)
from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.constraints import select_by_loc
from skillmodels.common.model_spec import ModelSpec
from skillmodels.common.utilities import reduce_n_periods
from skillmodels.test_data.model2 import MODEL2, MODEL2_CHS_OPTIONS


def test_apply_neutral_defaults_fills_correction_categories() -> None:
    """Neutral defaults must seed `investment_eq` and `kappa`.

    The first-stage (`investment_eq`) and control-function (`kappa`)
    coefficients are not produced by the moment / AMN overrides for every
    model, so the neutral defaults must cover them; otherwise the seeded
    start point keeps NaNs and `optimagic` rejects it. They seed to 0 — no
    first-stage relationship and no correction initially.
    """
    index = pd.MultiIndex.from_tuples(
        [
            ("investment_eq", 2, "inv", "fac1"),
            ("investment_eq", 2, "inv", "constant"),
            ("kappa", 1, "fac1", "cf"),
            ("transition", 0, "fac1", "fac1"),
        ],
        names=["category", "aug_period", "name1", "name2"],
    )
    params = pd.DataFrame({"value": [np.nan] * len(index)}, index=index)
    free = params["value"].isna()

    _apply_neutral_defaults(params, free, n_mixtures=1)

    assert not params["value"].isna().any()
    assert params.loc[("investment_eq", 2, "inv", "fac1"), "value"] == 0.0
    assert params.loc[("kappa", 1, "fac1", "cf"), "value"] == 0.0


def test_apply_neutral_defaults_seeds_higher_order_terms_small() -> None:
    """Higher-order terms seed to a small 0.01, not the linear defaults.

    Translog interactions / squares (`"fac1 * fac2"`, `"fac1 ** 2"`) and
    higher-order control-function terms (`"cf * fac1"`, `"cf ** 2"`) are not
    produced by the linear AMN/Spearman seeds. They get a small start so the
    optimiser explores away from zero without the higher-order monomials
    dominating the seeded production function.
    """
    index = pd.MultiIndex.from_tuples(
        [
            ("transition", 0, "fac1", "fac1"),
            ("transition", 0, "fac1", "fac1 * fac2"),
            ("transition", 0, "fac1", "fac1 ** 2"),
            ("kappa", 1, "fac1", "cf"),
            ("kappa", 1, "fac1", "cf * fac1"),
            ("kappa", 1, "fac1", "cf ** 2"),
        ],
        names=["category", "aug_period", "name1", "name2"],
    )
    params = pd.DataFrame({"value": [np.nan] * len(index)}, index=index)
    free = params["value"].isna()

    _apply_neutral_defaults(params, free, n_mixtures=1)

    # Linear terms keep their category defaults ...
    assert params.loc[("transition", 0, "fac1", "fac1"), "value"] == 0.5
    assert params.loc[("kappa", 1, "fac1", "cf"), "value"] == 0.0
    # ... higher-order terms (a space in `name2`) seed small.
    assert params.loc[("transition", 0, "fac1", "fac1 * fac2"), "value"] == 0.01
    assert params.loc[("transition", 0, "fac1", "fac1 ** 2"), "value"] == 0.01
    assert params.loc[("kappa", 1, "fac1", "cf * fac1"), "value"] == 0.01
    assert params.loc[("kappa", 1, "fac1", "cf ** 2"), "value"] == 0.01


@pytest.fixture
def model2_short() -> ModelSpec:
    spec = reduce_n_periods(MODEL2, new_n_periods=3)
    assert isinstance(spec, ModelSpec)
    return spec


@pytest.fixture
def model2_data() -> pd.DataFrame:
    return pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta").set_index(
        ["caseid", "period"]
    )


def test_default_strategy_is_amn() -> None:
    """`CHSEstimationOptions().start_params_strategy` defaults to "amn"."""
    assert CHSEstimationOptions().start_params_strategy == "amn"


def test_template_filled_with_spearman_strategy(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """`start_params_strategy="spearman"` returns a fully-populated template."""
    inputs = get_maximization_inputs(
        model2_short,
        model2_data,
        chs_options=CHSEstimationOptions(start_params_strategy="spearman"),
    )
    template = inputs["params_template"]
    assert not template["value"].isna().any()


def test_strategy_none_leaves_nan(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """`start_params_strategy="none"` reproduces the legacy NaN behaviour."""
    inputs = get_maximization_inputs(
        model2_short,
        model2_data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    template = inputs["params_template"]
    assert template["value"].isna().any()


def test_filled_template_yields_finite_loglike(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """The moment-seeded template produces a finite log-likelihood."""
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    val = inputs["loglike"](inputs["params_template"])
    assert np.isfinite(val)


def test_loadings_seeded_from_data_not_constant(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Loadings vary across measurements (Spearman seed, not flat 1.0)."""
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    template = inputs["params_template"]
    loadings = template.loc["loadings", "value"]
    free = (
        template.loc["loadings", "lower_bound"]
        != template.loc["loadings", "upper_bound"]
    )
    free_loadings = loadings[free].to_numpy()
    assert (free_loadings != free_loadings[0]).any()
    assert not np.allclose(free_loadings, 1.0)


def test_meas_sds_seeded_from_data_not_constant(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Measurement SDs vary across indicators (residual SD seed, not 0.5)."""
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    template = inputs["params_template"]
    meas_sds = template.loc["meas_sds", "value"].to_numpy()
    assert (meas_sds != meas_sds[0]).any()
    assert (meas_sds > 0).all()


def test_initial_cholcovs_diagonal_is_positive(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Initial-cov diagonals are positive (sqrt(latent_var))."""
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    template = inputs["params_template"]
    cholcov = template.loc["initial_cholcovs", "value"]
    diag_mask = pd.Series(
        [name2.split("-")[0] == name2.split("-")[1] for *_, name2 in cholcov.index],
        index=cholcov.index,
    )
    assert (cholcov[diag_mask] > 0).all()


def test_fixed_params_pin_survives_moment_fill(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Entries set by `fixed_params` keep their pinned value."""
    fixed_idx = pd.MultiIndex.from_tuples(
        [("transition", 0, "fac1", "fac3"), ("transition", 1, "fac1", "fac3")],
        names=["category", "period", "name1", "name2"],
    )
    fixed_df = pd.DataFrame({"value": [0.0, 0.0]}, index=fixed_idx)
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS, fixed_params=fixed_df
    )
    template = inputs["params_template"]
    assert template.loc[("transition", 0, "fac1", "fac3"), "value"] == 0.0
    assert template.loc[("transition", 1, "fac1", "fac3"), "value"] == 0.0


def test_explicit_strategy_argument_via_helper(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """The standalone helper produces the same fills as the wired-in spearman path."""
    from skillmodels.common.constraints import (  # noqa: PLC0415
        project_to_probability_constraints,
    )

    inputs_raw = get_maximization_inputs(
        model2_short,
        model2_data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    template_raw = inputs_raw["params_template"]
    filled = get_spearman_start_params(model2_short, model2_data, template_raw)
    # The wired-in path renormalizes free entries of every
    # ProbabilityConstraint to sum to one after the strategy step;
    # apply the same projection here so the two paths can be compared.
    filled = project_to_probability_constraints(
        params_template=filled, constraints=inputs_raw["constraints"]
    )

    inputs_spearman = get_maximization_inputs(
        model2_short,
        model2_data,
        chs_options=CHSEstimationOptions(start_params_strategy="spearman"),
    )
    template_spearman = inputs_spearman["params_template"]

    pd.testing.assert_series_equal(filled["value"], template_spearman["value"])


def test_helper_does_not_overwrite_user_set_values(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """If the caller already set a non-NaN value, the helper preserves it."""
    inputs = get_maximization_inputs(
        model2_short,
        model2_data,
        chs_options=CHSEstimationOptions(start_params_strategy="none"),
    )
    template = inputs["params_template"]
    sentinel_loc = template.index[template["value"].isna()][0]
    template.loc[sentinel_loc, "value"] = 999.0
    filled = get_spearman_start_params(model2_short, model2_data, template)
    assert filled.loc[sentinel_loc, "value"] == 999.0


def test_transition_coefficients_seeded_via_ols(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Free transition rows get AMN-style OLS seeds, not constant 0.5."""
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    template = inputs["params_template"]
    free_trans = template.loc["transition"]
    free_mask = free_trans["lower_bound"] != free_trans["upper_bound"]
    free_values = free_trans.loc[free_mask, "value"]
    assert (free_values != 0.5).any()


def test_shock_sds_seeded_via_residual_variance(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Free shock_sds rows get residual-variance seeds, not flat 0.5."""
    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS
    )
    template = inputs["params_template"]
    free_sds = template.loc["shock_sds"]
    free_mask = free_sds["lower_bound"] != free_sds["upper_bound"]
    free_values = free_sds.loc[free_mask, "value"]
    assert (free_values != 0.5).any()


def test_pool_equality_groups_averages_unpinned() -> None:
    """Members of an `om.EqualityConstraint` group are averaged."""
    idx = pd.MultiIndex.from_tuples(
        [
            ("meas_sds", 0, "z1", "-"),
            ("meas_sds", 1, "z1", "-"),
            ("meas_sds", 2, "z1", "-"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    params = pd.DataFrame({"value": [0.2, 0.4, 0.6]}, index=idx)
    constraints: list[om.constraints.Constraint] = [
        om.EqualityConstraint(
            selector=functools.partial(select_by_loc, loc=idx),
        ),
    ]
    out = pool_equality_groups(params, constraints)
    assert list(out["value"]) == pytest.approx([0.4, 0.4, 0.4])


def test_pool_equality_groups_respects_pinned() -> None:
    """If any group member is pinned, that value propagates to the rest."""
    idx = pd.MultiIndex.from_tuples(
        [
            ("meas_sds", 0, "z1", "-"),
            ("meas_sds", 1, "z1", "-"),
            ("meas_sds", 2, "z1", "-"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    params = pd.DataFrame({"value": [0.2, 0.4, 0.6]}, index=idx)
    pinned = pd.Series([False, True, False], index=idx)
    constraints: list[om.constraints.Constraint] = [
        om.EqualityConstraint(
            selector=functools.partial(select_by_loc, loc=idx),
        ),
    ]
    out = pool_equality_groups(params, constraints, keep_pinned_values=pinned)
    assert list(out["value"]) == pytest.approx([0.4, 0.4, 0.4])
