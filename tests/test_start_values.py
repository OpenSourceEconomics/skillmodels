"""Tests for `skillmodels.start_values.get_moment_based_start_params`."""

import functools

import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.chs.maximization_inputs import get_maximization_inputs
from skillmodels.config import TEST_DATA_DIR
from skillmodels.constraints import select_by_loc
from skillmodels.model_spec import ModelSpec
from skillmodels.start_values import (
    get_moment_based_start_params,
    pool_equality_groups,
)
from skillmodels.test_data.model2 import MODEL2
from skillmodels.types import EstimationOptions
from skillmodels.utilities import reduce_n_periods


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


def test_default_strategy_is_moment_based() -> None:
    """`EstimationOptions().start_params_strategy` defaults to moment_based."""
    assert EstimationOptions().start_params_strategy == "moment_based"


def test_template_filled_with_moment_based_default(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Default `get_maximization_inputs` returns a fully-populated template."""
    inputs = get_maximization_inputs(model2_short, model2_data)
    template = inputs["params_template"]
    assert not template["value"].isna().any()


def test_strategy_none_leaves_nan(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """`start_params_strategy="none"` reproduces the legacy NaN behaviour."""
    spec_none = model2_short.with_estimation_options(
        EstimationOptions(start_params_strategy="none")
    )
    inputs = get_maximization_inputs(spec_none, model2_data)
    template = inputs["params_template"]
    assert template["value"].isna().any()


def test_filled_template_yields_finite_loglike(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """The moment-seeded template produces a finite log-likelihood."""
    inputs = get_maximization_inputs(model2_short, model2_data)
    val = inputs["loglike"](inputs["params_template"])
    assert np.isfinite(val)


def test_loadings_seeded_from_data_not_constant(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Loadings vary across measurements (Spearman seed, not flat 1.0)."""
    inputs = get_maximization_inputs(model2_short, model2_data)
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
    inputs = get_maximization_inputs(model2_short, model2_data)
    template = inputs["params_template"]
    meas_sds = template.loc["meas_sds", "value"].to_numpy()
    assert (meas_sds != meas_sds[0]).any()
    assert (meas_sds > 0).all()


def test_initial_cholcovs_diagonal_is_positive(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Initial-cov diagonals are positive (sqrt(latent_var))."""
    inputs = get_maximization_inputs(model2_short, model2_data)
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
    inputs = get_maximization_inputs(model2_short, model2_data, fixed_params=fixed_df)
    template = inputs["params_template"]
    assert template.loc[("transition", 0, "fac1", "fac3"), "value"] == 0.0
    assert template.loc[("transition", 1, "fac1", "fac3"), "value"] == 0.0


def test_explicit_strategy_argument_via_helper(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """The standalone helper produces the same fills as the wired-in path."""
    spec_none = model2_short.with_estimation_options(
        EstimationOptions(start_params_strategy="none")
    )
    inputs_raw = get_maximization_inputs(spec_none, model2_data)
    template_raw = inputs_raw["params_template"]
    filled = get_moment_based_start_params(spec_none, model2_data, template_raw)

    inputs_default = get_maximization_inputs(model2_short, model2_data)
    template_default = inputs_default["params_template"]

    pd.testing.assert_series_equal(filled["value"], template_default["value"])


def test_helper_does_not_overwrite_user_set_values(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """If the caller already set a non-NaN value, the helper preserves it."""
    spec_none = model2_short.with_estimation_options(
        EstimationOptions(start_params_strategy="none")
    )
    inputs = get_maximization_inputs(spec_none, model2_data)
    template = inputs["params_template"]
    sentinel_loc = template.index[template["value"].isna()][0]
    template.loc[sentinel_loc, "value"] = 999.0
    filled = get_moment_based_start_params(spec_none, model2_data, template)
    assert filled.loc[sentinel_loc, "value"] == 999.0


def test_transition_coefficients_seeded_via_ols(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Free transition rows get AMN-style OLS seeds, not constant 0.5."""
    inputs = get_maximization_inputs(model2_short, model2_data)
    template = inputs["params_template"]
    free_trans = template.loc["transition"]
    free_mask = free_trans["lower_bound"] != free_trans["upper_bound"]
    free_values = free_trans.loc[free_mask, "value"]
    assert (free_values != 0.5).any()


def test_shock_sds_seeded_via_residual_variance(
    model2_short: ModelSpec, model2_data: pd.DataFrame
) -> None:
    """Free shock_sds rows get residual-variance seeds, not flat 0.5."""
    inputs = get_maximization_inputs(model2_short, model2_data)
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
