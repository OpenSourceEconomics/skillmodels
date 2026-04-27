"""Smoke tests for the CNLSY MATLAB data loader."""

from pathlib import Path

import numpy as np
import pytest

from .load_cnlsy import (
    INCOME_MEASURE,
    INV_MEASURES,
    MC_MEASURES,
    MN_MEASURES,
    SKILL_MEASURES,
    load_measurements,
)

_DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "complete_7_9_11.xls"


pytestmark = pytest.mark.skipif(
    not _DEFAULT_DATA_PATH.exists(),
    reason=f"CNLSY reference data not available at {_DEFAULT_DATA_PATH}",
)


@pytest.fixture(scope="module")
def cnlsy_data():
    return load_measurements(_DEFAULT_DATA_PATH)


def test_cnlsy_has_expected_shape(cnlsy_data) -> None:
    assert len(cnlsy_data) == 1403 * 3
    assert cnlsy_data.index.names == ["caseid", "period"]


def test_cnlsy_skill_measurements_are_standardised_per_period(cnlsy_data) -> None:
    for period in (0, 1, 2):
        panel = cnlsy_data.xs(period, level="period")
        for col in SKILL_MEASURES:
            values = panel[col].to_numpy()
            assert np.isclose(values.mean(), 0.0, atol=1e-8)
            assert np.isclose(values.std(), 1.0, atol=1e-8)


def test_cnlsy_mc_mn_filled_only_in_period_zero(cnlsy_data) -> None:
    period_zero = cnlsy_data.xs(0, level="period")
    for col in (*MC_MEASURES, *MN_MEASURES):
        assert period_zero[col].notna().all()
    for period in (1, 2):
        panel = cnlsy_data.xs(period, level="period")
        for col in (*MC_MEASURES, *MN_MEASURES):
            assert panel[col].isna().all()


def test_cnlsy_investment_filled_in_periods_zero_and_one(cnlsy_data) -> None:
    for period in (0, 1):
        panel = cnlsy_data.xs(period, level="period")
        for col in INV_MEASURES:
            assert panel[col].notna().all()
    panel_two = cnlsy_data.xs(2, level="period")
    for col in INV_MEASURES:
        assert panel_two[col].isna().all()


def test_cnlsy_log_income_period_two_holds_period_one(cnlsy_data) -> None:
    """Period 2 log income is hold-last-value from period 1 (faminc9).

    The shipped file has no ``faminc11``. Filling with period 1's value
    lets CHS's ``process_data`` consume the frame without raising on
    missing observed factors; AF does not read ``log_income`` at
    period 2 so the imputed values do not affect its likelihood.
    """
    period_one = cnlsy_data.xs(1, level="period")[INCOME_MEASURE]
    period_two = cnlsy_data.xs(2, level="period")[INCOME_MEASURE]
    np.testing.assert_array_equal(period_two.to_numpy(), period_one.to_numpy())
