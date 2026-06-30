"""Configuration constants and paths for skillmodels."""

from pathlib import Path

# `__file__` lives in src/skillmodels/common/config.py; test_data sits in
# src/skillmodels/test_data so resolve one level up.
TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
REGRESSION_VAULT = (
    Path(__file__).resolve().parent.parent.parent.parent / "tests" / "regression_vault"
)

# Long-format CNLSY measurements used by the AF 2025 application; produced
# by ``matlab_ces_repro/load_cnlsy.py`` from the bundled ``complete_7_9_11.xls``
# (CNLSY public-use data, BLS). Read with
# ``pd.read_csv(CNLSY_DATA_PATH).set_index(["caseid", "period"])``.
CNLSY_DATA_PATH = TEST_DATA_DIR / "cnlsy_7_9_11.csv"
