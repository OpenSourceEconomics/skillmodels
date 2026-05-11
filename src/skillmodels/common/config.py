"""Configuration constants and paths for skillmodels."""

from pathlib import Path

# `__file__` lives in src/skillmodels/common/config.py; test_data sits in
# src/skillmodels/test_data so resolve one level up.
TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
REGRESSION_VAULT = (
    Path(__file__).resolve().parent.parent.parent.parent / "tests" / "regression_vault"
)
