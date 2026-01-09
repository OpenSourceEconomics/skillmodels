"""Configuration constants and paths for skillmodels."""

from pathlib import Path

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
REGRESSION_VAULT = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "regression_vault"
)
