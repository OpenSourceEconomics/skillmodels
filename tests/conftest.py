"""Shared test fixtures and helpers.

Tests opt in to whole-package beartype via `beartype.claw.beartype_package`
here so that annotation drift on *internal* helpers surfaces as a
`BeartypeCallHintParamViolation` during the test run. The perimeter
decorators (`skillmodels._beartype_conf`) keep raising project-specific
exception classes for *user-facing* parameter violations; the claw-installed
checks below are for everything in between.

`skillmodels.chs.qr` is skipped because it relies on JAX's `@custom_jvp`
decorator, which beartype.claw wraps in a way that strips the
`.defjvp` attribute that the second-stage `@qr_gpu.defjvp` decoration
needs. No annotations in that module are user-facing.
"""

from beartype import BeartypeConf
from beartype.claw import beartype_package

# Mirror the perimeter conf's PEP-484 numeric tower so `int` satisfies
# `float`-typed parameters. Without this every `value=1` call site
# (e.g. `FixedConstraintWithValue(value=1)`) trips the claw checker.
beartype_package(
    "skillmodels",
    conf=BeartypeConf(
        is_pep484_tower=True,
        claw_skip_package_names=("skillmodels.chs.qr",),
    ),
)

from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from skillmodels.common.config import TEST_DATA_DIR  # noqa: E402
from skillmodels.test_data.model2 import MODEL2  # noqa: E402

REGRESSION_VAULT = Path(__file__).parent / "regression_vault"


@pytest.fixture
def model2():
    """Return MODEL2 test model specification."""
    return MODEL2


@pytest.fixture
def model2_data():
    """Load model2 simulated data with (caseid, period) index."""
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta")
    return data.set_index(["caseid", "period"])


@pytest.fixture
def model2_params():
    """Load regression vault params for one_stage_anchoring."""
    params = pd.read_csv(REGRESSION_VAULT / "one_stage_anchoring.csv")
    return params.set_index(["category", "period", "name1", "name2"])


@pytest.fixture
def model2_with_endogenous():
    """Model2 with fac3 set as endogenous factor."""
    fac3 = MODEL2.factors["fac3"]
    new_fac3 = replace(fac3, is_endogenous=True)
    new_factors = dict(MODEL2.factors) | {"fac3": new_fac3}
    return MODEL2._replace(
        factors=new_factors,
        stagemap=None,
        anchoring=None,
    )
