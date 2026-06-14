"""Regression tests for the AF kappa-parameter scope guard.

These pin the documented AF scope boundary (kappa_t = 0, exogenous
investment): a caller cannot silently smuggle a `kappa`/`kappa_t`
parameter, while standard parameter categories still pass.
"""

import pandas as pd
import pytest

from skillmodels.af.validate import fail_if_unsupported_kappa_params


def test_fail_if_unsupported_kappa_params_rejects_kappa_in_start_params() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("kappa", 0, "skill", "-")],
        names=["category", "period", "name1", "name2"],
    )
    start = pd.DataFrame({"value": [0.3]}, index=idx)
    with pytest.raises(NotImplementedError, match="kappa"):
        fail_if_unsupported_kappa_params(start, None, None)


def test_fail_if_unsupported_kappa_params_rejects_kappa_in_fixed_params() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("kappa_t", 0, "skill", "-")],
        names=["category", "period", "name1", "name2"],
    )
    fixed = pd.DataFrame({"value": [0.0]}, index=idx)
    with pytest.raises(NotImplementedError):
        fail_if_unsupported_kappa_params(None, fixed, None)


def test_fail_if_unsupported_kappa_params_allows_standard_categories() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("shock_sds", 0, "skill", "-"), ("investment_sds", 0, "inv", "-")],
        names=["category", "period", "name1", "name2"],
    )
    start = pd.DataFrame({"value": [0.5, 0.5]}, index=idx)
    # Must NOT raise.
    fail_if_unsupported_kappa_params(start, None, None)
