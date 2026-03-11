"""Tests for types module."""

import pickle
from types import MappingProxyType

import pytest

from skillmodels.types import FactorInfo, _make_immutable


def test_make_immutable_list_to_tuple() -> None:
    assert _make_immutable([1, 2]) == (1, 2)


def test_make_immutable_set_to_frozenset() -> None:
    assert _make_immutable({1, 2}) == frozenset({1, 2})


def test_mapping_proxy_pickle_roundtrip() -> None:
    mp = MappingProxyType({"a": 1, "b": [2, 3]})
    result = pickle.loads(pickle.dumps(mp))
    assert dict(result) == {"a": 1, "b": (2, 3)}


def test_factor_info_from_flags_all_false() -> None:
    info = FactorInfo.from_flags(is_endogenous=False, is_correction=False)
    assert info.is_state


def test_factor_info_from_flags_correction() -> None:
    info = FactorInfo.from_flags(is_endogenous=True, is_correction=True)
    assert info.is_correction


def test_factor_info_from_flags_both_true_raises() -> None:
    with pytest.raises(ValueError, match=r"correction.*endogenous"):
        FactorInfo.from_flags(is_endogenous=False, is_correction=True)
