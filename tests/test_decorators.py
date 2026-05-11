"""Tests for decorators."""

import jax.numpy as jnp
import pytest

from skillmodels.common.decorators import (
    extract_params,
    jax_array_output,
    register_params,
)


def test_extract_params_decorator_only_key() -> None:
    @extract_params(key="a")
    def f(x, params):
        return x * params

    assert f(x=3, params={"a": 4, "b": 5}) == 12


def test_extract_params_direct_call_only_key() -> None:
    def f(x, params):
        return x * params

    g = extract_params(f, key="a")

    assert g(x=3, params={"a": 4, "b": 5}) == 12


def test_extract_params_decorator_only_names() -> None:
    @extract_params(names=["c", "d"])
    def f(x, params):
        return x * params["c"]

    assert f(x=3, params=[4, 5]) == 12


def test_extract_params_direct_call_only_names() -> None:
    def f(x, params):
        return x * params["c"]

    g = extract_params(f, names=["c", "d"])
    assert g(x=3, params=[4, 5]) == 12


def test_extract_params_decorator_key_and_names() -> None:
    @extract_params(key="a", names=["c", "d"])
    def f(x, params):
        return x * params["c"]

    assert f(x=3, params={"a": [4, 5], "b": [5, 6]}) == 12


def test_extract_params_direct_call_key_and_names() -> None:
    def f(x, params):
        return x * params["c"]

    g = extract_params(f, key="a", names=["c", "d"])
    assert g(x=3, params={"a": [4, 5], "b": [5, 6]}) == 12


def test_jax_array_output_decorator() -> None:
    @jax_array_output
    def f():
        return (1, 2, 3)

    assert isinstance(f(), jnp.ndarray)


def test_jax_array_output_direct_call() -> None:
    def f():
        return (1, 2, 3)

    g = jax_array_output(f)

    assert isinstance(g(), jnp.ndarray)


def test_register_params_decorator() -> None:
    @register_params(params=["a", "b", "c"])
    def f() -> str:
        return "bla"

    assert f.__registered_params__ == ["a", "b", "c"]
    assert f() == "bla"


def test_register_params_direct_call() -> None:
    def f() -> str:
        return "bla"

    g = register_params(f, params=["a", "b", "c"])
    assert g.__registered_params__ == ["a", "b", "c"]  # ty: ignore[unresolved-attribute]
    assert g() == "bla"


def test_extract_params_no_key_no_names_raises() -> None:
    with pytest.raises(ValueError, match="cannot both be None"):
        extract_params(key=None, names=None)(lambda **kw: kw)
