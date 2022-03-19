import jax.numpy as jnp

from skillmodels.decorators import extract_params
from skillmodels.decorators import jax_array_output
from skillmodels.decorators import register_params


def test_extract_params_decorator():
    @extract_params(key="a")
    def f(x, params):
        return x * params

    assert f(x=3, params={"a": 4, "b": 5}) == 12


def test_extract_params_direct_call():
    def f(x, params):
        return x * params

    g = extract_params(f, key="a")

    assert g(x=3, params={"a": 4, "b": 5}) == 12


def test_jax_array_output_decorator():
    @jax_array_output
    def f():
        return (1, 2, 3)

    assert isinstance(f(), jnp.ndarray)


def test_jax_array_output_direct_call():
    def f():
        return (1, 2, 3)

    g = jax_array_output(f)

    assert isinstance(g(), jnp.ndarray)


def test_register_params_decorator():
    @register_params(params=["a", "b", "c"])
    def f():
        return "bla"

    assert f.__registered_params__ == ["a", "b", "c"]
    assert f() == "bla"


def test_register_params_direct_call():
    def f():
        return "bla"

    g = register_params(f, params=["a", "b", "c"])
    assert g.__registered_params__ == ["a", "b", "c"]
    assert g() == "bla"
