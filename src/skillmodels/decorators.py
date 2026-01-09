import functools
from collections.abc import Callable  # noqa: TC003
from typing import Any

import jax.numpy as jnp
from jax import Array


def extract_params(
    func: Callable | None = None,
    *,
    key: str | None = None,
    names: list[str] | None = None,
) -> Callable:
    """Process params before passing them to func.

    Note: The resulting function is keyword only!

    Args:
        func: The function to be decorated, or None if using decorator with arguments.
        key: If key is not None, we assume params is a dictionary of which
            only the params[key] should be passed into func.
        names: If names is provided, we assume that params
            (or params[key]) should be converted to a dictionary with names as keys
            before passing them to func.

    """

    def decorator_extract_params(func: Callable) -> Callable:
        if key is not None and names is None:

            @functools.wraps(func)
            def wrapper_extract_params(**kwargs: Any) -> Any:
                internal_kwargs = kwargs.copy()
                internal_kwargs["params"] = kwargs["params"][key]
                return func(**internal_kwargs)

        elif key is None and names is not None:

            @functools.wraps(func)
            def wrapper_extract_params(**kwargs: Any) -> Any:
                internal_kwargs = kwargs.copy()
                internal_kwargs["params"] = dict(
                    zip(names, kwargs["params"], strict=False)
                )
                return func(**internal_kwargs)

        elif key is not None and names is not None:

            @functools.wraps(func)
            def wrapper_extract_params(**kwargs: Any) -> Any:
                internal_kwargs = kwargs.copy()
                internal_kwargs["params"] = dict(
                    zip(names, kwargs["params"][key], strict=False)
                )
                return func(**internal_kwargs)

        else:
            raise ValueError("key and names cannot both be None.")

        return wrapper_extract_params

    if callable(func):
        return decorator_extract_params(func)
    return decorator_extract_params


def jax_array_output(func: Callable) -> Callable:
    """Convert tuple output to list output."""

    @functools.wraps(func)
    def wrapper_jax_array_output(*args: Any, **kwargs: Any) -> Array:
        raw = func(*args, **kwargs)
        out = jnp.array(raw)
        return out

    return wrapper_jax_array_output


def register_params(
    func: Callable | None = None,
    *,
    params: list[str] | None = None,
) -> Callable:
    def decorator_register_params(func: Callable) -> Callable:
        func.__registered_params__ = params  # ty: ignore[unresolved-attribute]
        return func

    if callable(func):
        return decorator_register_params(func)
    return decorator_register_params
