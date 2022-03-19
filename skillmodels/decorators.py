import functools

import jax.numpy as jnp


def extract_params(func=None, *, key=None):
    """Extract the ``key`` entry from params before passing it to func.

    Note: The resulting function is keyword only!

    """

    def decorator_extract_params(func):
        @functools.wraps(func)
        def wrapper_extract_params(**kwargs):
            internal_kwargs = kwargs.copy()
            internal_kwargs["params"] = kwargs["params"][key]
            return func(**internal_kwargs)

        return wrapper_extract_params

    if callable(func):
        return decorator_extract_params(func)
    else:
        return decorator_extract_params


def jax_array_output(func):
    """Convert tuple output to list output."""

    @functools.wraps(func)
    def wrapper_jax_array_output(*args, **kwargs):
        raw = func(*args, **kwargs)
        out = jnp.array(raw)
        return out

    return wrapper_jax_array_output


def register_params(func=None, *, params=None):
    def decorator_register_params(func):
        func.__registered_params__ = params
        return func

    if callable(func):
        return decorator_register_params(func)
    else:
        return decorator_register_params
