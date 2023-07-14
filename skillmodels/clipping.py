import jax
import jax.numpy as jnp


def soft_clipping(arr, lower=None, upper=None, lower_hardness=1, upper_hardness=1):
    """Clip values in an array elementwise using a soft maximum to avoid kinks.

    Clipping from below is taking a maximum between two values. Clipping
    from above is taking a minimum, but it can be rewritten as taking a maximum after
    switching the signs.

    To smooth out the kinks introduced by normal clipping, we first rewrite all clipping
    operations to taking maxima. Then we replace the normal maximum by the soft maximum.

    For background on the soft maximum check out this
    `article by John Cook: <https://www.johndcook.com/soft_maximum.pdf>`_

    Note that contrary to the name, the soft maximum can be calculated using
    ``scipy.special.logsumexp``. ``scipy.special.softmax`` is the gradient of
    ``scipy.special.logsumexp``.


    Args:
        arr (jax.numpy.array): Array that is clipped elementwise.
        lower (float): The value at which the array is clipped from below.
        upper (float): The value at which the array is clipped from above.
        lower_hardness (float): Scaling factor that is applied inside the soft maximum.
            High values imply a closer approximation of the real maximum.
        upper_hardness (float): Scaling factor that is applied inside the soft maximum.
            High values imply a closer approximation of the real maximum.

    """
    shape = arr.shape
    flat = arr.flatten()
    dim = len(flat)
    if lower is not None:
        helper = jnp.column_stack([flat, jnp.full(dim, lower)])
        flat = (
            jax.scipy.special.logsumexp(lower_hardness * helper, axis=1)
            / lower_hardness
        )
    if upper is not None:
        helper = jnp.column_stack([-flat, jnp.full(dim, -upper)])
        flat = (
            -jax.scipy.special.logsumexp(upper_hardness * helper, axis=1)
            / upper_hardness
        )
    return flat.reshape(shape)
