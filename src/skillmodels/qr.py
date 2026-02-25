import jax
import jax.numpy as jnp


@jax.custom_jvp
def qr_gpu(a: jax.Array):
    """Custom implementation of the QR Decomposition."""
    r, tau = jnp.linalg.qr(a, mode="raw")

    return jnp.triu(r.mT[: tau.shape[0]])


def _householder(r: jax.Array, tau: jax.Array):
    """Custom implementation of the Householder Product.

    Uses the outputs of jnp.linalg.qr with mode = "raw" to calculate Q. This is needed
    because the JAX implementation is extremely slow for a batch of small matrices.
    """
    m = r.shape[0]
    n = tau.shape[0]
    r = jnp.tril(jnp.fill_diagonal(r, 1, inplace=False))
    # Calculate Householder Vector which is saved in the lower triangle of R
    v1 = jnp.expand_dims(r[:, 0], 1)
    h = jnp.eye(m) - tau[0] * (v1 @ jnp.transpose(v1))
    # Multiply all Householder Vectors Q = H(1)*H(2)...*H(n)
    for i in range(1, n):
        vi = jnp.expand_dims(r[:, i], 1)
        h = h - tau[i] * (h @ vi) @ jnp.transpose(vi)
    return h[:, :n]


def _apply_householder_t(r: jax.Array, tau: jax.Array, a: jax.Array):
    """Custom implementation of the Householder Product.

    Uses the outputs of jnp.linalg.qr with mode = "raw" to calculate Q. This is needed
    because the JAX implementation is extremely slow for a batch of small matrices.
    """
    n = tau.shape[0]
    r = jnp.tril(jnp.fill_diagonal(r, 1, inplace=False))
    # Calculate Householder Vector which is saved in the lower triangle of R
    v1 = jnp.expand_dims(r[:, n - 1], 1)
    h = a - tau[n - 1] * a @ v1 @ jnp.transpose(v1)
    # Multiply all Householder Vectors Q = H(1)*H(2)...*H(n)
    for i in range(n - 2, -1, -1):
        vi = jnp.expand_dims(r[:, i], 1)
        h = h - tau[i] * (h @ vi) @ jnp.transpose(vi)
    return h[:, :n]


def _t(x: jax.Array) -> jax.Array:
    """Transpose batched Matrix."""
    return jax.lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


def _h(x: jax.Array) -> jax.Array:
    """Hermitian Transpose of a Matrix."""
    return _t(x).conj()


def _tril(m: jax.Array, k: int = 0) -> jax.Array:
    """Select lower Triangle of a Matrix."""
    *_, dim_n, dim_m = m.shape
    mask = jnp.tri(dim_n, dim_m, k, bool)
    return jax.lax.select(jax.lax.broadcast(mask, m.shape[:-2]), m, jnp.zeros_like(m))


@qr_gpu.defjvp
def qr_jvp_rule(primals, tangents):
    """Calculates the derivative of the custom QR composition."""
    # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
    (x,) = primals
    (dx,) = tangents
    r_raw, tau = jnp.linalg.qr(x, mode="raw")
    r = jnp.triu(r_raw.mT[: tau.shape[0]])
    dx_rinv = jax.lax.linalg.triangular_solve(r, dx)  # Right side solve by default
    qt_dx_rinv = _apply_householder_t(r_raw.mT, tau, dx_rinv)
    qt_dx_rinv_lower = _tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - _h(qt_dx_rinv_lower)  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    n = x.shape[-1]
    i = jax.lax.expand_dims(jnp.eye(n, n), range(qt_dx_rinv.ndim - 2))
    do = do + i * (qt_dx_rinv - qt_dx_rinv.real.astype(qt_dx_rinv.dtype))
    dr = (qt_dx_rinv - do) @ r
    return (r), (dr)
