import jax
import jax.numpy as jnp
from jax import Array


@jax.custom_jvp
def qr_gpu(a: Array) -> tuple[Array, Array]:
    """Custom implementation of the QR Decomposition."""
    r, tau = jnp.linalg.qr(a, mode="raw")

    q = _householder(r.mT, tau)
    return q, jnp.triu(r.mT[: tau.shape[0]])


def _householder(r: Array, tau: Array) -> Array:
    """Custom implementation of the Householder Product.

    Uses the outputs of jnp.linalg.qr with mode = "raw" to calculate Q. This is needed
    because the JAX implementation is extremely slow for a batch of small matrices.
    """
    m = r.shape[0]
    n = tau.shape[0]
    # Calculate Householder Vector which is saved in the lower triangle of R
    v1 = jnp.expand_dims(r[:, 0], 1)
    v1 = v1.at[0:0].set(0)
    v1 = v1.at[0].set(1)
    h = jnp.eye(m) - tau[0] * (v1 @ jnp.transpose(v1))
    # Multiply all Householder Vectors Q = H(1)*H(2)...*H(n)
    for i in range(1, n):
        vi = jnp.expand_dims(r[:, i], 1)
        vi = vi.at[0:i].set(0)
        vi = vi.at[i].set(1)
        h = h - tau[i] * (h @ vi) @ jnp.transpose(vi)
    return h[:, :n]


def _t(x: Array) -> Array:
    """Transpose batched Matrix."""
    return jax.lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


def _h(x: Array) -> Array:
    """Hermitian Transpose of a Matrix."""
    return _t(x).conj()


def _tril(m: Array, k: int = 0) -> Array:
    """Select lower Triangle of a Matrix."""
    *_, dim_n, dim_m = m.shape
    mask = jnp.tri(dim_n, dim_m, k, bool)
    return jax.lax.select(jax.lax.broadcast(mask, m.shape[:-2]), m, jnp.zeros_like(m))


@qr_gpu.defjvp
def qr_jvp_rule(
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Calculates the derivative of the custom QR composition."""
    # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
    (x,) = primals
    (dx,) = tangents
    q, r = qr_gpu(x)
    dx_rinv = jax.lax.linalg.triangular_solve(r, dx)  # Right side solve by default
    qt_dx_rinv = _h(q) @ dx_rinv
    qt_dx_rinv_lower = _tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - _h(qt_dx_rinv_lower)  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    n = x.shape[-1]
    i = jax.lax.expand_dims(jnp.eye(n, n), range(qt_dx_rinv.ndim - 2))
    do = do + i * (qt_dx_rinv - qt_dx_rinv.real.astype(qt_dx_rinv.dtype))
    dq = q @ (do - qt_dx_rinv) + dx_rinv
    dr = (qt_dx_rinv - do) @ r
    return (q, r), (dq, dr)
