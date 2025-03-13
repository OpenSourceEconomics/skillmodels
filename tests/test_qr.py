import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
from skillmodels.qr import qr_gpu

def test_qr():
    factorized = np.random.uniform(low=-1, high=3, size=(7, 7))
    cov = factorized @ factorized.T * 0.5 + np.eye(7)
    q_gpu, r_gpu = qr_gpu(cov)
    q_jax, r_jax = jnp.linalg.qr(cov)
    def f_jax(A):
        q,r = jnp.linalg.qr(A)
        return jnp.sum(r) + jnp.sum(q)
    def f_gpu(A):
        q,r = qr_gpu(A)
        return jnp.sum(r) + jnp.sum(q)
    grad_qr_jax = jax.grad(f_jax)
    grad_qr_gpu = jax.grad(f_gpu)
    grad_gpu = grad_qr_gpu(cov)
    grad_jax = grad_qr_jax(cov)
    aaae(q_gpu, q_jax)
    aaae(r_gpu, r_jax)
    aaae(grad_gpu, grad_jax)
    aaae(grad_gpu, grad_jax)