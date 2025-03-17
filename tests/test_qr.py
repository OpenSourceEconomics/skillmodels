import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.qr import qr_gpu

SEED = 20


@pytest.fixture
def cov_matrix():
    fixedrng = np.random.default_rng(SEED)
    factorized = fixedrng.uniform(low=-1, high=3, size=(7, 7))
    cov = factorized @ factorized.T * 0.5 + np.eye(7)
    return cov


def test_q(cov_matrix):
    q_gpu, _ = qr_gpu(cov_matrix)
    q_jax, _ = jnp.linalg.qr(cov_matrix)
    aaae(q_gpu, q_jax)


def test_r(cov_matrix):
    _, r_gpu = qr_gpu(cov_matrix)
    _, r_jax = jnp.linalg.qr(cov_matrix)
    aaae(r_gpu, r_jax)


def test_grad_qr(cov_matrix):
    def f_jax(a):
        q, r = jnp.linalg.qr(a)
        return jnp.sum(r) + jnp.sum(q)

    def f_gpu(a):
        q, r = qr_gpu(a)
        return jnp.sum(r) + jnp.sum(q)

    grad_qr_jax = jax.grad(f_jax)
    grad_qr_gpu = jax.grad(f_gpu)
    grad_gpu = grad_qr_gpu(cov_matrix)
    grad_jax = grad_qr_jax(cov_matrix)
    aaae(grad_gpu, grad_jax)
