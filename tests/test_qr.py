"""Tests for custom QR decomposition."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.typing import NDArray

from skillmodels.chs.qr import qr_gpu

SEED = 20


@pytest.fixture
def cov_matrix() -> NDArray[np.floating]:
    """Create a covariance matrix for testing."""
    fixedrng = np.random.default_rng(SEED)
    factorized = fixedrng.uniform(low=-1, high=3, size=(7, 7))
    return factorized @ factorized.T * 0.5 + np.eye(7)


def test_q(cov_matrix: NDArray[np.floating]) -> None:
    """Test Q matrix from QR decomposition matches JAX implementation."""
    q_gpu, _ = qr_gpu(cov_matrix)
    q_jax, _ = jnp.linalg.qr(cov_matrix)
    aaae(q_gpu, q_jax)


def test_r(cov_matrix: NDArray[np.floating]) -> None:
    """Test R matrix from QR decomposition matches JAX implementation."""
    _, r_gpu = qr_gpu(cov_matrix)
    _, r_jax = jnp.linalg.qr(cov_matrix)
    aaae(r_gpu, r_jax)


def test_grad_qr(cov_matrix: NDArray[np.floating]) -> None:
    """Test gradient of QR decomposition matches JAX implementation."""

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
