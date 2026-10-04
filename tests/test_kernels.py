import numpy as np
import pytest

from core.types import KernelSpec
from kernels import get_kernel
from kernels.base import SUPPORTED


def _rand(rng, n=40, d=3):
    return rng.normal(size=(n, d))


def test_all_kernels_symmetric():
    rng = np.random.default_rng(0)
    X = _rand(rng)
    for name in SUPPORTED:
        k = get_kernel(KernelSpec(name, gamma=0.7, degree=3, coef0=0.1))
        G = k.gram(X)
        assert G.shape == (40, 40)
        assert np.allclose(G, G.T, atol=1e-10), f"{name} not symmetric"


def test_rbf_diagonal_is_one():
    rng = np.random.default_rng(1)
    X = _rand(rng)
    G = get_kernel(KernelSpec("rbf", gamma=2.0)).gram(X)
    assert np.allclose(np.diag(G), 1.0, atol=1e-12)


def test_invalid_kernel_name():
    with pytest.raises(ValueError):
        get_kernel(KernelSpec("does_not_exist"))


def test_poly_degree_odd_allows_negative():
    rng = np.random.default_rng(2)
    X = _rand(rng)
    G = get_kernel(KernelSpec("poly", gamma=1.0, degree=3, coef0=-2.0)).gram(X)
    # degree 3 + negative coef0 allows negative entries -> not all positive
    assert np.any(G < 0)


def test_kernel_output_dtype():
    rng = np.random.default_rng(3)
    X = _rand(rng)
    G = get_kernel(KernelSpec("linear")).gram(X)
    assert G.dtype == np.float64
