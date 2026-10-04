"""Kernel Gram-matrix builders with symmetry/PD invariants.

All kernels are positive-definite by construction; the symmetry invariant is
asserted at call time so a numerically broken kernel fails loudly (NumericsError)
rather than silently corrupting the SVM dual.
"""

from __future__ import annotations

import numpy as np

from core.errors import NumericsError
from core.types import KernelSpec

__all__ = ["SUPPORTED", "Kernel", "get_kernel"]


SUPPORTED = ("linear", "rbf", "poly", "sigmoid")


def _linear(
    X: np.ndarray, Y: np.ndarray, gamma: float, degree: int, coef0: float
) -> np.ndarray:
    return X @ Y.T


def _rbf(
    X: np.ndarray, Y: np.ndarray, gamma: float, degree: int, coef0: float
) -> np.ndarray:
    if gamma <= 0.0:
        gamma = 1.0
    xx = np.sum(X * X, axis=1)[:, None]
    yy = np.sum(Y * Y, axis=1)[None, :]
    d2 = xx + yy - 2.0 * (X @ Y.T)
    d2 = np.clip(d2, 0.0, None)
    return np.exp(-gamma * d2)


def _poly(
    X: np.ndarray, Y: np.ndarray, gamma: float, degree: int, coef0: float
) -> np.ndarray:
    g = gamma if gamma > 0.0 else 1.0
    base = g * (X @ Y.T) + coef0
    base = np.clip(base, 0.0, None) if degree % 2 == 0 else base
    return np.power(base, degree)


def _sigmoid(
    X: np.ndarray, Y: np.ndarray, gamma: float, degree: int, coef0: float
) -> np.ndarray:
    g = gamma if gamma > 0.0 else 1.0
    return np.tanh(g * (X @ Y.T) + coef0)


_FN = {"linear": _linear, "rbf": _rbf, "poly": _poly, "sigmoid": _sigmoid}


class Kernel:
    """A concrete kernel bound to a :class:`KernelSpec`."""

    def __init__(self, spec: KernelSpec) -> None:
        if spec.name not in _FN:
            raise ValueError(f"unsupported kernel {spec.name!r}; supported={SUPPORTED}")
        self.spec = spec
        self.name = spec.name
        self.gamma = float(spec.gamma)
        self.degree = int(spec.degree)
        self.coef0 = float(spec.coef0)
        self._fn = _FN[spec.name]

    def gram(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = X if Y is None else np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
            raise NumericsError("kernel inputs must be 2D with equal feature dim")
        G = self._fn(X, Y, self.gamma, self.degree, self.coef0)
        if Y is None:
            if not np.allclose(G, G.T, atol=1e-9, rtol=0.0):
                raise NumericsError(f"kernel {self.name} Gram matrix not symmetric")
            # diagonal must be k(x,x) == positive constant (for rbf/poly/sigmoid: 1.0)
            diag = np.diag(G)
            if not np.allclose(diag, diag[0], atol=1e-9):
                raise NumericsError(f"kernel {self.name} has non-constant diagonal")
        return G

    def describe(self) -> str:
        return self.spec.key()


def get_kernel(spec: KernelSpec) -> Kernel:
    return Kernel(spec)
