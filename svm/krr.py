"""Kernel Ridge Regression classifier (Savers et al. 1998) — closed-form.

For an OvR target t in {-1,+1}, alpha = (K + lam I)^{-1} t, f(x)=K(x,.)@alpha.
Pure numpy, deterministic, the secondary offline solver used by the fusion
flagship for stability cross-checks.
"""

from __future__ import annotations

import numpy as np

from core.types import KernelSpec
from kernels import get_kernel


class KRRClassifier:
    def __init__(self, kernel_spec: KernelSpec, C: float = 1.0) -> None:
        self.kernel_spec = kernel_spec
        self.C = float(C)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> KRRClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.alpha_ = []
        self.lam_ = max(1.0 / (2.0 * max(self.C, 1e-12)), 1e-4)
        K = get_kernel(self.kernel_spec).gram(X)
        n = X.shape[0]
        A = K + self.lam_ * np.eye(n)
        self.X_train = X
        for c in self.classes_:
            t = np.where(y == c, 1.0, -1.0).astype(np.float64)
            alpha = np.linalg.solve(A, t)
            self.alpha_.append(alpha)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Kt = get_kernel(self.kernel_spec).gram(X, self.X_train)  # (m, n)
        out = np.stack([Kt @ a for a in self.alpha_], axis=1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return self.classes_[np.argmax(df, axis=1).astype(int)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        df = df - df.max(axis=1, keepdims=True)
        e = np.exp(df)
        return e / e.sum(axis=1, keepdims=True)
