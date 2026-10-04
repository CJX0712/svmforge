"""Least-Squares SVM (Suykens & Vandewalle 1999) — closed-form, pure numpy.

LS-SVM replaces the inequality constraints of the max-margin SVM with
equality constraints, yielding a single linear system instead of a QP.
Because it has a closed-form solution it is the most numerically stable of
our offline solvers and serves as the workhorse inside the fusion flagship.
"""

from __future__ import annotations

import numpy as np

from core.types import KernelSpec
from kernels import get_kernel


def _solve_binary_lssvm(K: np.ndarray, yb: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """Return (alpha, b) for the LS-SVM primal dual system.

    Min 1/2||w||^2 + 1/2 gamma * sum e^2  s.t.  y_i(phi·x_i + b) = 1 - e_i.
    gamma = C (larger C => harder margin). lam = 1/gamma is the ridge term.
    """
    n = K.shape[0]
    lam = max(1.0 / max(C, 1e-12), 1e-4)
    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[0, 1:] = yb
    A[1:, 0] = yb
    A[1:, 1:] = yb[:, None] * yb[None, :] * K + lam * np.eye(n)
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[1:] = 1.0
    sol = np.linalg.solve(A, rhs)
    b = float(sol[0])
    alpha = sol[1:]
    return alpha, b


class LSVM:
    """Binary LS-SVM classifier (labels auto-mapped to {-1,+1})."""

    def __init__(self, kernel_spec: KernelSpec, C: float = 1.0) -> None:
        self.kernel_spec = kernel_spec
        self.C = float(C)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSVM:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"LSVM needs exactly 2 classes, got {len(classes)}")
        yb = np.where(y == classes[1], 1.0, -1.0).astype(np.float64)
        K = get_kernel(self.kernel_spec).gram(X)
        alpha, b = _solve_binary_lssvm(K, yb, self.C)
        sv = np.abs(alpha) > 1e-9
        self.X_train = X
        self.classes_ = classes
        self.alpha = alpha
        self.b = b
        self.X_sv = X[sv]
        self.alpha_sv = alpha[sv]
        self.yb_sv = yb[sv]
        self.n_support_ = int(sv.sum())
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Kt = get_kernel(self.kernel_spec).gram(X, self.X_sv)
        return Kt @ (self.alpha_sv * self.yb_sv) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        out = np.where(df >= 0, self.classes_[1], self.classes_[0])
        return out.astype(self.classes_.dtype)


class LSVMClassifier:
    """One-vs-Rest multiclass wrapper around :class:`LSVM`."""

    def __init__(self, kernel_spec: KernelSpec, C: float = 1.0) -> None:
        self.kernel_spec = kernel_spec
        self.C = C
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSVMClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.binary_ = []
        for c in self.classes_:
            yb = (y == c).astype(int)
            clf = LSVM(self.kernel_spec, C=self.C)
            clf.fit(X, yb)
            self.binary_.append(clf)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.stack([clf.decision_function(X) for clf in self.binary_], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return self.classes_[np.argmax(df, axis=1).astype(int)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        df = df - df.max(axis=1, keepdims=True)
        e = np.exp(df)
        return e / e.sum(axis=1, keepdims=True)
