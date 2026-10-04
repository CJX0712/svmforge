"""BinarySMO — Platt (1998) Sequential Minimal Optimization, pure numpy.

This is the canonical maximum-margin SVM solver. It is the offline Tier-1
core: no sklearn, no downloads. Verifiable invariants (see tests/):
  * dual objective is monotonically non-decreasing across the run,
  * KKT conditions hold at convergence (violation <= tol),
  * on a 2-D toy set the decision boundary matches scikit-learn SVC.
"""

from __future__ import annotations

import numpy as np

from core.errors import NumericsError
from core.types import KernelSpec
from kernels import get_kernel


class BinarySMO:
    """Binary SVM via SMO. Labels are auto-mapped to {-1, +1}."""

    def __init__(
        self,
        kernel_spec: KernelSpec,
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 5000,
    ) -> None:
        self.kernel_spec = kernel_spec
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self._fitted = False

    # ---- training -------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> BinarySMO:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"BinarySMO needs exactly 2 classes, got {len(classes)}")
        yb = np.where(y == classes[1], 1.0, -1.0).astype(np.float64)
        kernel = get_kernel(self.kernel_spec)
        K = kernel.gram(X)
        n = X.shape[0]
        if K.shape != (n, n):
            raise NumericsError("Gram matrix shape mismatch")

        alpha = np.zeros(n, dtype=np.float64)
        b = 0.0
        E = -yb.copy()  # f == b == 0 initially -> E = f - yb = -yb
        passes = 0
        it = 0
        while passes < 12 and it < self.max_iter:
            changed = 0
            for i in range(n):
                Ei = E[i]
                viol_i = (yb[i] * Ei < -self.tol and alpha[i] < self.C) or (
                    yb[i] * Ei > self.tol and alpha[i] > 0
                )
                if not viol_i:
                    continue
                # second-choice heuristic: maximise |Ei - Ej|
                j = int(np.argmin(E)) if Ei > 0 else int(np.argmax(E))
                if j == i:
                    j = (i + 1) % n
                Ej = E[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if yb[i] != yb[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - self.C)
                    H = min(self.C, ai_old + aj_old)
                if L == H:
                    continue
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    # numerically degenerate working pair; try a random partner
                    continue
                aj_new = np.clip(aj_old - yb[j] * (Ei - Ej) / eta, L, H)
                if abs(aj_new - aj_old) < 1e-5:
                    continue
                ai_new = ai_old + yb[i] * yb[j] * (aj_old - aj_new)
                bi = (
                    b
                    - Ei
                    - yb[i] * (ai_new - ai_old) * K[i, i]
                    - yb[j] * (aj_new - aj_old) * K[i, j]
                )
                bj = (
                    b
                    - Ej
                    - yb[i] * (ai_new - ai_old) * K[i, j]
                    - yb[j] * (aj_new - aj_old) * K[j, j]
                )
                if 0.0 < ai_new < self.C:
                    b_new = bi
                elif 0.0 < aj_new < self.C:
                    b_new = bj
                else:
                    b_new = 0.5 * (bi + bj)
                delta = (
                    yb[i] * (ai_new - ai_old) * K[:, i]
                    + yb[j] * (aj_new - aj_old) * K[:, j]
                )
                db = b_new - b
                alpha[i], alpha[j] = ai_new, aj_new
                E = E + delta + db
                b = b_new
                changed += 1
                it += 1
            if changed == 0:
                passes += 1
            else:
                passes = 0

        sv = alpha > 1e-5
        self.X_train = X
        self.yb = yb
        self.classes_ = classes
        self.alpha = alpha
        self.b = float(b)
        self.sv_idx = np.where(sv)[0]
        self.X_sv = X[sv]
        self.yb_sv = yb[sv]
        self.alpha_sv = alpha[sv]
        self.n_support_ = int(sv.sum())
        self.K_train_ = K
        self._fitted = True
        return self

    # ---- inference ------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() first")
        X = np.asarray(X, dtype=np.float64)
        Kt = get_kernel(self.kernel_spec).gram(X, self.X_sv)
        return Kt @ (self.alpha_sv * self.yb_sv) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        out = np.where(df >= 0, self.classes_[1], self.classes_[0])
        return out.astype(self.classes_.dtype)

    # ---- invariants (used by tests + pipeline self-check) ---------------
    def dual_objective(self) -> float:
        """W(alpha) = sum alpha - 0.5 * alpha^T (yb yb^T * K) alpha."""
        yK = self.yb[:, None] * self.K_train_
        return float(np.sum(self.alpha) - 0.5 * self.alpha @ yK @ self.alpha)

    def check_kkt(self, tol: float | None = None) -> float:
        """Return the maximum KKT violation at the fitted solution."""
        tol = self.tol if tol is None else tol
        f = self.K_train_ @ (self.alpha * self.yb) + self.b
        E = f - self.yb  # E_i
        viol = np.zeros_like(self.alpha)
        for i in range(len(self.alpha)):
            yiE = self.yb[i] * E[i]
            a = self.alpha[i]
            if a < 1e-9:  # alpha == 0  ->  yiE >= -tol
                viol[i] = max(0.0, -yiE - tol)
            elif a > self.C - 1e-9:  # alpha == C  ->  yiE <= tol
                viol[i] = max(0.0, yiE - tol)
            else:  # 0 < alpha < C  ->  |yiE| <= tol
                viol[i] = max(0.0, abs(yiE) - tol)
        return float(np.max(viol))


class SMOClassifier:
    """One-vs-Rest multiclass wrapper around BinarySMO."""

    def __init__(
        self,
        kernel_spec: KernelSpec,
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 5000,
    ):
        self.kernel_spec = kernel_spec
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> SMOClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.binary_ = []
        for c in self.classes_:
            yb = (y == c).astype(int)
            clf = BinarySMO(
                self.kernel_spec, C=self.C, tol=self.tol, max_iter=self.max_iter
            )
            clf.fit(X, yb)
            self.binary_.append(clf)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.stack([clf.decision_function(X) for clf in self.binary_], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        idx = np.argmax(df, axis=1)
        return self.classes_[idx.astype(int)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        df = df - df.max(axis=1, keepdims=True)
        e = np.exp(df)
        return e / e.sum(axis=1, keepdims=True)
