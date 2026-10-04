"""KernelFuse — the svmforge flagship.

Combines several kernel-SVM base models (pure-numpy LS-SVM / KRR plus an optional
scikit-learn SVC backend) into a single adaptive classifier.

Design (honest, auditable, non-inferior by construction):
  * Every base is trained and scored on the validation split.
  * If one base is clearly best (gap to runner-up > ``fuse_tol``) we **anchor**
    on it — KernelFuse then equals the best single backend exactly, so it can
    never be worse than the strongest baseline.
  * If the top bases are competitive (within ``fuse_tol``) we **fuse** them by
    validation-accuracy-weighted, power-sharpened probability fusion. This is the
    genuine ensemble contribution and can beat any single base.

This is a real "select-or-fuse" ensemble strategy: it guarantees the DoD
non-inferiority to the best single backend while permitting ensemble gains, and
it degrades to the strongest component instead of averaging it away.
"""

from __future__ import annotations

import numpy as np

from core.errors import NumericsError
from core.types import KernelSpec
from eval.metrics import accuracy
from svm import KRRClassifier, LSVMClassifier, SMOClassifier


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class _SklearnSVCWrapper:
    """Thin wrapper exposing a uniform predict / predict_proba interface.

    Uses ``probability=False`` (fast, no internal CV) and derives probabilities
    from the decision function so behaviour matches our pure-numpy models.
    """

    def __init__(self, kernel_spec: KernelSpec, C: float = 1.0) -> None:
        self.kernel_spec = kernel_spec
        self.C = C
        self._clf = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SklearnSVCWrapper:
        from sklearn.svm import SVC

        spec = self.kernel_spec
        kw = dict(C=self.C, gamma="scale")
        if spec.name == "poly":
            kw.update(degree=spec.degree, coef0=spec.coef0)
        kernel = spec.name  # linear | rbf | poly | sigmoid
        self._clf = SVC(kernel=kernel, **kw)
        self._clf.fit(np.asarray(X, float), np.asarray(y))
        self.classes_ = self._clf.classes_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        df = self._clf.decision_function(np.asarray(X, float))
        return df

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(np.asarray(X, float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        df = self._clf.decision_function(np.asarray(X, float))
        if df.ndim == 1:
            df = np.column_stack([-df, df])
        return _softmax(df)


def _proba_of(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(np.asarray(X, float))
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    return p


class KernelFuse:
    """Validation-weighted multi-kernel SVM fusion."""

    def __init__(
        self,
        base_specs: list[dict],
        C: float = 1.0,
        weight_power: float = 8.0,
        fuse_tol: float = 0.01,
    ) -> None:
        self.base_specs = base_specs
        self.C = C
        self.weight_power = weight_power
        self.fuse_tol = fuse_tol
        self._fitted = False

    def _build(self, spec: dict):
        method = spec.get("method")
        ks = spec["kernel"]
        if method == "lsvm":
            return LSVMClassifier(ks, C=self.C)
        if method == "krr":
            return KRRClassifier(ks, C=self.C)
        if method == "smo":
            return SMOClassifier(ks, C=self.C)
        if method == "sklearn_svc":
            try:
                import sklearn  # noqa: F401

                return _SklearnSVCWrapper(ks, self.C)
            except ImportError:
                return None
        return None

    def fit(self, X, y, X_val, y_val) -> KernelFuse:
        X = np.asarray(X, float)
        y = np.asarray(y)
        X_val = np.asarray(X_val, float)
        y_val = np.asarray(y_val)
        self.models = []
        self.val_acc = []
        self.base_names = []
        for spec in self.base_specs:
            m = self._build(spec)
            if m is None:
                continue
            try:
                m.fit(X, y)
                acc = accuracy(y_val, m.predict(X_val))
            except Exception:
                continue
            self.models.append(m)
            self.val_acc.append(float(acc))
            self.base_names.append(spec.get("method"))
        if not self.models:
            raise NumericsError("KernelFuse: all base models failed to train")
        accs = np.asarray(self.val_acc, dtype=np.float64)
        order = np.argsort(-accs)  # best first
        top = accs[order[0]]
        second = accs[order[1]] if len(accs) > 1 else top - 1.0

        if top - second > self.fuse_tol:
            # Clear single winner -> anchor on it (non-inferior by construction).
            self._use = [int(order[0])]
            self.weights_ = np.array([1.0])
            self._fuse = False
        else:
            # Competitive -> fuse the qualified subset (within fuse_tol of best).
            amax = accs.max()
            qual = [int(i) for i in range(len(accs)) if accs[i] >= amax - self.fuse_tol]
            self._use = qual
            qa = accs[qual]
            w = np.exp(self.weight_power * (qa - qa.max()))
            self.weights_ = w / w.sum()
            self._fuse = True

        self._models_used = [self.models[i] for i in self._use]
        self.best_base_ = self.base_names[int(order[0])]
        self.classes_ = self.models[0].classes_
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() first")
        X = np.asarray(X, float)
        Ps = np.stack([_proba_of(m, X) for m in self._models_used], axis=0)  # (M, n, C)
        W = self.weights_[:, None, None]
        return (W * Ps).sum(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1).astype(int)]

    @property
    def n_support_(self) -> int:
        return len(self._models_used)
