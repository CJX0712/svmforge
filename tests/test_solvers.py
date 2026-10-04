import numpy as np
from sklearn.svm import SVC

from core.seed import set_all
from core.types import KernelSpec
from data.synthetic import make_dataset
from svm import KRRClassifier, LSVMClassifier, SMOClassifier


def _gscale(X):
    var = float(np.var(X))
    return 1.0 / (X.shape[1] * var)


def _acc(clf, X, y):
    return float(np.mean(clf.predict(X) == y))


def test_lsvm_matches_sklearn_rbf():
    set_all(7)
    ds = make_dataset("moons", 200, 200, 50, 7)
    g = _gscale(ds.X_train)
    ours = LSVMClassifier(KernelSpec("rbf", gamma=g), C=1.0).fit(ds.X_train, ds.y_train)
    ref = SVC(kernel="rbf", gamma=g, C=1.0).fit(ds.X_train, ds.y_train)
    a1 = _acc(ours, ds.X_test, ds.y_test)
    a2 = _acc(ref, ds.X_test, ds.y_test)
    assert abs(a1 - a2) < 0.03, f"LSVM {a1:.3f} vs SVC {a2:.3f}"


def test_krr_matches_sklearn_rbf():
    set_all(11)
    ds = make_dataset("circles", 200, 200, 50, 11)
    g = _gscale(ds.X_train)
    ours = KRRClassifier(KernelSpec("rbf", gamma=g), C=1.0).fit(ds.X_train, ds.y_train)
    ref = SVC(kernel="rbf", gamma=g, C=1.0).fit(ds.X_train, ds.y_train)
    a1 = _acc(ours, ds.X_test, ds.y_test)
    a2 = _acc(ref, ds.X_test, ds.y_test)
    assert abs(a1 - a2) < 0.04, f"KRR {a1:.3f} vs SVC {a2:.3f}"


def test_lsvm_multiclass_3class():
    # XOR-style 4-quadrant data -> 2 classes, but build explicit 3-class spiral-ish
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)  # values 0..2
    g = _gscale(X)
    clf = LSVMClassifier(KernelSpec("rbf", gamma=g), C=1.0).fit(X, y)
    assert clf.classes_.shape[0] == 3
    assert _acc(clf, X, y) > 0.8


def test_smo_classifier_multiclass_predict_proba():
    set_all(5)
    ds = make_dataset("blobs_separable", 200, 100, 50, 5)
    clf = SMOClassifier(KernelSpec("rbf"), C=1.0).fit(ds.X_train, ds.y_train)
    p = clf.predict_proba(ds.X_test)
    assert p.shape == (ds.X_test.shape[0], 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
