import numpy as np

from core.types import KernelSpec
from svm.smo import BinarySMO


def _separable():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def test_smo_separable_perfect():
    X, y = _separable()
    clf = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0, max_iter=5000)
    clf.fit(X, y)
    assert np.mean(clf.predict(X) == y) > 0.999
    assert clf.check_kkt() <= clf.tol * 2


def test_smo_deterministic():
    X, y = _separable()
    a = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0).fit(X, y)
    b = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0).fit(X, y)
    assert np.allclose(a.alpha, b.alpha, atol=1e-9)
    assert a.n_support_ == b.n_support_


def test_smo_matches_sklearn_on_toy():
    from sklearn.svm import SVC

    X, y = _separable()
    clf = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0, max_iter=8000).fit(X, y)
    sk = SVC(kernel="rbf", gamma=1.0, C=1.0).fit(X, y)
    # decision boundaries should agree on the grid corners
    assert np.mean(clf.predict(X) == sk.predict(X)) > 0.99


def test_smo_dual_objective_finite():
    X, y = _separable()
    clf = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0).fit(X, y)
    obj = clf.dual_objective()
    assert np.isfinite(obj)
    # For a correct max-margin solution with C=1 the dual should be positive here
    assert obj > 0.0
