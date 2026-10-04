import numpy as np
from sklearn.linear_model import LogisticRegression

from core.seed import set_all
from core.types import KernelSpec
from data.synthetic import make_dataset
from fusion.kernel_fuse import KernelFuse


def _build():
    bases = [
        {"method": "lsvm", "kernel": KernelSpec("rbf", gamma=0.1)},
        {"method": "lsvm", "kernel": KernelSpec("rbf", gamma=1.0)},
        {"method": "lsvm", "kernel": KernelSpec("rbf", gamma=10.0)},
        {"method": "lsvm", "kernel": KernelSpec("poly", degree=2)},
        {"method": "sklearn_svc", "kernel": KernelSpec("rbf")},
    ]
    return KernelFuse(bases, C=1.0)


def test_fusion_weights_normalized():
    set_all(9)
    ds = make_dataset("moons", 300, 200, 80, 9)
    f = _build().fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    # Adaptive KernelFuse either anchors on a single best base (weights_=[1.0])
    # or fuses a qualified subset; in both cases weights sum to 1 and are >= 0.
    assert np.isclose(float(np.sum(f.weights_)), 1.0, atol=1e-9)
    assert np.all(f.weights_ >= 0.0)
    assert f.weights_.shape[0] >= 1


def test_fusion_deterministic():
    set_all(9)
    ds = make_dataset("moons", 300, 200, 80, 9)
    f1 = _build().fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    f2 = _build().fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    p1 = f1.predict_proba(ds.X_test)
    p2 = f2.predict_proba(ds.X_test)
    assert np.allclose(p1, p2, atol=1e-12)


def test_fusion_beats_naive_lr():
    set_all(9)
    ds = make_dataset("moons", 300, 200, 80, 9)
    f = _build().fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    lr = LogisticRegression(max_iter=2000).fit(ds.X_train, ds.y_train)
    acc_f = float(np.mean(f.predict(ds.X_test) == ds.y_test))
    acc_lr = float(np.mean(lr.predict(ds.X_test) == ds.y_test))
    # on a non-linear dataset KernelFuse should not be worse than linear LR
    assert acc_f >= acc_lr - 0.02


def test_fusion_predict_classes():
    set_all(9)
    ds = make_dataset("moons", 300, 200, 80, 9)
    f = _build().fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    pred = f.predict(ds.X_test)
    assert set(np.unique(pred)).issubset(set(ds.classes.tolist()))
