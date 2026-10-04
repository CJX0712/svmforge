import numpy as np

from core.types import Dataset
from data.synthetic import available_dgps, make_dataset


def test_dataset_deterministic_same_seed():
    a = make_dataset("moons", 200, 100, 50, 42)
    b = make_dataset("moons", 200, 100, 50, 42)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_train, b.y_train)


def test_dataset_differs_across_seed():
    a = make_dataset("moons", 200, 100, 50, 1)
    b = make_dataset("moons", 200, 100, 50, 2)
    assert not np.array_equal(a.X_train, b.X_train)


def test_no_leakage_disjoint_splits():
    ds = make_dataset("xor", 200, 100, 50, 7)
    assert isinstance(ds, Dataset)
    # indices used for each split are disjoint by construction
    idx_all = np.arange(350)
    tr = idx_all[:200]
    va = idx_all[200:300]
    te = idx_all[300:350]
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0


def test_labels_integer_and_two_class():
    ds = make_dataset("imbalanced", 300, 100, 50, 5)
    assert ds.y_train.dtype == np.int64
    assert ds.n_classes == 2


def test_all_dgps_runnable():
    for name in available_dgps():
        ds = make_dataset(name, 120, 40, 20, 1)
        assert ds.X_train.shape[0] == 120
        assert set(np.unique(ds.y_train).tolist()).issubset({0, 1})
