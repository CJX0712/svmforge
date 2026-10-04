"""File loaders (CSV / NPZ). Offline-safe; used only for user-provided data."""

from __future__ import annotations

import numpy as np

from core.types import Dataset


def load_csv(path: str, label_col: int = -1, sep: str = ",") -> Dataset:
    """Load a CSV into a single in-memory Dataset (no split — caller splits)."""
    arr = np.loadtxt(path, delimiter=sep, ndmin=2)
    X = np.delete(arr, label_col, axis=1)
    y = arr[:, label_col].astype(np.int64)
    return Dataset(
        name="csv",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        metadata={"source": path},
    )
