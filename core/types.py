"""Shared data types for svmforge.

All arrays are float64 / int64. Labels are integer-encoded 0..(C-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """A train/val/test split plus metadata.

    y_train / y_test are integer labels in 0..(n_classes-1). X_* are (n, d).
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X_train = np.asarray(self.X_train, dtype=np.float64)
        self.y_train = np.asarray(self.y_train, dtype=np.int64)
        self.X_test = np.asarray(self.X_test, dtype=np.float64)
        self.y_test = np.asarray(self.y_test, dtype=np.int64)
        if self.X_val is not None:
            self.X_val = np.asarray(self.X_val, dtype=np.float64)
            self.y_val = np.asarray(self.y_val, dtype=np.int64)

    @property
    def n_classes(self) -> int:
        return len(np.unique(np.concatenate([self.y_train, self.y_test])))

    @property
    def classes(self) -> np.ndarray:
        return np.unique(np.concatenate([self.y_train, self.y_test]))

    @property
    def dim(self) -> int:
        return int(self.X_train.shape[1])

    def labels_are_binary(self) -> bool:
        return self.n_classes == 2


@dataclass
class KernelSpec:
    """Specification of a kernel and its hyperparameters."""

    name: str  # linear | rbf | poly | sigmoid
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 0.0

    def key(self) -> str:
        return f"{self.name}-g{self.gamma:g}-d{self.degree}-c{self.coef0:g}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
        }


@dataclass
class MetricReport:
    """Per-dataset, per-method evaluation metrics."""

    dataset: str
    method: str
    seed: int
    accuracy: float = float("nan")
    macro_f1: float = float("nan")
    auc: float = float("nan")  # binary only; nan otherwise
    ece: float = float("nan")  # expected calibration error
    n_train: int = 0
    n_support: int = 0
    elapsed_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "auc": self.auc,
            "ece": self.ece,
            "n_train": self.n_train,
            "n_support": self.n_support,
            "elapsed_sec": self.elapsed_sec,
        }
