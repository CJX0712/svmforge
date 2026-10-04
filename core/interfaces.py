"""Protocol interfaces — the architectural contracts of svmforge.

Every module depends only on these abstractions, never on concrete classes,
so backends (pure-numpy vs sklearn) are interchangeable and independently
verifiable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Classifier(Protocol):
    """A binary or multiclass classifier."""

    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def decision_function(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...


@runtime_checkable
class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Kernel(Protocol):
    """A positive-definite kernel: k(x, x') over pairs of rows."""

    name: str

    def gram(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray: ...


@runtime_checkable
class Backend(Protocol):
    """A trainable model backend selected by availability."""

    name: str

    def available(self) -> bool: ...

    def train(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...
