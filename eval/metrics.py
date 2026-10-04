"""Pure-numpy evaluation metrics with documented edge-case handling.

All functions accept numpy arrays. Binary AUC uses the exact Mann-Whitney U
formulation; ECE uses predicted-class confidence bins. NaN is returned (never a
fabricated number) when a metric is undefined for the label set.
"""

from __future__ import annotations

import numpy as np

__all__ = ["accuracy", "auc", "ece", "macro_f1"]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray | None = None
) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 0:
        return float("nan")
    f1s = []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def auc(y_true_binary: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true_binary).astype(int).ravel()
    s = np.asarray(score).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[ys == 1]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(p_pred, dtype=np.float64)
    if p.ndim == 1:
        pred = (p >= 0.5).astype(int)
        conf = np.where(p >= 0.5, p, 1.0 - p)
    else:
        pred = np.argmax(p, axis=1)
        conf = p.max(axis=1)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (
            (conf >= lo) & (conf < hi)
            if i < n_bins - 1
            else (conf >= lo) & (conf <= hi)
        )
        k = int(m.sum())
        if k == 0:
            continue
        acc = float(np.mean(pred[m] == y[m]))
        tot += k * abs(acc - conf[m].mean())
    return float(tot / n)
