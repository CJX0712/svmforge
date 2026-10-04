"""Platt scaling (Platt 1999) — pure-numpy logistic calibration of decision values.

Maps raw decision values f to P(y=positive) via 1/(1+exp(A f + B)). Trained
with a bounded Newton method (Iris data prior for stability). Used by the
fusion flagship to put heterogeneous base models on a comparable probability
scale before weighted combination.
"""

from __future__ import annotations

import numpy as np


def _prior(count1: int, count2: int) -> tuple[float, float]:
    """Iris (2001) smoothing constants for the target probabilities."""
    if count1 + count2 <= 1:
        return 0.0, 0.0
    pos = (count1 + 1.0) / (count1 + count2 + 2.0)
    neg = (count2 + 1.0) / (count1 + count2 + 2.0)
    return np.log(pos / (1.0 - pos)), np.log(neg / (1.0 - neg))


def platt_fit(
    decision: np.ndarray, y_binary: np.ndarray, max_iter: int = 100
) -> tuple[float, float]:
    """Fit (A, B). y_binary in {0,1} where 1 = positive class."""
    f = np.asarray(decision, dtype=np.float64).ravel()
    t = np.asarray(y_binary, dtype=np.float64).ravel()
    hi = float(np.sum(t >= 0.5))
    lo = float(np.sum(t < 0.5))
    A0, B0 = _prior(hi, lo)
    A, B = 0.0, 0.0
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(A0 * f + B0))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        # gradient of cross-entropy w.r.t (A0, B0)
        dA = np.sum((p - t) * p * (1.0 - p) * f)
        dB = np.sum((p - t) * p * (1.0 - p))
        # Hessian (diagonal approx via second derivative of sigmoid*... )
        # Use simplified Newton with row-wise second derivative.
        pp = p * (1.0 - p)
        Haa = np.sum(pp * (1.0 - 2.0 * p) * f * f)
        Hab = np.sum(pp * (1.0 - 2.0 * p) * f)
        Hbb = np.sum(pp * (1.0 - 2.0 * p))
        # Newton step on the convex negative-log-likelihood
        # grad of NLL = -(dA, dB); Hessian = -(Haa, Hab, Hbb)
        det = Haa * Hbb - Hab * Hab
        if abs(det) < 1e-12:
            break
        stepA = -(Hbb * dA - Hab * dB) / det
        stepB = -(-Hab * dA + Haa * dB) / det
        A0 += stepA
        B0 += stepB
        if abs(stepA) < 1e-7 and abs(stepB) < 1e-7:
            break
    A, B = float(A0), float(B0)
    return A, B


def platt_predict(A: float, B: float, decision: np.ndarray) -> np.ndarray:
    f = np.asarray(decision, dtype=np.float64).ravel()
    p = 1.0 / (1.0 + np.exp(A * f + B))
    return np.clip(p, 0.0, 1.0)


def calibrate_binary(
    model, X_val: np.ndarray, y_val_binary: np.ndarray
) -> tuple[float, float]:
    """Fit Platt (A,B) for a model exposing decision_function on a binary task."""
    f = model.decision_function(X_val).ravel()
    return platt_fit(f, y_val_binary.astype(int))
