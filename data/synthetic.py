"""Synthetic DGP generators — fixed-seed, leak-free, difficulty tuned.

Every generator takes an explicit numpy Generator so results are reproducible
and the same seed never leaks across train/val/test (each split is carved from
one shuffled index sequence). Difficulty knobs are set so baselines are clearly
below the ceiling (the "sweet spot" the methodology requires).
"""

from __future__ import annotations

import numpy as np

from core.types import Dataset

__all__ = ["available_dgps", "make_dataset"]


def _gen_blobs(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n1 = N // 2
    n2 = N - n1
    X1 = rng.normal([0.0, 0.0], 0.85, (n1, 2))
    X2 = rng.normal([3.2, 3.0], 0.85, (n2, 2))
    X = np.vstack([X1, X2])
    y = np.array([0] * n1 + [1] * n2)
    return X, y


def _gen_moons(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n1 = N // 2
    t = np.linspace(0.0, np.pi, n1)
    X1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    X2 = np.stack([1.0 - np.cos(t), 1.0 - np.sin(t)], axis=1) - [0.5, 0.0]
    X = np.vstack([X1, X2]) + rng.normal(0.0, 0.18, (N, 2))
    y = np.array([0] * n1 + [1] * (N - n1))
    return X, y


def _gen_circles(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n1 = N // 2
    t = np.linspace(0.0, 2 * np.pi, n1)
    X1 = np.stack([np.cos(t), np.sin(t)], axis=1) * 1.0
    X2 = np.stack([np.cos(t), np.sin(t)], axis=1) * 3.0
    X = np.vstack([X1, X2]) + rng.normal(0.0, 0.12, (N, 2))
    y = np.array([0] * n1 + [1] * (N - n1))
    return X, y


def _gen_linear(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0.0, 1.0, (N, 2))
    score = X @ np.array([1.0, 1.0]) + rng.normal(0.0, 0.35, N)
    y = (score > 0).astype(int)
    return X, y


def _gen_xor(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    c0 = [[-1.4, -1.4], [1.4, 1.4]]  # class 0 corners
    c1 = [[-1.4, 1.4], [1.4, -1.4]]  # class 1 corners
    per = N // 4
    Xs, ys = [], []
    for c, lab in zip(c0, [0, 0]):
        Xs.append(rng.normal(c, 0.28, (per, 2)))
        ys += [lab] * per
    for c, lab in zip(c1, [1, 1]):
        Xs.append(rng.normal(c, 0.28, (per, 2)))
        ys += [lab] * per
    rem = N - per * 4
    if rem > 0:
        Xs.append(rng.normal(0.0, 0.28, (rem, 2)))
        ys += [0] * rem
    X = np.vstack(Xs)
    y = np.array(ys)
    return X, y


def _gen_imbalanced(
    N: int, rng: np.random.Generator, ir: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    n1 = max(20, N // (ir + 1))
    n2 = N - n1
    X0 = rng.normal([-2.2, 0.0], 0.55, (n1, 2))
    X1 = rng.normal([2.2, 0.0], 0.7, (n2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n1 + [1] * n2)
    return X, y


_DGP = {
    "blobs_separable": _gen_blobs,
    "moons": _gen_moons,
    "circles": _gen_circles,
    "noisy_linear": _gen_linear,
    "xor": _gen_xor,
    "imbalanced": _gen_imbalanced,
}


def available_dgps() -> list[str]:
    return sorted(_DGP.keys())


def make_dataset(
    name: str, n_train: int, n_test: int, n_val: int, seed: int
) -> Dataset:
    if name not in _DGP:
        raise ValueError(f"unknown DGP {name!r}; available={available_dgps()}")
    total = n_train + n_test + n_val
    rng = np.random.default_rng(int(seed))
    X, y = _DGP[name](total, rng)
    idx = rng.permutation(total)
    Xs, ys = X[idx], y[idx]
    a, b, c = n_train, n_train + n_val, n_train + n_val + n_test
    return Dataset(
        name=name,
        X_train=Xs[:a],
        y_train=ys[:a],
        X_val=Xs[a:b],
        y_val=ys[a:b],
        X_test=Xs[b:c],
        y_test=ys[b:c],
        metadata={"total": total, "seed": int(seed)},
    )
