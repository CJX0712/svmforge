"""Deterministic seeding — the single entry point for reproducibility.

Rule: never call ``np.random.seed`` ad hoc in domain code. Use
``with_seed`` / ``rng`` so every random stream is derived from one master
seed and is bit-for-bit reproducible across runs.
"""

from __future__ import annotations

import random

import numpy as np

_MASTER = 20261004
_ACTIVE = _MASTER


def set_all(seed: int = _MASTER) -> None:
    """Set the global master seed for legacy global RNGs (sklearn, numpy)."""
    global _ACTIVE
    _ACTIVE = int(seed)
    np.random.seed(int(seed))
    random.seed(int(seed))


def get_master() -> int:
    return _ACTIVE


def rng(seed: int) -> np.random.Generator:
    """Return a fresh, independent numpy Generator for ``seed``."""
    return np.random.default_rng(int(seed))


def py_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def with_seed(seed: int):
    """Context manager restoring global RNG state afterwards (defensive)."""

    prev_np = np.random.get_state()
    prev_py = random.getstate()
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(prev_np)
        random.setstate(prev_py)
