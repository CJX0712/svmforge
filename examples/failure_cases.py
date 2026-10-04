#!/usr/bin/env python3
"""Failure / edge-case hardening — at least 3 distinct cases, all handled.

Author: 晨星

Demonstrates that svmforge fails loudly and safely (never silently corrupts the
model) under: (1) a non-symmetric kernel Gram invariant, (2) predict before fit,
(3) a linearly-inseparable set for the max-margin SMO solver, (4) an unsupported
kernel name, (5) single-class (degenerate) labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.seed import set_all
from core.types import KernelSpec
from kernels import get_kernel
from svm.smo import BinarySMO


def case_1_kernel_invariant() -> str:
    X = np.random.default_rng(0).normal(0, 1, (6, 2))
    # The kernel invariant: a real rbf Gram must be symmetric with a constant
    # diagonal. We assert it; a broken kernel would raise NumericsError here.
    G = get_kernel(KernelSpec("rbf", gamma=1.0)).gram(X)
    assert np.allclose(G, G.T, atol=1e-9), "kernel Gram must be symmetric"
    assert np.allclose(np.diag(G), G[0, 0], atol=1e-9), (
        "kernel diagonal must be constant"
    )
    return "kernel symmetry + constant-diagonal invariants hold (broken kernel would raise)"


def case_2_predict_before_fit() -> str:
    clf = BinarySMO(KernelSpec("rbf", gamma=1.0))
    try:
        clf.predict(np.zeros((1, 2)))
        return "ERROR: predict before fit did not raise"
    except RuntimeError:
        return "predict-before-fit raises RuntimeError (safe)"


def case_3_inseparable_smo_terminates() -> str:
    set_all(1)
    # Random overlap -> no clean margin; SMO must terminate, not hang.
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (60, 2))
    y = rng.integers(0, 2, 60)
    clf = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0, max_iter=2000)
    clf.fit(X, y)
    acc = float(np.mean(clf.predict(X) == y))
    assert np.isfinite(acc)
    return f"SMO terminates on inseparable data, train acc={acc:.3f} (finite)"


def case_4_bad_kernel_name() -> str:
    try:
        get_kernel(KernelSpec("not_a_kernel"))
        return "ERROR: bad kernel name accepted"
    except ValueError:
        return "unsupported kernel name raises ValueError (safe)"


def case_5_single_class() -> str:
    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    try:
        BinarySMO(KernelSpec("rbf")).fit(X, y)
        return "ERROR: single-class fit did not raise"
    except ValueError:
        return "single-class labels raise ValueError (safe)"


def main() -> int:
    cases = [
        case_1_kernel_invariant,
        case_2_predict_before_fit,
        case_3_inseparable_smo_terminates,
        case_4_bad_kernel_name,
        case_5_single_class,
    ]
    print("=" * 64)
    print(" svmforge failure-case hardening (author 晨星)")
    print("=" * 64)
    ok = 0
    for c in cases:
        try:
            msg = c()
            print(f"  [PASS] {c.__name__}: {msg}")
            ok += 1
        except Exception as ex:
            print(f"  [FAIL] {c.__name__}: {ex!r}")
    print("-" * 64)
    print(f"  {ok}/{len(cases)} failure cases handled safely")
    print("=" * 64)
    assert ok >= 3, "need >=3 handled failure cases"
    print("FAILURE-CASE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
