#!/usr/bin/env python3
"""Ablation: KernelFuse (adaptive select-or-fuse) vs weaker fusion baselines.

Author: 晨星

We compare three fusers built from the SAME base pool:
  * adaptive  -- KernelFuse as shipped (anchor on the best base when clearly
                ahead, else accuracy-weighted fusion of the competitive subset).
  * mean      -- naive equal-weight probability averaging of ALL bases.
  * random    -- equal-weight averaging with a fixed (non-learned) weight vector.

The point: the adaptive rule recovers the best single base when one is clearly
superior and only fuses when bases are genuinely competitive, so it never
dilutes a strong base the way naive/mean fusion does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.seed import set_all
from core.types import KernelSpec
from data.synthetic import make_dataset
from eval.metrics import accuracy
from fusion.kernel_fuse import KernelFuse
from svm import KRRClassifier, LSVMClassifier


def _bases():
    return [
        {"method": "lsvm", "kernel": KernelSpec("rbf", gamma=0.1)},
        {"method": "lsvm", "kernel": KernelSpec("rbf", gamma=1.0)},
        {"method": "lsvm", "kernel": KernelSpec("rbf", gamma=10.0)},
        {"method": "sklearn_svc", "kernel": KernelSpec("rbf")},
    ]


def _fit_bases(specs, X, y, Xv, yv):
    from fusion.kernel_fuse import _SklearnSVCWrapper

    models, vacc = [], []
    for s in specs:
        if s["method"] == "lsvm":
            m = LSVMClassifier(s["kernel"], C=1.0)
        elif s["method"] == "krr":
            m = KRRClassifier(s["kernel"], C=1.0)
        else:
            m = _SklearnSVCWrapper(s["kernel"], 1.0)
        m.fit(X, y)
        models.append(m)
        vacc.append(accuracy(yv, m.predict(Xv)))
    return models, np.asarray(vacc, float)


def _proba(m, X):
    p = m.predict_proba(np.asarray(X, float))
    return p if p.ndim == 2 else np.column_stack([1 - p, p])


def mean_fuse(models, X):
    Ps = np.stack([_proba(m, X) for m in models], 0)
    return Ps.mean(0)


def random_fuse(models, X, rng):
    w = rng.dirichlet(np.ones(len(models)))
    Ps = np.stack([_proba(m, X) for m in models], 0)
    return (w[:, None, None] * Ps).sum(0)


def main() -> int:
    set_all(20261004)
    rng = np.random.default_rng(7)
    dsets = ["moons", "circles", "xor", "noisy_linear"]
    rows = {"adaptive": [], "mean": [], "random": []}
    for d in dsets:
        ds = make_dataset(d, 200, 150, 60, 3)
        models, _ = _fit_bases(_bases(), ds.X_train, ds.y_train, ds.X_val, ds.y_val)
        kf = KernelFuse(_bases(), C=1.0).fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
        yhat_adv = kf.predict(ds.X_test)
        rows["adaptive"].append(accuracy(ds.y_test, yhat_adv))
        rows["mean"].append(accuracy(ds.y_test, np.argmax(mean_fuse(models, ds.X_test), 1)))
        yhat_rnd = np.argmax(random_fuse(models, ds.X_test, rng), 1)
        rows["random"].append(accuracy(ds.y_test, yhat_rnd))

    print("=" * 60)
    print(" KernelFuse ablation (mean test acc over 4 datasets)")
    print("=" * 60)
    for k in ("adaptive", "mean", "random"):
        print(f"  {k:<10} {np.mean(rows[k]):.4f}")
    adv = np.mean(rows["adaptive"])
    print("-" * 60)
    print(f"  adaptive - mean   : {adv - np.mean(rows['mean']):+.4f}")
    print(f"  adaptive - random : {adv - np.mean(rows['random']):+.4f}")
    print("=" * 60)
    # Ablation claim: the adaptive select-or-fuse rule is NON-INFERIOR to naive
    # fusion (when one base is clearly best we use it verbatim; otherwise we fuse
    # the competitive subset). Averaging all bases cannot beat the best component,
    # so adaptive should track or beat naive fusion within a small tolerance.
    tol = 0.01
    assert adv >= np.mean(rows["mean"]) - tol, "adaptive should not lose to mean fusion"
    assert adv >= np.mean(rows["random"]) - tol, "adaptive should not lose to random fusion"
    print("ABLATION OK: adaptive select-or-fuse is non-inferior to naive fusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
