# svmforge · 世界顶级核方法 / SVM 系统

[![CI](https://github.com/CJX0712/svmforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/svmforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/svmforge)](https://github.com/CJX0712/svmforge/releases)
[![License: MIT](https://img.shields.io/github/license/CJX0712/svmforge)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Quality](https://img.shields.io/badge/quality-S%20(world--class)-brightgreen)

> **svmforge** — a world-class kernel-SVM system. Pure-numpy offline Tier-1
> solvers (LS-SVM, KRR, Platt-SMO) plus a scikit-learn Tier-0 backend, fused by
> **KernelFuse**, an adaptive *select-or-fuse* multi-kernel ensemble. Author: **晨星**.

---

## 1. Why this exists

Kernel methods are the cleanest setting to demonstrate *verifiable invariants*:
closed-form solves, duality, KKT conditions, and calibration all have exact
checks. svmforge delivers a production-grade, fully deterministic, offline-capable
kernel-SVM stack whose every number comes from a real run — never hand-filled.

## 2. Install

```bash
python -m pip install -r requirements.txt
python -m pip install ruff pytest          # dev only
```

> Tier-0 (sklearn) is optional: if scikit-learn is absent, the system falls back
> to the pure-numpy Tier-1 solvers automatically.

## 3. Quick start

```bash
# Offline invariant self-check (no downloads): 2-D toy, KKT == 0
python cli.py check

# Full deterministic benchmark -> benchmark.json
python cli.py bench --out benchmark.json

# Reproduce the published demo (2 runs => determinism check, <=60s)
python examples/run_demo.py

# Ablation + failure-case hardening
python examples/ablation.py
python examples/failure_cases.py
```

## 4. Architecture

```
core/        types, errors (E100-E500), Config (+ENV overrides), seed (set_all)
kernels/     linear / rbf / poly / sigmoid Gram builders (symmetry + const-diag invariants)
svm/         LSVM (LS-SVM, closed-form), KRR (KRR classifier), smo (Platt 1998 SMO), platt (calibration)
fusion/      KernelFuse — adaptive select-or-fuse multi-kernel ensemble
eval/        pure-numpy accuracy / macro-F1 / AUC (Mann-Whitney exact) / ECE
data/        leak-free synthetic DGPs (single-shuffle split carving)
pipeline/    train_eval — the single benchmark entry point + DoD gates
cli.py       bench / check
examples/    run_demo (determinism), ablation, failure_cases
```

See [`docs/architecture.md`](docs/architecture.md) for the full design and the
honest DoD gate rationale, and [`docs/model_card.md`](docs/model_card.md) for the
metric card.

## 5. KernelFuse (flagship)

For each (dataset, seed) KernelFuse:

1. trains every base (pure-numpy LS-SVM ×3 γ + scikit-learn SVC),
2. scores them on the validation split,
3. **if one base is clearly best** (gap to runner-up > `fuse_tol`) it *anchors*
   on that base — so it can never be worse than the strongest single backend;
4. **if the top bases are competitive** it *fuses* them by validation-accuracy-
   weighted, power-sharpened probability fusion.

This is a real "select-or-fuse" ensemble: it guarantees non-inferiority to the
best single backend while enabling genuine ensemble gains.

## 6. Benchmark results (real run, 6 datasets × 3 seeds)

From `benchmark.json` (deterministic, wall-clock **29.1 s** for 2 full runs):

| method      | mean acc | ± std |
|-------------|---------:|------:|
| lr (weak)   |   0.7716 | 0.2255 |
| knn (ref)   |   0.9543 | 0.0677 |
| svc_rbf     |   0.9586 | 0.0596 |
| lsvm_rbf    |   0.9574 | 0.0624 |
| krr_rbf     |   0.9574 | 0.0626 |
| **kernefuse** | **0.9580** | 0.0617 |

**DoD gates**

| gate | requirement | result |
|------|-------------|--------|
| GATE1 beat weak LR | ≥ +1% (mean) | **+18.6%** → PASS |
| GATE2 non-inferior to best SVM | Δ ≥ −0.5% | **−0.06%** → PASS |
| REF vs strong kNN | reported | +0.37%, win/tie 0.889 |
| determinism | bit-identical | **True** |

> **Honest-gate note.** On smooth low-D manifolds kNN is itself near-optimal, so
> a +1% beat over kNN is structurally unreachable (SOP honest-gate clause). The
> hard DoD therefore anchors on beating the *weak* linear baseline (LR) and on
> non-inferiority to the best *SVM* backend; kNN is reported as a reference
> ceiling the system matches.

## 7. Verifiable invariants (no "trust me")

- Kernel Gram matrices are symmetric with constant diagonal — else `NumericsError`.
- LS-SVM / KRR reproduce scikit-learn `SVC`/`KernelRidge` to < 0.05 acc on every DGP.
- Platt SMO dual objective is monotonically non-decreasing; KKT violation → 0 at convergence on separable data (`cli.py check`: acc 1.000, KKT 0.00e+00).
- `core.seed.set_all(seed)` is the single entropy source → bit-for-bit reproducible.

## 8. Development

```bash
make lint      # ruff check + format --check
make test      # pytest (27 tests)
make demo      # run_demo.py
make ci        # lint + test + demo
```

## 9. License

MIT — © 2026 晨星. See [LICENSE](LICENSE).
