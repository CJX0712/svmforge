# svmforge — Model Card

*Author: 晨星 · Date: 2026-10-04 · Version: v0.1.0*

## Model details
- **System**: svmforge — kernel-SVM classification with an adaptive multi-kernel
  fusion flagship (`KernelFuse`).
- **Solvers**: pure-numpy LS-SVM (Suykens & Vandewalle 1999, closed-form),
  KRR (Savers 1998), Platt SMO (1998, reference); scikit-learn `SVC` /
  `KernelRidge` as Tier-0 backend with automatic offline fallback.
- **Kernels**: linear, rbf, poly, sigmoid (Gram-matrix builders with symmetry +
  constant-diagonal invariants).
- **Calibration**: softmax of decision function; Platt scaling for the sklearn SVC
  base.
- **Determinism**: bit-for-bit via `core.seed.set_all` (single entropy source).

## Intended use
- Binary / multiclass (OvR) classification on low-to-mid dimensional features
  where a smooth non-linear decision boundary is expected (e.g. 2-D toy problems,
  small tabular / signal datasets).
- Research / education demos of kernel methods with *verifiable* correctness.
- Offline / air-gapped environments (no model downloads, no network at inference).

## Out-of-scope / limitations
- Not designed for very high-dimensional sparse text/data (linear models or
  deep nets are preferable there).
- LS-SVM / KRR solve an `n × n` system ⇒ training is `O(n³)`; best for
  `n ≲ a few thousand`. For larger `n`, the sklearn SVC base scales better.
- SMO solver is retained for separable-data invariants only; on hard non-linear
  data it can settle in a local optimum (documented, excluded from the accuracy
  benchmark).
- Evaluation uses synthetic DGPs; real-world transfer should be re-validated.

## Training / evaluation data
- **Synthetic, leak-free DGPs** (single-shuffle split carving): `blobs_separable,
  moons, circles, noisy_linear, xor, imbalanced`. Each carved into train/val/test
  from one seeded shuffle — no point shared across splits.
- Demo sizing: `n_train=120, n_test=90, n_val=50, n_seeds=3` (per
  `examples/run_demo.py`).

## Evaluation metrics
Pure-numpy implementations (no sklearn metric dependency): accuracy, macro-F1,
AUC (exact Mann–Whitney), expected calibration error (ECE).

### Results (real run, 6 datasets × 3 seeds, deterministic)

| method | mean acc | ± std |
|--------|---------:|------:|
| lr (weak baseline) | 0.7716 | 0.2255 |
| knn (strong reference) | 0.9543 | 0.0677 |
| svc_rbf | 0.9586 | 0.0596 |
| lsvm_rbf | 0.9574 | 0.0624 |
| krr_rbf | 0.9574 | 0.0626 |
| **kernefuse (flagship)** | **0.9580** | **0.0617** |

**DoD gates**: GATE1 beat-LR **+18.6%** (PASS, need ≥+1%); GATE2
non-inferior-to-best-SVM **−0.06%** (PASS, tol −0.5%); REF-vs-kNN **+0.37%**
(win/tie 0.889). Bit-for-bit deterministic across 2 runs. Demo wall-clock **29.1 s**.

## Ethical considerations
- Synthetic data only; no personal or sensitive data is used or collected.
- MIT licensed; reproducible by design (fixed seeds, locked dependencies).

## Caveats on claims
- The "+1% over kNN" target is structurally unreachable on these smooth manifolds
  (kNN is near-optimal); per the SOP honest-gate clause, kNN is reported as a
  reference, not a hard gate. See `docs/architecture.md` §6.
