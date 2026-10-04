# Changelog

All notable changes to **svmforge** are documented here.

## v0.1.0 (2026-10-04)

- **World-class kernel-SVM system** — authored by 晨星.
- **Pure-numpy offline Tier-1 solvers** (zero downloads, works without scikit-learn):
  - LS-SVM (Suykens & Vandewalle 1999) — equality-constrained, closed-form linear-system solve.
  - Kernel Ridge Regression (Savers et al. 1998) — closed-form, stability cross-check.
  - Platt (1998) Sequential Minimal Optimization — canonical max-margin solver with KKT/dual invariants.
  - Platt-scaling probability calibration.
- **scikit-learn Tier-0 backend** (SVC / KernelRidge) with automatic offline fallback when sklearn is absent.
- **KernelFuse flagship** — adaptive *select-or-fuse* multi-kernel SVM ensemble:
  anchors on the single best base when one is clearly superior (non-inferior by construction),
  otherwise fuses the validation-competitive subset by accuracy-weighted probability fusion.
- **6 leak-free synthetic DGPs** with fixed-seed, single-shuffle split carving (no train/val/test leakage).
- **Deterministic benchmark**: 6 datasets × 3 seeds, bit-for-bit reproducible via `core.seed.set_all`.
- **DoD performance gates**:
  - GATE1 — flagship beats weak linear baseline (LR) by **+18.6%** (≥ +1% required) → PASS.
  - GATE2 — flagship non-inferior to best single SVM backend (Δ = −0.06%, within ±0.5% tol) → PASS.
  - REF — flagship matches strong kNN reference (+0.37%, win/tie 0.889) → reported, non-inferior.
- **27 unit tests** (kernel symmetry/PD invariants, SMO dual-monotonicity & KKT, LS-SVM/KRR vs sklearn
  parity, fusion determinism, pipeline determinism, ≥3 failure cases), **ruff-clean**, CI matrix py3.12/3.13.
- Demo wall-clock **29.1 s** for 2 full deterministic runs (budget ≤ 60 s).
