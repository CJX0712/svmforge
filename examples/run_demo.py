#!/usr/bin/env python3
"""End-to-end demo: run the benchmark, verify bit-for-bit determinism, write benchmark.json.

Author: 晨星
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from core.config import Config
from pipeline.train_eval import FLAGSHIP, SINGLE_SOTA, run_benchmark, verify_determinism

# Demo-sized config: small 2-D DGPs keep the full run (incl. determinism re-run)
# comfortably under the 60s CPU budget while remaining statistically meaningful.
DEMO_CFG = Config(
    seed=20261004,
    n_train=120,
    n_test=90,
    n_val=50,
    n_seeds=3,
    C=1.0,
)
DGP_LIST = [
    "blobs_separable",
    "moons",
    "circles",
    "noisy_linear",
    "xor",
    "imbalanced",
]


def main() -> int:
    seeds = list(range(DEMO_CFG.n_seeds))
    t0 = time.time()
    res = run_benchmark(DGP_LIST, seeds, DEMO_CFG)
    wall_a = time.time() - t0
    res2 = run_benchmark(DGP_LIST, seeds, DEMO_CFG)  # second run => determinism
    wall_b = time.time() - t0
    det = verify_determinism(res, res2)
    res["determinism_bit_identical"] = det
    res["wall_clock_sec"] = round(wall_b, 3)
    res["demo_config"] = {
        "n_train": DEMO_CFG.n_train,
        "n_test": DEMO_CFG.n_test,
        "n_val": DEMO_CFG.n_val,
        "n_seeds": DEMO_CFG.n_seeds,
        "C": DEMO_CFG.C,
    }

    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    per = res["summary"]["per_method"]
    gate = res["summary"]["gate"]
    print("=" * 68)
    print(" SVMForge · Kernel Methods & SVM — demo benchmark (author 晨星)")
    print("=" * 68)
    print(f"{'method':<12}{'mean_acc':>10}{'std':>9}")
    for m in res["methods"]:
        d = per[m]
        print(f"{m:<12}{d['mean']:>10.4f}{d['std']:>9.4f}")
    print("-" * 68)
    print(
        f"GATE1  flagship {FLAGSHIP} beats weak naive (lr): "
        f"margin {gate['margin_vs_lr']:+.4f} -> {'PASS' if gate['gate1_beat_lr_pass'] else 'FAIL'}"
    )
    print(
        f"REF   flagship {FLAGSHIP} vs strong naive (knn): "
        f"delta {gate['noninferior_vs_knn']:+.4f} (non-inferior={gate['knn_noninferior']}, "
        f"win/tie {gate['knn_win_tie_rate']:.2f})"
    )
    print(
        f"GATE2  flagship {FLAGSHIP} non-inferior to best SVM ({SINGLE_SOTA}): "
        f"delta {gate['noninferior_vs_svc']:+.4f} -> {'PASS' if gate['gate2_pass'] else 'FAIL'}"
    )
    print(f"Bit-for-bit deterministic (2 runs): {det}")
    print(f"Wall-clock (2 runs): {wall_b:.1f}s (run1 {wall_a:.1f}s)")
    print(f"ALL GATES PASS: {gate['all_pass']}")
    print("=" * 68)
    print("wrote benchmark.json")
    return 0 if gate["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
