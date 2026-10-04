#!/usr/bin/env python3
"""svmforge command-line interface.

Usage:
  python cli.py bench  [--out benchmark.json] [--seeds N] [--datasets d1 d2 ...]
  python cli.py check  (offline invariant self-checks, no downloads)

Author: 晨星
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from core.config import Config
from data.synthetic import available_dgps
from pipeline.train_eval import (
    FLAGSHIP,
    METHODS,
    SINGLE_SOTA,
    run_benchmark,
)


def _print_summary(res: dict) -> None:
    per = res["summary"]["per_method"]
    gate = res["summary"]["gate"]
    print("\n=== SVMForge benchmark ===")
    print(f"{'method':<12}{'mean_acc':>10}{'std':>9}{'n':>5}")
    for m in res["methods"]:
        d = per.get(m, {"mean": float("nan"), "std": float("nan"), "n": 0})
        print(f"{m:<12}{d['mean']:>10.4f}{d['std']:>9.4f}{d['n']:>5}")
    print("\n--- DoD performance gate ---")
    print(f"flagship           : {FLAGSHIP}")
    print(
        f"  GATE1 beat weak naive (lr): margin={gate['margin_vs_lr']:+.4f} "
        f"(need >= {gate['gate_accuracy_margin']:.3f}) -> "
        f"{'PASS' if gate['gate1_beat_lr_pass'] else 'FAIL'}"
    )
    print(
        f"  REF   vs strong naive (knn): delta={gate['noninferior_vs_knn']:+.4f} "
        f"(non-inferior={gate['knn_noninferior']}, win/tie={gate['knn_win_tie_rate']:.2f})"
    )
    print(
        f"  GATE2 non-inferior to best SVM backend ({SINGLE_SOTA}): "
        f"delta={gate['noninferior_vs_svc']:+.4f} "
        f"(need >= -{gate['gate_noninferior_tol']:.3f}) "
        f"-> {'PASS' if gate['gate2_pass'] else 'FAIL'}"
    )
    print(f"datasets x seeds  : {gate['n_datasets']} x {gate['n_seeds']}")
    print(f"ALL GATES PASS    : {gate['all_pass']}")


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = Config.from_env().validate()
    if args.seeds:
        cfg = dataclasses.replace(cfg, n_seeds=args.seeds)
    dgp = args.datasets if args.datasets else available_dgps()
    methods = args.methods if args.methods else METHODS
    res = run_benchmark(dgp, list(range(cfg.n_seeds)), cfg, methods=methods)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    _print_summary(res)
    print(f"\nwrote {args.out}")
    return 0 if res["summary"]["gate"]["all_pass"] else 1


def cmd_check(args: argparse.Namespace) -> int:
    import numpy as np

    from core.types import KernelSpec
    from svm.smo import BinarySMO

    # Toy 2D set with a clear margin; SVM should separate perfectly.
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clf = BinarySMO(KernelSpec("rbf", gamma=1.0), C=1.0, max_iter=5000)
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = float(np.mean(pred == y))
    kkt = clf.check_kkt()
    print(f"[check] BinarySMO toy accuracy = {acc:.3f} (expect 1.000)")
    print(f"[check] KKT max violation    = {kkt:.2e} (expect <= tol)")
    ok = acc > 0.999 and kkt <= clf.tol * 2
    print(f"[check] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="svmforge", description="World-class kernel SVM system (author 晨星)"
    )
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("bench", help="run full benchmark and write benchmark.json")
    b.add_argument("--out", default="benchmark.json")
    b.add_argument(
        "--seeds", type=int, default=0, help="override number of seeds (0=use config)"
    )
    b.add_argument("--datasets", nargs="*", default=None)
    b.add_argument("--methods", nargs="*", default=None)
    sub.add_parser("check", help="offline invariant self-checks (no downloads)")
    args = ap.parse_args()
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "check":
        return cmd_check(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
