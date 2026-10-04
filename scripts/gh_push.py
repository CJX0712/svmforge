#!/usr/bin/env python3
"""Three-tier GitHub delivery for svmforge (author 晨星).

Tier 0 — local secret scan (abort if a credential pattern is found).
Tier 1 — ensure the remote repo exists and push the commit + tag.
Tier 2 — publish a GitHub Release with the live benchmark summary.

Usage:
  python scripts/gh_push.py [--repo CJX0712/svmforge] [--version v0.1.0] [--branch main]

The script is idempotent: an existing repo is reused, an existing tag is skipped.
It shells out to `git` and `gh`; both must be on PATH and `gh` must be authed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_RE = re.compile(
    r"(sk-[0-9a-zA-Z]{20}|api[_-]?key|secret|password|token)\s*[:=]",
    re.IGNORECASE,
)


def run(cmd, check=True, capture=True):
    print(f"[gh_push] $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=capture, text=True)
    if check and res.returncode != 0:
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(f"command failed: {' '.join(cmd)}")
    return res


def tier0_secret_scan() -> None:
    print("[gh_push] Tier 0: secret scan")
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".lock", ".pyc"} or ".git" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if SECRET_RE.search(text):
            hits.append(str(p))
    if hits:
        print("::error:: possible secret in:", file=sys.stderr)
        for h in hits:
            print("  ", h, file=sys.stderr)
        sys.exit(1)
    print("[gh_push] Tier 0: clean")


def git_ensure_identity() -> None:
    import contextlib

    with contextlib.suppress(Exception):
        run(["git", "config", "user.email"], check=False)
    email = run(["git", "config", "user.email"], check=False).stdout.strip()
    name = run(["git", "config", "user.name"], check=False).stdout.strip()
    if not email:
        run(["git", "config", "user.email", "chenxing@svmforge.local"])
    if not name:
        run(["git", "config", "user.name", "晨星"])


def tier1_push(repo: str, version: str, branch: str) -> None:
    print(f"[gh_push] Tier 1: ensure repo {repo} exists")
    res = run(["gh", "repo", "view", repo], check=False)
    if res.returncode != 0:
        desc = "svmforge — world-class kernel-SVM system (author 晨星)"
        run(["gh", "repo", "create", repo, "--public", "--description", desc])
    else:
        print(f"[gh_push] repo {repo} already exists")

    run(["git", "init"], check=False)
    git_ensure_identity()
    run(["git", "checkout", "-B", branch], check=False)
    run(["git", "add", "-A"])
    # Ship the benchmark artifact as DoD evidence even though .gitignore excludes it.
    run(["git", "add", "-f", "benchmark.json"], check=False)
    status = run(["git", "status", "--porcelain"], check=False).stdout.strip()
    if status:
        msg = f"svmforge {version}: world-class kernel-SVM system (author 晨星)"
        run(["git", "commit", "-m", msg])
    else:
        print("[gh_push] nothing to commit")

    run(["git", "remote", "remove", "origin"], check=False)
    run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"])
    run(["git", "push", "-u", "origin", branch, "--force"])

    tag_res = run(["git", "tag", "-l", version], check=False).stdout.strip()
    if version not in tag_res.split():
        run(["git", "tag", version])
        run(["git", "push", "origin", version])
    else:
        print(f"[gh_push] tag {version} exists, skipping")


def tier2_release(repo: str, version: str) -> None:
    print(f"[gh_push] Tier 2: release {version}")
    notes = build_release_notes(version)
    # gh release create is idempotent-ish; if it exists, skip.
    existing = run(["gh", "release", "list", "--repo", repo], check=False).stdout
    if version in existing:
        print(f"[gh_push] release {version} exists, skipping")
        return
    run(
        [
            "gh",
            "release",
            "create",
            version,
            "--repo",
            repo,
            "--title",
            f"svmforge {version}",
            "--notes",
            notes,
        ]
    )


def build_release_notes(version: str = "v0.1.0") -> str:
    bench = ROOT / "benchmark.json"
    if not bench.exists():
        return f"svmforge {version}\n\n(See benchmark.json for live results.)"
    data = json.loads(bench.read_text(encoding="utf-8"))
    g = data["summary"]["gate"]
    per = data["summary"]["per_method"]
    lines = [
        f"# svmforge {version}",
        "",
        "World-class kernel-SVM system — author 晨星.",
        "",
        "## DoD performance gates (6 datasets x 3 seeds, deterministic)",
        f"- GATE1 beat weak LR: **+{g['margin_vs_lr'] * 100:.1f}%** (need >= 1%) -> "
        f"{'PASS' if g['gate1_beat_lr_pass'] else 'FAIL'}",
        f"- GATE2 non-inferior to best SVM backend: **{g['noninferior_vs_svc'] * 100:+.2f}%** "
        f"(tol -0.5%) -> {'PASS' if g['gate2_pass'] else 'FAIL'}",
        f"- REF vs strong kNN: {g['noninferior_vs_knn'] * 100:+.2f}% "
        f"(win/tie {g['knn_win_tie_rate']:.2f})",
        f"- Bit-for-bit deterministic: {data.get('determinism_bit_identical')}",
        f"- Wall-clock (2 runs): {data.get('wall_clock_sec')}s",
        "",
        "## Mean accuracy by method",
    ]
    for m in data["methods"]:
        d = per[m]
        lines.append(f"- `{m}`: {d['mean']:.4f} ± {d['std']:.4f}")
    lines += [
        "",
        "## Reproduce",
        "```bash",
        "pip install -r requirements.txt",
        "python examples/run_demo.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="CJX0712/svmforge")
    ap.add_argument("--version", default="v0.1.0")
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    tier0_secret_scan()
    tier1_push(args.repo, args.version, args.branch)
    tier2_release(args.repo, args.version)
    print(
        f"[gh_push] done -> https://github.com/{args.repo}/releases/tag/{args.version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
