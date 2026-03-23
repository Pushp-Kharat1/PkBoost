#!/usr/bin/env python3
"""
Autoresearch loop for PkBoost — SVP company scoring use case.

Inspired by Karpathy's autoresearch: an AI agent proposes targeted Rust changes,
compiles, benchmarks, and keeps improvements. Runs autonomously overnight.

Usage:
    python experiment_loop.py               # run 50 experiments
    python experiment_loop.py --n 100       # run 100 experiments
    python experiment_loop.py --baseline    # just measure baseline score

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY must be set
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

PKBOOST_DIR = Path(__file__).parent.parent
AUTORESEARCH_DIR = Path(__file__).parent
PROGRAM_FILE = AUTORESEARCH_DIR / "program.md"
LOG_FILE = AUTORESEARCH_DIR / "experiment_log.jsonl"
BENCHMARK_SCRIPT = AUTORESEARCH_DIR / "benchmark.py"

PYTHON = "uv"
MATURIN = "uv"

# Source files the agent is allowed to edit
SOURCE_FILES = [
    "src/loss.rs",
    "src/model.rs",
    "src/tree.rs",
    "src/histogram_builder.rs",
]


# ── Compile & benchmark helpers ───────────────────────────────────────────────

def cargo_check() -> tuple[bool, str]:
    """Fast syntax/type check (~5-10s). Run before paying for full build."""
    r = subprocess.run(
        ["cargo", "check", "--lib"],
        cwd=PKBOOST_DIR, capture_output=True, text=True, timeout=120,
    )
    return r.returncode == 0, r.stderr


def maturin_build() -> tuple[bool, str]:
    """Full release build (~60-120s)."""
    r = subprocess.run(
        [MATURIN, "run", "maturin", "develop", "--release"],
        cwd=PKBOOST_DIR, capture_output=True, text=True, timeout=600,
    )
    return r.returncode == 0, r.stderr


def run_benchmark(env_overrides: dict | None = None) -> tuple[dict | None, str]:
    """Run benchmark.py and parse the JSON result from stdout."""
    env = {**os.environ}
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    r = subprocess.run(
        [PYTHON, "run", "python", str(BENCHMARK_SCRIPT)],
        cwd=PKBOOST_DIR, capture_output=True, text=True, timeout=900, env=env,
    )
    if r.returncode != 0:
        return None, r.stderr + r.stdout

    # Grab the last JSON line (training output may have other lines)
    for line in reversed(r.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), ""
            except json.JSONDecodeError:
                pass
    return None, f"No JSON found in output:\n{r.stdout}"


def git_revert_sources():
    """Revert only the editable source files to HEAD."""
    subprocess.run(
        ["git", "checkout", "HEAD", "--"] + SOURCE_FILES,
        cwd=PKBOOST_DIR, check=True,
    )


def read_sources() -> dict[str, str]:
    return {f: (PKBOOST_DIR / f).read_text() for f in SOURCE_FILES}


def apply_change(file_rel: str, find: str, replace: str) -> bool:
    """Apply a find-replace patch to a source file. Returns True on success."""
    path = PKBOOST_DIR / file_rel
    original = path.read_text()
    if find not in original:
        return False
    path.write_text(original.replace(find, replace, 1))
    return True


def log_experiment(record: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Claude API call ────────────────────────────────────────────────────────────

def propose_change(
    client: anthropic.Anthropic,
    program: str,
    sources: dict[str, str],
    best_score: float,
    history: list[dict],
) -> dict:
    """Ask Claude to propose one focused change. Returns parsed proposal."""

    source_block = "\n\n".join(
        f"### {path}\n```rust\n{content}\n```"
        for path, content in sources.items()
    )

    history_block = ""
    if history:
        lines = []
        for e in history[-8:]:
            status = "KEPT ✓" if e.get("kept") else "REVERTED"
            score = e.get("score")
            score_str = f"{score:.5f}" if score is not None else "failed"
            lines.append(f"- [{status}] {e.get('hypothesis', '?')} → PR AUC {score_str}")
        history_block = "\n\nRecent experiment history:\n" + "\n".join(lines)

    prompt = f"""{program}

---

## Current Source Code
{source_block}

## Current Best PR AUC: {best_score:.5f}
{history_block}

---

## Your Task

Propose exactly ONE focused change to improve PR AUC on this benchmark.
Think carefully about what is most likely to help given the extreme class imbalance (3400:1).

Respond in exactly this format — no other text:

HYPOTHESIS: <one sentence explaining what you expect and why>
FILE: <relative path, e.g. src/loss.rs>

FIND:
```rust
<exact existing code block to replace — must match the source exactly, including whitespace>
```

REPLACE:
```rust
<the new code to substitute in>
```
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text

    # Parse the structured response
    hypothesis_m = re.search(r"^HYPOTHESIS:\s*(.+)$", text, re.MULTILINE)
    file_m = re.search(r"^FILE:\s*(.+)$", text, re.MULTILINE)
    find_m = re.search(r"FIND:\s*```rust\s*(.*?)```", text, re.DOTALL)
    replace_m = re.search(r"REPLACE:\s*```rust\s*(.*?)```", text, re.DOTALL)

    return {
        "raw": text,
        "hypothesis": hypothesis_m.group(1).strip() if hypothesis_m else "unknown",
        "file": file_m.group(1).strip() if file_m else None,
        "find": find_m.group(1) if find_m else None,
        "replace": replace_m.group(1) if replace_m else None,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of experiments")
    parser.add_argument("--baseline", action="store_true", help="Just measure baseline")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    program = PROGRAM_FILE.read_text()

    print("=" * 60)
    print("PkBoost Autoresearch Loop")
    print(f"Target: PR AUC on Series A model (3400:1 imbalance)")
    print(f"Log: {LOG_FILE}")
    print("=" * 60)

    # ── Build & get baseline ──────────────────────────────────────────────────
    print("\nBuilding baseline...")
    ok, err = maturin_build()
    if not ok:
        print(f"Build failed:\n{err}")
        sys.exit(1)

    print("Running baseline benchmark...")
    result, err = run_benchmark()
    if result is None:
        print(f"Benchmark failed:\n{err}")
        sys.exit(1)

    best_score = result["pr_auc"]
    print(f"Baseline PR AUC: {best_score:.5f}  (ROC AUC: {result['roc_auc']:.4f})")
    log_experiment({"type": "baseline", "timestamp": datetime.now().isoformat(), **result})

    if args.baseline:
        return

    history = []
    n_kept = 0

    for exp_num in range(1, args.n + 1):
        print(f"\n{'─' * 60}")
        print(f"Experiment {exp_num}/{args.n}  (best so far: {best_score:.5f})")

        sources = read_sources()
        proposal = propose_change(client, program, sources, best_score, history)

        hypothesis = proposal["hypothesis"]
        print(f"Hypothesis: {hypothesis}")

        record: dict = {
            "type": "experiment",
            "exp_num": exp_num,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "file": proposal["file"],
        }

        # Validate proposal
        if not proposal["file"] or not proposal["find"] or not proposal["replace"]:
            print("  SKIP: malformed proposal (missing FILE/FIND/REPLACE)")
            record["outcome"] = "malformed"
            log_experiment(record)
            history.append({**record, "kept": False, "score": None})
            continue

        # Apply the patch
        applied = apply_change(proposal["file"], proposal["find"], proposal["replace"])
        if not applied:
            print("  SKIP: FIND block not found in source (whitespace mismatch?)")
            record["outcome"] = "patch_failed"
            log_experiment(record)
            history.append({**record, "kept": False, "score": None})
            continue

        # Fast compile check
        ok, err = cargo_check()
        if not ok:
            print("  REVERT: cargo check failed")
            record["outcome"] = "compile_error"
            record["error"] = err[:500]
            git_revert_sources()
            log_experiment(record)
            history.append({**record, "kept": False, "score": None})
            continue

        # Full build
        print("  Building (release)...")
        t0 = time.time()
        ok, err = maturin_build()
        build_time = time.time() - t0
        if not ok:
            print(f"  REVERT: maturin build failed ({build_time:.0f}s)")
            record["outcome"] = "build_error"
            record["error"] = err[:500]
            git_revert_sources()
            log_experiment(record)
            history.append({**record, "kept": False, "score": None})
            continue

        # Benchmark
        print(f"  Benchmarking... (build took {build_time:.0f}s)")
        bench_result, err = run_benchmark()
        if bench_result is None:
            print("  REVERT: benchmark failed")
            record["outcome"] = "benchmark_error"
            record["error"] = err[:500]
            git_revert_sources()
            log_experiment(record)
            history.append({**record, "kept": False, "score": None})
            continue

        score = bench_result["pr_auc"]
        delta = score - best_score
        record["score"] = score
        record["delta"] = delta
        record["bench"] = bench_result

        if score > best_score + 1e-5:
            best_score = score
            n_kept += 1
            record["outcome"] = "kept"
            record["kept"] = True
            print(f"  KEPT ✓  PR AUC: {score:.5f}  (+{delta:+.5f})  [{n_kept} improvements so far]")
        else:
            record["outcome"] = "reverted"
            record["kept"] = False
            print(f"  REVERT  PR AUC: {score:.5f}  ({delta:+.5f})")
            git_revert_sources()
            # Rebuild with reverted code so next iteration starts clean
            maturin_build()

        log_experiment(record)
        history.append(record)

    print(f"\n{'=' * 60}")
    print(f"Done. {n_kept} improvements found in {args.n} experiments.")
    print(f"Final best PR AUC: {best_score:.5f}")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
