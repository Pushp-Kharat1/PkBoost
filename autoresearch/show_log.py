#!/usr/bin/env python3
"""Render autoresearch experiment_log.jsonl as a pretty terminal table."""

import json
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent / "experiment_log.jsonl"

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
DIM    = "\033[2m"
CYAN   = "\033[36m"


def load_records():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_FILE
    if not path.exists():
        sys.exit(f"Log not found: {path}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt_delta(delta):
    if delta is None:
        return "      —"
    sign = "+" if delta >= 0 else ""
    color = GREEN if delta > 1e-5 else (RED if delta < -1e-5 else DIM)
    return f"{color}{sign}{delta:+.5f}{RESET}"


def fmt_score(score):
    if score is None:
        return "    failed"
    return f"{score:.5f}"


def fmt_outcome(outcome, kept):
    if outcome == "kept":
        return f"{GREEN}{BOLD}KEPT ✓  {RESET}"
    if outcome == "reverted":
        return f"{RED}reverted{RESET}"
    if outcome == "compile_error":
        return f"{YELLOW}compile ✗{RESET}"
    if outcome == "patch_failed":
        return f"{YELLOW}patch ✗ {RESET}"
    if outcome == "malformed":
        return f"{YELLOW}malform.{RESET}"
    return f"{DIM}{outcome:<8}{RESET}"


def main():
    records = load_records()

    baselines = [r for r in records if r["type"] == "baseline"]
    experiments = [r for r in records if r["type"] == "experiment"]

    # Print baselines
    if baselines:
        b = baselines[-1]  # most recent baseline
        print(f"\n{BOLD}Baseline{RESET}  PR AUC {CYAN}{b['pr_auc']:.5f}{RESET}"
              f"  ROC AUC {b['roc_auc']:.5f}"
              f"  {DIM}{b['timestamp'][:16]}{RESET}")

    if not experiments:
        print("No experiments yet.")
        return

    best = max((e["score"] for e in experiments if e.get("score") is not None), default=None)

    # Column widths
    H_EXP      = " # "
    H_RESULT   = "Result   "
    H_PRAUC    = " PR AUC "
    H_DELTA    = "  Delta  "
    H_FILE     = "File           "
    H_HYP      = "Hypothesis"

    sep = "─"
    print()
    print(f"{BOLD}{H_EXP}  {H_RESULT}  {H_PRAUC}  {H_DELTA}  {H_FILE}  {H_HYP}{RESET}")
    print(sep * 110)

    for e in experiments:
        num     = e.get("exp_num", "?")
        outcome = e.get("outcome", "?")
        kept    = e.get("kept", False)
        score   = e.get("score")
        delta   = e.get("delta")
        file_   = e.get("file") or ""
        hyp     = e.get("hypothesis", "")

        # Truncate hypothesis to fit terminal
        hyp_short = hyp[:62] + "…" if len(hyp) > 63 else hyp

        file_short = file_.replace("src/", "")[:14]

        star = f"{GREEN}★{RESET}" if score is not None and score == best else " "
        print(
            f"{star}{num:>2}  "
            f"{fmt_outcome(outcome, kept)}  "
            f"{fmt_score(score)}  "
            f"{fmt_delta(delta)}  "
            f"{DIM}{file_short:<14}{RESET}  "
            f"{hyp_short}"
        )

    kept_count = sum(1 for e in experiments if e.get("kept"))
    total = len(experiments)
    best_score = max((e["score"] for e in experiments if e.get("score") is not None), default=None)
    print(sep * 110)
    print(f"{BOLD}{kept_count}/{total} improvements kept{RESET}"
          + (f"  │  best PR AUC {GREEN}{BOLD}{best_score:.5f}{RESET}" if best_score else ""))
    print()


if __name__ == "__main__":
    main()
